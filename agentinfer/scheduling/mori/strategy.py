# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""MORI sticky GPU/CPU/Waiting placement over committed server-side facts."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, replace

from agentinfer.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from agentinfer.scheduling.cache_placement import CacheClass, ProgramCacheClassObserver
from agentinfer.scheduling.domain import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramView,
    TokenObservationSource,
)
from agentinfer.scheduling.events import SchedulingEvent, StrategyDiagnostic
from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.mori.capacity import CapacityClaim, CapacityLedger, FootprintProvider
from agentinfer.scheduling.mori.config import MORIConfig
from agentinfer.scheduling.mori.diagnostics import MORIDiagnosticState
from agentinfer.scheduling.mori.factors import MORIGlobalFactors, MORIProgramFactors, MORITier
from agentinfer.scheduling.request_diagnostics import RequestDiagnostics
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from agentinfer.scheduling.strategy import SchedulingCycleOrder, SchedulingStrategy
from agentinfer.scheduling.transitions import TransitionController, TransitionResult

MORIFactors = StrategyFactors[MORIGlobalFactors, MORIProgramFactors]

logger = logging.getLogger(__name__)

# Optional residency commitments remain bounded even with unknown future tools.
_GPU_SERVICE_MAX_SECONDS = 30.0
_GPU_SERVICE_ACTING_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class _VictimPlan:
    """One exact candidate-added GPU plan in legacy demotion order."""

    victims: tuple[ProgramView, ...]
    final_used: int
    overshoot: int
    known_context_cost: int
    unknown_refs: tuple[ProgramRef, ...]


class MORIStrategy(SchedulingStrategy[MORIGlobalFactors, MORIProgramFactors]):
    """Periodic relative-idleness placement with current-cycle waiting compensation.

    The strategy does not predict a future return, prefetch KV, or perform a
    transfer. It commits logical protection classes; native vLLM remains the
    owner of allocation, on-demand loading, offload jobs, fences, and physical
    safety validation.
    """

    def __init__(self, config: MORIConfig) -> None:
        self.config = config
        self.counters: Counter[str] = Counter()
        self.decision_cpu_seconds = 0.0
        self.batch_policy_cpu_seconds = 0.0
        self.writeback_audit_cpu_seconds = 0.0
        self.writeback_capture_cpu_seconds = 0.0
        self.repair_ledger_cpu_seconds = 0.0
        self.replacement_planner_cpu_seconds = 0.0
        self.replacement_telemetry_cpu_seconds = 0.0
        self._factors: MORIFactors | None = None
        self._classes: dict[ProgramRef, CacheClass] = {}
        self._cache_class_epoch = 0
        self._observers: list[ProgramCacheClassObserver] = []
        self._footprints: FootprintProvider | None = None
        self._emit_replacement_telemetry = False
        self._slow_diagnostics = (
            MORIDiagnosticState(self, config.slow_request_threshold_seconds)
            if config.slow_request_threshold_seconds > 0
            else None
        )
        self._request_diagnostics = (
            RequestDiagnostics(
                threshold_seconds=config.slow_request_threshold_seconds,
                repeat_seconds=config.slow_request_repeat_seconds,
                context_provider=self._slow_diagnostics.context,
            )
            if self._slow_diagnostics is not None
            else None
        )

    def request_diagnostics(self) -> RequestDiagnostics | None:
        """Return the one bounded lifecycle tracker, or None when disabled."""
        return self._request_diagnostics

    def bind_observability(self, enabled: bool) -> None:
        """Emit per-plan evidence only under the runtime's explicit opt-in."""
        self._emit_replacement_telemetry = enabled

    @property
    def uses_request_branches(self) -> bool:
        """Use an independent server-owned branch for each overlapping request."""
        return True

    def bind_footprints(self, provider: FootprintProvider | None) -> None:
        """Attach native block evidence, or detach it at adapter shutdown."""
        self._footprints = provider

    def on_context_pending(self, program: ProgramRef) -> None:
        """Drop stale capacity proof without exposing the prior cache while this request waits."""
        if self._slow_diagnostics is not None:
            self._slow_diagnostics.release(program)
        if self._footprints is not None:
            self._footprints.invalidate_footprint(program)
        if self._factors is not None:
            self._request_schedule(self._factors, time.monotonic())

    def on_context_replaced(self, program: ProgramRef) -> None:
        """Atomically hand typed cache ownership to the prompt entering native scheduling."""
        if self._footprints is not None:
            self._footprints.replace_context(program)
        if self._factors is not None:
            self._request_schedule(self._factors, time.monotonic())

    def on_context_growth(
        self, program: ProgramRef, now: float, *, previous_tokens: int, current_tokens: int, block_size: int
    ) -> None:
        """Wake only when rounded context plus decode headroom grows, preserving block proof."""
        buffer = self.config.decode_buffer_tokens
        if self._factors is not None and self._round_up(previous_tokens + buffer, block_size) < self._round_up(
            current_tokens + buffer, block_size
        ):
            self._request_schedule(self._factors, now)

    def on_execution_changed(self, factors: MORIFactors, program: ProgramRef, now: float, *, running: bool) -> None:
        """Accumulate native execution intervals, excluding admission and preemption waits."""
        sidecar = factors.for_program(program)
        if sidecar is None:
            return
        if running:
            if (
                self.config.gpu_service_requests > 0
                and sidecar.gpu_service_started_at_monotonic_s is None
                and sidecar.tier is MORITier.GPU
            ):
                sidecar = replace(sidecar, gpu_service_started_at_monotonic_s=now)
                self.counters["gpu_service_started"] += 1
            if sidecar.reasoning_wait_seconds or sidecar.reasoning_wait_started_at_monotonic_s is not None:
                sidecar = replace(sidecar, reasoning_wait_seconds=0.0, reasoning_wait_started_at_monotonic_s=None)
            if not sidecar.execution_observed:
                sidecar = replace(
                    sidecar,
                    execution_observed=True,
                    pending_reasoning_seconds=0.0,
                    reasoning_started_at_monotonic_s=now,
                )
            elif sidecar.reasoning_started_at_monotonic_s is None:
                sidecar = replace(sidecar, reasoning_started_at_monotonic_s=now)
        elif sidecar.execution_observed and sidecar.reasoning_started_at_monotonic_s is not None:
            sidecar = replace(
                sidecar,
                pending_reasoning_seconds=sidecar.pending_reasoning_seconds
                + max(0.0, now - sidecar.reasoning_started_at_monotonic_s),
                reasoning_started_at_monotonic_s=None,
            )
        self._set_factor(factors, program, sidecar)

    @property
    def cycle_order(self) -> SchedulingCycleOrder:
        return SchedulingCycleOrder.REPAIR_THEN_RESUME

    @property
    def requires_program_identity(self) -> bool:
        return True

    @property
    def observes_streaming_context_growth(self) -> bool:
        """Account for generated tokens when repairing MORI reservations."""
        return True

    @property
    def cache_class_epoch(self) -> int:
        return self._cache_class_epoch

    def bind_factors(self, factors: MORIFactors) -> None:
        """Bind the one factor collection owned by the containing runtime."""
        self._factors = factors
        self._classes = {ref: self._class_for_tier(sidecar.tier) for ref, sidecar in factors.program_factors}

    def subscribe_cache_class_observer(self, observer: ProgramCacheClassObserver) -> None:
        """Register an engine-thread observer once."""
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe_cache_class_observer(self, observer: ProgramCacheClassObserver) -> None:
        """Remove an observer without disturbing other native policy indexes."""
        if observer in self._observers:
            self._observers.remove(observer)

    def cache_class_for_program(self, program: ProgramRef) -> CacheClass:
        """Return the committed class, failing safe to INACTIVE for an unknown generation."""
        return self._classes.get(program, CacheClass.INACTIVE)

    def idleness_for_program(self, program: ProgramRef, now_monotonic_s: float | None = None) -> float:
        """Return causal relative idleness for diagnostics and native policy tests."""
        factors = self._factors
        if factors is None:
            return 0.0
        sidecar = factors.for_program(program)
        if sidecar is None:
            return 0.0
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        return self._idleness(sidecar, now)

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        if self._slow_diagnostics is not None:
            self._slow_diagnostics.observe(snapshot)
        started = time.thread_time()
        try:
            view = snapshot.program(candidate)
            if view is None:
                raise ValueError("MORI admission candidate is absent from the snapshot")
            sidecar = self._factor(strategy_factors, candidate)
            # Ingress is the only implicit owner-creation boundary. Reads of an
            # old snapshot must never recreate a retired owner's accounting.
            if view.state is ProgramState.PAUSED and view.status is ProgramStatus.REASONING:
                sidecar = replace(
                    sidecar,
                    reasoning_wait_started_at_monotonic_s=(
                        view.wait_started_at_monotonic_s
                        if view.wait_started_at_monotonic_s is not None
                        else snapshot.observed_at_monotonic_s
                    ),
                )
            self._set_factor(strategy_factors, candidate, sidecar)
            required = self._gpu_target(view, snapshot)

            if view.state is ProgramState.ACTIVE and sidecar.tier is MORITier.GPU:
                # GPU membership is the admission decision. Growth is repaired by
                # the placement loop; never retain an unstarted request behind a
                # second capacity gate while still charging it to the GPU tier.
                self._request_schedule(strategy_factors, snapshot.observed_at_monotonic_s)
                self.counters["admission_existing_gpu"] += 1
                return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_existing_gpu", required_tokens=required)

            # Ingress uses only existing slack. Capacity repair is coalesced
            # into the host's next planning cycle, never repeated per waiter.
            self._request_schedule(strategy_factors, snapshot.observed_at_monotonic_s)
            gpu, _cpu = self._ledgers(snapshot, strategy_factors)
            other_waiters = any(
                p.ref != candidate and p.state is ProgramState.PAUSED and p.status is ProgramStatus.REASONING
                for p in snapshot.programs
            )
            if not other_waiters and gpu.used + gpu.incremental(
                self._claim(view, snapshot, gpu=True)
            ) <= self._gpu_quota(snapshot):
                result = transitions.admit(candidate, reason="mori_free_capacity", backend_id=snapshot.backend_id)
                self._require_applied(result)
                return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_free_capacity", required_tokens=required)

            available = self._gpu_quota(snapshot) - gpu.used
            result = transitions.queue(candidate, reason="mori_capacity_queue")
            self._require_applied(result)
            self.counters["admission_queued"] += 1
            return AdmissionOutcome(
                AdmissionDisposition.QUEUED,
                "mori_capacity_queue",
                required_tokens=required,
                deficit_tokens=max(0, required - max(0, available)),
                transitioned_programs=(candidate,),
            )
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
    ) -> None:
        if self._slow_diagnostics is not None:
            self._slow_diagnostics.observe(snapshot)
        started = time.thread_time()
        try:
            demoted = self._select_pending(snapshot, strategy_factors, transitions)
            self._retire_inactive(snapshot, strategy_factors, transitions)
            # The input snapshot predates these accepted pauses. Retire only
            # after all planning scans, using their committed post-pause state.
            for program in demoted:
                self._retire_program(program, strategy_factors, transitions)
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def _select_pending(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
    ) -> tuple[ProgramView, ...]:
        """Rebalance ready inference against idle GPU occupancy each round.

        Prefer slack, then reclaim strictly more idle Acting owners. If no GPU
        Reasoning can make progress, permit a feasible request to reclaim Acting
        space even at a score tie or disadvantage. This prevents an idle engine
        from waiting on its own cached parents. Never preempt Reasoning here.
        """
        candidates = [
            p
            for p in snapshot.programs
            if p.state is ProgramState.PAUSED and p.status is ProgramStatus.REASONING and not p.marked_for_pause
        ]
        if not candidates:
            return ()
        gpu, cpu = self._ledgers(snapshot, strategy_factors)
        quota = self._gpu_quota(snapshot)
        if gpu.used > quota:
            self.counters["promotion_blocked_capacity_debt"] += 1
            if self._slow_diagnostics is not None:
                for candidate in candidates:
                    decision = self._slow_diagnostics.begin(
                        candidate,
                        snapshot,
                        strategy_factors,
                        gpu,
                        None,
                        rank=None,
                        admitted_before=0,
                    )
                    if decision is not None:
                        decision["reason"] = "global_capacity_debt"
            return ()
        now = snapshot.observed_at_monotonic_s
        candidates.sort(key=lambda p: self._promotion_key(p, strategy_factors, snapshot, now))
        reserved = self._aged_waiter(snapshot)
        if reserved is not None:
            candidates.sort(key=lambda p: p.ref != reserved.ref)
        victims = [
            p
            for p in snapshot.programs
            if p.ref in gpu.claims
            and p.state is ProgramState.ACTIVE
            and p.status is ProgramStatus.ACTING
            and not p.marked_for_pause
            and self._factor(strategy_factors, p.ref).tier is MORITier.GPU
            and self._factor(strategy_factors, p.ref).pending_tier is None
            and not self._gpu_service_protected(p, strategy_factors, now)
        ]
        victims.sort(key=lambda p: self._demotion_key(p, strategy_factors, now))
        victim_idleness = {
            victim.ref: self._idleness(self._factor(strategy_factors, victim.ref), now) for victim in victims
        }
        exchange_valid = True
        demoted: list[ProgramView] = []
        admitted_before = 0
        for rank, candidate in enumerate(candidates):
            self.counters["planning_candidates"] += 1
            exchange_victims: list[ProgramView] = []
            placements: tuple[tuple[ProgramRef, str], ...] = ()
            result: TransitionResult | None = None
            plan_details: dict[str, object] | None = None
            capture_plan_details = False
            claim = self._claim(candidate, snapshot, gpu=True)
            decision = (
                self._slow_diagnostics.begin(
                    candidate, snapshot, strategy_factors, gpu, claim, rank=rank, admitted_before=admitted_before
                )
                if self._slow_diagnostics is not None
                else None
            )
            if claim.tokens > quota:
                self.counters["promotion_exceeds_total_quota"] += 1
                if decision is not None:
                    decision["reason"] = "claim_exceeds_total_quota"
                continue
            reason = "mori_relative_rebalance"
            if gpu.used + gpu.incremental(claim) > quota:
                if not exchange_valid:
                    self.counters["promotion_no_fit"] += 1
                    if decision is not None:
                        decision["reason"] = "exchange_invalid"
                    continue
                candidate_idle = self._idleness(self._factor(strategy_factors, candidate.ref), now)
                acting = [victim for victim in victims if victim.ref in gpu.claims]
                eligible = [victim for victim in acting if victim_idleness[victim.ref] > candidate_idle]
                if reserved is not None and candidate.ref == reserved.ref:
                    eligible = acting
                    reason = "mori_ready_reservation"
                has_reasoning = any(
                    program.ref in gpu.claims and program.status is ProgramStatus.REASONING
                    for program in snapshot.programs
                )
                candidate_ledger = gpu.copy()
                candidate_ledger.add(candidate.ref, claim)
                relative_required: int | None = None
                all_acting_required: int | None = None
                if decision is not None:
                    relative_required = self._used_after_removing(candidate_ledger, eligible)
                    all_acting_required = self._used_after_removing(candidate_ledger, acting)
                    decision["relief"] = {
                        "eligible_acting_prefix": len(eligible),
                        "all_acting_count": len(acting),
                        "already_removed_acting_count": 0,
                        "required_after_eligible_tokens": relative_required,
                        "required_after_all_acting_tokens": all_acting_required,
                        "has_gpu_reasoning": has_reasoning,
                        "idle_gpu_progress_override": not has_reasoning and relative_required > quota,
                        "exchange_valid": exchange_valid,
                    }
                planner_started = time.thread_time()
                legacy = self._legacy_victim_plan(candidate_ledger, eligible, quota)
                if legacy is None and not has_reasoning:
                    eligible = acting
                    reason = "mori_idle_gpu_progress"
                    legacy = self._legacy_victim_plan(candidate_ledger, eligible, quota)
                if legacy is None:
                    planner_elapsed = time.thread_time() - planner_started
                    self.replacement_planner_cpu_seconds += planner_elapsed
                    self.counters["promotion_no_fit"] += 1
                    if decision is not None:
                        assert all_acting_required is not None
                        decision["reason"] = "idleness_limited" if all_acting_required <= quota else "no_all_acting_fit"
                        decision["replacement_plan"] = {
                            "candidate_ref": self._ref_text(candidate.ref),
                            "candidate_rank": rank,
                            "eligibility_mode": "relative" if has_reasoning else "idle_gpu_progress",
                            "eligible_refs": [self._ref_text(victim.ref) for victim in eligible],
                            "eligible_count": len(eligible),
                            "candidate_added_used_tokens": candidate_ledger.used,
                            "quota_tokens": quota,
                            "fallback_reason": "no_feasible_legacy_prefix",
                            "planner_cpu_seconds": planner_elapsed,
                        }
                    if reserved is not None and candidate.ref == reserved.ref:
                        # Do not spend capacity being drained for this ready
                        # request on a younger, smaller continuation.
                        self.counters["ready_reservation_blocked_cycles"] += 1
                        break
                    continue
                self.counters["replacement_plans_considered"] += 1
                capture_plan_details = decision is not None or self._emit_replacement_telemetry
                chosen, plan_details = self._bounded_replacement_plan(
                    candidate_ledger,
                    eligible,
                    legacy,
                    quota,
                    strategy_factors,
                    capture_trials=capture_plan_details,
                )
                planner_elapsed = time.thread_time() - planner_started
                self.replacement_planner_cpu_seconds += planner_elapsed
                exchange_victims.extend(chosen.victims)
                if capture_plan_details:
                    plan_details["planner_cpu_seconds"] = planner_elapsed
                    plan_details["eligibility_mode"] = (
                        "idle_gpu_progress" if reason == "mori_idle_gpu_progress" else "relative"
                    )
                    plan_details["has_gpu_reasoning"] = has_reasoning
                    plan_details["candidate_ref"] = self._ref_text(candidate.ref)
                    plan_details["candidate_rank"] = rank
                    plan_details["candidate_added_used_tokens"] = candidate_ledger.used
                    plan_details["quota_tokens"] = quota
                if decision is not None:
                    decision["replacement_plan"] = plan_details
            transition_error: str | None = None
            try:
                for victim in exchange_victims:
                    self.counters["planning_victims"] += 1
                    if not self._pause_for_exchange(victim, strategy_factors, transitions, gpu, reason=reason):
                        # Accepted earlier pauses remain real; a rejected one
                        # invalidates speculative exchanges for the rest of the
                        # round. Subsequent admissions may still use real slack.
                        exchange_valid = False
                        self.counters["replacement_transition_rejections"] += 1
                        break
                    demoted.append(replace(victim, state=ProgramState.PAUSED, backend_id=None, marked_for_pause=False))
                    self.counters["rebalance_demotions"] += 1
                    if reason == "mori_idle_gpu_progress":
                        self.counters["idle_gpu_progress_demotions"] += 1
                if gpu.used + gpu.incremental(claim) > quota:
                    if decision is not None:
                        decision["reason"] = "demotion_rejected"
                else:
                    result = (transitions.resume if candidate.step_count else transitions.admit)(
                        candidate.ref, reason="mori_cycle_promote", backend_id=snapshot.backend_id
                    )
                    if result.applied:
                        gpu.add(candidate.ref, claim)
                        cpu.remove(candidate.ref)
                        self.counters["promotion_applied"] += 1
                        admitted_before += 1
                    elif exchange_victims:
                        self.counters["replacement_transition_rejections"] += 1
            except Exception as error:
                transition_error = f"{type(error).__name__}: {error}"
                if exchange_victims:
                    self.counters["replacement_transition_exceptions"] += 1
                raise
            finally:
                # Spend CPU relief only after admission commits. Rejected
                # promotions keep their CPU claim; accepted pauses still need
                # a destination based on that unchanged committed capacity.
                # Only victims whose pause transition applied have left ``gpu``.
                accepted_victims = [victim for victim in exchange_victims if victim.ref not in gpu.claims]
                placements = self._place_exchange_victims(accepted_victims, snapshot, strategy_factors, cpu)
                if capture_plan_details and plan_details is not None:
                    plan_details["transition_rejected"] = (
                        transition_error is not None or result is None or not result.applied
                    )
                    plan_details["transition_applied"] = result is not None and result.applied
                    plan_details["transition_reason"] = (
                        result.reason
                        if result is not None
                        else "transition_exception"
                        if transition_error is not None
                        else "demotion_rejected"
                    )
                    plan_details["transition_exception"] = transition_error
                    plan_details["final_placements"] = [
                        {"ref": self._ref_text(ref), "tier": tier} for ref, tier in placements
                    ]
                    if self._emit_replacement_telemetry:
                        telemetry_started = time.thread_time()
                        encoded_plan = json.dumps(plan_details, sort_keys=True, separators=(",", ":"))
                        logger.info("MORI replacement plan %s", encoded_plan)
                        self.replacement_telemetry_cpu_seconds += time.thread_time() - telemetry_started
                        self.counters["replacement_telemetry_records"] += 1
                        self.counters["replacement_telemetry_bytes"] += len(encoded_plan.encode("utf-8"))
            if decision is not None:
                decision["reason"] = (
                    "promotion_applied" if result is not None and result.applied else "promotion_rejected"
                )
                decision["transition_reason"] = result.reason if result is not None else "demotion_rejected"
            if reserved is not None and candidate.ref == reserved.ref and (result is None or not result.applied):
                break
        return tuple(demoted)

    @staticmethod
    def _ref_text(ref: ProgramRef) -> str:
        return f"{ref.program_id}:{ref.generation}"

    @staticmethod
    def _backend_context_known(victim: ProgramView) -> bool:
        return victim.tokens.source is TokenObservationSource.BACKEND_USAGE

    def _victim_observation(self, factors: MORIFactors, victim: ProgramView) -> dict[str, object]:
        """Return production cost and cache shadow facts without affecting selection."""
        sidecar = self._factor(factors, victim.ref)
        return {
            "ref": self._ref_text(victim.ref),
            "source": victim.tokens.source.value,
            "context_tokens": victim.tokens.estimated_context_tokens,
            "hbm_cached_prefix_exact_present": sidecar.last_hbm_cached_prefix_exact_present,
            "hbm_cached_prefix_tokens": (
                sidecar.last_hbm_cached_prefix_tokens if sidecar.last_hbm_cached_prefix_exact_present else None
            ),
            "hbm_hit_ref_zero_exact_present": sidecar.last_hbm_hit_ref_zero_exact_present,
            "hbm_hit_ref_zero_tokens": (
                sidecar.last_hbm_hit_ref_zero_tokens if sidecar.last_hbm_hit_ref_zero_exact_present else None
            ),
        }

    @classmethod
    def _plan_from_ledger(
        cls,
        victims: tuple[ProgramView, ...],
        ledger: CapacityLedger,
        quota: int,
    ) -> _VictimPlan:
        return _VictimPlan(
            victims=victims,
            final_used=ledger.used,
            overshoot=quota - ledger.used,
            known_context_cost=sum(
                victim.tokens.estimated_context_tokens for victim in victims if cls._backend_context_known(victim)
            ),
            unknown_refs=tuple(victim.ref for victim in victims if not cls._backend_context_known(victim)),
        )

    @classmethod
    def _legacy_victim_plan(
        cls,
        candidate_ledger: CapacityLedger,
        eligible: list[ProgramView],
        quota: int,
    ) -> _VictimPlan | None:
        """Build the exact first feasible prefix without skipping zero-relief owners."""
        projected = candidate_ledger.copy()
        selected: list[ProgramView] = []
        for victim in eligible:
            selected.append(victim)
            projected.remove(victim.ref)
            if projected.used <= quota:
                return cls._plan_from_ledger(tuple(selected), projected, quota)
        return None

    @staticmethod
    def _used_after_removing(candidate_ledger: CapacityLedger, victims: list[ProgramView]) -> int:
        projected = candidate_ledger.copy()
        for victim in victims:
            projected.remove(victim.ref)
        return projected.used

    def _bounded_replacement_plan(
        self,
        candidate_ledger: CapacityLedger,
        eligible: list[ProgramView],
        legacy: _VictimPlan,
        quota: int,
        factors: MORIFactors,
        *,
        capture_trials: bool,
    ) -> tuple[_VictimPlan, dict[str, object]]:
        """Try deterministic known-context swaps inside the legacy safety envelope."""
        legacy_refs = {victim.ref for victim in legacy.victims}
        selected = {victim.ref: victim for victim in legacy.victims}
        legacy_index = {victim.ref: index for index, victim in enumerate(eligible)}
        planned = candidate_ledger.copy()
        for victim in legacy.victims:
            planned.remove(victim.ref)
        trials: list[dict[str, object]] = []
        swaps: list[dict[str, object]] = []
        accepted_swap_count = 0
        outgoing = sorted(
            (victim for victim in legacy.victims if self._backend_context_known(victim)),
            key=lambda victim: (-victim.tokens.estimated_context_tokens, legacy_index[victim.ref]),
        )
        for outgoing_victim in outgoing:
            if outgoing_victim.ref not in selected:
                continue
            best: tuple[int, int, int, ProgramRef, ProgramView, int] | None = None
            for incoming in eligible:
                if incoming.ref in legacy_refs or incoming.ref in selected:
                    continue
                if not self._backend_context_known(incoming):
                    self.counters["replacement_unknown_skipped"] += 1
                    continue
                incoming_cost = incoming.tokens.estimated_context_tokens
                outgoing_cost = outgoing_victim.tokens.estimated_context_tokens
                if incoming_cost >= outgoing_cost:
                    self.counters["replacement_non_improving_guard"] += 1
                    if incoming_cost == outgoing_cost:
                        self.counters["replacement_equal_cost_guard"] += 1
                    continue
                self.counters["replacement_trials"] += 1
                trial_used = planned.used_after_swap(
                    candidate_ledger.claims[outgoing_victim.ref],
                    incoming.ref,
                )
                trial_overshoot = quota - trial_used
                outcome = "accepted_candidate"
                if trial_used > quota:
                    self.counters["replacement_capacity_guard"] += 1
                    outcome = "capacity_guard"
                elif trial_overshoot > legacy.overshoot:
                    self.counters["replacement_overshoot_guard"] += 1
                    outcome = "overshoot_guard"
                else:
                    key = (
                        incoming_cost,
                        trial_overshoot,
                        legacy_index[incoming.ref],
                        incoming.ref,
                        incoming,
                        trial_used,
                    )
                    if best is None or key[:4] < best[:4]:
                        best = key
                if capture_trials:
                    trials.append(
                        {
                            "outgoing_ref": self._ref_text(outgoing_victim.ref),
                            "incoming_ref": self._ref_text(incoming.ref),
                            "outgoing_context_tokens": outgoing_cost,
                            "incoming_context_tokens": incoming_cost,
                            "final_used_tokens": trial_used,
                            "overshoot_tokens": trial_overshoot,
                            "outcome": outcome,
                        }
                    )
            if best is None:
                continue
            incoming = best[4]
            planned.add(outgoing_victim.ref, candidate_ledger.claims[outgoing_victim.ref])
            planned.remove(incoming.ref)
            del selected[outgoing_victim.ref]
            selected[incoming.ref] = incoming
            accepted_swap_count += 1
            if capture_trials:
                swaps.append(
                    {
                        "outgoing_ref": self._ref_text(outgoing_victim.ref),
                        "incoming_ref": self._ref_text(incoming.ref),
                        "outgoing_context_tokens": outgoing_victim.tokens.estimated_context_tokens,
                        "incoming_context_tokens": incoming.tokens.estimated_context_tokens,
                        "final_used_tokens": planned.used,
                        "overshoot_tokens": quota - planned.used,
                    }
                )

        chosen_victims = tuple(victim for victim in eligible if victim.ref in selected)
        chosen = self._plan_from_ledger(chosen_victims, planned, quota)
        changed = tuple(victim.ref for victim in chosen.victims) != tuple(victim.ref for victim in legacy.victims)
        guard_ok = (
            len(chosen.victims) == len(legacy.victims)
            and chosen.final_used <= quota
            and chosen.overshoot <= legacy.overshoot
            and (not changed or chosen.known_context_cost < legacy.known_context_cost)
            and chosen.unknown_refs == legacy.unknown_refs
        )
        fallback_reason = "no_improving_swap"
        if changed and guard_ok:
            self.counters["replacement_plans_changed"] += 1
            self.counters["replacement_swaps"] += accepted_swap_count
            fallback_reason = "changed"
        else:
            self.counters["replacement_plans_legacy_fallback"] += 1
            if not guard_ok:
                self.counters["replacement_internal_guard_fallback"] += 1
                fallback_reason = "internal_guard_failed"
            chosen = legacy
            swaps = []
            changed = False
        reasoning_victim_count = sum(victim.status is ProgramStatus.REASONING for victim in chosen.victims)
        self.counters["replacement_reasoning_victims"] += reasoning_victim_count
        if not capture_trials:
            return chosen, {}
        details: dict[str, object] = {
            "eligible_refs": [self._ref_text(victim.ref) for victim in eligible],
            "eligible_count": len(eligible),
            "victim_observations": [self._victim_observation(factors, victim) for victim in eligible],
            "legacy_refs": [self._ref_text(victim.ref) for victim in legacy.victims],
            "legacy_count": len(legacy.victims),
            "legacy_final_used_tokens": legacy.final_used,
            "legacy_overshoot_tokens": legacy.overshoot,
            "legacy_known_context_cost": legacy.known_context_cost,
            "chosen_refs": [self._ref_text(victim.ref) for victim in chosen.victims],
            "chosen_count": len(chosen.victims),
            "chosen_final_used_tokens": chosen.final_used,
            "chosen_overshoot_tokens": chosen.overshoot,
            "chosen_known_context_cost": chosen.known_context_cost,
            "known_context_cost_delta": chosen.known_context_cost - legacy.known_context_cost,
            "changed": changed,
            "fallback_reason": fallback_reason,
            "replacement_trials": trials,
            "accepted_swaps": swaps,
            "reasoning_victim_count": reasoning_victim_count,
        }
        return chosen, details

    def _pause_for_exchange(
        self,
        victim: ProgramView,
        factors: MORIFactors,
        transitions: TransitionController,
        gpu: CapacityLedger,
        *,
        reason: str,
    ) -> bool:
        """Pause an Acting lease while retaining protection until the exchange resolves.

        The caller holds the scheduling lock for this synchronous hook. Keep
        the GPU cache class through the pause so observers cannot retire the
        owner or initiate a CPU retry using a speculative destination. Native
        blocks and transfer pins remain entirely owned by the backend.
        """
        previous = self._factor(factors, victim.ref)
        self._set_factor(factors, victim.ref, replace(previous, pending_tier=MORITier.GPU))
        try:
            result = transitions.pause(victim.ref, reason=reason)
        except Exception:
            self._set_factor(factors, victim.ref, previous)
            raise
        if not result.applied:
            self._set_factor(factors, victim.ref, previous)
            self.counters["demotion_rejected"] += 1
            return False
        gpu.remove(victim.ref)
        return True

    def _place_exchange_victims(
        self,
        victims: list[ProgramView],
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        cpu: CapacityLedger,
    ) -> tuple[tuple[ProgramRef, str], ...]:
        """Commit paused victims against CPU capacity after the admission result."""
        placements: list[tuple[ProgramRef, str]] = []
        for victim in victims:
            claim = self._claim(victim, snapshot, gpu=False)
            target, replacements = self._cpu_destination(victim, snapshot, factors, cpu, claim)
            sidecar = self._factor(factors, victim.ref)
            self._set_factor(factors, victim.ref, replace(sidecar, tier=target, pending_tier=None))
            self._commit_cpu_destination(victim, factors, cpu, claim, target, replacements)
            placements.append((victim.ref, target.value))
        return tuple(placements)

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
    ) -> None:
        if self._slow_diagnostics is not None:
            self._slow_diagnostics.observe(snapshot)
        started = time.thread_time()
        try:
            self.counters["repair_cycles"] += 1
            reads_before = self.counters["planning_program_reads"]
            ledger_started = time.thread_time()
            try:
                gpu, cpu = self._ledgers(snapshot, strategy_factors)
            finally:
                self.repair_ledger_cpu_seconds += time.thread_time() - ledger_started
                self.counters["repair_programs_observed"] += self.counters["planning_program_reads"] - reads_before
            self.counters["repair_ledger_builds"] += 1
            self._repair_cpu_capacity(snapshot, strategy_factors, cpu)
            self.counters["repair_program_reads_avoided"] += len(snapshot.programs)
            self._repair_gpu_capacity(snapshot, strategy_factors, transitions, gpu, cpu)
            self._reserve_aged_waiter(snapshot, strategy_factors, transitions, gpu, cpu)
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        """Forget completed owners only after committed placement leaves both tiers."""
        program = snapshot.program(completed)
        if program is not None:
            self._retire_program(program, strategy_factors, transitions)

    def _retire_inactive(
        self, snapshot: SchedulingSnapshot, factors: MORIFactors, transitions: TransitionController
    ) -> None:
        """Retire at the end of a hook, after all capacity scans have finished."""
        for program in snapshot.programs:
            self._retire_program(program, factors, transitions)

    def _retire_program(self, program: ProgramView, factors: MORIFactors, transitions: TransitionController) -> None:
        sidecar = factors.for_program(program.ref)
        if (
            program.state is ProgramState.PAUSED
            and program.status is ProgramStatus.ACTING
            and sidecar is not None
            and sidecar.tier is MORITier.WAITING
            and sidecar.pending_tier is None
        ):
            result = transitions.retire(program.ref, reason="mori_inactive_acting")
            if result.applied:
                self.counters["owners_retired"] += 1

    def on_cache_pressure(self, strategy_factors: MORIFactors, now_monotonic_s: float) -> None:
        """Wake ordinary placement evaluation; pressure alone does not select a victim."""
        self._request_schedule(strategy_factors, now_monotonic_s)

    def on_schedule_cycle_start(self, snapshot: SchedulingSnapshot, strategy_factors: MORIFactors) -> None:
        del snapshot
        current = strategy_factors.global_factors
        strategy_factors.set_global_factors(
            replace(current, schedule_cycles=current.schedule_cycles + 1, schedule_requested_at=None)
        )
        self.counters["schedule_cycles"] += 1

    def next_check_at(self, strategy_factors: MORIFactors) -> float | None:
        """Wake on demand or capacity changes without waiting for a periodic tick."""
        return strategy_factors.global_factors.schedule_requested_at

    @staticmethod
    def _request_schedule(strategy_factors: MORIFactors, now: float) -> None:
        current = strategy_factors.global_factors
        previous = current.schedule_requested_at
        strategy_factors.set_global_factors(
            replace(current, schedule_requested_at=now if previous is None else min(previous, now))
        )

    def diagnostics(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
    ) -> tuple[StrategyDiagnostic, ...]:
        now = snapshot.observed_at_monotonic_s
        tiers = Counter(self._factor(strategy_factors, program.ref).tier.value for program in snapshot.programs)
        gpu_reserved = self._gpu_reserved_tokens(snapshot, strategy_factors)
        cpu_reserved = self._cpu_reserved_tokens(snapshot, strategy_factors)
        prompt_tokens = self.counters["cache_prompt_tokens"]
        prefix_hit_tokens = self.counters["cache_prefix_hit_tokens"]
        hbm_hit_tokens = self.counters["cache_hbm_hit_tokens"]
        mean_idleness = sum(
            self._idleness(self._factor(strategy_factors, program.ref), now) for program in snapshot.programs
        ) / max(1, len(snapshot.programs))
        return (
            StrategyDiagnostic(
                "mori_placement",
                "periodic_relative_idleness",
                (
                    ("mode", self.config.mode.value),
                    ("window_size", self.config.window_size),
                    ("enable_wait_credit", self.config.enable_wait_credit),
                    ("ready_wait_limit_seconds", self.config.ready_wait_limit_seconds),
                    ("gpu_service_requests", self.config.gpu_service_requests),
                    ("history_gated_gpu_service", self.config.history_gated_gpu_service),
                    ("decode_batch_tokens", self.config.decode_batch_tokens),
                    ("batch_policy_cpu_seconds", self.batch_policy_cpu_seconds),
                    ("audit_writeback_sources", self.config.audit_writeback_sources),
                    ("writeback_audit_cpu_seconds", self.writeback_audit_cpu_seconds),
                    ("repair_writeback_endpoints", self.config.repair_writeback_endpoints),
                    ("writeback_capture_cpu_seconds", self.writeback_capture_cpu_seconds),
                    ("fast_capacity_accounting", self.config.fast_capacity_accounting),
                    ("footprint_snapshot_hits", getattr(self._footprints, "footprint_snapshot_hits", 0)),
                    ("footprint_snapshot_misses", getattr(self._footprints, "footprint_snapshot_misses", 0)),
                    ("decode_buffer_tokens", self.config.decode_buffer_tokens),
                    ("gpu_programs", tiers[MORITier.GPU.value]),
                    ("cpu_programs", tiers[MORITier.CPU.value]),
                    ("waiting_programs", tiers[MORITier.WAITING.value]),
                    ("gpu_reserved_tokens", gpu_reserved),
                    ("gpu_quota_tokens", snapshot.total_kv_tokens),
                    ("gpu_capacity_debt_tokens", max(0, gpu_reserved - self._gpu_quota(snapshot))),
                    ("cpu_reserved_tokens", cpu_reserved),
                    ("cpu_quota_tokens", snapshot.total_dram_kv_tokens),
                    ("native_hbm_used_tokens", snapshot.native_used_kv_tokens),
                    ("native_dram_used_tokens", snapshot.used_dram_kv_tokens),
                    ("pending_h2d_tokens", snapshot.pending_h2d_kv_tokens),
                    ("pending_d2h_tokens", snapshot.pending_d2h_kv_tokens),
                    ("cache_observed_requests", self.counters["cache_observed_requests"]),
                    ("cache_prompt_tokens", prompt_tokens),
                    ("cache_prefix_hit_tokens", prefix_hit_tokens),
                    ("cache_hbm_hit_tokens", hbm_hit_tokens),
                    ("cache_hbm_ref_zero_hit_tokens", self.counters["cache_hbm_ref_zero_hit_tokens"]),
                    ("cache_prefix_hit_ratio", prefix_hit_tokens / prompt_tokens if prompt_tokens else 0.0),
                    ("cache_hbm_hit_ratio", hbm_hit_tokens / prompt_tokens if prompt_tokens else 0.0),
                    ("mean_idleness", mean_idleness),
                    (
                        "max_wait_seconds",
                        max(
                            (
                                self._wait_seconds(program, now)
                                for program in snapshot.programs
                                if program.state is ProgramState.PAUSED and program.status is ProgramStatus.REASONING
                            ),
                            default=0.0,
                        ),
                    ),
                    ("cache_class_epoch", self.cache_class_epoch),
                    ("decision_cpu_seconds", self.decision_cpu_seconds),
                    ("repair_ledger_cpu_seconds", self.repair_ledger_cpu_seconds),
                    ("replacement_planner_cpu_seconds", self.replacement_planner_cpu_seconds),
                    ("replacement_telemetry_cpu_seconds", self.replacement_telemetry_cpu_seconds),
                    *(
                        (name, self.counters[name])
                        for name in (
                            "planning_program_reads",
                            "repair_cycles",
                            "repair_ledger_builds",
                            "repair_programs_observed",
                            "repair_program_reads_avoided",
                            "planning_candidates",
                            "planning_victims",
                            "promotion_no_fit",
                            "promotion_exceeds_total_quota",
                            "promotion_applied",
                            "gpu_debt_waiting_lazy_relief",
                            "demotion_to_cpu",
                            "demotion_to_waiting",
                            "cpu_relative_replacements",
                            "owners_retired",
                            "rebalance_demotions",
                            "idle_gpu_progress_demotions",
                            "ready_reservation_cycles",
                            "ready_reservation_marks",
                            "ready_reservation_pauses",
                            "ready_reservation_blocked_cycles",
                            "ready_reservation_transition_exceptions",
                            "gpu_service_started",
                            "gpu_service_completed_requests",
                            "gpu_service_protection_checks",
                            "gpu_service_history_eligible_checks",
                            "gpu_service_history_rejected_checks",
                            "decode_batch_capped_iterations",
                            "decode_batch_reordered_iterations",
                            "decode_batch_assigned_tokens",
                            "decode_batch_max_budget_tokens",
                            "decode_batch_native_ceiling_tokens",
                            "decode_batch_max_assigned_tokens",
                            "decode_batch_native_errors",
                            "writeback_audit_snapshot_intents",
                            "writeback_audit_candidates",
                            "writeback_audit_missing_before",
                            "writeback_audit_matched_intents",
                            "writeback_audit_matched_missing",
                            "writeback_audit_sources_lost",
                            "writeback_audit_lost_with_capacity",
                            "writeback_audit_zero_capacity_intents",
                            "writeback_audit_cancelled_intents",
                            "writeback_audit_unmatched_intents",
                            "writeback_audit_submitted_blocks",
                            "writeback_audit_native_errors",
                            "writeback_audit_completed_blocks",
                            "writeback_audit_completed_noninitial_blocks",
                            "writeback_audit_abandoned_blocks",
                            "writeback_audit_invalid_completions",
                            "writeback_capture_requests",
                            "writeback_capture_blocks",
                            "writeback_capture_noninitial_blocks",
                            "fast_capacity_ledger_builds",
                            "replacement_plans_considered",
                            "replacement_plans_changed",
                            "replacement_plans_legacy_fallback",
                            "replacement_trials",
                            "replacement_swaps",
                            "replacement_unknown_skipped",
                            "replacement_non_improving_guard",
                            "replacement_equal_cost_guard",
                            "replacement_capacity_guard",
                            "replacement_overshoot_guard",
                            "replacement_internal_guard_fallback",
                            "replacement_transition_rejections",
                            "replacement_transition_exceptions",
                            "replacement_reasoning_victims",
                            "replacement_telemetry_records",
                            "replacement_telemetry_bytes",
                        )
                    ),
                ),
            ),
        )

    def on_request_pending(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is not None:
            self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)

    def on_program_status_changed(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if (
            event.program is not None
            and event.current_status is ProgramStatus.ACTING
            and event.reason == "request_cancelled"
        ):
            sidecar = self._factor(strategy_factors, event.program)
            elapsed = (
                max(0.0, event.occurred_at_monotonic_s - sidecar.reasoning_started_at_monotonic_s)
                if sidecar.reasoning_started_at_monotonic_s is not None
                else 0.0
            )
            self._set_factor(
                strategy_factors,
                event.program,
                replace(
                    sidecar,
                    pending_reasoning_seconds=(sidecar.pending_reasoning_seconds if sidecar.execution_observed else 0.0)
                    + elapsed,
                    reasoning_started_at_monotonic_s=None,
                    reasoning_wait_seconds=0.0,
                    reasoning_wait_started_at_monotonic_s=None,
                    acting_started_at_monotonic_s=event.occurred_at_monotonic_s,
                ),
            )
            self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
            return
        if event.program is None or event.current_status is not ProgramStatus.REASONING:
            return
        sidecar = self._factor(strategy_factors, event.program)
        cycles = sidecar.cycles
        if sidecar.acting_started_at_monotonic_s is not None:
            acting = max(0.0, event.occurred_at_monotonic_s - sidecar.acting_started_at_monotonic_s)
            cycles = (*cycles, (sidecar.pending_reasoning_seconds, acting))[-self.config.window_size :]
            self.counters["cycles_completed"] += 1
        updated = replace(
            sidecar,
            cycles=cycles,
            pending_reasoning_seconds=0.0,
            reasoning_wait_seconds=0.0,
            reasoning_wait_started_at_monotonic_s=(
                event.occurred_at_monotonic_s if event.current_state is ProgramState.PAUSED else None
            ),
            execution_observed=False,
            acting_started_at_monotonic_s=None,
            reasoning_started_at_monotonic_s=(
                event.occurred_at_monotonic_s if event.current_state is ProgramState.ACTIVE else None
            ),
        )
        self._set_factor(strategy_factors, event.program, updated)

    def on_program_admitted(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        self._on_gpu_transition(strategy_factors, event)

    def on_program_resumed(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        self._on_gpu_transition(strategy_factors, event)

    def on_request_finished(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
        sidecar = self._factor(strategy_factors, event.program)
        latency = self._event_non_negative_float(event.field("request_latency_seconds"))
        scheduler_wait = self._event_non_negative_float(event.field("scheduler_queue_seconds"))
        prompt_tokens = self._event_non_negative_int(event.field("prompt_tokens"))
        prefix_hit_tokens = min(
            prompt_tokens,
            self._event_non_negative_int(event.field("cached_prefix_tokens")),
        )
        hbm_hit_tokens = min(
            prompt_tokens,
            self._event_non_negative_int(event.field("hbm_cached_prefix_tokens")),
        )
        hbm_ref_zero_hit_tokens = min(
            hbm_hit_tokens,
            self._event_non_negative_int(event.field("hbm_hit_ref_zero_tokens")),
        )
        hbm_exact_present = event.field("hbm_cached_prefix_exact_present") is True
        hbm_ref_zero_exact_present = event.field("hbm_hit_ref_zero_exact_present") is True
        self.counters["cache_observed_requests"] += 1
        self.counters["cache_prompt_tokens"] += prompt_tokens
        self.counters["cache_prefix_hit_tokens"] += prefix_hit_tokens
        self.counters["cache_hbm_hit_tokens"] += hbm_hit_tokens
        self.counters["cache_hbm_ref_zero_hit_tokens"] += hbm_ref_zero_hit_tokens
        reasoning_finished = event.current_status is ProgramStatus.ACTING
        reasoning = sidecar.pending_reasoning_seconds
        if reasoning_finished:
            reasoning = (sidecar.pending_reasoning_seconds if sidecar.execution_observed else 0.0) + (
                max(0.0, event.occurred_at_monotonic_s - sidecar.reasoning_started_at_monotonic_s)
                if sidecar.reasoning_started_at_monotonic_s is not None
                else 0.0
                if sidecar.execution_observed
                else max(0.0, latency - scheduler_wait)
            )
        acting_started = event.occurred_at_monotonic_s if reasoning_finished else sidecar.acting_started_at_monotonic_s
        updated = replace(
            sidecar,
            pending_reasoning_seconds=reasoning,
            gpu_service_completed_requests=(
                sidecar.gpu_service_completed_requests + 1
                if self.config.gpu_service_requests > 0
                and sidecar.tier is MORITier.GPU
                and sidecar.gpu_service_started_at_monotonic_s is not None
                and self._event_non_negative_int(event.field("completion_tokens")) > 0
                else sidecar.gpu_service_completed_requests
            ),
            acting_started_at_monotonic_s=acting_started,
            reasoning_started_at_monotonic_s=(None if reasoning_finished else sidecar.reasoning_started_at_monotonic_s),
            reasoning_wait_seconds=(0.0 if reasoning_finished else sidecar.reasoning_wait_seconds),
            reasoning_wait_started_at_monotonic_s=(
                None if reasoning_finished else sidecar.reasoning_wait_started_at_monotonic_s
            ),
            last_hbm_cached_prefix_tokens=hbm_hit_tokens if hbm_exact_present else None,
            last_hbm_hit_ref_zero_tokens=(hbm_ref_zero_hit_tokens if hbm_ref_zero_exact_present else None),
            last_hbm_cached_prefix_exact_present=hbm_exact_present,
            last_hbm_hit_ref_zero_exact_present=hbm_ref_zero_exact_present,
        )
        self._set_factor(strategy_factors, event.program, updated)

        if (
            self.config.gpu_service_requests > 0
            and sidecar.tier is MORITier.GPU
            and sidecar.gpu_service_started_at_monotonic_s is not None
            and self._event_non_negative_int(event.field("completion_tokens")) > 0
        ):
            self.counters["gpu_service_completed_requests"] += 1

    def on_program_paused(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
        sidecar = self._factor(strategy_factors, event.program)
        target = sidecar.pending_tier or MORITier.WAITING
        self._set_factor(
            strategy_factors,
            event.program,
            replace(
                sidecar,
                tier=target,
                pending_tier=None,
                reasoning_started_at_monotonic_s=None,
                reasoning_wait_seconds=0.0,
                reasoning_wait_started_at_monotonic_s=(
                    event.occurred_at_monotonic_s if event.current_status is ProgramStatus.REASONING else None
                ),
            ),
        )

    def on_program_released(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        if self._slow_diagnostics is not None:
            self._slow_diagnostics.release(event.program)
        self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
        strategy_factors.remove_program_factors(event.program)
        self._classes.pop(event.program, None)
        self._cache_class_epoch += 1
        for observer in tuple(self._observers):
            observer.on_program_released(event.program)

    def _on_gpu_transition(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sidecar = self._factor(strategy_factors, event.program)
        self._set_factor(
            strategy_factors,
            event.program,
            replace(
                sidecar,
                tier=MORITier.GPU,
                pending_tier=None,
                reasoning_wait_seconds=0.0,
                reasoning_wait_started_at_monotonic_s=None,
                gpu_service_started_at_monotonic_s=None,
                gpu_service_completed_requests=0,
                reasoning_started_at_monotonic_s=(
                    event.occurred_at_monotonic_s
                    if event.current_status is ProgramStatus.REASONING
                    else sidecar.reasoning_started_at_monotonic_s
                ),
            ),
        )

    def _claim(self, program: ProgramView, snapshot: SchedulingSnapshot, *, gpu: bool) -> CapacityClaim:
        """Construct a rounded promise, discounting only plausible current-prompt block proof."""
        tokens = self._gpu_target(program, snapshot) if gpu else self._dram_target(program, snapshot)
        quantum = (snapshot.hbm_block_size_tokens if gpu else snapshot.dram_chunk_size_tokens) or 1
        blocks = self._footprints.footprint(program.ref, "gpu" if gpu else "cpu") if self._footprints else frozenset()
        # More evidence than this prompt can contain means it is stale or from an
        # unsupported multi-group layout. Do not guess which subset is current.
        if len(blocks) * quantum > self._round_up(program.tokens.estimated_context_tokens, quantum):
            blocks = frozenset()
        return CapacityClaim(tokens, blocks)

    def _ledgers(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> tuple[CapacityLedger, CapacityLedger]:
        """Build GPU commitments and CPU commitments plus pending destinations in one scan."""
        gpu = CapacityLedger(snapshot.hbm_block_size_tokens or 1, fast=self.config.fast_capacity_accounting)
        cpu = CapacityLedger(snapshot.dram_chunk_size_tokens or 1, fast=self.config.fast_capacity_accounting)
        self.counters["fast_capacity_ledger_builds"] += int(self.config.fast_capacity_accounting)
        for program in snapshot.programs:
            self.counters["planning_program_reads"] += 1
            sidecar = self._factor(factors, program.ref)
            if sidecar.tier is MORITier.GPU:
                gpu.add(program.ref, self._claim(program, snapshot, gpu=True))
            if sidecar.tier is MORITier.CPU or sidecar.pending_tier is MORITier.CPU:
                cpu.add(program.ref, self._claim(program, snapshot, gpu=False))
        return gpu, cpu

    def _demote(
        self,
        victim: ProgramView,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        transitions: TransitionController,
        gpu: CapacityLedger,
        cpu: CapacityLedger,
        *,
        reason: str,
    ) -> bool:
        """Reserve a destination; only finished Acting leases release GPU commitments now."""
        cpu_claim = self._claim(victim, snapshot, gpu=False)
        target, cpu_replacements = self._cpu_destination(victim, snapshot, factors, cpu, cpu_claim)
        previous = self._factor(factors, victim.ref)
        self._set_factor(factors, victim.ref, replace(previous, pending_tier=target))
        operation = transitions.pause if victim.status is ProgramStatus.ACTING else transitions.mark_for_pause
        result = operation(victim.ref, reason=reason)
        if not result.applied:
            self._set_factor(factors, victim.ref, previous)
            self.counters["demotion_rejected"] += 1
            return False
        self._commit_cpu_destination(victim, factors, cpu, cpu_claim, target, cpu_replacements)
        if victim.status is ProgramStatus.ACTING:
            gpu.remove(victim.ref)
        return True

    def _commit_cpu_destination(
        self,
        victim: ProgramView,
        factors: MORIFactors,
        cpu: CapacityLedger,
        claim: CapacityClaim,
        target: MORITier,
        replacements: tuple[ProgramView, ...],
    ) -> None:
        """Apply one destination's CPU claims and relative replacements exactly once."""
        for replaced in replacements:
            resident = self._factor(factors, replaced.ref)
            self._set_factor(factors, replaced.ref, replace(resident, tier=MORITier.WAITING))
            cpu.remove(replaced.ref)
            self.counters["cpu_relative_replacements"] += 1
        if target is MORITier.CPU:
            cpu.add(victim.ref, claim)
        self.counters[f"demotion_to_{target.value}"] += 1

    def _cpu_destination(
        self,
        candidate: ProgramView,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        cpu: CapacityLedger,
        claim: CapacityClaim,
    ) -> tuple[MORITier, tuple[ProgramView, ...]]:
        """Place a more valuable demotion across the CPU/Waiting boundary atomically."""
        if claim.tokens > self._dram_quota(snapshot):
            return MORITier.WAITING, ()
        quota = self._dram_quota(snapshot)
        if cpu.used + cpu.incremental(claim) <= quota:
            return MORITier.CPU, ()

        now = snapshot.observed_at_monotonic_s
        candidate_idle = self._idleness(self._factor(factors, candidate.ref), now)
        residents = [
            program
            for program in snapshot.programs
            if program.ref != candidate.ref
            and program.ref in cpu.claims
            and self._factor(factors, program.ref).tier is MORITier.CPU
            and self._factor(factors, program.ref).pending_tier is None
        ]
        residents.sort(
            key=lambda program: (
                -self._idleness(self._factor(factors, program.ref), now),
                program.ref.program_id,
                program.ref.generation,
            )
        )
        projected = cpu.copy()
        replacements: list[ProgramView] = []
        for resident in residents:
            if self._idleness(self._factor(factors, resident.ref), now) <= candidate_idle:
                break
            replacements.append(resident)
            projected.remove(resident.ref)
            if projected.used + projected.incremental(claim) <= quota:
                return MORITier.CPU, tuple(replacements)
        return MORITier.WAITING, ()

    def _repair_gpu_capacity(
        self,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        transitions: TransitionController,
        gpu: CapacityLedger,
        cpu: CapacityLedger,
    ) -> None:
        """Plan enough marginal relief to cover hard debt without spending pending releases."""
        quota = self._gpu_quota(snapshot)
        if gpu.used <= quota:
            return
        projected = gpu.copy()
        for program in snapshot.programs:
            sidecar = self._factor(factors, program.ref)
            if sidecar.pending_tier is not None:
                projected.remove(program.ref)
        if projected.used <= quota:
            self.counters["gpu_debt_waiting_lazy_relief"] += 1
            return
        victims = [
            p
            for p in snapshot.programs
            if p.ref in gpu.claims and not p.marked_for_pause and self._factor(factors, p.ref).pending_tier is None
        ]
        victims.sort(key=lambda p: self._demotion_key(p, factors, snapshot.observed_at_monotonic_s))
        for victim in victims:
            self.counters["planning_victims"] += 1
            if self._demote(victim, snapshot, factors, transitions, gpu, cpu, reason="mori_capacity_debt"):
                projected.remove(victim.ref)
            if projected.used <= quota:
                return
        self.counters["gpu_capacity_debt_unresolved"] += 1

    def _repair_cpu_capacity(self, snapshot: SchedulingSnapshot, factors: MORIFactors, cpu: CapacityLedger) -> None:
        """Relinquish lower-priority CPU claims until the logical union fits its quota."""
        quota = self._dram_quota(snapshot)
        if cpu.used <= quota:
            return
        victims = [p for p in snapshot.programs if p.ref in cpu.claims]
        victims.sort(
            key=lambda p: (-self._idleness(self._factor(factors, p.ref), snapshot.observed_at_monotonic_s), p.ref)
        )
        retained_claims: list[tuple[ProgramRef, CapacityClaim]] = []
        for victim in victims:
            sidecar = self._factor(factors, victim.ref)
            claim = cpu.claims[victim.ref]
            replacement = (
                replace(sidecar, tier=MORITier.WAITING)
                if sidecar.tier is MORITier.CPU
                else replace(sidecar, pending_tier=MORITier.WAITING)
            )
            self._set_factor(factors, victim.ref, replacement)
            cpu.remove(victim.ref)
            if replacement.tier is MORITier.CPU or replacement.pending_tier is MORITier.CPU:
                # In-tree writers never produce CPU/CPU, but injected factors can.
                # Preserve the old post-CPU rebuild result without rescanning the
                # snapshot or changing this loop's historical stopping point.
                retained_claims.append((victim.ref, claim))
            self.counters["cpu_capacity_to_waiting"] += 1
            if cpu.used <= quota:
                break
        for ref, claim in retained_claims:
            cpu.add(ref, claim)

    def _gpu_service_protected(self, program: ProgramView, factors: MORIFactors, now: float) -> bool:
        """Keep a bounded useful-service lease, with at most one second of Acting grace.

        Admission and restoration alone earn no service. Only observed native
        execution starts this clock; real output-bearing completions consume
        its request budget. Hard capacity repair deliberately ignores leases.
        """
        if self.config.gpu_service_requests <= 0:
            return False
        sidecar = self._factor(factors, program.ref)
        started = sidecar.gpu_service_started_at_monotonic_s
        if (
            sidecar.tier is not MORITier.GPU
            or started is None
            or sidecar.gpu_service_completed_requests >= self.config.gpu_service_requests
        ):
            return False
        if self.config.history_gated_gpu_service:
            # Completed cycles contain only past, actually observed tool gaps.
            # Unknown or slow-return owners receive no speculative retention.
            # This is a reuse hint, never a tool completion clock or a promise
            # of service: all existing native-progress and expiry gates remain.
            eligible = bool(sidecar.cycles) and sum(acting for _, acting in sidecar.cycles) <= (
                len(sidecar.cycles) * _GPU_SERVICE_ACTING_GRACE_SECONDS
            )
            self.counters[
                "gpu_service_history_eligible_checks" if eligible else "gpu_service_history_rejected_checks"
            ] += 1
            if not eligible:
                return False
        deadline = started + _GPU_SERVICE_MAX_SECONDS
        if program.status is ProgramStatus.ACTING:
            if sidecar.acting_started_at_monotonic_s is None:
                return False
            deadline = min(deadline, sidecar.acting_started_at_monotonic_s + _GPU_SERVICE_ACTING_GRACE_SECONDS)
        if now >= deadline:
            return False
        self._request_schedule(factors, deadline)
        self.counters["gpu_service_protection_checks"] += 1
        return True

    def _aged_waiter(self, snapshot: SchedulingSnapshot) -> ProgramView | None:
        """Select a causally ready oldest request independently of cache presence."""
        limit = self.config.ready_wait_limit_seconds
        if limit <= 0:
            return None
        now = snapshot.observed_at_monotonic_s
        return min(
            (
                program
                for program in snapshot.programs
                if program.state is ProgramState.PAUSED
                and program.status is ProgramStatus.REASONING
                and not program.marked_for_pause
                and program.wait_started_at_monotonic_s is not None
                and self._wait_seconds(program, now) >= limit
                and self._gpu_target(program, snapshot) <= self._gpu_quota(snapshot)
            ),
            key=lambda program: (program.wait_started_at_monotonic_s, program.ref),
            default=None,
        )

    def _reserve_aged_waiter(
        self,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        transitions: TransitionController,
        gpu: CapacityLedger,
        cpu: CapacityLedger,
    ) -> None:
        """Drain a feasible oldest waiter's capacity at natural request boundaries.

        The projected ledger chooses lazy victims only. Their real GPU claims
        and native transfer/source pins remain intact until completion; resume
        still checks the real ledger. A failed transition never earns relief.
        """
        if self.config.ready_wait_limit_seconds <= 0:
            return
        candidate = self._aged_waiter(snapshot)
        if candidate is None:
            now = snapshot.observed_at_monotonic_s
            for program in snapshot.programs:
                if (
                    program.state is ProgramState.PAUSED
                    and program.status is ProgramStatus.REASONING
                    and program.wait_started_at_monotonic_s is not None
                    and self._gpu_target(program, snapshot) <= self._gpu_quota(snapshot)
                ):
                    deadline = program.wait_started_at_monotonic_s + self.config.ready_wait_limit_seconds
                    if deadline > now:
                        self._request_schedule(factors, deadline)
            return
        quota = self._gpu_quota(snapshot)
        projected = gpu.copy()
        projected.add(candidate.ref, self._claim(candidate, snapshot, gpu=True))
        for program in snapshot.programs:
            if self._factor(factors, program.ref).pending_tier is not None:
                projected.remove(program.ref)
        if projected.used <= quota:
            return
        self.counters["ready_reservation_cycles"] += 1
        victims = [
            program
            for program in snapshot.programs
            if program.ref in gpu.claims
            and program.state is ProgramState.ACTIVE
            and not program.marked_for_pause
            and self._factor(factors, program.ref).pending_tier is None
            and not self._gpu_service_protected(program, factors, snapshot.observed_at_monotonic_s)
        ]
        victims.sort(key=lambda program: self._demotion_key(program, factors, snapshot.observed_at_monotonic_s))
        for victim in victims:
            previous = self._factor(factors, victim.ref)
            try:
                applied = self._demote(
                    victim, snapshot, factors, transitions, gpu, cpu, reason="mori_ready_reservation"
                )
            except Exception:
                current = self._factor(factors, victim.ref)
                # A committed pause already changed tier through its event and
                # must remain real. A speculative destination on the old tier
                # is not a release or a CPU backup. A committed lazy mark, if
                # any, still drains naturally to WAITING at completion.
                if current.tier is previous.tier and current.pending_tier is not None:
                    self._set_factor(factors, victim.ref, replace(current, pending_tier=previous.pending_tier))
                self.counters["ready_reservation_transition_exceptions"] += 1
                raise
            if applied:
                projected.remove(victim.ref)
                counter = (
                    "ready_reservation_pauses" if victim.status is ProgramStatus.ACTING else "ready_reservation_marks"
                )
                self.counters[counter] += 1
            if projected.used <= quota:
                return

    def _promotion_key(
        self,
        program: ProgramView,
        factors: MORIFactors,
        snapshot: SchedulingSnapshot,
        now: float,
    ) -> tuple[int, int, float, str, int]:
        """Order by MORI tier, new context size, idleness and stable identity."""
        tier = self._factor(factors, program.ref).tier
        priority = 0 if tier is MORITier.CPU else 1 if program.step_count > 0 else 2
        new_context = self._gpu_target(program, snapshot) if priority == 2 else 0
        return (
            priority,
            new_context,
            self._idleness(self._factor(factors, program.ref), now),
            program.ref.program_id,
            program.ref.generation,
        )

    @staticmethod
    def _wait_seconds(program: ProgramView, now: float) -> float:
        """Measure only the current pending request's wait, excluding idle time."""
        started = program.wait_started_at_monotonic_s
        return max(0.0, now - started) if started is not None else 0.0

    def _demotion_key(
        self,
        program: ProgramView,
        factors: MORIFactors,
        now: float,
    ) -> tuple[int, float, str, int]:
        return (
            0 if program.status is ProgramStatus.ACTING else 1,
            -self._idleness(self._factor(factors, program.ref), now),
            program.ref.program_id,
            program.ref.generation,
        )

    def _remaining_gpu_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int:
        return self._gpu_quota(snapshot) - self._gpu_reserved_tokens(snapshot, factors)

    def _gpu_reserved_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int:
        return self._ledgers(snapshot, factors)[0].used

    def _cpu_reserved_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int:
        return self._ledgers(snapshot, factors)[1].used

    def _gpu_target(self, program: ProgramView, snapshot: SchedulingSnapshot) -> int:
        block_size = snapshot.hbm_block_size_tokens or program.tokens.block_size_tokens or 1
        context = max(0, program.tokens.estimated_context_tokens)
        rounded_context = self._round_up(context, block_size)
        target = self._round_up(context + self.config.decode_buffer_tokens, block_size)
        # The speculative decode buffer alone must not make a physically
        # feasible prompt wait forever. Native max-model-length checks and
        # allocation remain authoritative for actual growth.
        return max(rounded_context, min(target, self._gpu_quota(snapshot)))

    @staticmethod
    def _dram_target(program: ProgramView, snapshot: SchedulingSnapshot) -> int:
        chunk_size = snapshot.dram_chunk_size_tokens or program.tokens.block_size_tokens or 1
        return MORIStrategy._round_up(max(0, program.tokens.estimated_context_tokens), chunk_size)

    @staticmethod
    def _gpu_quota(snapshot: SchedulingSnapshot) -> int:
        if snapshot.total_kv_tokens is None:
            raise RuntimeError("MORI requires a rank-local HBM token quota")
        return snapshot.total_kv_tokens

    @staticmethod
    def _dram_quota(snapshot: SchedulingSnapshot) -> int:
        if snapshot.total_dram_kv_tokens is None:
            raise RuntimeError("full MORI requires a rank-local DRAM token quota")
        return snapshot.total_dram_kv_tokens

    @staticmethod
    def _round_up(value: int, quantum: int) -> int:
        return ((value + quantum - 1) // quantum) * quantum

    def _idleness(self, sidecar: MORIProgramFactors, now: float) -> float:
        partial_reasoning = 0.0
        partial_acting = 0.0
        wait_credit = 0.0
        has_partial = False
        if sidecar.acting_started_at_monotonic_s is not None:
            has_partial = True
            partial_reasoning = sidecar.pending_reasoning_seconds
            partial_acting = max(0.0, now - sidecar.acting_started_at_monotonic_s)
        elif sidecar.execution_observed:
            has_partial = True
            partial_reasoning = sidecar.pending_reasoning_seconds
            if sidecar.reasoning_started_at_monotonic_s is not None:
                partial_reasoning += max(0.0, now - sidecar.reasoning_started_at_monotonic_s)
        # Fairness credit is separate from the factual window and exists only
        # while a Reasoning request waits outside GPU. Admission removes it;
        # native queueing before first execution is neither credit nor work.
        if (
            self.config.enable_wait_credit
            and sidecar.tier is not MORITier.GPU
            and sidecar.acting_started_at_monotonic_s is None
            and sidecar.reasoning_wait_started_at_monotonic_s is not None
        ):
            wait_credit = max(0.0, now - sidecar.reasoning_wait_started_at_monotonic_s)
        keep = max(0, self.config.window_size - int(has_partial))
        cycles = sidecar.cycles[-keep:] if keep else ()
        reasoning = sum(item[0] for item in cycles) + partial_reasoning + wait_credit
        acting = sum(item[1] for item in cycles) + partial_acting
        total = reasoning + acting
        return acting / total if total > 0 else 0.0

    def _factor(self, factors: MORIFactors, ref: ProgramRef) -> MORIProgramFactors:
        """Read without materializing an owner from a stale snapshot or callback."""
        return factors.for_program(ref) or MORIProgramFactors()

    def _set_factor(self, factors: MORIFactors, ref: ProgramRef, sidecar: MORIProgramFactors) -> None:
        previous_class = self._classes.get(ref, CacheClass.INACTIVE)
        current_class = self._class_for_tier(sidecar.tier)
        factors.set_program_factors(ref, sidecar)
        self._classes[ref] = current_class
        if previous_class is current_class:
            return
        self._cache_class_epoch += 1
        for observer in tuple(self._observers):
            observer.on_cache_class_changed(ref, previous_class, current_class)

    @staticmethod
    def _class_for_tier(tier: MORITier) -> CacheClass:
        if tier is MORITier.GPU:
            return CacheClass.BUSY
        if tier is MORITier.CPU:
            return CacheClass.IDLE
        return CacheClass.INACTIVE

    @staticmethod
    def _event_non_negative_float(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        normalized = float(value)
        return max(0.0, normalized) if math.isfinite(normalized) else 0.0

    @staticmethod
    def _event_non_negative_int(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return max(0, value)

    @staticmethod
    def _require_applied(result: TransitionResult) -> None:
        if not result.applied:
            raise RuntimeError(f"MORI transition rejected: {result.kind.value}: {result.reason}")
