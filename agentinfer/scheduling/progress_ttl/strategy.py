# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Progress-TTL strategy over the shared AgentInfer scheduling contracts.

The strategy guarantees short consecutive service segments, retains acting Programs only while avoided cold-prefill
cost justifies it, resumes pending reasoning before idle acting work, and repairs capacity by segment-tenure priority.
It owns decision factors only and requests every core Program-state mutation through ``TransitionController``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

from agentinfer.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus, ProgramView
from agentinfer.scheduling.events import SchedulingEvent, SchedulingEventKind, StrategyDiagnostic
from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.progress_ttl.config import ProgressTTLConfig
from agentinfer.scheduling.progress_ttl.factors import ProgressTTLProgramFactors
from agentinfer.scheduling.progress_ttl.rolling_stats import ProgressTTLGlobalFactors
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from agentinfer.scheduling.strategy import SchedulingStrategy
from agentinfer.scheduling.transitions import TransitionController, TransitionResult

ProgressTTLFactors = StrategyFactors[ProgressTTLGlobalFactors, ProgressTTLProgramFactors]
logger = logging.getLogger(__name__)


class ProgressTTLStrategy(SchedulingStrategy[ProgressTTLGlobalFactors, ProgressTTLProgramFactors]):
    """Concrete Progress-TTL admission, resume, capacity-repair, and deadline policy."""

    def __init__(self, config: ProgressTTLConfig) -> None:
        self.config = config

    def diagnostics(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
    ) -> tuple[StrategyDiagnostic, ...]:
        """Describe the current Program mix and capacity projection without changing policy state."""
        active = self._active_programs(snapshot)
        paused = [program for program in snapshot.programs if program.state is ProgramState.PAUSED]
        remaining_rounds = sum(
            max(
                0.0,
                self._target_growth_rounds(self._program_factors(strategy_factors, program.ref).is_privileged)
                - float(self._effective_segment_rounds(strategy_factors, program.ref)),
            )
            for program in active
        )
        rolling_growth = max(0.0, strategy_factors.global_factors.avg_input_token_growth_per_round)
        capacity_growth = self._capacity_growth_per_round(strategy_factors)
        projected_reserve = self.config.capacity_safety_margin_tokens + math.ceil(capacity_growth * remaining_rounds)
        effective_capacity = (
            int(snapshot.total_kv_tokens * self.config.resume_capacity_ratio)
            if snapshot.total_kv_tokens is not None
            else None
        )
        return (
            StrategyDiagnostic(
                "progress_ttl_state",
                "periodic_sample",
                (
                    ("active", len(active)),
                    ("active_reasoning", sum(program.status is ProgramStatus.REASONING for program in active)),
                    ("active_acting", sum(program.status is ProgramStatus.ACTING for program in active)),
                    ("paused", len(paused)),
                    ("paused_reasoning", sum(program.status is ProgramStatus.REASONING for program in paused)),
                    ("paused_acting", sum(program.status is ProgramStatus.ACTING for program in paused)),
                    ("waiting_programs", len(snapshot.waiting_programs)),
                    ("used_tokens", sum(self._capacity_tokens(program) for program in active)),
                    ("total_kv_tokens", snapshot.total_kv_tokens),
                    ("resume_effective_capacity", effective_capacity),
                    ("rolling_growth", rolling_growth),
                    ("capacity_growth", capacity_growth),
                    ("remaining_growth_rounds", remaining_rounds),
                    ("projected_active_reserve", projected_reserve),
                ),
            ),
        )

    def event_diagnostics(
        self,
        event: SchedulingEvent,
        strategy_factors: ProgressTTLFactors,
    ) -> tuple[StrategyDiagnostic, ...]:
        """Describe a newly armed acting TTL after request accounting is committed."""
        if event.program is None or event.kind is not SchedulingEventKind.REQUEST_FINISHED:
            return ()
        sidecar = self._program_factors(strategy_factors, event.program)
        if (
            event.current_status is not ProgramStatus.ACTING
            or sidecar.acting_since_monotonic_s is None
            or sidecar.ttl_deadline_monotonic_s is None
        ):
            return ()
        return (
            StrategyDiagnostic(
                "progress_ttl_armed",
                "request_finished",
                (
                    ("program", event.program.program_id),
                    ("generation", event.program.generation),
                    ("total_tokens", event.field("total_tokens")),
                    ("acting_since", sidecar.acting_since_monotonic_s),
                    ("ttl_seconds", sidecar.ttl_deadline_monotonic_s - sidecar.acting_since_monotonic_s),
                    ("ttl_deadline", sidecar.ttl_deadline_monotonic_s),
                    ("is_privileged", sidecar.is_privileged),
                ),
            ),
        )

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        """Admit new work only when its context and minimum-segment growth reserve both fit."""
        view = snapshot.program(candidate)
        if view is None:
            raise ValueError("admission candidate is absent from the scheduling snapshot")
        self._reconcile_privileges(snapshot, strategy_factors)
        required = self._required_tokens(view)
        if view.state is ProgramState.ACTIVE and view.backend_id == snapshot.backend_id:
            return AdmissionOutcome(AdmissionDisposition.ADMITTED, "already_active", required_tokens=required)
        relationship_source = self._privileged_relationship_source(snapshot, strategy_factors, view)
        if relationship_source is not None:
            return self._admit_by_privilege_handoff(
                snapshot,
                strategy_factors,
                transitions,
                view,
                relationship_source,
                required,
            )
        candidate_privileged = not snapshot.waiting_programs and self._can_promote(
            snapshot,
            strategy_factors,
            view,
        )
        reserve = self._continuous_growth_reserve_tokens(
            tuple(self._active_programs(snapshot)),
            strategy_factors,
            candidate=view,
            candidate_privileged=candidate_privileged,
        )
        if candidate in snapshot.waiting_programs:
            return self._queue(transitions, candidate, "already_waiting", required, reserve)
        if snapshot.waiting_programs:
            return self._queue(transitions, candidate, "waiting_queue_precedence", required, reserve)
        remaining = self._remaining_tokens(snapshot, self.config.resume_capacity_ratio)
        if remaining is None:
            return self._queue(transitions, candidate, "capacity_unknown", required, reserve)
        required_with_reserve = required + reserve
        if remaining < required_with_reserve:
            return self._queue(
                transitions,
                candidate,
                "insufficient_capacity",
                required,
                reserve,
                required_with_reserve - remaining,
            )
        result = transitions.admit(candidate, reason="direct_capacity", backend_id=snapshot.backend_id)
        self._require_applied(result)
        if candidate_privileged:
            self._set_privileged(strategy_factors, view.ref, True, reason="direct_admission_promote")
        return AdmissionOutcome(
            AdmissionDisposition.ADMITTED,
            "direct_capacity",
            required_tokens=required,
            reserve_tokens=reserve,
            transitioned_programs=(candidate,),
        )

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
    ) -> None:
        """Resume score-ordered Programs, using eligible acting capacity only when required."""
        self._reconcile_privileges(snapshot, strategy_factors)
        remaining = self._remaining_tokens(snapshot, self.config.resume_capacity_ratio)
        if remaining is None:
            return
        active = self._active_programs(snapshot)
        waiting = [
            program
            for ref in snapshot.waiting_programs
            if (program := snapshot.program(ref)) is not None
            and program.state is ProgramState.PAUSED
            and program.status is ProgramStatus.REASONING
        ]
        waiting.sort(key=lambda program: self._resume_key(program, strategy_factors, snapshot.observed_at_monotonic_s))
        victims = (
            [
                program
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.ACTING
                and program.backend_id == snapshot.backend_id
                and self._program_factors(strategy_factors, program.ref).is_evictable_after_min_rounds
                and not self._program_factors(strategy_factors, program.ref).is_privileged
            ]
            if self.config.resume_reclaim_acting_programs
            else []
        )
        used_victims: set[ProgramRef] = set()
        for program in waiting:
            required = self._required_tokens(program)
            if self._force_resume_due(program, strategy_factors, snapshot.observed_at_monotonic_s):
                wait_started = self._program_factors(strategy_factors, program.ref).wait_started_at_monotonic_s
                waited_seconds = snapshot.observed_at_monotonic_s - wait_started if wait_started is not None else 0.0
                logger.warning(
                    "Progress-TTL force resume program=%s generation=%d waited_seconds=%.1f",
                    program.ref.program_id,
                    program.ref.generation,
                    waited_seconds,
                )
                self._require_applied(
                    transitions.resume(
                        program.ref,
                        reason="progress_ttl_force_resume_timeout",
                        backend_id=snapshot.backend_id,
                    )
                )
                remaining -= required
                active.append(program)
                continue
            reserve = self._continuous_growth_reserve_tokens(tuple(active), strategy_factors, candidate=program)
            required_with_reserve = required + reserve
            if remaining < required_with_reserve:
                deficit = required_with_reserve - remaining
                available = [victim for victim in victims if victim.ref not in used_victims]
                selected = self._select_layered_capacity_fit(
                    (
                        [
                            victim
                            for victim in available
                            if self._program_factors(strategy_factors, victim.ref).segment_served_rounds
                            >= self.config.target_max_segment_rounds
                        ],
                        [
                            victim
                            for victim in available
                            if self._program_factors(strategy_factors, victim.ref).segment_served_rounds
                            < self.config.target_max_segment_rounds
                        ],
                    ),
                    deficit,
                )
                if selected is None:
                    continue
                selected_refs = {victim.ref for victim in selected}
                for victim in selected:
                    self._require_applied(transitions.pause(victim.ref, reason="progress_ttl_resume_reclaim"))
                    used_victims.add(victim.ref)
                    remaining += self._capacity_tokens(victim)
                active = [item for item in active if item.ref not in selected_refs]
                reserve = self._continuous_growth_reserve_tokens(tuple(active), strategy_factors, candidate=program)
                required_with_reserve = required + reserve
                if remaining < required_with_reserve:
                    continue
            self._require_applied(
                transitions.resume(program.ref, reason="progress_ttl_resume", backend_id=snapshot.backend_id)
            )
            remaining -= required
            active.append(program)

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
    ) -> None:
        """Restore pause headroom by segment-tenure priority across both Program statuses."""
        self._reconcile_privileges(snapshot, strategy_factors)
        self._promote_progress_privileges(snapshot, strategy_factors)
        active = [program for program in self._active_programs(snapshot) if not program.marked_for_pause]
        remaining = self._remaining_tokens(snapshot, self.config.pause_capacity_ratio)
        future_paused = sum(
            self._capacity_tokens(program)
            for program in snapshot.programs
            if program.state is ProgramState.ACTIVE
            and program.backend_id == snapshot.backend_id
            and program.marked_for_pause
        )
        lookahead_relief = self._pause_lookahead_tokens_per_active_program(strategy_factors)
        required_headroom = (
            self.config.capacity_safety_margin_tokens
            + lookahead_relief * len(active)
            + self._privileged_extra_reserve_tokens(active, strategy_factors)
        )
        if remaining is None or remaining + future_paused >= required_headroom:
            return
        deficit = required_headroom - remaining - future_paused
        candidates = [
            program for program in active if not self._program_factors(strategy_factors, program.ref).is_privileged
        ]
        while deficit > 0:
            if not candidates:
                demoted = self._demote_privileged_for_capacity(active, strategy_factors)
                if demoted is None:
                    return
                candidates.append(demoted)
            ordered = sorted(
                candidates,
                key=lambda program: (
                    -self._pause_priority_score(
                        program,
                        strategy_factors,
                        snapshot.observed_at_monotonic_s,
                    ),
                    -self._capacity_tokens(program),
                    program.ref.program_id,
                    program.ref.generation,
                ),
            )
            for victim in ordered:
                if victim.status is ProgramStatus.ACTING:
                    result = transitions.pause(victim.ref, reason="progress_ttl_capacity_repair")
                else:
                    result = transitions.mark_for_pause(victim.ref, reason="progress_ttl_capacity_repair")
                self._require_applied(result)
                candidates.remove(victim)
                deficit -= self._capacity_tokens(victim) + lookahead_relief
                if deficit <= 0:
                    return

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        """Yield a maximum-round ordinary Program when queued resume demand is not covered by pending pauses."""
        program = snapshot.program(completed)
        if (
            program is None
            or program.state is not ProgramState.ACTIVE
            or program.status is not ProgramStatus.ACTING
            or program.backend_id != snapshot.backend_id
            or self._program_factors(strategy_factors, completed).is_privileged
            or self._effective_segment_rounds(strategy_factors, completed) < self.config.target_max_segment_rounds
        ):
            return
        waiting = [
            candidate
            for ref in snapshot.waiting_programs
            if (candidate := snapshot.program(ref)) is not None
            and candidate.state is ProgramState.PAUSED
            and candidate.status is ProgramStatus.REASONING
        ]
        if not waiting:
            return
        top = min(
            waiting,
            key=lambda candidate: self._resume_key(
                candidate,
                strategy_factors,
                snapshot.observed_at_monotonic_s,
            ),
        )
        future_paused = sum(
            self._capacity_tokens(candidate)
            for candidate in snapshot.programs
            if candidate.state is ProgramState.ACTIVE
            and candidate.backend_id == snapshot.backend_id
            and candidate.marked_for_pause
        )
        if self._required_tokens(top) <= future_paused:
            return
        self._require_applied(transitions.pause(completed, reason="progress_ttl_max_round_queue_pressure"))

    def next_check_at(self, strategy_factors: ProgressTTLFactors) -> float | None:
        """Return the earliest live acting TTL deadline."""
        deadlines = (
            deadline
            for _, state in strategy_factors.program_factors
            for deadline in (
                state.ttl_deadline_monotonic_s,
                (
                    state.acting_since_monotonic_s + self.config.ttl_max_seconds
                    if state.acting_since_monotonic_s is not None
                    else None
                ),
                state.release_deadline_monotonic_s,
            )
            if deadline is not None
        )
        return min(deadlines, default=None)

    def handle_scheduled_check(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: ProgressTTLFactors,
        transitions: TransitionController,
    ) -> None:
        """Pause stale acting Programs and release paused Programs after their retention TTL."""
        self._reconcile_privileges(snapshot, strategy_factors)
        now = snapshot.observed_at_monotonic_s
        protected_targets: set[ProgramRef] = set()
        for program in snapshot.programs:
            sidecar = self._program_factors(strategy_factors, program.ref)
            fallback_deadline = (
                sidecar.acting_since_monotonic_s + self._ttl_seconds(program.tokens.estimated_context_tokens)
                if sidecar.acting_since_monotonic_s is not None
                else None
            )
            acting_deadline_due = (
                sidecar.ttl_deadline_monotonic_s is not None and sidecar.ttl_deadline_monotonic_s <= now
            ) or (fallback_deadline is not None and fallback_deadline <= now)
            invalid_acting_clock = (
                program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.ACTING
                and sidecar.acting_since_monotonic_s is None
            )
            if (
                (acting_deadline_due or invalid_acting_clock)
                and program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.ACTING
                and program.ref not in protected_targets
            ):
                target = self._same_task_ttl_privilege_target(snapshot, program) if sidecar.is_privileged else None
                self._require_applied(transitions.pause(program.ref, reason="progress_ttl_expired"))
                self._set_privileged(strategy_factors, program.ref, False, reason="ttl_pause_source")
                if target is None:
                    continue
                if target.state is ProgramState.PAUSED:
                    self._require_applied(
                        transitions.resume(
                            target.ref,
                            reason="progress_ttl_privilege_ttl_handoff",
                            backend_id=snapshot.backend_id,
                        )
                    )
                self._set_privileged(strategy_factors, target.ref, True, reason="ttl_handoff")
                protected_targets.add(target.ref)
        for program in snapshot.programs:
            sidecar = self._program_factors(strategy_factors, program.ref)
            if (
                program.state is ProgramState.PAUSED
                and program.status is ProgramStatus.ACTING
                and sidecar.release_deadline_monotonic_s is not None
                and sidecar.release_deadline_monotonic_s <= now
            ):
                self._require_applied(transitions.release(program.ref, reason="progress_ttl_paused_retention_expired"))

    def on_request_pending(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Record the start of a scheduling wait and invalidate an acting TTL."""
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                wait_started_at_monotonic_s=(
                    current.wait_started_at_monotonic_s
                    if current.wait_started_at_monotonic_s is not None
                    else event.occurred_at_monotonic_s
                ),
                request_wait_started_at_monotonic_s=event.occurred_at_monotonic_s,
                acting_since_monotonic_s=None,
                ttl_deadline_monotonic_s=None,
                release_deadline_monotonic_s=None,
            ),
        )

    def on_program_admitted(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Start a fresh service segment after initial admission."""
        self._start_segment(strategy_factors, event, resume=False)

    def on_program_resumed(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Start a fresh service segment after resume without counting a round yet."""
        self._start_segment(strategy_factors, event, resume=True)

    def on_program_status_changed(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Invalidate TTL as soon as another reasoning request arrives."""
        if event.program is None or event.current_status is not ProgramStatus.REASONING:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.set_program_factors(
            event.program,
            replace(current, acting_since_monotonic_s=None, ttl_deadline_monotonic_s=None),
        )

    def on_request_finished(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Count a completed round and arm TTL only after all overlapping reasoning is done."""
        if event.program is None:
            return
        prompt_tokens = self._event_int(event.field("prompt_tokens"))
        completion_tokens = self._event_int(event.field("completion_tokens"))
        total_tokens = self._event_int(event.field("total_tokens")) or prompt_tokens + completion_tokens
        previous_context_tokens = self._event_int(event.field("previous_context_tokens"))
        context_growth_tokens = max(0, total_tokens - previous_context_tokens)
        global_factors = strategy_factors.global_factors
        inter_request_gap_seconds = self._event_float(event.field("inter_request_gap_seconds"))
        global_factors.update_request(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_latency_seconds=self._event_float(event.field("request_latency_seconds")),
            active_programs=self._event_int(event.field("active_programs")),
            waiting_programs=self._event_int(event.field("waiting_programs")),
            input_token_growth=context_growth_tokens,
            inter_request_gap_seconds=inter_request_gap_seconds,
        )
        current = self._program_factors(strategy_factors, event.program)
        rounds = current.segment_served_rounds + 1
        ttl_deadline = None
        acting_since = None
        if event.current_status is ProgramStatus.ACTING:
            acting_since = event.occurred_at_monotonic_s
            ttl_deadline = event.occurred_at_monotonic_s + self._ttl_seconds(total_tokens)
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                segment_served_rounds=rounds,
                segment_prompt_tokens=current.segment_prompt_tokens + prompt_tokens,
                segment_completion_tokens=current.segment_completion_tokens + completion_tokens,
                last_request_prompt_tokens=prompt_tokens,
                is_evictable_after_min_rounds=rounds >= self.config.target_min_segment_rounds,
                acting_since_monotonic_s=acting_since,
                ttl_deadline_monotonic_s=ttl_deadline,
                release_deadline_monotonic_s=None,
                last_inter_request_gap_seconds=inter_request_gap_seconds,
            ),
        )

    def on_program_paused(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Freeze the completed service segment for later resume scoring."""
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.global_factors.update_pause(served_rounds=current.segment_served_rounds)
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                segment_served_rounds=0,
                segment_started_at_monotonic_s=None,
                segment_prompt_tokens=0,
                segment_completion_tokens=0,
                last_segment_served_rounds=current.segment_served_rounds,
                last_segment_prompt_tokens=current.segment_prompt_tokens,
                last_segment_completion_tokens=current.segment_completion_tokens,
                is_evictable_after_min_rounds=False,
                wait_started_at_monotonic_s=event.occurred_at_monotonic_s,
                request_wait_started_at_monotonic_s=None,
                acting_since_monotonic_s=None,
                ttl_deadline_monotonic_s=None,
                release_deadline_monotonic_s=(event.occurred_at_monotonic_s + self.config.paused_program_ttl_seconds),
                last_pause_at_monotonic_s=event.occurred_at_monotonic_s,
                pause_reason=event.reason,
                is_privileged=False,
                privilege_reason=f"{event.reason}_pause_clear",
            ),
        )

    def on_program_marked_for_pause(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Invalidate an acting deadline when a reasoning Program is committed to pause."""
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.set_program_factors(event.program, replace(current, ttl_deadline_monotonic_s=None))

    def on_program_released(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent) -> None:
        """Remove terminal sidecars and their deadlines."""
        if event.program is not None:
            strategy_factors.remove_program_factors(event.program)

    def _start_segment(self, strategy_factors: ProgressTTLFactors, event: SchedulingEvent, *, resume: bool) -> None:
        if event.program is None:
            return
        current = self._program_factors(strategy_factors, event.program)
        strategy_factors.set_program_factors(
            event.program,
            replace(
                current,
                segment_served_rounds=0,
                segment_started_at_monotonic_s=event.occurred_at_monotonic_s,
                segment_prompt_tokens=0,
                segment_completion_tokens=0,
                is_evictable_after_min_rounds=False,
                wait_started_at_monotonic_s=None,
                request_wait_started_at_monotonic_s=None,
                acting_since_monotonic_s=None,
                ttl_deadline_monotonic_s=None,
                release_deadline_monotonic_s=None,
                last_resume_at_monotonic_s=(
                    event.occurred_at_monotonic_s if resume else current.last_resume_at_monotonic_s
                ),
                resume_reason=event.reason if resume else current.resume_reason,
            ),
        )

    def _resume_key(self, program: ProgramView, state: ProgressTTLFactors, now: float) -> tuple[int, float, str, int]:
        sidecar = self._program_factors(state, program.ref)
        waiting_seconds = (
            max(0.0, now - sidecar.wait_started_at_monotonic_s)
            if sidecar.wait_started_at_monotonic_s is not None
            else 0.0
        )
        request_wait_started = sidecar.request_wait_started_at_monotonic_s
        request_waiting_seconds = max(0.0, now - request_wait_started) if request_wait_started is not None else 0.0
        if (
            program.status is ProgramStatus.REASONING
            and request_waiting_seconds >= self.config.force_resume_timeout_seconds
        ):
            tier = -1
        else:
            tier = 0 if program.status is ProgramStatus.REASONING else 1
        stats = state.global_factors
        avg_latency = max(stats.avg_request_latency_seconds, 1e-3)
        avg_active = max(stats.avg_active_programs, 0.0)
        avg_waiting = max(stats.avg_waiting_programs, 0.0)
        fairness_credit = (
            ((waiting_seconds + sidecar.last_inter_request_gap_seconds) / avg_latency)
            * avg_active
            / max(avg_active + avg_waiting, 1.0)
        )
        fairness = fairness_credit * self.config.resume_fairness_weight
        service_debit = max(1, sidecar.last_segment_served_rounds) * self.config.resume_resource_penalty_weight
        prompt_scale = self._resource_scale(sidecar.last_segment_prompt_tokens, state.global_factors.avg_prompt_tokens)
        completion_scale = self._resource_scale(
            sidecar.last_segment_completion_tokens,
            state.global_factors.avg_completion_tokens,
        )
        score = (fairness - service_debit) * prompt_scale * completion_scale
        return tier, -score, program.ref.program_id, program.ref.generation

    def _force_resume_due(self, program: ProgramView, state: ProgressTTLFactors, now: float) -> bool:
        """Return whether a retained reasoning request reached the historical waiter timeout."""
        wait_started = self._program_factors(state, program.ref).request_wait_started_at_monotonic_s
        return (
            program.status is ProgramStatus.REASONING
            and wait_started is not None
            and now - wait_started >= self.config.force_resume_timeout_seconds
        )

    def _active_programs(self, snapshot: SchedulingSnapshot) -> list[ProgramView]:
        """Return Programs currently consuming logical capacity on this backend."""
        return [
            program
            for program in snapshot.programs
            if program.state is ProgramState.ACTIVE and program.backend_id == snapshot.backend_id
        ]

    def _privilege_enabled(self) -> bool:
        """Return whether bounded workflow privilege is explicitly enabled."""
        return self.config.privileged_lookahead_rounds > 0

    def _privileged_program_limit(self, snapshot: SchedulingSnapshot) -> int:
        """Reserve at most half of the backend's full-context slots for privileged Programs."""
        if snapshot.total_kv_tokens is None or snapshot.total_kv_tokens <= 0:
            return 1
        full_context_slots = snapshot.total_kv_tokens // self.config.privileged_max_context_tokens
        return max(1, full_context_slots // 2)

    def _reconcile_privileges(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors) -> None:
        """Clear stale, duplicate-task, and over-limit privilege sidecars from fresh runtime facts."""
        privileged: list[ProgramView] = []
        for ref, sidecar in state.program_factors:
            if not sidecar.is_privileged:
                continue
            program = snapshot.program(ref)
            if (
                not self._privilege_enabled()
                or program is None
                or program.state is not ProgramState.ACTIVE
                or program.backend_id != snapshot.backend_id
                or program.task_id is None
            ):
                self._set_privileged(state, ref, False, reason="privilege_reconcile_clear")
                continue
            privileged.append(program)
        privileged.sort(
            key=lambda program: (
                -self._program_factors(state, program.ref).segment_served_rounds,
                -self._capacity_tokens(program),
                program.ref.program_id,
                program.ref.generation,
            )
        )
        retained_tasks: set[str] = set()
        retained_count = 0
        limit = self._privileged_program_limit(snapshot)
        for program in privileged:
            task_id = program.task_id
            if task_id is None or task_id in retained_tasks or retained_count >= limit:
                self._set_privileged(state, program.ref, False, reason="privilege_limit_clear")
                continue
            retained_tasks.add(task_id)
            retained_count += 1

    def _can_promote(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors, candidate: ProgramView) -> bool:
        """Return whether a candidate can occupy a backend and task privilege slot."""
        if not self._privilege_enabled() or candidate.task_id is None:
            return False
        privileged = [
            program
            for program in snapshot.programs
            if self._program_factors(state, program.ref).is_privileged
            and program.state is ProgramState.ACTIVE
            and program.backend_id == snapshot.backend_id
        ]
        if any(program.task_id == candidate.task_id and program.ref != candidate.ref for program in privileged):
            return False
        return len(privileged) < self._privileged_program_limit(snapshot)

    def _set_privileged(self, state: ProgressTTLFactors, ref: ProgramRef, enabled: bool, *, reason: str) -> None:
        """Update one policy-owned privilege bit and emit a stable experiment diagnostic."""
        current = self._program_factors(state, ref)
        if current.is_privileged is enabled and current.privilege_reason == reason:
            return
        state.set_program_factors(ref, replace(current, is_privileged=enabled, privilege_reason=reason))
        logger.info(
            "Progress-TTL privilege %s program=%s generation=%d reason=%s",
            "enabled" if enabled else "disabled",
            ref.program_id,
            ref.generation,
            reason,
        )

    def _promote_progress_privileges(self, snapshot: SchedulingSnapshot, state: ProgressTTLFactors) -> None:
        """Fill available slots from the most-progressed active reasoning Programs."""
        if not self._privilege_enabled():
            return
        while True:
            candidates = [
                program
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE
                and program.status is ProgramStatus.REASONING
                and program.backend_id == snapshot.backend_id
                and not self._program_factors(state, program.ref).is_privileged
                and self._can_promote(snapshot, state, program)
            ]
            if not candidates:
                return
            selected = max(
                candidates,
                key=lambda program: (
                    program.step_count,
                    self._capacity_tokens(program),
                    self._program_factors(state, program.ref).segment_served_rounds,
                    program.ref.program_id,
                    program.ref.generation,
                ),
            )
            self._set_privileged(state, selected.ref, True, reason="scheduled_progress_promote")

    def _privileged_relationship_source(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        candidate: ProgramView,
    ) -> ProgramView | None:
        """Return an explicitly related privileged active-acting source in historical priority order."""
        if not self._privilege_enabled() or candidate.task_id is None:
            return None

        def relation_tier(source: ProgramView) -> int | None:
            if candidate.parent_program_id == source.ref.program_id:
                return 0
            if candidate.parent_program_id is not None and source.parent_program_id == candidate.parent_program_id:
                return 1
            if source.parent_program_id == candidate.ref.program_id:
                return 1
            return None

        sources: list[tuple[int, ProgramView]] = []
        for source in snapshot.programs:
            tier = relation_tier(source)
            if (
                tier is not None
                and source.ref != candidate.ref
                and source.task_id == candidate.task_id
                and source.state is ProgramState.ACTIVE
                and source.status is ProgramStatus.ACTING
                and source.backend_id == snapshot.backend_id
                and not source.marked_for_pause
                and self._program_factors(state, source.ref).is_privileged
            ):
                sources.append((tier, source))
        if not sources:
            return None
        return min(sources, key=lambda item: (item[0], item[1].ref.program_id, item[1].ref.generation))[1]

    def _admit_by_privilege_handoff(
        self,
        snapshot: SchedulingSnapshot,
        state: ProgressTTLFactors,
        transitions: TransitionController,
        candidate: ProgramView,
        source: ProgramView,
        required_tokens: int,
    ) -> AdmissionOutcome:
        """Transfer a task's service privilege by pausing its safe acting source.

        This bounded priority path intentionally bypasses the ordinary capacity watermark. The next capacity-repair
        cycle restores headroom by pausing ordinary Programs first, or by demoting a privileged Program if necessary.
        """
        self._require_applied(transitions.pause(source.ref, reason="progress_ttl_privilege_source_handoff"))
        self._set_privileged(state, source.ref, False, reason="relationship_handoff_source")
        candidate_state = self._program_factors(state, candidate.ref)
        if candidate.step_count > 0 or candidate_state.last_pause_at_monotonic_s is not None:
            target_result = transitions.resume(
                candidate.ref,
                reason="progress_ttl_privilege_relationship_resume",
                backend_id=snapshot.backend_id,
            )
        else:
            target_result = transitions.admit(
                candidate.ref,
                reason="progress_ttl_privilege_relationship_admission",
                backend_id=snapshot.backend_id,
            )
        self._require_applied(target_result)
        self._set_privileged(state, candidate.ref, True, reason="relationship_handoff_target")
        active_after_handoff = tuple(
            program for program in self._active_programs(snapshot) if program.ref not in {source.ref, candidate.ref}
        )
        reserve = self._continuous_growth_reserve_tokens(
            active_after_handoff,
            state,
            candidate=candidate,
            candidate_privileged=True,
        )
        return AdmissionOutcome(
            AdmissionDisposition.ADMITTED,
            "privileged_relationship_handoff",
            required_tokens=required_tokens,
            reserve_tokens=reserve,
            transitioned_programs=(source.ref, candidate.ref),
        )

    def _same_task_ttl_privilege_target(
        self,
        snapshot: SchedulingSnapshot,
        source: ProgramView,
    ) -> ProgramView | None:
        """Choose active reasoning, paused reasoning, then active acting as a TTL privilege target."""
        if source.task_id is None:
            return None
        candidates: list[tuple[int, ProgramView]] = []
        for target in snapshot.programs:
            if target.ref == source.ref or target.task_id != source.task_id:
                continue
            if target.state is ProgramState.ACTIVE and target.status is ProgramStatus.REASONING:
                candidates.append((0, target))
            elif target.state is ProgramState.PAUSED and target.status is ProgramStatus.REASONING:
                candidates.append((1, target))
            elif target.state is ProgramState.ACTIVE and target.status is ProgramStatus.ACTING:
                candidates.append((2, target))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item[0],
                item[1].wait_started_at_monotonic_s is None,
                (item[1].wait_started_at_monotonic_s if item[1].wait_started_at_monotonic_s is not None else math.inf),
                item[1].ref.program_id,
                item[1].ref.generation,
            ),
        )[1]

    def _privileged_extra_reserve_tokens(
        self,
        active: list[ProgramView],
        state: ProgressTTLFactors,
    ) -> int:
        """Return capped growth beyond ordinary pause lookahead for active privileged Programs."""
        extra_rounds = max(
            0.0,
            self.config.privileged_lookahead_rounds - self.config.pause_capacity_lookahead_rounds,
        )
        growth = self._capacity_growth_per_round(state)
        per_program = math.ceil(growth * extra_rounds)
        return sum(
            min(
                per_program,
                max(
                    0,
                    self.config.privileged_max_context_tokens - program.tokens.estimated_context_tokens,
                ),
            )
            for program in active
            if self._program_factors(state, program.ref).is_privileged
        )

    def _demote_privileged_for_capacity(
        self,
        active: list[ProgramView],
        state: ProgressTTLFactors,
    ) -> ProgramView | None:
        """Demote the least-progressed privilege only when ordinary victims cannot restore safety."""
        candidates = [program for program in active if self._program_factors(state, program.ref).is_privileged]
        if not candidates:
            return None
        demoted = min(
            candidates,
            key=lambda program: (
                program.step_count,
                self._program_factors(state, program.ref).segment_served_rounds,
                self._capacity_tokens(program),
                program.ref.program_id,
                program.ref.generation,
            ),
        )
        self._set_privileged(state, demoted.ref, False, reason="capacity_demote")
        return demoted

    def _continuous_growth_reserve_tokens(
        self,
        active_programs: tuple[ProgramView, ...],
        strategy_factors: ProgressTTLFactors,
        *,
        candidate: ProgramView,
        candidate_privileged: bool = False,
    ) -> int:
        """Reserve context growth until every post-activation Program reaches the minimum segment length.

        The candidate starts a fresh service segment and therefore contributes all configured minimum rounds. Active
        Programs contribute only their remaining rounds. Admission and resume call this same projection so neither
        path can fill capacity with current contexts while omitting the near-term growth required by the policy.
        """
        remaining_rounds = self._target_growth_rounds(candidate_privileged)
        for program in active_programs:
            if program.ref == candidate.ref:
                continue
            served = self._program_factors(strategy_factors, program.ref).segment_served_rounds
            target = self._target_growth_rounds(self._program_factors(strategy_factors, program.ref).is_privileged)
            remaining_rounds += max(0.0, target - float(served))
        growth = self._capacity_growth_per_round(strategy_factors)
        return self.config.capacity_safety_margin_tokens + math.ceil(growth * remaining_rounds)

    def _capacity_growth_per_round(self, state: ProgressTTLFactors) -> float:
        """Return fixed deployment growth when enabled, otherwise the rolling workload estimate."""
        if self.config.use_fixed_input_token_growth:
            return float(self.config.fixed_input_token_growth_per_round)
        return max(0.0, state.global_factors.avg_input_token_growth_per_round)

    def _target_growth_rounds(self, privileged: bool) -> float:
        """Return the shared float round domain, preserving fractional privileged lookahead."""
        if privileged:
            return max(0.0, self.config.privileged_lookahead_rounds)
        return float(self.config.target_min_segment_rounds)

    def _remaining_tokens(self, snapshot: SchedulingSnapshot, capacity_ratio: float) -> int | None:
        if snapshot.total_kv_tokens is None:
            return None
        effective_capacity = int(snapshot.total_kv_tokens * capacity_ratio)
        used = sum(
            self._capacity_tokens(program)
            for program in snapshot.programs
            if program.state is ProgramState.ACTIVE and program.backend_id == snapshot.backend_id
        )
        return effective_capacity - used

    def _pause_lookahead_tokens_per_active_program(self, strategy_factors: ProgressTTLFactors) -> int:
        """Return recent context growth reserved for each continuing active Program."""
        growth = self._capacity_growth_per_round(strategy_factors)
        return math.ceil(growth * self.config.pause_capacity_lookahead_rounds)

    def _required_tokens(self, program: ProgramView) -> int:
        return (
            program.tokens.estimated_context_tokens
            + program.tokens.estimated_next_round_tokens
            + self.config.decode_buffer_tokens
        )

    def _effective_segment_rounds(self, state: ProgressTTLFactors, ref: ProgramRef) -> int:
        """Return current segment rounds used by the policy's soft progress boundaries."""
        return self._program_factors(state, ref).segment_served_rounds

    def _capacity_tokens(self, program: ProgramView) -> int:
        return program.tokens.estimated_context_tokens + self.config.decode_buffer_tokens

    def _pause_priority_score(
        self,
        program: ProgramView,
        state: ProgressTTLFactors,
        now_monotonic_s: float,
    ) -> float:
        """Return segment tenure scaled by current input reload cost for capacity repair."""
        sidecar = self._program_factors(state, program.ref)
        segment_started_at = sidecar.segment_started_at_monotonic_s
        duration = max(0.0, now_monotonic_s - segment_started_at) if segment_started_at is not None else 0.0
        input_tokens = program.tokens.estimated_context_tokens
        if program.status is ProgramStatus.ACTING and sidecar.last_request_prompt_tokens:
            input_tokens = sidecar.last_request_prompt_tokens
        return duration * self._resource_scale(input_tokens, state.global_factors.avg_prompt_tokens)

    def _ttl_seconds(self, total_tokens: int) -> float:
        uncached = max(0, int(total_tokens * self.config.uncached_ratio_default))
        cold_prefill = self.config.ttl_prefill_seconds_per_1k_uncached_tokens * uncached / 1000.0
        decode_interference = (1.0 - self.config.ttl_decode_throughput_alpha) * cold_prefill
        total_impact = (cold_prefill + decode_interference) * self.config.ttl_impact_multiplier
        return min(self.config.ttl_max_seconds, max(self.config.ttl_min_seconds, total_impact))

    def _select_capacity_fit(
        self,
        programs: list[ProgramView],
        deficit: int,
        *,
        per_program_relief_tokens: int = 0,
    ) -> tuple[ProgramView, ...] | None:
        if deficit <= 0:
            return ()
        reachable: dict[int, tuple[ProgramView, ...]] = {0: ()}
        best: tuple[int, tuple[ProgramView, ...]] | None = None
        for program in sorted(programs, key=lambda item: (item.ref.program_id, item.ref.generation)):
            tokens = self._capacity_tokens(program) + per_program_relief_tokens
            additions: dict[int, tuple[ProgramView, ...]] = {}
            for freed, selected in tuple(reachable.items()):
                new_freed = freed + tokens
                candidate = (*selected, program)
                if new_freed >= deficit:
                    key = (new_freed, len(candidate), tuple(item.ref for item in candidate))
                    if best is None or key < (best[0], len(best[1]), tuple(item.ref for item in best[1])):
                        best = (new_freed, candidate)
                else:
                    existing = additions.get(new_freed, reachable.get(new_freed))
                    candidate_key = (len(candidate), tuple(item.ref for item in candidate))
                    if existing is None or candidate_key < (len(existing), tuple(item.ref for item in existing)):
                        additions[new_freed] = candidate
            reachable.update(additions)
        return best[1] if best is not None else None

    def _select_layered_capacity_fit(
        self,
        layers: tuple[list[ProgramView], ...],
        deficit: int,
    ) -> tuple[ProgramView, ...] | None:
        """Prefer older progress tiers while minimizing over-release within each tier."""
        selected: list[ProgramView] = []
        remaining = deficit
        for layer in layers:
            fit = self._select_capacity_fit(layer, remaining)
            if fit is not None:
                return (*selected, *fit)
            selected.extend(layer)
            remaining -= sum(self._capacity_tokens(program) for program in layer)
            if remaining <= 0:
                return tuple(selected)
        return None

    @staticmethod
    def _program_factors(state: ProgressTTLFactors, ref: ProgramRef) -> ProgressTTLProgramFactors:
        current = state.for_program(ref)
        if current is None:
            current = ProgressTTLProgramFactors()
            state.set_program_factors(ref, current)
        return current

    @staticmethod
    def _resource_scale(tokens: int, average_tokens: float) -> float:
        ratio = max(float(tokens) / max(average_tokens, 1e-6), 1e-6)
        return max(0.5, 1.0 / math.sqrt(ratio))

    @staticmethod
    def _event_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float) and math.isfinite(value):
            return max(0, int(value))
        return 0

    @staticmethod
    def _event_float(value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return 0.0
        result = float(value)
        return max(0.0, result) if math.isfinite(result) else 0.0

    @staticmethod
    def _require_applied(result: TransitionResult) -> None:
        if not result.applied:
            raise RuntimeError(f"Progress-TTL transition rejected: {result.kind.value}: {result.reason}")

    @staticmethod
    def _queue(
        transitions: TransitionController,
        candidate: ProgramRef,
        reason: str,
        required_tokens: int,
        reserve_tokens: int = 0,
        deficit_tokens: int = 0,
    ) -> AdmissionOutcome:
        result = transitions.queue(candidate, reason=reason)
        ProgressTTLStrategy._require_applied(result)
        return AdmissionOutcome(
            AdmissionDisposition.QUEUED,
            reason,
            required_tokens=required_tokens,
            reserve_tokens=reserve_tokens,
            deficit_tokens=deficit_tokens,
            transitioned_programs=(candidate,),
        )
