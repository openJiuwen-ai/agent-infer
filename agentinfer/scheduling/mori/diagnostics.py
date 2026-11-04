# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Slow-only MORI decision evidence, separate from placement state and policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus, ProgramView
from agentinfer.scheduling.mori.capacity import CapacityClaim, CapacityLedger
from agentinfer.scheduling.mori.factors import MORIProgramFactors, MORITier
from agentinfer.scheduling.snapshot import SchedulingSnapshot

if TYPE_CHECKING:
    from agentinfer.scheduling.mori.strategy import MORIFactors, MORIStrategy


def score_components(
    sidecar: MORIProgramFactors, now: float, window_size: int, *, wait_credit_enabled: bool = True
) -> dict[str, object]:
    """Explain the exact current-window score without changing factual history."""
    partial_reasoning = 0.0
    partial_acting = 0.0
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
    wait_credit = 0.0
    waiting = (
        wait_credit_enabled
        and sidecar.tier is not MORITier.GPU
        and sidecar.acting_started_at_monotonic_s is None
        and sidecar.reasoning_wait_started_at_monotonic_s is not None
    )
    if waiting:
        wait_credit = max(0.0, now - sidecar.reasoning_wait_started_at_monotonic_s)
    keep = max(0, window_size - int(has_partial))
    cycles = sidecar.cycles[-keep:] if keep else ()
    completed_reasoning = sum(cycle[0] for cycle in cycles)
    completed_acting = sum(cycle[1] for cycle in cycles)
    reasoning = completed_reasoning + partial_reasoning + wait_credit
    acting = completed_acting + partial_acting
    return {
        "idleness": acting / (acting + reasoning) if acting + reasoning else 0.0,
        "retained_cycles": len(cycles),
        "dropped_cycles": len(sidecar.cycles) - len(cycles),
        "completed_reasoning_seconds": completed_reasoning,
        "completed_acting_seconds": completed_acting,
        "partial_reasoning_seconds": partial_reasoning,
        "partial_acting_seconds": partial_acting,
        "current_wait_credit_seconds": wait_credit,
        "wait_credit_enabled": wait_credit_enabled,
        "wait_affects_idleness": waiting,
        "execution_observed": sidecar.execution_observed,
    }


class MORIDiagnosticState:
    """Keep one snapshot reference and at most one decision per live generation."""

    def __init__(self, strategy: MORIStrategy, threshold_seconds: float) -> None:
        self.strategy = strategy
        self.threshold_seconds = threshold_seconds
        self.snapshot: SchedulingSnapshot | None = None
        self.decisions: dict[ProgramRef, dict[str, object]] = {}

    def observe(self, snapshot: SchedulingSnapshot) -> None:
        """Capture an existing immutable snapshot without scanning its Programs."""
        self.snapshot = snapshot

    def begin(
        self,
        program: ProgramView,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        gpu: CapacityLedger,
        claim: CapacityClaim | None,
        *,
        rank: int | None,
        admitted_before: int,
    ) -> dict[str, object] | None:
        """Build detailed evidence only once per five seconds for an old waiter."""
        now = snapshot.observed_at_monotonic_s
        started = program.wait_started_at_monotonic_s
        if started is None or now - started < self.threshold_seconds:
            return None
        previous = self.decisions.get(program.ref)
        if previous is not None and now - float(previous["observed_at_monotonic_s"]) < 5.0:
            return None
        if claim is None:
            claim = self.strategy._claim(program, snapshot, gpu=True)
        factor = self.strategy._factor(factors, program.ref)
        decision: dict[str, object] = {
            "scope": "exact_candidate_evaluation",
            "observed_at_monotonic_s": now,
            "reason": "evaluating",
            "tier": factor.tier.value,
            "score": score_components(
                factor,
                now,
                self.strategy.config.window_size,
                wait_credit_enabled=self.strategy.config.enable_wait_credit,
            ),
            "capacity": {
                "gpu_committed_tokens": gpu.used,
                "gpu_quota_tokens": snapshot.total_kv_tokens,
                "claim_tokens": claim.tokens,
                "incremental_claim_tokens": gpu.incremental(claim),
            },
            "ranking": {
                "candidate_rank": rank,
                "admitted_before_in_cycle": admitted_before,
                "priority_group": "cpu"
                if factor.tier is MORITier.CPU
                else "returning"
                if program.step_count
                else "new",
                "ordering_is_not_causal_attribution": True,
            },
        }
        self.decisions[program.ref] = decision
        return decision

    def release(self, program: ProgramRef) -> None:
        self.decisions.pop(program, None)

    def context(self, program: ProgramRef, now: float) -> dict[str, object]:
        """Reconstruct current factors over the last snapshot only when emitting."""
        strategy = self.strategy
        factors = strategy._factors
        snapshot = self.snapshot
        context: dict[str, object] = {}
        if factors is None:
            return context
        factor = factors.for_program(program)
        if factor is not None:
            context["current_program"] = {
                "tier": factor.tier.value,
                "pending_tier": factor.pending_tier.value if factor.pending_tier else None,
                "generation": program.generation,
            }
            context["current_score"] = score_components(
                factor,
                now,
                strategy.config.window_size,
                wait_credit_enabled=strategy.config.enable_wait_credit,
            )
        decision = self.decisions.get(program)
        if decision is not None:
            context["last_decision"] = decision
            context["last_decision_age_seconds"] = max(0.0, now - float(decision["observed_at_monotonic_s"]))
        if snapshot is None:
            return context
        # Do not call _ledgers: diagnostic reads must not increment policy counters.
        gpu = CapacityLedger(snapshot.hbm_block_size_tokens or 1)
        pending = []
        counts = {"gpu_reasoning": 0, "gpu_acting": 0, "paused_reasoning": 0}
        for view in snapshot.programs:
            sidecar = factors.for_program(view.ref)
            if sidecar is None:
                continue
            if sidecar.tier is MORITier.GPU:
                gpu.add(view.ref, strategy._claim(view, snapshot, gpu=True))
                counts["gpu_reasoning" if view.status is ProgramStatus.REASONING else "gpu_acting"] += 1
                if sidecar.pending_tier is not None:
                    pending.append(view.ref)
            if view.state is ProgramState.PAUSED and view.status is ProgramStatus.REASONING:
                counts["paused_reasoning"] += 1
        after_pending = gpu.copy()
        for ref in pending:
            after_pending.remove(ref)
        context["latest_snapshot"] = {
            "observed_at_monotonic_s": snapshot.observed_at_monotonic_s,
            "age_seconds": max(0.0, now - snapshot.observed_at_monotonic_s),
            "native_used_tokens": snapshot.native_used_kv_tokens,
            "native_waiting_tokens": snapshot.native_waiting_kv_tokens,
            "native_dram_used_tokens": snapshot.used_dram_kv_tokens,
            "pending_h2d_tokens": snapshot.pending_h2d_kv_tokens,
            "pending_d2h_tokens": snapshot.pending_d2h_kv_tokens,
        }
        context["reconstructed_capacity"] = {
            "scope": "latest_snapshot_views_with_current_factors_and_footprints_not_exact_decision",
            "gpu_committed_tokens": gpu.used,
            "gpu_quota_tokens": snapshot.total_kv_tokens,
            "cpu_quota_tokens": snapshot.total_dram_kv_tokens,
            "pending_release_programs": len(pending),
            "gpu_committed_after_pending_release_tokens": after_pending.used,
            **counts,
        }
        return context
