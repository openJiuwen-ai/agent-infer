# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""MORI budget derived from Program estimates and effective host capacities."""

from dataclasses import dataclass

from agentinfer.scheduling.domain import ProgramRef, ProgramStatus
from agentinfer.scheduling.mori.factors import MoriFactors
from agentinfer.scheduling.snapshot import SchedulingSnapshot

BUFFER_PER_PROGRAM = 100
SHARED_PREFIX_ACTIVE_WINDOW_SECONDS = 300.0


@dataclass(frozen=True)
class MoriLedger:
    snapshot: SchedulingSnapshot
    factors: MoriFactors

    @property
    def hbm_room(self) -> int | float:
        """Port ThunderAgent BackendState.remaining_capacity for the local rank."""
        capacity = self.snapshot.total_kv_tokens
        if capacity is None:
            return float("inf")
        state = self.factors.global_factors
        active_tokens = sum(self.factors.for_program(ref).tokens for ref in state.active)
        return capacity - active_tokens - len(state.active) * BUFFER_PER_PROGRAM + state.shared_tokens

    @property
    def dram_room(self) -> int:
        """Port ThunderAgent BackendState.cpu_remaining_capacity."""
        capacity = self.snapshot.total_dram_kv_tokens
        return capacity - sum(self.factors.global_factors.warm.values()) if capacity else 0

    @property
    def future_release(self) -> int:
        return sum(self.factors.for_program(ref).future_credit or 0 for ref in self.factors.global_factors.active)

    def required(self, ref: ProgramRef) -> int:
        return self.factors.for_program(ref).tokens + BUFFER_PER_PROGRAM

    def update_shared_prefix(self, now: float) -> None:
        """Port ThunderAgent shared_prefix_tokens using a Runtime clock sample."""
        state = self.factors.global_factors
        counts: dict[str, int] = {}
        for ref in state.active:
            program = self.factors.for_program(ref)
            active = self.snapshot.program(ref).status is ProgramStatus.REASONING or (
                program.acting_since is not None and now - program.acting_since < SHARED_PREFIX_ACTIVE_WINDOW_SECONDS
            )
            if active:
                counts[program.prefix_group] = counts.get(program.prefix_group, 0) + 1
        state.shared_tokens = sum((count - 1) * state.prefix_lengths.get(group, 0) for group, count in counts.items())
