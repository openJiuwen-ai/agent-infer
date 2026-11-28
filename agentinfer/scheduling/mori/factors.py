# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""MORI token estimates, timing history, and ordered placement indexes."""

from collections import deque
from dataclasses import dataclass, field

from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.factors import StrategyFactors


@dataclass
class MoriProgramFactors:
    steps: int = 0
    context_chars: int = 0
    tokens: int = 0
    prefix_group: str = ""
    sampled_prefix: bool = False
    acting_since: float | None = None
    reasoning_since: float | None = None
    acting_history: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    reasoning_history: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    future_credit: int | None = None

    def idleness(self, now: float) -> float:
        """Port ThunderAgent Program.idleness and the Router's missing-history default."""
        acting = sum(self.acting_history)
        reasoning = sum(self.reasoning_history)
        if self.acting_since is not None:
            acting += max(0.0, now - self.acting_since)
        if self.reasoning_since is not None:
            reasoning += max(0.0, now - self.reasoning_since)
        duration = acting + reasoning
        return acting / duration if duration > 0 else 1.0

    def start_reasoning(self, now: float) -> None:
        if self.acting_since is not None:
            self.acting_history.append(max(0.0, now - self.acting_since))
        self.acting_since = None

    def finish_reasoning(self, now: float) -> None:
        if self.reasoning_since is not None:
            self.reasoning_history.append(max(0.0, now - self.reasoning_since))
        self.reasoning_since = None
        self.acting_since = now


@dataclass
class MoriGlobalFactors:
    char_to_token_ratio: float = 5.0
    ratio_initialized: bool = False
    prefix_lengths: dict[str, int] = field(default_factory=dict)
    shared_tokens: int = 0
    active: dict[ProgramRef, None] = field(default_factory=dict)
    warm: dict[ProgramRef, int] = field(default_factory=dict)
    waiting: dict[ProgramRef, None] = field(default_factory=dict)


MoriFactors = StrategyFactors[MoriGlobalFactors, MoriProgramFactors]
