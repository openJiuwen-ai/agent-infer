# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Explicit MORI construction parallel to the Progress-TTL builder."""

from __future__ import annotations

from dataclasses import dataclass

from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.mori.config import MORIConfig
from agentinfer.scheduling.mori.factors import MORIGlobalFactors, MORIProgramFactors
from agentinfer.scheduling.mori.strategy import MORIStrategy


@dataclass(frozen=True)
class MORIComponents:
    """Concrete MORI strategy and its empty initial decision factors."""

    strategy: MORIStrategy
    initial_factors: StrategyFactors[MORIGlobalFactors, MORIProgramFactors]


def build_mori_strategy(config: MORIConfig | None = None) -> MORIComponents:
    """Construct MORI and bind its cache-class view to the initial factors."""
    strategy = MORIStrategy(config or MORIConfig())
    factors = StrategyFactors[MORIGlobalFactors, MORIProgramFactors](global_factors=MORIGlobalFactors())
    strategy.bind_factors(factors)
    return MORIComponents(strategy=strategy, initial_factors=factors)
