# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Construct a MORI policy and its empty decision factors."""

from dataclasses import dataclass

from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.mori.config import MoriConfig
from agentinfer.scheduling.mori.factors import MoriFactors, MoriGlobalFactors
from agentinfer.scheduling.mori.strategy import MoriStrategy


@dataclass(frozen=True)
class MoriComponents:
    strategy: MoriStrategy
    initial_factors: MoriFactors


def build_mori_strategy(config: MoriConfig | None = None) -> MoriComponents:
    return MoriComponents(MoriStrategy(config or MoriConfig()), StrategyFactors(MoriGlobalFactors()))
