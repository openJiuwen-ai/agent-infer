# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Explicit Progress-TTL construction without a multi-policy registry."""

from __future__ import annotations

from dataclasses import dataclass

from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.progress_ttl.config import ProgressTTLConfig
from agentinfer.scheduling.progress_ttl.factors import ProgressTTLProgramFactors
from agentinfer.scheduling.progress_ttl.rolling_stats import ProgressTTLGlobalFactors
from agentinfer.scheduling.progress_ttl.strategy import ProgressTTLStrategy


@dataclass(frozen=True)
class ProgressTTLComponents:
    """Concrete Progress-TTL strategy and its empty initial decision factors."""

    strategy: ProgressTTLStrategy
    initial_factors: StrategyFactors[ProgressTTLGlobalFactors, ProgressTTLProgramFactors]


def build_progress_ttl_strategy(config: ProgressTTLConfig | None = None) -> ProgressTTLComponents:
    """Construct Progress-TTL and its initial decision factors from immutable configuration."""
    resolved = config or ProgressTTLConfig()
    return ProgressTTLComponents(
        strategy=ProgressTTLStrategy(resolved),
        initial_factors=StrategyFactors(global_factors=ProgressTTLGlobalFactors()),
    )
