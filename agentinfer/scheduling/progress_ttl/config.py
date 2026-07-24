# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validated configuration for the Progress-TTL scheduling policy.

The policy combines a minimum consecutive-service guarantee, capacity-aware pause/resume, and an acting TTL derived
from estimated cold-prefill cost. Transport, backend polling, and vLLM-specific fields are deliberately excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressTTLConfig:
    """Policy-owned Progress-TTL parameters."""

    target_min_segment_rounds: int = 7
    target_max_segment_rounds: int = 12
    ttl_min_seconds: float = 10.0
    ttl_max_seconds: float = 120.0
    ttl_prefill_seconds_per_1k_uncached_tokens: float = 0.29
    ttl_impact_multiplier: float = 2.0
    ttl_decode_throughput_alpha: float = 0.15
    uncached_ratio_default: float = 0.8
    resume_fairness_weight: float = 1.0
    resume_resource_penalty_weight: float = 1.0
    capacity_safety_margin_tokens: int = 0
    resume_capacity_ratio: float = 1.0
    pause_capacity_ratio: float = 1.0
    pause_capacity_lookahead_rounds: float = 0.0
    privileged_lookahead_rounds: float = 0.0
    privileged_max_context_tokens: int = 262_144
    decode_buffer_tokens: int = 100
    force_resume_timeout_seconds: float = 1800.0
    paused_program_ttl_seconds: float = 1800.0

    def __post_init__(self) -> None:
        """Reject values that make deadlines or capacity projections invalid."""
        if self.target_min_segment_rounds <= 0:
            raise ValueError("target_min_segment_rounds must be positive")
        if self.target_max_segment_rounds < self.target_min_segment_rounds:
            raise ValueError("target_max_segment_rounds must be >= target_min_segment_rounds")
        finite_non_negative = (
            self.ttl_min_seconds,
            self.ttl_max_seconds,
            self.ttl_prefill_seconds_per_1k_uncached_tokens,
            self.ttl_impact_multiplier,
            self.resume_fairness_weight,
            self.resume_resource_penalty_weight,
            self.uncached_ratio_default,
            self.force_resume_timeout_seconds,
            self.paused_program_ttl_seconds,
            self.pause_capacity_lookahead_rounds,
            self.privileged_lookahead_rounds,
        )
        if any(not math.isfinite(value) or value < 0 for value in finite_non_negative):
            raise ValueError("Progress-TTL time, cost, and weight values must be finite and non-negative")
        if self.ttl_max_seconds < self.ttl_min_seconds:
            raise ValueError("ttl_max_seconds must be >= ttl_min_seconds")
        if not math.isfinite(self.ttl_decode_throughput_alpha) or not 0 <= self.ttl_decode_throughput_alpha <= 1:
            raise ValueError("ttl_decode_throughput_alpha must be between 0 and 1")
        if not math.isfinite(self.resume_capacity_ratio) or not 0 < self.resume_capacity_ratio <= 1:
            raise ValueError("resume_capacity_ratio must be in (0, 1]")
        if not math.isfinite(self.pause_capacity_ratio) or not 0 < self.pause_capacity_ratio <= 1:
            raise ValueError("pause_capacity_ratio must be in (0, 1]")
        if self.capacity_safety_margin_tokens < 0 or self.decode_buffer_tokens < 0:
            raise ValueError("Progress-TTL token margins must be non-negative")
        if self.privileged_max_context_tokens <= 0:
            raise ValueError("privileged_max_context_tokens must be positive")
