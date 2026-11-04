# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Committed history and placement factors maintained by MORI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MORITier(str, Enum):
    """Logical cache protection tier, independent of Program execution state."""

    GPU = "gpu"
    CPU = "cpu"
    WAITING = "waiting"


@dataclass(frozen=True)
class MORIGlobalFactors:
    """Policy-wide committed counters used by diagnostics."""

    schedule_cycles: int = 0
    schedule_requested_at: float | None = None

    def __post_init__(self) -> None:
        if self.schedule_cycles < 0:
            raise ValueError("MORI schedule_cycles must be non-negative")
        if self.schedule_requested_at is not None and (
            not math.isfinite(self.schedule_requested_at) or self.schedule_requested_at < 0
        ):
            raise ValueError("MORI schedule request time must be finite and non-negative")


@dataclass(frozen=True)
class MORIProgramFactors:
    """Recent reasoning/acting cycles and generation-safe logical placement."""

    tier: MORITier = MORITier.WAITING
    cycles: tuple[tuple[float, float], ...] = ()
    pending_reasoning_seconds: float = 0.0
    acting_started_at_monotonic_s: float | None = None
    reasoning_started_at_monotonic_s: float | None = None
    # Non-GPU Reasoning fairness credit only; cleared at admission, never history.
    reasoning_wait_seconds: float = 0.0
    reasoning_wait_started_at_monotonic_s: float | None = None
    pending_tier: MORITier | None = None
    execution_observed: bool = False
    gpu_service_started_at_monotonic_s: float | None = None
    gpu_service_completed_requests: int = 0
    # Shadow-only last-request cache evidence. These fields are never ranking
    # inputs; exact-presence bits prevent a combined-cache fallback from being
    # misreported as a local-HBM observation.
    last_hbm_cached_prefix_tokens: int | None = None
    last_hbm_hit_ref_zero_tokens: int | None = None
    last_hbm_cached_prefix_exact_present: bool = False
    last_hbm_hit_ref_zero_exact_present: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tier, MORITier):
            raise ValueError("MORI tier must be a MORITier")
        if (
            isinstance(self.gpu_service_completed_requests, bool)
            or not isinstance(self.gpu_service_completed_requests, int)
            or self.gpu_service_completed_requests < 0
        ):
            raise ValueError("MORI GPU service completion count must be a non-negative integer")
        if self.pending_tier is not None and not isinstance(self.pending_tier, MORITier):
            raise ValueError("MORI pending_tier must be a MORITier when supplied")
        cache_counts = (self.last_hbm_cached_prefix_tokens, self.last_hbm_hit_ref_zero_tokens)
        if any(value is not None and value < 0 for value in cache_counts):
            raise ValueError("MORI shadow cache observations must be non-negative")
        if not isinstance(self.last_hbm_cached_prefix_exact_present, bool) or not isinstance(
            self.last_hbm_hit_ref_zero_exact_present, bool
        ):
            raise ValueError("MORI shadow cache exact-presence fields must be boolean")
        values = [self.pending_reasoning_seconds, self.reasoning_wait_seconds]
        values.extend(value for cycle in self.cycles for value in cycle)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("MORI timing and idleness values must be finite and non-negative")
        timestamps = (
            self.acting_started_at_monotonic_s,
            self.reasoning_started_at_monotonic_s,
            self.reasoning_wait_started_at_monotonic_s,
            self.gpu_service_started_at_monotonic_s,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in timestamps):
            raise ValueError("MORI timestamps must be finite and non-negative")
