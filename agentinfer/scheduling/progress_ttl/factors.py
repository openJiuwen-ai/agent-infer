# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Per-Program scheduling decision factors maintained by Progress-TTL.

Global rolling workload values live in ``rolling_stats``. This module retains segment progress, fairness timestamps,
TTL deadlines, and bounded privilege used by admission, resume, pause, handoff, and release decisions without replacing
the Program lifecycle facts owned by the Scheduler runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressTTLProgramFactors:
    """Decision factors for one exact Program generation's progress, TTL, privilege, and fairness."""

    segment_served_rounds: int = 0
    segment_started_at_monotonic_s: float | None = None
    segment_prompt_tokens: int = 0
    segment_completion_tokens: int = 0
    last_request_prompt_tokens: int = 0
    last_segment_served_rounds: int = 0
    last_segment_prompt_tokens: int = 0
    last_segment_completion_tokens: int = 0
    is_evictable_after_min_rounds: bool = False
    is_privileged: bool = False
    privilege_reason: str | None = None
    wait_started_at_monotonic_s: float | None = None
    request_wait_started_at_monotonic_s: float | None = None
    acting_since_monotonic_s: float | None = None
    ttl_deadline_monotonic_s: float | None = None
    release_deadline_monotonic_s: float | None = None
    last_pause_at_monotonic_s: float | None = None
    last_resume_at_monotonic_s: float | None = None
    pause_reason: str | None = None
    resume_reason: str | None = None
    last_inter_request_gap_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate counters, request gap, and optional monotonic timestamps."""
        counters = (
            self.segment_served_rounds,
            self.segment_prompt_tokens,
            self.segment_completion_tokens,
            self.last_request_prompt_tokens,
            self.last_segment_served_rounds,
            self.last_segment_prompt_tokens,
            self.last_segment_completion_tokens,
        )
        if min(counters) < 0:
            raise ValueError("Progress-TTL Program counters must be non-negative")
        timestamps = (
            self.segment_started_at_monotonic_s,
            self.wait_started_at_monotonic_s,
            self.request_wait_started_at_monotonic_s,
            self.acting_since_monotonic_s,
            self.ttl_deadline_monotonic_s,
            self.release_deadline_monotonic_s,
            self.last_pause_at_monotonic_s,
            self.last_resume_at_monotonic_s,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in timestamps):
            raise ValueError("Progress-TTL timestamps must be finite and non-negative")
        if not math.isfinite(self.last_inter_request_gap_seconds) or self.last_inter_request_gap_seconds < 0:
            raise ValueError("last_inter_request_gap_seconds must be finite and non-negative")
