# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Reusable bounded rolling statistics for the Progress-TTL policy.

This is the single shared implementation used by Router-hosted and engine-embedded Progress-TTL runtimes. It preserves
the original PT request-window sizing, cold-start threshold, gradual eviction, sample sanitization, and pause window.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestStatSample:
    """Sanitized per-request observation used by windowed PT statistics."""

    prompt_tokens: int
    completion_tokens: int | None
    total_tokens: int
    request_latency_seconds: float
    active_programs: int
    waiting_programs: int
    input_token_growth: int | None
    inter_request_gap_seconds: float


@dataclass
class ProgressTTLGlobalFactors:
    """Policy-wide decision factors derived from bounded recent request and pause windows."""

    request_count: int = 0
    request_window_size: int = 64
    min_request_window_size: int = 64
    max_request_window_size: int = 320
    request_window_per_active_program: int = 10
    max_evictions_per_update: int = 5
    min_update_window_fraction: float = 0.5
    avg_prompt_tokens: float = 1024.0
    avg_completion_tokens: float = 256.0
    avg_total_tokens: float = 1280.0
    avg_router_queue_seconds: float = 0.0
    avg_request_latency_seconds: float = 1.0
    avg_active_programs: float = 1.0
    avg_waiting_programs: float = 0.0
    avg_segment_served_rounds_on_pause: float = 1.0
    avg_input_token_growth_per_round: float = 1024.0
    avg_inter_request_gap_seconds: float = 0.0
    avg_reasoning_seconds: float = 1.0
    avg_acting_seconds: float = 0.0
    cold_prefill_cost_seconds: float = 1.0
    _request_samples: deque[RequestStatSample] = field(default_factory=deque, init=False, repr=False)
    _prompt_token_sum: int = field(default=0, init=False, repr=False)
    _completion_token_sum: int = field(default=0, init=False, repr=False)
    _completion_token_count: int = field(default=0, init=False, repr=False)
    _total_token_sum: int = field(default=0, init=False, repr=False)
    _request_latency_sum: float = field(default=0.0, init=False, repr=False)
    _active_program_sum: int = field(default=0, init=False, repr=False)
    _waiting_program_sum: int = field(default=0, init=False, repr=False)
    _input_token_growth_sum: int = field(default=0, init=False, repr=False)
    _input_token_growth_count: int = field(default=0, init=False, repr=False)
    _inter_request_gap_sum: float = field(default=0.0, init=False, repr=False)
    _pause_served_rounds: deque[int] = field(default_factory=deque, init=False, repr=False)
    _pause_served_rounds_sum: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate bounded-window controls and finite initial decision values."""
        if self.request_count < 0:
            raise ValueError("request_count must be non-negative")
        if not 1 <= self.min_request_window_size <= self.request_window_size <= self.max_request_window_size:
            raise ValueError("request window sizes must satisfy 1 <= min <= current <= max")
        if self.request_window_per_active_program <= 0 or self.max_evictions_per_update <= 0:
            raise ValueError("request window growth and eviction limits must be positive")
        if not math.isfinite(self.min_update_window_fraction) or not 0 < self.min_update_window_fraction <= 1:
            raise ValueError("min_update_window_fraction must be in (0, 1]")
        decision_values = (
            self.avg_prompt_tokens,
            self.avg_completion_tokens,
            self.avg_total_tokens,
            self.avg_router_queue_seconds,
            self.avg_request_latency_seconds,
            self.avg_active_programs,
            self.avg_waiting_programs,
            self.avg_segment_served_rounds_on_pause,
            self.avg_input_token_growth_per_round,
            self.avg_inter_request_gap_seconds,
            self.avg_reasoning_seconds,
            self.avg_acting_seconds,
            self.cold_prefill_cost_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in decision_values):
            raise ValueError("initial Progress-TTL decision values must be finite and non-negative")

    def update_request(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        request_latency_seconds: float,
        active_programs: int,
        waiting_programs: int,
        input_token_growth: int,
        inter_request_gap_seconds: float,
    ) -> None:
        """Update request-level rolling averages using one sanitized sample."""
        sample = self._sanitize_request_sample(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_latency_seconds=request_latency_seconds,
            active_programs=active_programs,
            waiting_programs=waiting_programs,
            input_token_growth=input_token_growth,
            inter_request_gap_seconds=inter_request_gap_seconds,
        )
        self.request_count += 1
        self.request_window_size = self._target_request_window_size(active_programs)
        self._append_request_sample(sample)
        self._evict_request_samples_to_target()
        self._refresh_request_averages_if_ready()

    def update_pause(self, *, served_rounds: int) -> None:
        """Update bounded segment statistics when a Program is paused."""
        sanitized_rounds = max(served_rounds, 0)
        self._pause_served_rounds.append(sanitized_rounds)
        self._pause_served_rounds_sum += sanitized_rounds
        while len(self._pause_served_rounds) > self.max_request_window_size:
            self._pause_served_rounds_sum -= self._pause_served_rounds.popleft()
        if len(self._pause_served_rounds) >= self._min_samples_for_update(self.request_window_size):
            self.avg_segment_served_rounds_on_pause = self._pause_served_rounds_sum / len(self._pause_served_rounds)

    def _target_request_window_size(self, active_programs: int) -> int:
        active_count = max(1, active_programs)
        target = active_count * max(1, self.request_window_per_active_program)
        return min(self.max_request_window_size, max(self.min_request_window_size, target))

    def _sanitize_request_sample(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        request_latency_seconds: float,
        active_programs: int,
        waiting_programs: int,
        input_token_growth: int,
        inter_request_gap_seconds: float,
    ) -> RequestStatSample:
        if not math.isfinite(request_latency_seconds) or not math.isfinite(inter_request_gap_seconds):
            raise ValueError("request latency and inter-request gap must be finite")
        prompt_value = max(prompt_tokens, 0)
        total_value = max(total_tokens, 0)
        completion_value = self._sanitize_completion_tokens(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_value,
            total_tokens=total_value,
        )
        return RequestStatSample(
            prompt_tokens=prompt_value,
            completion_tokens=completion_value,
            total_tokens=total_value,
            request_latency_seconds=max(request_latency_seconds, 0.0),
            active_programs=max(active_programs, 0),
            waiting_programs=max(waiting_programs, 0),
            input_token_growth=self._sanitize_input_token_growth(
                input_token_growth=input_token_growth,
                previous_total_tokens=max(0, total_value - max(input_token_growth, 0)),
            ),
            inter_request_gap_seconds=max(inter_request_gap_seconds, 0.0),
        )

    @staticmethod
    def _sanitize_completion_tokens(*, completion_tokens: int, prompt_tokens: int, total_tokens: int) -> int | None:
        completion_value = max(completion_tokens, 0)
        if total_tokens <= 0:
            return None
        if prompt_tokens <= 0 and completion_value >= total_tokens:
            return None
        if completion_value > total_tokens:
            return None
        return completion_value

    @staticmethod
    def _sanitize_input_token_growth(*, input_token_growth: int, previous_total_tokens: int) -> int | None:
        if previous_total_tokens <= 0 or input_token_growth <= 0:
            return None
        if input_token_growth < previous_total_tokens:
            return input_token_growth
        return previous_total_tokens

    def _append_request_sample(self, sample: RequestStatSample) -> None:
        self._request_samples.append(sample)
        self._prompt_token_sum += sample.prompt_tokens
        if sample.completion_tokens is not None:
            self._completion_token_sum += sample.completion_tokens
            self._completion_token_count += 1
        self._total_token_sum += sample.total_tokens
        self._request_latency_sum += sample.request_latency_seconds
        self._active_program_sum += sample.active_programs
        self._waiting_program_sum += sample.waiting_programs
        if sample.input_token_growth is not None:
            self._input_token_growth_sum += sample.input_token_growth
            self._input_token_growth_count += 1
        self._inter_request_gap_sum += sample.inter_request_gap_seconds

    def _remove_request_sample(self) -> None:
        sample = self._request_samples.popleft()
        self._prompt_token_sum -= sample.prompt_tokens
        if sample.completion_tokens is not None:
            self._completion_token_sum -= sample.completion_tokens
            self._completion_token_count -= 1
        self._total_token_sum -= sample.total_tokens
        self._request_latency_sum -= sample.request_latency_seconds
        self._active_program_sum -= sample.active_programs
        self._waiting_program_sum -= sample.waiting_programs
        if sample.input_token_growth is not None:
            self._input_token_growth_sum -= sample.input_token_growth
            self._input_token_growth_count -= 1
        self._inter_request_gap_sum -= sample.inter_request_gap_seconds

    def _evict_request_samples_to_target(self) -> None:
        excess = max(0, len(self._request_samples) - self.request_window_size)
        for _ in range(min(self.max_evictions_per_update, excess)):
            self._remove_request_sample()

    def _refresh_request_averages_if_ready(self) -> None:
        sample_count = len(self._request_samples)
        if sample_count < self._min_samples_for_update(self.request_window_size):
            return
        self.avg_prompt_tokens = self._prompt_token_sum / sample_count
        if self._completion_token_count > 0:
            self.avg_completion_tokens = self._completion_token_sum / self._completion_token_count
        self.avg_total_tokens = self._total_token_sum / sample_count
        self.avg_request_latency_seconds = self._request_latency_sum / sample_count
        self.avg_active_programs = self._active_program_sum / sample_count
        self.avg_waiting_programs = self._waiting_program_sum / sample_count
        if self._input_token_growth_count > 0:
            self.avg_input_token_growth_per_round = self._input_token_growth_sum / self._input_token_growth_count
        self.avg_inter_request_gap_seconds = self._inter_request_gap_sum / sample_count

    def _min_samples_for_update(self, window_size: int) -> int:
        return max(1, int(window_size * self.min_update_window_fraction))
