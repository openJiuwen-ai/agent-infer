# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Bounded slow-request lifecycle warnings without hot-path logging or snapshots."""

from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from agentinfer.scheduling.domain import ProgramRef

_LOGGER = logging.getLogger(__name__)
_MAX_SCAN = 256
_MAX_EVENT_CHARS = 8192


@dataclass
class _RequestTiming:
    program: ProgramRef
    external_program_id: str | None
    arrived_at: float
    prompt_tokens: int | None
    admitted_at: float | None = None
    first_native_at: float | None = None
    first_output_at: float | None = None
    native_running: bool | None = None
    native_preemptions: int = 0
    cached_prefix_tokens: int | None = None
    hbm_cached_prefix_tokens: int | None = None
    last_warning_at: float | None = None


class RequestDiagnostics:
    """Track live requests in O(1) per lifecycle event and warn on slow first output.

    The strategy opts in by returning one tracker; a missing tracker is the
    disabled path. Polling rotates a bounded number of live entries, so bursts
    cannot monopolize the engine and repeatedly warned heads cannot starve peers.
    Finished entries leave both indexes immediately; no historical queue remains.
    Context is obtained only for emitted warnings and must contain diagnostic
    facts, never prompt contents. This class does not build scheduling snapshots.
    """

    def __init__(
        self,
        *,
        threshold_seconds: float = 30.0,
        repeat_seconds: float = 60.0,
        poll_interval_seconds: float = 5.0,
        max_events_per_poll: int = 8,
        context_provider: Callable[[ProgramRef, float], dict[str, object]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        for name, value in (
            ("threshold_seconds", threshold_seconds),
            ("repeat_seconds", repeat_seconds),
            ("poll_interval_seconds", poll_interval_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(max_events_per_poll, bool)
            or not isinstance(max_events_per_poll, int)
            or max_events_per_poll <= 0
        ):
            raise ValueError("max_events_per_poll must be a positive integer")
        self.threshold_seconds = threshold_seconds
        self.repeat_seconds = repeat_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.max_events_per_poll = max_events_per_poll
        self._context_provider = context_provider
        self._logger = logger if logger is not None else _LOGGER
        self._requests: OrderedDict[str, _RequestTiming] = OrderedDict()
        self._program_requests: dict[ProgramRef, set[str]] = {}
        self._next_poll_at = 0.0

    @property
    def active_count(self) -> int:
        """Return the number of live tracked attempts, excluding all history."""
        return len(self._requests)

    def arrive(
        self,
        request_id: str,
        program: ProgramRef,
        now: float,
        *,
        external_program_id: str | None = None,
        prompt_tokens: int | None = None,
    ) -> None:
        """Record one accepted attempt before its immediate admission decision."""
        if request_id in self._requests:
            return
        self._requests[request_id] = _RequestTiming(program, external_program_id, now, prompt_tokens)
        self._program_requests.setdefault(program, set()).add(request_id)

    def admitted(self, request_id: str, now: float) -> None:
        """Record handoff from the scheduler's RequestPool to the native adapter."""
        timing = self._requests.get(request_id)
        if timing is not None and timing.admitted_at is None:
            timing.admitted_at = now

    def native_execution(self, request_id: str, now: float, *, running: bool) -> None:
        """Observe the first native schedule and subsequent preemption boundaries."""
        timing = self._requests.get(request_id)
        if timing is None or timing.native_running is running:
            return
        if running and timing.first_native_at is None:
            timing.first_native_at = now
        elif not running and timing.native_running is True:
            timing.native_preemptions += 1
        timing.native_running = running

    def cache_observation(
        self,
        request_id: str,
        *,
        cached_prefix_tokens: int,
        hbm_cached_prefix_tokens: int | None = None,
    ) -> None:
        """Retain scalar native cache facts without inspecting cache ownership."""
        timing = self._requests.get(request_id)
        if timing is not None:
            timing.cached_prefix_tokens = cached_prefix_tokens
            timing.hbm_cached_prefix_tokens = hbm_cached_prefix_tokens

    def first_output(self, request_id: str, now: float) -> None:
        """Emit at most one first-output summary, and only for a slow request."""
        timing = self._requests.get(request_id)
        if timing is None or timing.first_output_at is not None:
            return
        timing.first_output_at = now
        if now - timing.arrived_at >= self.threshold_seconds:
            self._emit(request_id, timing, now, "first_output")

    def finish(self, request_id: str, now: float, *, outcome: str = "completed") -> None:
        """Forget an attempt; summarize only slow first output or an unfinished wait."""
        timing = self._requests.pop(request_id, None)
        if timing is None:
            return
        siblings = self._program_requests[timing.program]
        siblings.discard(request_id)
        if not siblings:
            del self._program_requests[timing.program]
        first_output_or_end = timing.first_output_at if timing.first_output_at is not None else now
        if first_output_or_end - timing.arrived_at >= self.threshold_seconds:
            self._emit(request_id, timing, now, outcome)

    def release(self, program: ProgramRef, now: float) -> None:
        """Remove only the exact released generation, including abnormal cleanup."""
        for request_id in tuple(self._program_requests.get(program, ())):
            self.finish(request_id, now, outcome="released")

    def poll(self, now: float) -> None:
        """Warn about due slow attempts with bounded scanning and output."""
        if now < self._next_poll_at:
            return
        self._next_poll_at = now + self.poll_interval_seconds
        emitted = 0
        for _ in range(min(len(self._requests), _MAX_SCAN)):
            request_id, timing = next(iter(self._requests.items()))
            self._requests.move_to_end(request_id)
            if timing.first_output_at is not None:
                continue
            if now - timing.arrived_at < self.threshold_seconds:
                continue
            if timing.last_warning_at is not None and now - timing.last_warning_at < self.repeat_seconds:
                continue
            self._emit(request_id, timing, now, "poll")
            emitted += 1
            if emitted >= self.max_events_per_poll:
                break

    @staticmethod
    def _phase(timing: _RequestTiming) -> str:
        if timing.admitted_at is None:
            return "request_pool"
        if timing.native_running is False:
            return "native_waiting"
        if timing.first_output_at is not None:
            return "native_decode"
        return "native_prefill" if timing.first_native_at is not None else "native_waiting"

    def _emit(self, request_id: str, timing: _RequestTiming, now: float, trigger: str) -> None:
        timing.last_warning_at = now
        context: object = None
        if self._context_provider is not None:
            try:
                context = _bounded_value(self._context_provider(timing.program, now))
            except Exception as exc:
                # Diagnostics must not change scheduling if optional context fails.
                context = {"context_error": type(exc).__name__}
        native_wait = None
        if timing.admitted_at is not None:
            if timing.first_native_at is not None:
                native_wait = max(0.0, timing.first_native_at - timing.admitted_at)
            elif timing.first_output_at is None:
                native_wait = max(0.0, now - timing.admitted_at)
        row = {
            "event": "mori_slow_request",
            "trigger": trigger[:64],
            "request_id": request_id[:128],
            "program_id": timing.program.program_id[:128],
            "generation": timing.program.generation,
            "external_program_id": timing.external_program_id[:128] if timing.external_program_id is not None else None,
            "phase": self._phase(timing),
            "elapsed_seconds": max(0.0, now - timing.arrived_at),
            "request_pool_seconds": max(
                0.0, (timing.admitted_at if timing.admitted_at is not None else now) - timing.arrived_at
            ),
            "native_wait_seconds": native_wait,
            # This elapsed interval includes native preemptions; it is not GPU compute time.
            "scheduled_to_first_output_seconds": (
                max(
                    0.0,
                    (timing.first_output_at if timing.first_output_at is not None else now) - timing.first_native_at,
                )
                if timing.first_native_at is not None
                else None
            ),
            "decode_seconds": max(0.0, now - timing.first_output_at) if timing.first_output_at is not None else None,
            "arrival_at_monotonic_s": timing.arrived_at,
            "admitted_at_monotonic_s": timing.admitted_at,
            "first_native_at_monotonic_s": timing.first_native_at,
            "first_output_at_monotonic_s": timing.first_output_at,
            "first_output_observation": (
                "observed"
                if timing.first_output_at is not None
                else "missing_at_completion"
                if trigger == "completed"
                else "not_yet_observed"
            ),
            "ttft_seconds": (
                max(0.0, timing.first_output_at - timing.arrived_at) if timing.first_output_at is not None else None
            ),
            "native_running": timing.native_running,
            "native_preemptions": timing.native_preemptions,
            "prompt_tokens": timing.prompt_tokens,
            "cached_prefix_tokens": timing.cached_prefix_tokens,
            "hbm_cached_prefix_tokens": timing.hbm_cached_prefix_tokens,
            "context": context,
        }
        encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        if len(encoded) > _MAX_EVENT_CHARS:
            row["context"] = {"truncated": True}
            encoded = json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        self._logger.warning("MORI slow request %s", encoded)


def _bounded_value(value: object, depth: int = 0) -> object:
    """Bound optional structured context without serializing arbitrary objects."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 128 else "<integer out of range>"
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:256]
    if depth >= 3:
        return "<truncated>"
    if isinstance(value, Mapping):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 16:
                break
            if isinstance(key, str):
                result[key[:64]] = _bounded_value(item, depth + 1)
        return result
    if isinstance(value, list | tuple):
        return [_bounded_value(item, depth + 1) for item in value[:8]]
    return "<unsupported>"
