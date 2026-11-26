# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Wake the vLLM EngineCore owner for MORI's periodic check."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from threading import Timer

from agentinfer.agentcache.core.program_scheduling import MoriProgramAdmissionController

logger = logging.getLogger(__name__)
_DEADLINE = "agentinfer.mori.deadline"


class MoriTimer:
    """MORI-owned timer whose worker only posts to the Core input queue."""

    def __init__(self, admission: MoriProgramAdmissionController, wake: Callable[[], None]) -> None:
        self._admission = admission
        self._wake = wake
        self._timer: Timer | None = None

    def _arm(self, next_at: float) -> None:
        self._timer = Timer(max(0.0, next_at - time.monotonic()), self._wake)
        self._timer.daemon = True
        self._timer.start()

    def start(self) -> None:
        self._arm(time.monotonic() + self._admission.config.interval_seconds)

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def on_deadline(self) -> None:
        """Observe and run a tick on the Core owner, then arm the next deadline."""
        started = time.monotonic()
        applied = False
        try:
            self._admission.observe_backend()
            applied = self._admission.tick()
        except Exception:
            logger.exception("MORI periodic check failed")
        # Keep the v3 timing rule in the embedded implementation. A successful
        # KV operation starts the interval at acknowledgement.
        start = time.monotonic() if applied else started
        self._arm(start + self._admission.config.interval_seconds)


def install_mori_timer() -> None:
    """Use vLLM's local wakeup queue so deadlines also run when the engine is idle."""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCoreProc

    run_busy_loop = EngineCoreProc.run_busy_loop
    if getattr(run_busy_loop, "_agentinfer_mori", False):
        return
    handle = EngineCoreProc._handle_client_request
    wakeup = EngineCoreRequestType.WAKEUP

    @wraps(handle)
    def handle_request(self, request_type, payload):
        if request_type == wakeup and payload == _DEADLINE:
            timer = getattr(self, "_agentinfer_mori_timer", None)
            if timer is not None and self.is_running():
                timer.on_deadline()
            return None
        return handle(self, request_type, payload)

    @wraps(run_busy_loop)
    def run_with_timer(self, *args, **kwargs):
        admission = getattr(getattr(self.scheduler, "_agentcache", None), "program_admission", None)
        if admission is None:
            return run_busy_loop(self, *args, **kwargs)

        def wake() -> None:
            self.input_queue.put_nowait((wakeup, _DEADLINE))

        timer = MoriTimer(admission, wake)
        self._agentinfer_mori_timer = timer
        timer.start()
        try:
            return run_busy_loop(self, *args, **kwargs)
        finally:
            timer.close()
            self._agentinfer_mori_timer = None

    vars(run_with_timer)["_agentinfer_mori"] = True
    EngineCoreProc._handle_client_request = handle_request
    EngineCoreProc.run_busy_loop = run_with_timer
