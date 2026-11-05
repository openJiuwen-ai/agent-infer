# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Internal optional clocks and event sinks; no public configuration or artifacts."""

import asyncio
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .planner import ReplayPlanNode, ReplayTaskPlan


class ReplayClock(Protocol):
    """Executor timing; a fake transport must return completion times on the same clock."""

    def monotonic(self) -> float:
        """Return seconds on a nondecreasing clock."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend until at least the nonnegative interval has elapsed."""
        ...


class RealClock:
    """Use the existing monotonic clock and asyncio scheduler by default."""

    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(asyncio.sleep)


class ReplayObserver(Protocol):
    """Optional internal event sink; event dictionaries are never public results."""

    def emit(self, event: dict[str, object]) -> None:
        """Submit without blocking/raising; the sink owner checks recording errors."""
        ...


def emit(
    observer: ReplayObserver | None,
    clock: ReplayClock | None,
    event: str,
    task: "ReplayTaskPlan",
    node: "ReplayPlanNode | None" = None,
    *,
    at: float | None = None,
    **fields: object,
) -> None:
    """Submit identity and time plus event-specific scalar diagnostic fields.

    A missing clock uses real monotonic time. Callers may supply an already
    sampled timestamp to keep the event and result identical. No sink means no
    clock sampling or event payload construction.
    """
    if observer is None:
        return
    if at is None:
        at = clock.monotonic() if clock is not None else time.monotonic()
    observer.emit(
        {
            "event": event,
            "t_monotonic": at,
            "task_index": getattr(task, "task_index", None),
            "task_id": task.task_id if hasattr(task, "task_id") else None,
            "runtime_session_id": task.runtime_session_id,
            "source_key": node.source_key if node is not None and hasattr(node, "source_key") else None,
            "runtime_request_id": node.runtime_request_id if node is not None else None,
            **fields,
        }
    )
