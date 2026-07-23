"""Aggregate Router event evidence."""

from collections import Counter
from collections.abc import Iterable, Mapping

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RouterMetrics:
    """Store Router event counts grouped by event name."""

    events: Mapping[str, int]


def aggregate_router_events(events: Iterable[Mapping[str, object]]) -> RouterMetrics:
    """Count raw Router events by their event field."""

    return RouterMetrics(dict(Counter(str(event.get("event", "unknown")) for event in events)))


def aggregate_router_window(
    start: Iterable[Mapping[str, object]], end: Iterable[Mapping[str, object]]
) -> RouterMetrics:
    """Count events added between cumulative Router snapshots."""

    before = Counter(str(event.get("event", "unknown")) for event in start)
    after = Counter(str(event.get("event", "unknown")) for event in end)
    if any(after[name] < count for name, count in before.items()):
        raise ValueError("Router event counters reset during benchmark run")
    return RouterMetrics(dict(after - before))
