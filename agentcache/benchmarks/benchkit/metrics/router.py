"""Aggregate Router event evidence."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterMetrics:
    """Store Router event counts grouped by event name."""

    events: Mapping[str, int]


def aggregate_router_events(events: Iterable[Mapping[str, object]]) -> RouterMetrics:
    """Count raw Router events by their event field."""

    return RouterMetrics(dict(Counter(str(event.get("event", "unknown")) for event in events)))
