# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Adjacent-request interval analysis for BenchKit ``requests.jsonl`` records."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from numpy import quantile


@dataclass(frozen=True)
class RequestPoint:
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class Distribution:
    count: int
    minimum: float | None
    mean: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    maximum: float | None


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def load_request_intervals(path: Path) -> tuple[float, ...]:
    """Load intervals between adjacent requests per ``(session_id, actor_id)`` group."""

    groups: dict[tuple[str, str], list[RequestPoint]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            session_id = row.get("session_id")
            actor_id = row.get("actor_id")
            if not isinstance(session_id, str) or not isinstance(actor_id, str):
                continue
            started_at = parse_timestamp(row.get("started_at"))
            finished_at = parse_timestamp(row.get("finished_at"))
            if started_at is None or finished_at is None:
                continue
            groups[(session_id, actor_id)].append(RequestPoint(started_at=started_at, finished_at=finished_at))

    intervals: list[float] = []
    for requests in groups.values():
        requests.sort(key=lambda item: item.started_at)
        for previous, current in zip(requests, requests[1:], strict=False):
            intervals.append((current.started_at - previous.finished_at).total_seconds())
    return tuple(intervals)


def distribution(values: Sequence[float]) -> Distribution:
    if not values:
        return Distribution(0, None, None, None, None, None, None, None)
    ordered = sorted(values)
    return Distribution(
        count=len(ordered),
        minimum=ordered[0],
        mean=statistics.fmean(ordered),
        p50=float(quantile(ordered, 0.50)),
        p90=float(quantile(ordered, 0.90)),
        p95=float(quantile(ordered, 0.95)),
        p99=float(quantile(ordered, 0.99)),
        maximum=ordered[-1],
    )


GLOBAL_INTERVAL_FIELDS = (
    "minimum",
    "mean",
    "p50",
    "p90",
    "p95",
    "p99",
    "maximum",
)


def global_interval_distribution(path: Path) -> Distribution:
    return distribution(load_request_intervals(path))
