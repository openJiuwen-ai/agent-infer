# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Task result aggregation."""

from collections.abc import Iterable

from numpy import quantile
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from ...agents.contracts import AgentRunOutcome, AgentRunResult


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TaskMetrics:
    """Store aggregate execution outcomes, patch counts, and durations."""

    completed: int
    failed: int
    with_patch: int
    duration_seconds: dict[str, float | None]


def aggregate_task_results(results: Iterable[AgentRunResult]) -> TaskMetrics:
    """Aggregate execution outcomes independently from correctness evidence."""

    rows = list(results)
    durations = [row.duration_seconds for row in rows]
    return TaskMetrics(
        sum(row.outcome == AgentRunOutcome.COMPLETED for row in rows),
        sum(row.outcome != AgentRunOutcome.COMPLETED for row in rows),
        sum(row.has_patch for row in rows),
        {
            "mean": sum(durations) / len(durations) if durations else None,
            "p50": float(quantile(durations, 0.5)) if durations else None,
            "p95": float(quantile(durations, 0.95)) if durations else None,
            "p99": float(quantile(durations, 0.99)) if durations else None,
        },
    )
