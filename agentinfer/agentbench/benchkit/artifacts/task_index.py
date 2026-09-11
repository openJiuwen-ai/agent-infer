# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Build a run-level task index from already-written task results.

The runner writes ``task_position`` into each task's ``result.json``. Index
rows are sorted by that position, and failed tasks are grouped by termination
reason, using ``"unknown"`` when no reason is recorded.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from .task_result import TaskResultArtifact

_TASK_RESULT_ADAPTER = TypeAdapter(TaskResultArtifact)


@dataclass(frozen=True)
class TaskIndexRow:
    """One task's position, identity, and failure cause."""

    task_position: int
    instance_id: str
    session_id: str
    outcome: str
    termination_reason: str | None
    error: dict[str, str] | None


@dataclass(frozen=True)
class FailureSummary:
    """Aggregate failed tasks by termination reason."""

    by_termination_reason: dict[str, int]
    total_failed: int


@dataclass(frozen=True)
class TaskIndex:
    """Run-level lightweight task-session mapping with failure roll-up."""

    schema_version: Literal["1"]
    run_id: str
    tasks: tuple[TaskIndexRow, ...]
    failure_summary: FailureSummary

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_task_index(run_id: str, rows: Iterable[tuple[int, TaskResultArtifact]]) -> TaskIndex:
    """Assemble a task index from positioned task results.

    Positions and session ids must be unique. Failed tasks without a termination
    reason are collected under the ``"unknown"`` bucket so that ``total_failed``
    always equals the reason-count total.
    """

    seen_positions: set[int] = set()
    seen_sessions: set[str] = set()
    index_rows: list[TaskIndexRow] = []
    by_reason: dict[str, int] = {}
    total_failed = 0
    for position, result in rows:
        if position in seen_positions:
            raise ValueError(f"duplicate task_position: {position}")
        seen_positions.add(position)
        if result.session_id in seen_sessions:
            raise ValueError(f"duplicate session_id: {result.session_id}")
        seen_sessions.add(result.session_id)
        if result.outcome == "failed":
            total_failed += 1
            bucket = result.termination_reason or "unknown"
            by_reason[bucket] = by_reason.get(bucket, 0) + 1
        index_rows.append(
            TaskIndexRow(
                position,
                result.instance_id,
                result.session_id,
                result.outcome,
                result.termination_reason,
                result.error,
            )
        )
    index_rows.sort(key=lambda row: row.task_position)
    return TaskIndex("1", run_id, tuple(index_rows), FailureSummary(by_reason, total_failed))


def load_task_results(output_dir: Path) -> list[TaskResultArtifact]:
    """Load and validate per-task ``result.json`` files under a run directory."""

    tasks_root = output_dir / "tasks"
    if not tasks_root.is_dir():
        return []
    results: list[TaskResultArtifact] = []
    for entry in sorted(tasks_root.iterdir()):
        result_path = entry / "result.json"
        if not result_path.is_file():
            continue
        try:
            results.append(_TASK_RESULT_ADAPTER.validate_json(result_path.read_bytes()))
        except (OSError, ValidationError) as exc:
            raise ValueError(f"task result does not match the expected schema: {result_path}") from exc
    return results


def build_run_task_index(run_id: str, output_dir: Path) -> TaskIndex:
    """Build a run-level index from already-written task results."""

    results = load_task_results(output_dir)
    return build_task_index(run_id, ((result.task_position, result) for result in results))
