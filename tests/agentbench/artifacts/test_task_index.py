# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify the run-level task-session mapping index and failure aggregation."""

import json
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.artifacts.task_index import (
    build_run_task_index,
    build_task_index,
    load_task_results,
)
from agentinfer.agentbench.benchkit.artifacts.task_result import (
    AgentIdentity,
    TaskResultArtifact,
)
from agentinfer.agentbench.benchkit.metrics.request import ObservedTopology


def _topology(session_id: str = "s") -> ObservedTopology:
    return ObservedTopology(session_id, (), {})


def _result(
    instance_id: str,
    session_id: str,
    outcome: str,
    termination_reason: str | None,
    *,
    task_position: int = 0,
    error: dict[str, str] | None = None,
) -> TaskResultArtifact:
    return TaskResultArtifact(
        schema_version="1",
        instance_id=instance_id,
        session_id=session_id,
        agent=AgentIdentity("claude", "single"),
        outcome=outcome,
        termination_reason=termination_reason,
        duration_seconds=10.0,
        has_patch=True,
        patch_bytes=5,
        topology=_topology(session_id),
        error=error,
        task_position=task_position,
        started_at="2026-08-13T00:00:00+00:00",
        finished_at="2026-08-13T00:01:00+00:00",
    )


def test_build_task_index_assigns_positions_and_failure_summary() -> None:
    rows = [
        (0, _result("a", "s0", "completed", None)),
        (1, _result("b", "s1", "failed", "timeout")),
        (2, _result("c", "s2", "failed", "idle_after_patch", error={"type": "X", "message": "m"})),
    ]

    index = build_task_index("run", rows)

    assert index.schema_version == "1"
    assert index.run_id == "run"
    assert [row.task_position for row in index.tasks] == [0, 1, 2]
    assert index.tasks[2].error == {"type": "X", "message": "m"}
    assert index.failure_summary.total_failed == 2
    assert index.failure_summary.by_termination_reason == {"timeout": 1, "idle_after_patch": 1}


def test_build_task_index_failed_without_termination_reason_bucketed_unknown() -> None:
    rows = [(0, _result("a", "s0", "failed", None))]

    index = build_task_index("run", rows)

    assert index.failure_summary.total_failed == 1
    assert index.failure_summary.by_termination_reason == {"unknown": 1}


def test_build_task_index_positions_need_not_be_contiguous() -> None:
    rows = [
        (0, _result("a", "s0", "completed", None)),
        (2, _result("b", "s2", "completed", None)),
        (5, _result("c", "s5", "failed", "timeout")),
    ]

    index = build_task_index("run", rows)

    assert [row.task_position for row in index.tasks] == [0, 2, 5]
    assert index.failure_summary.total_failed == 1


def test_build_task_index_rejects_duplicate_task_position() -> None:
    rows = [
        (0, _result("a", "s0", "completed", None)),
        (0, _result("b", "s1", "completed", None)),
    ]

    with pytest.raises(ValueError, match="duplicate task_position"):
        build_task_index("run", rows)


def test_build_task_index_rejects_duplicate_session_id() -> None:
    rows = [
        (0, _result("a", "s0", "completed", None)),
        (1, _result("b", "s0", "completed", None)),
    ]

    with pytest.raises(ValueError, match="duplicate session_id"):
        build_task_index("run", rows)


def test_build_task_index_handles_empty_input() -> None:
    index = build_task_index("run", [])

    assert index.tasks == ()
    assert index.failure_summary.total_failed == 0
    assert index.failure_summary.by_termination_reason == {}


def test_build_task_index_to_dict_round_trips() -> None:
    rows = [(0, _result("a", "s0", "completed", None))]

    index = build_task_index("run", rows)
    data = index.to_dict()
    serialized = json.dumps(data, ensure_ascii=False)
    parsed = json.loads(serialized)

    assert parsed["schema_version"] == "1"
    assert parsed["run_id"] == "run"
    assert parsed["tasks"][0]["instance_id"] == "a"
    assert parsed["failure_summary"]["total_failed"] == 0


def test_load_task_results_reads_position_from_result(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / "instance"
    task_dir.mkdir(parents=True)
    result = _result("instance", "sess", "completed", None, task_position=3)
    (task_dir / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")

    loaded = load_task_results(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].task_position == 3
    assert loaded[0].instance_id == "instance"


def test_build_run_task_index_reads_position_from_result_files(tmp_path: Path) -> None:
    # write two task results out of disk order; positions should drive the index order
    for position, instance_id in [(2, "b"), (0, "a")]:
        task_dir = tmp_path / "tasks" / instance_id
        task_dir.mkdir(parents=True)
        result = _result(instance_id, f"s{position}", "completed", None, task_position=position)
        (task_dir / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")

    index = build_run_task_index("run", tmp_path)

    assert [row.task_position for row in index.tasks] == [0, 2]
    assert [row.instance_id for row in index.tasks] == ["a", "b"]


def test_load_task_results_handles_missing_tasks_dir(tmp_path: Path) -> None:
    assert load_task_results(tmp_path) == []


def test_load_task_results_backfills_legacy_fields(tmp_path: Path) -> None:
    """A pre-change (schema-v1) ``result.json`` without ``task_position`` /
    ``started_at`` / ``finished_at`` must load (not raise), with the new
    fields backfilled to their sentinel defaults. This is the documented
    backward-compatibility contract for historical runs.
    """
    task_dir = tmp_path / "tasks" / "legacy"
    task_dir.mkdir(parents=True)
    legacy_result = {
        "schema_version": "1",
        "instance_id": "legacy",
        "session_id": "s-legacy",
        "agent": {"type": "claude", "profile": "single"},
        "outcome": "completed",
        "termination_reason": None,
        "duration_seconds": 5.0,
        "has_patch": False,
        "patch_bytes": 0,
        "registration": None,
        "cleanup": None,
        "topology": {"session_id": "s-legacy", "agent_ids": [], "agent_roles": {}},
        "error": None,
        # task_position / started_at / finished_at intentionally absent
    }
    (task_dir / "result.json").write_text(json.dumps(legacy_result, ensure_ascii=False), encoding="utf-8")

    loaded = load_task_results(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].instance_id == "legacy"
    assert loaded[0].task_position == -1
    assert loaded[0].started_at == ""
    assert loaded[0].finished_at == ""
