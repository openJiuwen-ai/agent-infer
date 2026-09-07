# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit import dataset


def test_load_tasks_reads_swebench_uppercase_test_fields(tmp_path: Path) -> None:
    index = tmp_path / "instances.jsonl"
    selection = tmp_path / "tasks.txt"
    index.write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "repo": "owner/repo",
                "base_commit": "abc123",
                "problem_statement": "fix it",
                "FAIL_TO_PASS": '["tests/test_a.py::test_fix"]',
                "PASS_TO_PASS": ["tests/test_a.py::test_existing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection.write_text("task-a\n", encoding="utf-8")

    tasks = dataset.load_tasks(index, selection)

    assert tasks[0].fail_to_pass == ("tests/test_a.py::test_fix",)
    assert tasks[0].pass_to_pass == ("tests/test_a.py::test_existing",)


def test_load_tasks_preserves_selection_order_and_applies_total(tmp_path: Path) -> None:
    index = tmp_path / "instances.jsonl"
    selection = tmp_path / "tasks.txt"
    index.write_text(
        "\n".join(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "repo": "owner/repo",
                    "base_commit": instance_id,
                    "problem_statement": "fix it",
                }
            )
            for instance_id in ("task-a", "task-b", "task-c")
        )
        + "\n",
        encoding="utf-8",
    )
    selection.write_text("# priority order\ntask-c\n\ntask-a\ntask-b\n", encoding="utf-8")

    tasks = dataset.load_tasks(index, selection, total=2)

    assert [task.instance_id for task in tasks] == ["task-c", "task-a"]


@pytest.mark.parametrize("instance_id", ["../escape", "/absolute", "C:\\escape", "CON", "task/name"])
def test_task_rejects_unsafe_instance_id(instance_id: str) -> None:
    with pytest.raises(ValueError, match="instance_id"):
        dataset.Task(instance_id, "owner/repo", "abc", "fix it")


def test_load_tasks_rejects_duplicate_selection(tmp_path: Path) -> None:
    index = tmp_path / "instances.jsonl"
    selection = tmp_path / "tasks.txt"
    index.write_text('{"instance_id":"task-a"}\n', encoding="utf-8")
    selection.write_text("task-a\ntask-a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate selected"):
        dataset.load_tasks(index, selection)


def test_prepare_swebench_writes_local_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dataset,
        "_load_swebench_verified",
        lambda: [
            {
                "instance_id": "task-a",
                "repo": "owner/repo",
                "base_commit": "abc123",
                "problem_statement": "fix it",
                "FAIL_TO_PASS": ["tests/test_a.py::test_fix"],
                "PASS_TO_PASS": ["tests/test_a.py::test_existing"],
            },
            {
                "instance_id": "task-b",
                "repo": "owner/repo",
                "base_commit": "def456",
                "problem_statement": "fix it too",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            },
        ],
    )

    prepared = dataset.prepare_swebench(tmp_path / "swebench")

    assert prepared.rows == 2
    assert prepared.index_path.read_text(encoding="utf-8").count("\n") == 2
    assert prepared.selection_path.read_text(encoding="utf-8").splitlines() == ["task-a", "task-b"]
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "princeton-nlp/SWE-bench_Verified"
    assert manifest["split"] == "test"
    assert manifest["rows"] == 2


def test_prepare_swebench_rejects_non_empty_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "swebench"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        dataset.prepare_swebench(output_dir)
