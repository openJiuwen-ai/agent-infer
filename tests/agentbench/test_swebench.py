# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit import dataset, swebench


def test_prepare_swebench_writes_local_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        swebench,
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

    prepared = dataset.prepare_dataset("swebench", tmp_path / "swebench")

    assert prepared.rows == 2
    assert prepared.index_path.read_text(encoding="utf-8").count("\n") == 2
    assert prepared.selection_path.read_text(encoding="utf-8").splitlines() == ["task-a", "task-b"]
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "princeton-nlp/SWE-bench_Verified"
    assert manifest["split"] == "test"
    assert manifest["rows"] == 2
    rows = [json.loads(line) for line in prepared.index_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["FAIL_TO_PASS"] == rows[0]["fail_to_pass"] == ["tests/test_a.py::test_fix"]
    assert rows[0]["PASS_TO_PASS"] == rows[0]["pass_to_pass"] == ["tests/test_a.py::test_existing"]
    tasks = dataset.load_tasks(prepared.index_path, prepared.selection_path)
    assert [task.instance_id for task in tasks] == ["task-a", "task-b"]
    assert tasks[0].repo == "owner/repo"
    assert tasks[0].base_commit == "abc123"
    assert tasks[0].fail_to_pass == ("tests/test_a.py::test_fix",)
    assert tasks[0].pass_to_pass == ("tests/test_a.py::test_existing",)


def test_prepare_swebench_rejects_non_empty_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "swebench"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        dataset.prepare_dataset("swebench", output_dir)
