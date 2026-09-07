# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify combining existing run summaries into a CSV export."""

import csv
import json
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.summarize import combine_summaries


def _run(tmp_path: Path, name: str, summary: object) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_combine_summaries_places_each_run_in_one_column(tmp_path: Path) -> None:
    run1 = _run(
        tmp_path,
        "run1",
        {
            "schema_version": "1",
            "run_id": "run1",
            "tasks": {"completed": 2},
            "requests": {"prefix_cache_hit_rate": None},
            "metadata": ["a", "b"],
        },
    )
    run2 = _run(
        tmp_path,
        "run2",
        {
            "schema_version": "1",
            "run_id": "run2",
            "tasks": {"completed": 3, "failed": 1},
            "lifecycle": {"error": ""},
        },
    )
    output = tmp_path / "combined.csv"

    combine_summaries([run2, run1], output=output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert list(next(iter(rows.values()))) == ["metric", "run2", "run1"]
    assert rows["tasks.completed"] == {"metric": "tasks.completed", "run2": "3", "run1": "2"}
    assert rows["tasks.failed"] == {"metric": "tasks.failed", "run2": "1", "run1": "N/A"}
    assert rows["metadata"]["run1"] == '["a","b"]'
    assert rows["requests.prefix_cache_hit_rate"]["run1"] == "null"
    assert rows["lifecycle.error"]["run2"] == '""'


def test_combine_summaries_validates_all_inputs_before_writing(tmp_path: Path) -> None:
    valid = _run(tmp_path, "valid", {"run_id": "valid"})
    invalid = _run(tmp_path, "invalid", ["not", "an", "object"])
    output = tmp_path / "combined.csv"

    with pytest.raises(ValueError, match="summary must be a JSON object"):
        combine_summaries([valid, invalid], output=output)

    assert not output.exists()


def test_combine_summaries_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    run1 = _run(tmp_path, "run1", {"run_id": "duplicate"})
    run2 = _run(tmp_path, "run2", {"run_id": "duplicate"})

    with pytest.raises(ValueError, match="run_id values must be unique"):
        combine_summaries([run1, run2], output=tmp_path / "combined.csv")


def test_combine_summaries_reports_missing_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        combine_summaries([run_dir], output=tmp_path / "combined.csv")
