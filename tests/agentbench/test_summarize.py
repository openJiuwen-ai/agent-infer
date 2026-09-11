# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify combining existing run summaries into a CSV export."""

import csv
import json
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.distribution_plot import _task_request_buckets
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


def _distribution_run(tmp_path: Path, name: str, value: float) -> Path:
    run_dir = _run(tmp_path, name, {"run_id": name, "tasks": {"completed": 1}})
    rows = []
    for level, metric, sample, instance_id, request_id in (
        ("task", "task_duration_seconds", value, "task-1", "N/A"),
        ("request", "latency_seconds", value, "task-1", "request-1"),
        ("request", "latency_seconds", value * 40, "task-2", "request-2"),
    ):
        rows.append(
            {
                "run_id": name,
                "level": level,
                "metric": metric,
                "value": sample,
                "unit": "seconds",
                "instance_id": instance_id,
                "session_id": "session",
                "actor_id": "N/A",
                "actor_role": "N/A",
                "request_id": request_id,
                "task_outcome": "completed",
                "has_patch": True,
                "request_status": "N/A" if level == "task" else "success",
                "value_status": "valid",
                "exclusion_reason": "N/A",
            }
        )
    with (run_dir / "distribution_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def test_combine_summaries_writes_multi_run_distribution_plots(tmp_path: Path) -> None:
    run1 = _distribution_run(tmp_path, "run1", 1.0)
    run2 = _distribution_run(tmp_path, "run2", 2.0)
    run3 = _distribution_run(tmp_path, "run3", 3.0)
    figures_dir = tmp_path / "figures"

    combine_summaries(
        [run1, run2, run3],
        output=tmp_path / "combined.csv",
        figures_dir=figures_dir,
    )

    plots = {path.name for path in figures_dir.glob("*.png")}
    assert plots == {"distribution_summary.png", "task_request_breakdown.png"}
    assert not (figures_dir / "distribution_samples.csv").exists()
    assert not (figures_dir / "plots").exists()


def test_task_request_buckets_rank_tasks_independently_per_run() -> None:
    rows = []
    for run_id, task_id, values in (
        ("baseline", "shared-slow", (40.0, 60.0)),
        ("baseline", "baseline-fast", (5.0,)),
        ("candidate", "shared-slow", (2.0,)),
        ("candidate", "candidate-slow", (20.0, 30.0)),
    ):
        rows.extend(
            {
                "run_id": run_id,
                "level": "request",
                "metric": "latency_seconds",
                "value": str(value),
                "instance_id": task_id,
                "value_status": "valid",
            }
            for value in values
        )

    run_ids, ranked_tasks, grouped = _task_request_buckets(rows)

    assert run_ids == ["baseline", "candidate"]
    assert ranked_tasks == {
        "baseline": ["baseline-fast", "shared-slow"],
        "candidate": ["shared-slow", "candidate-slow"],
    }
    assert grouped["baseline"][ranked_tasks["baseline"][0]][0].sum() == 1
    assert grouped["baseline"][ranked_tasks["baseline"][0]][1].sum() == 5
    assert grouped["candidate"][ranked_tasks["candidate"][0]][0].sum() == 1
    assert grouped["candidate"][ranked_tasks["candidate"][0]][1].sum() == 2


def test_distribution_analysis_requires_two_runs(tmp_path: Path) -> None:
    run = _distribution_run(tmp_path, "run", 1.0)

    with pytest.raises(ValueError, match="at least two runs"):
        combine_summaries(
            [run],
            output=tmp_path / "combined.csv",
            figures_dir=tmp_path / "analysis",
        )

    assert not (tmp_path / "combined.csv").exists()


def test_combine_summaries_validates_distribution_columns_before_output(tmp_path: Path) -> None:
    run1 = _distribution_run(tmp_path, "run1", 1.0)
    run2 = _distribution_run(tmp_path, "run2", 2.0)
    path = run2 / "distribution_samples.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name for name in rows[0] if name != "unit"])
        writer.writeheader()
        writer.writerows([{key: value for key, value in rows[0].items() if key != "unit"}])

    with pytest.raises(ValueError, match="columns do not match"):
        combine_summaries(
            [run1, run2],
            output=tmp_path / "combined.csv",
            figures_dir=tmp_path / "analysis",
        )

    assert not (tmp_path / "combined.csv").exists()
    assert not (tmp_path / "figures").exists()
