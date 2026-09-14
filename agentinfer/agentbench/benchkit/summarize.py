# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Combine existing benchmark summaries into a CSV export."""

import csv
import json
from pathlib import Path

from .common import atomic_write_csv


def combine_summaries(run_dirs: list[Path], *, output: Path, figures_dir: Path | None = None) -> None:
    """Combine existing run summaries and optionally plot their distribution samples."""

    records = [_load_record(run_dir.resolve()) for run_dir in run_dirs]
    run_ids = [str(summary.get("run_id", run_dir)) for run_dir, summary in records]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("summary run_id values must be unique")
    if figures_dir is not None:
        if len(run_dirs) < 2:
            raise ValueError("distribution analysis requires at least two runs")
        distribution = _load_distributions([run_dir.resolve() for run_dir in run_dirs], run_ids)
    else:
        distribution = None
    flattened = [_flatten(summary) for _, summary in records]
    metrics = sorted({metric for summary in flattened for metric in summary})
    rows = [
        {
            "metric": metric,
            **{run_id: summary.get(metric, "N/A") for run_id, summary in zip(run_ids, flattened, strict=False)},
        }
        for metric in metrics
    ]
    atomic_write_csv(output, rows, ["metric", *run_ids])
    if distribution is not None:
        from .distribution_plot import write_distribution_plots_from_rows

        write_distribution_plots_from_rows(distribution, figures_dir)


def _load_distributions(run_dirs: list[Path], run_ids: list[str]) -> list[dict[str, str]]:
    expected_columns = None
    rows = []
    for run_dir, run_id in zip(run_dirs, run_ids, strict=True):
        path = run_dir.resolve() / "distribution_samples.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            if not columns or "run_id" not in columns:
                raise ValueError(f"distribution samples missing run_id column: {path}")
            if expected_columns is None:
                expected_columns = columns
            elif columns != expected_columns:
                raise ValueError(f"distribution sample columns do not match: {path}")
            for row in reader:
                row["run_id"] = run_id
                rows.append(row)
    return rows


def _load_record(run_dir: Path) -> tuple[str, dict[str, object]]:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a JSON object: {summary_path}")
    return str(run_dir), summary


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict) and value:
        flattened = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, name))
        return flattened
    if value == "":
        return {prefix: '""'}
    if value is None:
        return {prefix: "null"}
    if isinstance(value, (dict, list, tuple)):
        return {prefix: json.dumps(value, ensure_ascii=False, separators=(",", ":"))}
    return {prefix: value}
