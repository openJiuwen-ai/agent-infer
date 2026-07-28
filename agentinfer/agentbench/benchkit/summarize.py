"""Combine existing benchmark summaries into a CSV export."""

import csv
import json
from pathlib import Path


def combine_summaries(run_dirs: list[Path], *, output: Path) -> None:
    """Combine existing run summaries without recomputing their metrics."""

    records = [_load_record(run_dir.resolve()) for run_dir in run_dirs]
    run_ids = [str(summary.get("run_id", run_dir)) for run_dir, summary in records]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("summary run_id values must be unique")
    flattened = [_flatten(summary) for _, summary in records]
    metrics = sorted({metric for summary in flattened for metric in summary})
    rows = [
        {
            "metric": metric,
            **{run_id: summary.get(metric, "N/A") for run_id, summary in zip(run_ids, flattened, strict=False)},
        }
        for metric in metrics
    ]
    _write_csv(output, rows, ["metric", *run_ids])


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


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
