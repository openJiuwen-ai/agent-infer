# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Compare finalized benchmark run summaries."""

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .artifacts import RunManifest, RunSummary
from .metrics.vllm import parse_prometheus

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
_METRIC_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("completed_tasks", ("tasks", "completed")),
    ("failed_tasks", ("tasks", "failed")),
    ("tasks_with_patch", ("tasks", "with_patch")),
    ("mean_task_duration_seconds", ("tasks", "duration_seconds", "mean")),
    ("p50_task_duration_seconds", ("tasks", "duration_seconds", "p50")),
    ("p95_task_duration_seconds", ("tasks", "duration_seconds", "p95")),
    ("p99_task_duration_seconds", ("tasks", "duration_seconds", "p99")),
    ("run_wall_time_seconds", ("run_wall_time_seconds",)),
    ("request_throughput_per_second", ("request_throughput_per_second",)),
    ("input_token_throughput_per_second", ("input_token_throughput_per_second",)),
    ("output_token_throughput_per_second", ("output_token_throughput_per_second",)),
    ("requests", ("requests", "requests")),
    ("successful_requests", ("requests", "successful_requests")),
    ("failed_requests", ("requests", "failed_requests")),
    ("input_tokens", ("requests", "input_tokens")),
    ("output_tokens", ("requests", "output_tokens")),
    ("cache_creation_input_tokens", ("requests", "cache_creation_input_tokens")),
    ("cached_input_tokens", ("requests", "cached_input_tokens")),
    ("prefix_cache_hit_rate", ("requests", "prefix_cache_hit_rate")),
    ("vllm_prefix_cache_hit_rate", ("vllm", "prefix_cache_hit_rate")),
    ("vllm_prompt_token_hit_rate", ("vllm", "prompt_token_hit_rate")),
    ("latency_mean_seconds", ("requests", "latency_seconds", "mean")),
    ("latency_p50_seconds", ("requests", "latency_seconds", "p50")),
    ("latency_p95_seconds", ("requests", "latency_seconds", "p95")),
    ("latency_p99_seconds", ("requests", "latency_seconds", "p99")),
    ("ttft_mean_seconds", ("requests", "ttft_seconds", "mean")),
    ("ttft_p50_seconds", ("requests", "ttft_seconds", "p50")),
    ("ttft_p95_seconds", ("requests", "ttft_seconds", "p95")),
    ("ttft_p99_seconds", ("requests", "ttft_seconds", "p99")),
)
_VLLM_COMPONENTS = ("queue_time", "prefill_time", "decode_time", "inference_time")
_EXECUTION_METRIC_ROOTS = {
    "tasks",
    "requests",
    "vllm",
    "request_throughput_per_second",
    "input_token_throughput_per_second",
    "output_token_throughput_per_second",
}
_SUMMARY_ADAPTER = TypeAdapter(RunSummary)
_MANIFEST_ADAPTER = TypeAdapter(RunManifest)


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    try:
        manifest = _MANIFEST_ADAPTER.validate_json(manifest_path.read_bytes())
        summary_data = json.loads(path.read_bytes())
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"run artifacts in {run_dir} do not match the finalized schemas") from exc
    if not isinstance(summary_data, dict):
        raise ValueError(f"run artifacts in {run_dir} do not match the finalized schemas")
    if manifest.status != "completed":
        raise ValueError(f"run in {run_dir} did not complete successfully")
    if manifest.finished_at is None:
        raise ValueError(f"completed run in {run_dir} has no finish timestamp")
    if "run_wall_time_seconds" not in summary_data:
        warnings.warn(
            "comparing legacy summary without run-level throughput metrics; manifest fallback is deprecated",
            DeprecationWarning,
            stacklevel=2,
        )
        wall_time = (manifest.finished_at - manifest.created_at).total_seconds()
        if wall_time <= 0:
            raise ValueError(f"completed run in {run_dir} has non-positive wall time")
        requests = summary_data.get("requests")
        if not isinstance(requests, dict):
            raise ValueError(f"run artifacts in {run_dir} do not match the finalized schemas")
        summary_data.update(
            run_wall_time_seconds=wall_time,
            request_throughput_per_second=requests.get("requests", 0) / wall_time,
            input_token_throughput_per_second=requests.get("input_tokens", 0) / wall_time,
            output_token_throughput_per_second=requests.get("output_tokens", 0) / wall_time,
        )
    vllm = summary_data.get("vllm")
    if isinstance(vllm, dict):
        vllm.setdefault("prefix_cache_hit_rate", None)
        vllm.setdefault("prompt_token_hit_rate", None)
    try:
        summary = _SUMMARY_ADAPTER.validate_python(summary_data)
    except ValidationError as exc:
        raise ValueError(f"run artifacts in {run_dir} do not match the finalized schemas") from exc
    if summary.run_id != manifest.run_id or summary.run_id != run_dir.name:
        raise ValueError(f"run artifacts in {run_dir} identify different runs")
    if summary.lifecycle.status != "completed":
        raise ValueError(f"run in {run_dir} did not complete successfully")
    return asdict(summary)


def _as_dirs(runs: Path | str | list[Path] | list[str]) -> list[Path]:
    return [Path(run) for run in runs] if isinstance(runs, (list, tuple)) else [Path(runs)]


def _extract(summary: dict, keys: tuple[str, ...]) -> float | None:
    execution = summary.get("execution")
    if keys[0] in _EXECUTION_METRIC_ROOTS and isinstance(execution, dict) and execution.get("available") is False:
        return None
    current: Any = summary
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, (int, float)) else None


def _t_crit(n: int, confidence: float) -> float:
    if confidence != 0.95:
        raise ValueError("compare supports confidence=0.95 only")
    if n - 1 not in _T95:
        raise ValueError("compare supports at most 11 runs per arm")
    return _T95[n - 1]


def _aggregate(values: list[float | None], confidence: float) -> dict[str, Any] | None:
    samples = [value for value in values if value is not None]
    if not samples:
        return None
    mean = fmean(samples)
    if len(samples) == 1:
        return {"mean": mean, "stddev": 0.0, "ci": [mean, mean], "n": 1}
    stddev = stdev(samples)
    half_width = _t_crit(len(samples), confidence) * stddev / math.sqrt(len(samples))
    return {"mean": mean, "stddev": stddev, "ci": [mean - half_width, mean + half_width], "n": len(samples)}


def _comparison(base: dict[str, Any] | None, candidate: dict[str, Any] | None, confidence: float) -> dict[str, Any]:
    entry = {
        "baseline": base["mean"] if base else None,
        "candidate": candidate["mean"] if candidate else None,
        "baseline_ci": base["ci"] if base else None,
        "candidate_ci": candidate["ci"] if candidate else None,
        "n_baseline": base["n"] if base else 0,
        "n_candidate": candidate["n"] if candidate else 0,
    }
    if not base or not candidate:
        return entry
    delta = candidate["mean"] - base["mean"]
    entry["delta_absolute"] = delta
    entry["delta_percent"] = delta / base["mean"] * 100 if base["mean"] else None
    if base["n"] < 2 or candidate["n"] < 2:
        entry.update({"delta_ci": [delta, delta], "significant": False, "low_power": True})
        return entry
    standard_error = math.sqrt(base["stddev"] ** 2 / base["n"] + candidate["stddev"] ** 2 / candidate["n"])
    half_width = _t_crit(min(base["n"], candidate["n"]), confidence) * standard_error
    interval = [delta - half_width, delta + half_width]
    entry.update({"delta_ci": interval, "significant": interval[0] > 0 or interval[1] < 0, "low_power": False})
    return entry


def _cold_confirmed(run_dir: Path) -> bool:
    path = run_dir / "evidence" / "vllm_metrics_start.prom"
    if not path.exists():
        return False
    try:
        metrics = parse_prometheus(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    query_values = [
        value for (name, _labels), value in metrics.samples.items() if name == "vllm:prefix_cache_queries_total"
    ]
    return bool(query_values) and sum(query_values) < 1


def compare(
    baseline: Path | str | list[Path] | list[str],
    candidate: Path | str | list[Path] | list[str],
    *,
    as_json: bool = False,
    confidence: float = 0.95,
) -> str:
    if confidence != 0.95:
        raise ValueError("compare supports confidence=0.95 only")
    base_dirs = _as_dirs(baseline)
    candidate_dirs = _as_dirs(candidate)
    if not base_dirs:
        raise ValueError("baseline must contain at least one run directory")
    if not candidate_dirs:
        raise ValueError("candidate must contain at least one run directory")
    base_summaries = [load_summary(path) for path in base_dirs]
    candidate_summaries = [load_summary(path) for path in candidate_dirs]
    specs = list(_METRIC_SPECS)
    if any(summary["vllm"].get("available") for summary in base_summaries + candidate_summaries):
        specs.extend(
            (f"vllm_{name}_mean_seconds", ("vllm", "latency_breakdown_seconds", name, "mean"))
            for name in _VLLM_COMPONENTS
        )
    metrics = {
        label: _comparison(
            _aggregate([_extract(summary, path) for summary in base_summaries], confidence),
            _aggregate([_extract(summary, path) for summary in candidate_summaries], confidence),
            confidence,
        )
        for label, path in specs
    }
    cold_confirmed = {path: _cold_confirmed(path) for path in (*base_dirs, *candidate_dirs)}
    warnings = []
    for label, directories in (("baseline", base_dirs), ("candidate", candidate_dirs)):
        missing = [str(path) for path in directories if not cold_confirmed[path]]
        if missing:
            warnings.append(f"{label} cold start could not be confirmed for: {', '.join(missing)}")
    for label, directories, summaries in (
        ("baseline", base_dirs, base_summaries),
        ("candidate", candidate_dirs, candidate_summaries),
    ):
        unavailable = [
            str(path)
            for path, summary in zip(directories, summaries, strict=True)
            if isinstance(summary.get("execution"), dict) and summary["execution"].get("available") is False
        ]
        if unavailable:
            warnings.append(f"{label} execution metrics unavailable for: {', '.join(unavailable)}")
    report = {
        "baseline": [str(path) for path in base_dirs],
        "candidate": [str(path) for path in candidate_dirs],
        "warnings": warnings,
        "metrics": metrics,
        "metadata": {
            "baseline_router": base_summaries[0]["router"],
            "candidate_router": candidate_summaries[0]["router"],
            "baseline_execution": base_summaries[0]["execution"],
            "candidate_execution": candidate_summaries[0]["execution"],
            "confidence": confidence,
            "n_baseline": len(base_summaries),
            "n_candidate": len(candidate_summaries),
            "baseline_cold_confirmed": cold_confirmed[base_dirs[0]],
            "candidate_cold_confirmed": cold_confirmed[candidate_dirs[0]],
        },
    }
    if as_json:
        return json.dumps(report, indent=2)
    lines = ["AgentCache Benchmark Comparison"]
    lines.extend(f"WARNING: {warning}" for warning in warnings)
    for label, metric in metrics.items():
        lines.append(f"{label}: baseline={metric['baseline']} candidate={metric['candidate']}")
    lines.append(f"baseline router.applicable: {report['metadata']['baseline_router'].get('applicable')}")
    lines.append(f"candidate router.applicable: {report['metadata']['candidate_router'].get('applicable')}")
    return "\n".join(lines)
