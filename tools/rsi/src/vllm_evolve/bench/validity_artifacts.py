"""Bind raw remote telemetry to per-seed workload-validity verdicts."""
from __future__ import annotations

import json
from pathlib import Path

from vllm_evolve.bench.workload_validity import (
    UNSTABLE_OR_INCOMPLETE,
    evaluate_workload_validity,
)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _inside(row: dict, window: dict) -> bool:
    timestamp = float(row.get("captured_at_unix_s") or 0.0)
    return (
        float(window["started_at_unix_s"])
        <= timestamp
        <= float(window["ended_at_unix_s"])
    )


def _actual_sent_qps(raw_requests: list[dict], duration_s: float) -> float:
    submits = sorted(
        float(row["submit_time_s"])
        for row in raw_requests
        if row.get("submit_time_s") is not None
    )
    if not submits:
        return 0.0
    span = submits[-1] - submits[0] if len(submits) > 1 else duration_s
    return len(submits) / max(span, 1e-9)


def summarize_client_inflight(raw_requests: list[dict]) -> dict:
    """Return time-weighted client in-flight concurrency from raw timings."""
    events = []
    for row in raw_requests:
        submit = row.get("submit_time_s")
        end = row.get("end_time_s")
        if submit is None or end is None:
            continue
        events.append((float(submit), 1))
        events.append((float(end), -1))
    if not events:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    # Complete an ending request before admitting another at the same timestamp.
    events.sort(key=lambda item: (item[0], item[1]))
    level = 0
    maximum = 0
    segments = []
    previous = events[0][0]
    for timestamp, delta in events:
        duration = max(0.0, timestamp - previous)
        if duration:
            segments.append((level, duration))
        level += delta
        maximum = max(maximum, level)
        previous = timestamp
    total_duration = sum(duration for _, duration in segments)
    mean = (
        sum(level * duration for level, duration in segments) / total_duration
        if total_duration
        else float(maximum)
    )

    def weighted_percentile(q: float) -> float:
        if not segments or total_duration <= 0:
            return float(maximum)
        target = q * total_duration
        cumulative = 0.0
        for value, weight in sorted(segments):
            cumulative += weight
            if cumulative + 1e-12 >= target:
                return float(value)
        return float(segments[-1][0])

    return {
        "mean": mean,
        "p50": weighted_percentile(0.50),
        "p95": weighted_percentile(0.95),
        "max": int(maximum),
        "completed_timing_count": len(events) // 2,
        "duration_s": total_duration,
    }


def attach_workload_validity(
    eval_result: dict,
    *,
    status: dict,
    run_dir: str | Path,
    bench_config: dict,
) -> dict:
    """Augment one real result; raw files remain the authoritative evidence."""
    run = Path(run_dir)
    workload_spec = (
        bench_config.get("workload", {}).get("workload_spec", {})
        if isinstance(bench_config, dict) else {}
    )
    protocol = dict(workload_spec.get("validity") or {})
    if not protocol.get("required"):
        eval_result["workload_validity"] = {
            "required": False,
            "valid": False,
            "verdict": UNSTABLE_OR_INCOMPLETE,
            "reasons": ["formal saturated validity protocol was not configured"],
            "per_seed": [],
        }
        return eval_result

    windows_payload = {}
    windows_path = run / "measurement_windows.json"
    if windows_path.exists():
        windows_payload = json.loads(windows_path.read_text(encoding="utf-8"))
    windows = {
        int(window["seed"]): window
        for window in windows_payload.get("windows", [])
    }
    gpu_rows = _jsonl(run / "gpu_samples.jsonl")
    vllm_rows = _jsonl(run / "vllm_metrics.jsonl")
    raw_rows = _jsonl(run / "raw_requests.jsonl")
    per_seed_metrics = {
        int(row["seed"]): row
        for row in eval_result.get("raw_per_seed_metrics", [])
    }
    gpus = tuple(
        part.strip()
        for part in str(
            bench_config.get("environment", {}).get("gpus") or ""
        ).split(",")
        if part.strip()
    )
    max_num_seqs = int(
        bench_config.get("engine", {}).get("max_num_seqs") or 0
    )
    offered_qps = float(protocol.get("offered_qps") or 0.0)
    results = []
    expected_seeds = [int(seed) for seed in eval_result.get("seeds", [])]
    for seed in expected_seeds:
        window = windows.get(seed)
        metric_row = per_seed_metrics.get(seed, {})
        metrics = dict(metric_row.get("metrics") or {})
        if window is None:
            results.append({
                "seed": seed,
                "source": "real_vllm",
                "valid": False,
                "verdict": UNSTABLE_OR_INCOMPLETE,
                "reasons": ["measurement window is missing"],
            })
            continue
        seed_raw = [
            row for row in raw_rows if int(row.get("seed", -9999)) == seed
        ]
        duration = float(window.get("duration_s") or 0.0)
        result = evaluate_workload_validity(
            gpu_samples=[row for row in gpu_rows if _inside(row, window)],
            vllm_samples=[
                row
                for row in vllm_rows
                if int(row.get("seed", -9999)) == seed and _inside(row, window)
            ],
            selected_gpus=gpus,
            max_num_seqs=max_num_seqs,
            measured_duration_s=duration,
            measured_requests=int(window.get("measured_requests") or len(seed_raw)),
            completed_requests=int(metrics.get("num_completed") or 0),
            error_count=int(metrics.get("num_failed") or 0),
            offered_qps=offered_qps,
            actual_sent_qps=_actual_sent_qps(seed_raw, duration),
            plateau_reference=protocol.get("plateau_reference"),
            contaminated=bool(
                status.get("gpu_summary", {}).get("contaminated")
            ),
        )
        result["client_inflight"] = summarize_client_inflight(seed_raw)
        result["seed"] = seed
        results.append(result)

    first_invalid = next(
        (item for item in results if not item.get("valid")),
        None,
    )
    aggregate = {
        "required": True,
        "source": "real_vllm",
        "valid": bool(results) and first_invalid is None,
        "verdict": (
            "valid_saturated_real_vllm"
            if results and first_invalid is None
            else (
                first_invalid.get("verdict")
                if first_invalid else UNSTABLE_OR_INCOMPLETE
            )
        ),
        "per_seed": results,
    }
    eval_result["workload_validity"] = aggregate
    (run / "validity.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    return eval_result
