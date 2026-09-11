#!/usr/bin/env python3
"""Diagnose apparent AgentBench regressions from finalized run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

_REQUIRED_FILES = ("manifest.json", "summary.json", "sessions.csv", "task_index.json")
_OPTIONAL_EVIDENCE_SOURCES = (
    "agents",
    "distribution_samples",
    "environment",
    "request_trace",
    "source_control",
    "vllm_end",
    "vllm_start",
)
_COMPARE_CONFIG_PATHS = (
    ("experiment", "task_num"),
    ("experiment", "max_concurrency"),
    ("experiment", "task_timeout_seconds"),
    ("dataset", "name"),
    ("dataset", "index_path"),
    ("dataset", "selection_path"),
    ("agent", "type"),
    ("agent", "profile"),
    ("agent", "executable"),
    ("agent", "terminal_capture_interval_seconds"),
    ("agent", "tmux_startup_seconds"),
    ("backend", "type"),
    ("backend", "model"),
    ("backend", "endpoint"),
    ("request_proxy", "request_timeout_seconds"),
    ("request_proxy", "shutdown_timeout_seconds"),
    ("request_proxy", "startup_timeout_seconds"),
)
_LOG_PATTERNS = {
    "oom": "out of memory",
    "traceback": "traceback (most recent call last)",
    "server_restart": "counter_reset_detected",
    "force_resume": "progress-ttl force resume",
    "scheduler": "agentcacheasyncschedulerbridge",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value not in {"", "null", "N/A"}:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _ratio(candidate: float | int | None, baseline: float | int | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return float(candidate) / float(baseline)


def _percent(candidate: float | int | None, baseline: float | int | None) -> float | None:
    ratio = _ratio(candidate, baseline)
    return None if ratio is None else (ratio - 1.0) * 100.0


def _round(value: Any, digits: int = 3) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return value
    return round(value, digits)


def _read_sessions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = row.get("instance_id")
        if not instance_id or instance_id in result:
            raise ValueError(f"invalid or duplicate instance_id in {path}: {instance_id!r}")
        result[instance_id] = row
    return result


def _task_index(path: Path, run_id: str) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    if data.get("run_id") != run_id:
        raise ValueError(f"task_index run identity mismatch: {path}")
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"invalid task_index tasks: {path}")
    result: dict[str, dict[str, Any]] = {}
    positions: set[int] = set()
    session_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("instance_id"), str):
            raise ValueError(f"invalid task entry: {path}")
        instance_id = task["instance_id"]
        position = task.get("task_position")
        session_id = task.get("session_id")
        if instance_id in result:
            raise ValueError(f"duplicate instance_id in {path}: {instance_id!r}")
        if not isinstance(position, int) or position in positions:
            raise ValueError(f"invalid or duplicate task_position in {path}: {position!r}")
        if not isinstance(session_id, str) or not session_id or session_id in session_ids:
            raise ValueError(f"invalid or duplicate session_id in {path}: {session_id!r}")
        result[instance_id] = task
        positions.add(position)
        session_ids.add(session_id)
    return result


def _cold_confirmed(run_dir: Path) -> bool:
    path = run_dir / "evidence" / "vllm_metrics_start.prom"
    if not path.is_file():
        return False
    total = 0.0
    found = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("vllm:prefix_cache_queries_total"):
            continue
        try:
            total += float(line.rsplit(maxsplit=1)[-1])
            found = True
        except ValueError:
            return False
    return found and total < 1.0


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    missing = [name for name in _REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts in {run_dir}: {', '.join(missing)}")
    manifest = _load_json(run_dir / "manifest.json")
    summary = _load_json(run_dir / "summary.json")
    if manifest.get("status") != "completed" or summary.get("lifecycle", {}).get("status") != "completed":
        raise ValueError(f"run is not completed: {run_dir}")
    if manifest.get("run_id") != run_dir.name or summary.get("run_id") != run_dir.name:
        raise ValueError(f"run identity mismatch: {run_dir}")
    sessions = _read_sessions(run_dir / "sessions.csv")
    task_index = _task_index(run_dir / "task_index.json", run_dir.name)
    if set(sessions) != set(task_index):
        raise ValueError(f"sessions/task_index identities differ: {run_dir}")
    task_results: dict[str, dict[str, Any]] = {}
    for instance_id, index_entry in task_index.items():
        task_path = run_dir / "tasks" / instance_id / "result.json"
        if not task_path.is_file():
            raise FileNotFoundError(f"missing task result: {task_path}")
        task = _load_json(task_path)
        if task.get("instance_id") != instance_id:
            raise ValueError(f"task result identity mismatch: {task_path}")
        session = sessions[instance_id]
        expected = {
            "session_id": index_entry.get("session_id"),
            "outcome": index_entry.get("outcome"),
            "termination_reason": index_entry.get("termination_reason"),
            "task_position": index_entry.get("task_position"),
        }
        observed = {
            "session_id": task.get("session_id"),
            "outcome": task.get("outcome"),
            "termination_reason": task.get("termination_reason"),
            "task_position": task.get("task_position"),
        }
        if observed != expected:
            raise ValueError(f"task_index/task result fields differ: {task_path}")
        for field in ("session_id", "outcome", "termination_reason"):
            session_value = session.get(field)
            if session_value == "null":
                session_value = None
            if session_value != expected[field]:
                raise ValueError(f"sessions/task result fields differ for {instance_id}: {field}")
        task_results[instance_id] = task
    return {
        "path": str(run_dir),
        "manifest": manifest,
        "summary": summary,
        "sessions": sessions,
        "task_index": task_index,
        "task_results": task_results,
        "cold_confirmed": _cold_confirmed(run_dir),
    }


def _source_control(run: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(run["path"]) / "evidence" / "source_control.json"
    return _load_json(path) if path.is_file() else None


def _source_commit(run: dict[str, Any]) -> str | None:
    source_control = _source_control(run)
    if source_control is None:
        return None
    value = source_control.get("commit")
    return value if isinstance(value, str) else None


def _source_dirty(run: dict[str, Any]) -> bool | None:
    value = _nested(_source_control(run) or {}, ("dirty",))
    return value if isinstance(value, bool) else None


def _environment(run: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(run["path"]) / "evidence" / "environment.json"
    return _load_json(path) if path.is_file() else None


def _evidence_availability(run: dict[str, Any]) -> dict[str, bool | None]:
    evidence = run["manifest"].get("evidence")
    if not isinstance(evidence, list):
        return dict.fromkeys(_OPTIONAL_EVIDENCE_SOURCES)
    rows = {
        entry.get("source"): entry
        for entry in evidence
        if isinstance(entry, dict) and isinstance(entry.get("source"), str)
    }
    return {
        source: rows[source].get("available") is True if source in rows else None
        for source in _OPTIONAL_EVIDENCE_SOURCES
    }


def _task_order(run: dict[str, Any]) -> list[str]:
    entries = sorted(run["task_index"].values(), key=lambda item: item.get("task_position", -1))
    return [entry["instance_id"] for entry in entries]


def _canonical_config_value(path: tuple[str, ...], value: Any) -> Any:
    if path in {
        ("dataset", "index_path"),
        ("dataset", "selection_path"),
        ("cli", "config_path"),
    }:
        return Path(value).name if isinstance(value, str) else value
    return value


def _config_checks(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base_config = baseline["manifest"].get("config", {})
    candidate_config = candidate["manifest"].get("config", {})
    checks: list[dict[str, Any]] = []
    for path in _COMPARE_CONFIG_PATHS:
        base_value = _canonical_config_value(path, _nested(base_config, path))
        candidate_value = _canonical_config_value(path, _nested(candidate_config, path))
        checks.append(
            {
                "field": ".".join(path),
                "baseline": base_value,
                "candidate": candidate_value,
                "match": base_value == candidate_value,
            }
        )
    return checks


def _optional_config_check(field: str, baseline: Any, candidate: Any) -> dict[str, Any]:
    available = baseline is not None and candidate is not None
    return {
        "field": field,
        "baseline": baseline,
        "candidate": candidate,
        "match": baseline == candidate if available else None,
        "required": False,
    }


def comparability(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_commit_mismatch: bool = False,
) -> dict[str, Any]:
    checks = [{**check, "required": True} for check in _config_checks(baseline, candidate)]
    baseline_commit = _source_commit(baseline)
    candidate_commit = _source_commit(candidate)
    checks.append(
        {
            "field": "source_control.commit",
            "baseline": baseline_commit,
            "candidate": candidate_commit,
            "match": (allow_commit_mismatch or (baseline_commit is not None and baseline_commit == candidate_commit)),
            "required": True,
            "informational": allow_commit_mismatch,
        }
    )
    checks.extend(
        [
            _optional_config_check(
                "environment",
                _environment(baseline),
                _environment(candidate),
            ),
            _optional_config_check(
                "cli.entrypoint",
                _nested(baseline["summary"], ("cli", "entrypoint")),
                _nested(candidate["summary"], ("cli", "entrypoint")),
            ),
            _optional_config_check(
                "cli.config_path",
                _canonical_config_value(
                    ("cli", "config_path"),
                    _nested(baseline["summary"], ("cli", "config_path")),
                ),
                _canonical_config_value(
                    ("cli", "config_path"),
                    _nested(candidate["summary"], ("cli", "config_path")),
                ),
            ),
            {
                "field": "task_identity_and_order",
                "baseline": _task_order(baseline),
                "candidate": _task_order(candidate),
                "match": _task_order(baseline) == _task_order(candidate),
                "required": True,
            },
        ]
    )
    base_vllm = baseline["summary"].get("vllm", {})
    candidate_vllm = candidate["summary"].get("vllm", {})
    evidence_checks = {
        "baseline_sources": _evidence_availability(baseline),
        "candidate_sources": _evidence_availability(candidate),
        "baseline_vllm_available": base_vllm.get("available") is True,
        "candidate_vllm_available": candidate_vllm.get("available") is True,
        "baseline_counter_reset_absent": base_vllm.get("counter_reset_detected") is False,
        "candidate_counter_reset_absent": candidate_vllm.get("counter_reset_detected") is False,
        "baseline_lifecycle_verified": base_vllm.get("lifecycle_verified") is True,
        "candidate_lifecycle_verified": candidate_vllm.get("lifecycle_verified") is True,
        "baseline_execution_available": _nested(baseline["summary"], ("execution", "available")) is True,
        "candidate_execution_available": _nested(candidate["summary"], ("execution", "available")) is True,
        "baseline_cold_confirmed": baseline["cold_confirmed"],
        "candidate_cold_confirmed": candidate["cold_confirmed"],
    }
    workload_evidence_available = (
        evidence_checks["baseline_execution_available"] and evidence_checks["candidate_execution_available"]
    )
    serving_evidence_available = workload_evidence_available and all(
        evidence_checks[name]
        for name in (
            "baseline_vllm_available",
            "candidate_vllm_available",
            "baseline_counter_reset_absent",
            "candidate_counter_reset_absent",
            "baseline_lifecycle_verified",
            "candidate_lifecycle_verified",
        )
    )
    warnings: list[str] = []
    for arm in ("baseline", "candidate"):
        unavailable = [
            source for source, available in evidence_checks[f"{arm}_sources"].items() if available is not True
        ]
        if unavailable:
            warnings.append(f"{arm} optional evidence unavailable: {', '.join(unavailable)}")
    for check in checks:
        if not check["required"] and check["match"] is not True:
            state = "unavailable" if check["match"] is None else "differs"
            warnings.append(f"optional provenance {check['field']} {state}")
    for arm, run in (("baseline", baseline), ("candidate", candidate)):
        dirty = _source_dirty(run)
        if dirty is True:
            warnings.append(f"{arm} source checkout was dirty")
        elif dirty is None:
            warnings.append(f"{arm} source dirty state unavailable")
    if not evidence_checks["baseline_cold_confirmed"]:
        warnings.append("baseline cold start could not be confirmed")
    if not evidence_checks["candidate_cold_confirmed"]:
        warnings.append("candidate cold start could not be confirmed")
    return {
        "configuration_checks": checks,
        "configuration_comparable": all(
            check["match"] for check in checks if check["required"] and not check.get("informational")
        ),
        "evidence_checks": evidence_checks,
        "workload_evidence_available": workload_evidence_available,
        "serving_evidence_available": serving_evidence_available,
        "cold_pair_confirmed": evidence_checks["baseline_cold_confirmed"]
        and evidence_checks["candidate_cold_confirmed"],
        "warnings": warnings,
    }


def _summary_value(run: dict[str, Any], *path: str) -> float | None:
    return _number(_nested(run["summary"], path))


def summary_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "completed_tasks": ("tasks", "completed"),
        "failed_tasks": ("tasks", "failed"),
        "tasks_with_patch": ("tasks", "with_patch"),
        "run_wall_time_seconds": ("run_wall_time_seconds",),
        "requests": ("requests", "requests"),
        "successful_requests": ("requests", "successful_requests"),
        "failed_requests": ("requests", "failed_requests"),
        "input_tokens": ("requests", "input_tokens"),
        "output_tokens": ("requests", "output_tokens"),
        "request_throughput_per_second": ("request_throughput_per_second",),
        "input_token_throughput_per_second": ("input_token_throughput_per_second",),
        "output_token_throughput_per_second": ("output_token_throughput_per_second",),
        "task_duration_mean_seconds": ("tasks", "duration_seconds", "mean"),
        "task_duration_p50_seconds": ("tasks", "duration_seconds", "p50"),
        "task_duration_p95_seconds": ("tasks", "duration_seconds", "p95"),
        "task_duration_p99_seconds": ("tasks", "duration_seconds", "p99"),
        "latency_mean_seconds": ("requests", "latency_seconds", "mean"),
        "latency_p50_seconds": ("requests", "latency_seconds", "p50"),
        "latency_p95_seconds": ("requests", "latency_seconds", "p95"),
        "latency_p99_seconds": ("requests", "latency_seconds", "p99"),
        "ttft_mean_seconds": ("requests", "ttft_seconds", "mean"),
        "ttft_p99_seconds": ("requests", "ttft_seconds", "p99"),
        "vllm_queue_mean_seconds": ("vllm", "latency_breakdown_seconds", "queue_time", "mean"),
        "vllm_prefill_mean_seconds": ("vllm", "latency_breakdown_seconds", "prefill_time", "mean"),
        "vllm_decode_mean_seconds": ("vllm", "latency_breakdown_seconds", "decode_time", "mean"),
        "vllm_inference_mean_seconds": ("vllm", "latency_breakdown_seconds", "inference_time", "mean"),
        "vllm_prefix_cache_hit_rate": ("vllm", "prefix_cache_hit_rate"),
        "vllm_prompt_token_hit_rate": ("vllm", "prompt_token_hit_rate"),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        base = _summary_value(baseline, *path)
        cand = _summary_value(candidate, *path)
        result[name] = {
            "baseline": _round(base),
            "candidate": _round(cand),
            "delta": _round(cand - base) if base is not None and cand is not None else None,
            "delta_percent": _round(_percent(cand, base)),
        }
    return result


def task_diagnostics(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_rows = baseline["sessions"]
    candidate_rows = candidate["sessions"]
    common = sorted(set(base_rows) & set(candidate_rows))
    paired: list[dict[str, Any]] = []
    skipped: list[str] = []
    for instance_id in common:
        base = base_rows[instance_id]
        cand = candidate_rows[instance_id]
        base_duration = _number(base.get("task_duration_seconds"))
        candidate_duration = _number(cand.get("task_duration_seconds"))
        if base_duration is None or candidate_duration is None:
            skipped.append(instance_id)
            continue
        base_input = _number(base.get("input_tokens")) or 0.0
        candidate_input = _number(cand.get("input_tokens")) or 0.0
        base_requests = _number(base.get("requests")) or 0.0
        candidate_requests = _number(cand.get("requests")) or 0.0
        paired.append(
            {
                "instance_id": instance_id,
                "baseline_outcome": base.get("outcome"),
                "candidate_outcome": cand.get("outcome"),
                "baseline_termination_reason": base.get("termination_reason"),
                "candidate_termination_reason": cand.get("termination_reason"),
                "baseline_duration_seconds": _round(base_duration),
                "candidate_duration_seconds": _round(candidate_duration),
                "duration_delta_seconds": _round(candidate_duration - base_duration),
                "baseline_requests": int(base_requests),
                "candidate_requests": int(candidate_requests),
                "request_delta": int(candidate_requests - base_requests),
                "baseline_input_tokens": int(base_input),
                "candidate_input_tokens": int(candidate_input),
                "input_token_delta": int(candidate_input - base_input),
            }
        )
    durations = [item["duration_delta_seconds"] for item in paired]
    total_positive = sum(max(0.0, value) for value in durations)
    ranked = sorted(paired, key=lambda item: (-item["duration_delta_seconds"], item["instance_id"]))
    top = ranked[:10]
    top_positive = sum(max(0.0, item["duration_delta_seconds"]) for item in top)
    transitions = Counter(
        (
            f"{item['baseline_outcome']}:{item['baseline_termination_reason']}",
            f"{item['candidate_outcome']}:{item['candidate_termination_reason']}",
        )
        for item in paired
    )
    return {
        "common_task_count": len(common),
        "pairs_missing_duration": skipped,
        "baseline_only": sorted(set(base_rows) - set(candidate_rows)),
        "candidate_only": sorted(set(candidate_rows) - set(base_rows)),
        "candidate_slower_tasks": sum(value > 0 for value in durations),
        "candidate_faster_tasks": sum(value < 0 for value in durations),
        "median_duration_delta_seconds": _round(median(durations)) if durations else None,
        "positive_duration_delta_seconds": _round(total_positive),
        "top_10_positive_tail_share": _round(top_positive / total_positive) if total_positive else 0.0,
        "termination_transitions": [
            {"baseline": base, "candidate": cand, "count": count} for (base, cand), count in sorted(transitions.items())
        ],
        "top_candidate_slowdowns": top,
        "top_candidate_speedups": sorted(
            paired, key=lambda item: (item["duration_delta_seconds"], item["instance_id"])
        )[:10],
    }


def log_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "path": None, "counts": {}}
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"log not found: {path}")
    counts = dict.fromkeys(_LOG_PATTERNS, 0)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            for name, pattern in _LOG_PATTERNS.items():
                counts[name] += lowered.count(pattern)
    return {"provided": True, "path": str(path), "counts": counts}


def _classify(comparable: dict[str, Any], deltas: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    wall = deltas["run_wall_time_seconds"]["delta_percent"]
    inputs = deltas["input_tokens"]["delta_percent"]
    requests = deltas["requests"]["delta_percent"]
    latency = deltas["latency_mean_seconds"]["delta_percent"]
    inference = deltas["vllm_inference_mean_seconds"]["delta_percent"]
    queue = deltas["vllm_queue_mean_seconds"]["delta_percent"]
    p99 = deltas["latency_p99_seconds"]["delta_percent"]
    findings: list[dict[str, str]] = []
    if not comparable["configuration_comparable"] or not comparable["workload_evidence_available"]:
        return "INCONCLUSIVE", [
            {"confidence": "PROVEN", "claim": "Required configuration or execution-evidence gates failed."}
        ]
    if wall is not None and wall > 10 and inputs is not None and inputs > 25 and latency is not None and latency < 0:
        verdict = "WORKLOAD_AMPLIFICATION_WITH_TAIL"
        findings.append(
            {
                "confidence": "PROVEN",
                "claim": "Wall time increased while input-token work increased materially and mean request latency improved.",
            }
        )
    elif (
        wall is not None
        and wall > 10
        and inputs is not None
        and abs(inputs) <= 15
        and latency is not None
        and latency > 10
    ):
        verdict = "SERVING_REGRESSION"
        findings.append(
            {
                "confidence": "SUPPORTED",
                "claim": "Wall time and mean latency regressed under approximately matched token work.",
            }
        )
    else:
        verdict = "MIXED"
        if inputs is None:
            findings.append(
                {
                    "confidence": "UNKNOWN",
                    "claim": "Input-token evidence is missing, so a serving-cost verdict is not supported.",
                }
            )
    if comparable["serving_evidence_available"]:
        if inference is not None and inference < 0:
            findings.append(
                {
                    "confidence": "PROVEN",
                    "claim": "Mean vLLM inference time improved, contradicting a uniform decode slowdown.",
                }
            )
        if queue is not None and queue > 10:
            findings.append(
                {
                    "confidence": "PROVEN",
                    "claim": "Mean vLLM queue time regressed and contributes to the mixed serving result.",
                }
            )
    else:
        findings.append(
            {
                "confidence": "UNKNOWN",
                "claim": "Backend serving-cost attribution is unavailable because the vLLM evidence window is unhealthy.",
            }
        )
    if p99 is not None and p99 > 10:
        findings.append(
            {"confidence": "PROVEN", "claim": "Request p99 latency regressed, establishing a tail-latency problem."}
        )
    if requests is not None and requests > 10:
        findings.append({"confidence": "PROVEN", "claim": "The candidate executed materially more request turns."})
    findings.append(
        {
            "confidence": "UNKNOWN",
            "claim": "A single stochastic live-agent pair cannot isolate scheduler causality without repeated cold runs or controlled replay.",
        }
    )
    return verdict, findings


def analyze_pair(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    baseline_log: Path | None = None,
    candidate_log: Path | None = None,
    launcher_log: Path | None = None,
    allow_commit_mismatch: bool = False,
) -> dict[str, Any]:
    baseline = load_run(baseline_dir)
    candidate = load_run(candidate_dir)
    comparable = comparability(baseline, candidate, allow_commit_mismatch=allow_commit_mismatch)
    deltas = summary_deltas(baseline, candidate)
    tasks = task_diagnostics(baseline, candidate)
    verdict, findings = _classify(comparable, deltas)
    return {
        "baseline": baseline["path"],
        "candidate": candidate["path"],
        "verdict": verdict,
        "comparability": comparable,
        "summary_deltas": deltas,
        "task_diagnostics": tasks,
        "log_evidence": {
            "baseline_service": log_evidence(baseline_log),
            "candidate_service": log_evidence(candidate_log),
            "launcher": log_evidence(launcher_log),
        },
        "findings": findings,
        "boundaries": [
            "Tasks completed and patch presence are not SWE-bench correctness.",
            "Cache-counter rates are supporting telemetry, not causal proof of cache reuse or performance.",
            "One live-agent pair is low-power and trajectory-dependent.",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    deltas = report["summary_deltas"]
    tasks = report["task_diagnostics"]
    comparable = report["comparability"]
    lines = [
        "# AgentBench performance-regression investigation",
        "",
        f"**Verdict**: `{report['verdict']}`",
        "",
        f"- Baseline: `{report['baseline']}`",
        f"- Candidate: `{report['candidate']}`",
        f"- Configuration comparable: `{comparable['configuration_comparable']}`",
        f"- Workload evidence available: `{comparable['workload_evidence_available']}`",
        f"- Serving evidence available: `{comparable['serving_evidence_available']}`",
        f"- Cold pair confirmed: `{comparable['cold_pair_confirmed']}`",
        "",
        "## Evidence warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in comparable["warnings"])
    if not comparable["warnings"]:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Evidence-backed findings",
            "",
        ]
    )
    for finding in report["findings"]:
        lines.append(f"- **{finding['confidence']}** — {finding['claim']}")
    lines.extend(
        [
            "",
            "## Workload and serving decomposition",
            "",
            "| Metric | Baseline | Candidate | Delta % |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in (
        "run_wall_time_seconds",
        "completed_tasks",
        "requests",
        "input_tokens",
        "output_tokens",
        "failed_requests",
        "latency_mean_seconds",
        "latency_p99_seconds",
        "ttft_mean_seconds",
        "vllm_queue_mean_seconds",
        "vllm_prefill_mean_seconds",
        "vllm_decode_mean_seconds",
        "vllm_inference_mean_seconds",
    ):
        item = deltas[key]
        lines.append(
            f"| `{key}` | {_fmt(item['baseline'])} | {_fmt(item['candidate'])} | {_fmt(item['delta_percent'])}% |"
        )
    lines.extend(
        [
            "",
            "## Task/session localization",
            "",
            f"- Common tasks: {tasks['common_task_count']} (skipped for missing durations: {len(tasks['pairs_missing_duration'])})",
            f"- Candidate slower/faster tasks: {tasks['candidate_slower_tasks']} / {tasks['candidate_faster_tasks']}",
            f"- Median paired duration delta: {_fmt(tasks['median_duration_delta_seconds'])} seconds",
            f"- Top-10 share of positive duration deltas: {_fmt(tasks['top_10_positive_tail_share'])}",
            "",
            "| Instance | Baseline duration | Candidate duration | Delta | Input-token delta | Request delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in tasks["top_candidate_slowdowns"]:
        lines.append(
            f"| `{item['instance_id']}` | {item['baseline_duration_seconds']} | {item['candidate_duration_seconds']} | "
            f"{item['duration_delta_seconds']} | {item['input_token_delta']} | {item['request_delta']} |"
        )
    lines.extend(["", "## Interpretation boundaries", ""])
    lines.extend(f"- {boundary}" for boundary in report["boundaries"])
    if report.get("references"):
        lines.extend(["", "## Nearby-run references", "", "| Label | Verdict |", "|---|---|"])
        for label, reference in report["references"].items():
            verdict = reference.get("verdict") if "verdict" in reference else f"unavailable ({reference['error']})"
            lines.append(f"| {label} | {verdict} |")
    lines.extend(
        [
            "",
            "## Recommended follow-up",
            "",
            "Repeat cold baseline/candidate pairs and run controlled replay for the dominant task outliers. Hold task order, commit, model, profile, concurrency, timeout, TP, hardware, and service state fixed. Attribute scheduler causality only when the normalized and paired-task signal reproduces.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_reference(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or "," not in parts[1]:
        raise argparse.ArgumentTypeError("reference must be LABEL=BASELINE,CANDIDATE")
    baseline, candidate = parts[1].split(",", 1)
    return parts[0], Path(baseline), Path(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-service-log", type=Path)
    parser.add_argument("--candidate-service-log", type=Path)
    parser.add_argument("--launcher-log", type=Path)
    parser.add_argument("--allow-commit-mismatch", action="store_true")
    parser.add_argument("--reference", type=_parse_reference, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_pair(
        args.baseline,
        args.candidate,
        baseline_log=args.baseline_service_log,
        candidate_log=args.candidate_service_log,
        launcher_log=args.launcher_log,
        allow_commit_mismatch=args.allow_commit_mismatch,
    )
    report["references"] = {}
    for label, baseline, candidate in sorted(args.reference, key=lambda value: value[0]):
        try:
            report["references"][label] = analyze_pair(
                baseline, candidate, allow_commit_mismatch=args.allow_commit_mismatch
            )
        except (OSError, ValueError, KeyError) as error:
            report["references"][label] = {"error": f"{type(error).__name__}: {error}"}
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    source_dirs = {Path(report["baseline"]), Path(report["candidate"])}
    if any(
        output == source or source in output.parents for output in (output_json, output_md) for source in source_dirs
    ):
        raise ValueError("outputs must be outside source run directories")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    output_md.write_text(render_markdown(report), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
