# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Compare two completed Trace Replay runs for repeatability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentinfer.agentbench.benchkit.compare import _cold_confirmed, load_summary
from agentinfer.agentbench.benchkit.metrics.vllm import parse_prometheus

from .request_intervals import GLOBAL_INTERVAL_FIELDS, Distribution, global_interval_distribution


@dataclass(frozen=True)
class MetricCheck:
    label: str
    baseline: float | int | str | None
    candidate: float | int | str | None
    ok: bool
    detail: str
    informational: bool = False


@dataclass(frozen=True)
class StabilityReport:
    baseline_dir: Path
    candidate_dir: Path
    tolerance_ratio: float
    checks: tuple[MetricCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.ok for check in self.checks if not check.informational)


def _extract(summary: dict[str, Any], *keys: str) -> Any:
    current: Any = summary
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _cold_start_check(label: str, run_dir: Path) -> MetricCheck:
    confirmed = _cold_confirmed(run_dir)
    return MetricCheck(
        f"{label}.cold_start",
        confirmed,
        confirmed,
        confirmed,
        "prefix cache queries at run start < 1 (see evidence/vllm_metrics_start.prom)"
        if confirmed
        else f"cold start not confirmed under {run_dir / 'evidence'}",
        informational=True,
    )


def _process_start_time_seconds(run_dir: Path) -> tuple[float, ...] | None:
    path = run_dir / "evidence" / "vllm_metrics_start.prom"
    if not path.is_file():
        return None
    try:
        metrics = parse_prometheus(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    values = sorted(value for (name, _labels), value in metrics.samples.items() if name == "process_start_time_seconds")
    return tuple(values) if values else None


def _format_process_start_values(values: tuple[float, ...] | None) -> str | None:
    if values is None:
        return None
    return ",".join(f"{value:.3f}" for value in values)


def _process_start_check(baseline_dir: Path, candidate_dir: Path) -> MetricCheck:
    baseline_values = _process_start_time_seconds(baseline_dir)
    candidate_values = _process_start_time_seconds(candidate_dir)
    baseline_text = _format_process_start_values(baseline_values)
    candidate_text = _format_process_start_values(candidate_values)
    if baseline_values is None or candidate_values is None:
        return MetricCheck(
            "evidence.process_start_time_seconds",
            baseline_text,
            candidate_text,
            False,
            "missing process_start_time_seconds in one or both vllm_metrics_start.prom files",
            informational=True,
        )
    later_start = min(candidate_values) > max(baseline_values)
    detail = (
        "candidate run process start is later than baseline (distinct vLLM launches)"
        if later_start
        else "candidate process start is not strictly after baseline; verify vLLM was restarted between runs"
    )
    return MetricCheck(
        "evidence.process_start_time_seconds",
        baseline_text,
        candidate_text,
        later_start,
        detail,
        informational=True,
    )


def _exact_int(label: str, baseline: dict[str, Any], candidate: dict[str, Any], path: tuple[str, ...]) -> MetricCheck:
    left = _extract(baseline, *path)
    right = _extract(candidate, *path)
    ok = isinstance(left, int) and isinstance(right, int) and left == right
    detail = "exact match" if ok else f"expected equal ints, got {left!r} vs {right!r}"
    return MetricCheck(
        label, left if isinstance(left, int) else None, right if isinstance(right, int) else None, ok, detail
    )


def _within_ratio(
    label: str,
    baseline: float | None,
    candidate: float | None,
    *,
    tolerance: float,
    informational: bool = False,
) -> MetricCheck:
    if baseline is None or candidate is None:
        return MetricCheck(label, baseline, candidate, False, "missing numeric value", informational=informational)
    if baseline == 0.0 and candidate == 0.0:
        return MetricCheck(label, baseline, candidate, True, "both zero", informational=informational)
    if baseline == 0.0:
        ok = abs(candidate) <= tolerance
        detail = f"baseline zero; |candidate|={abs(candidate):.6g} (limit={tolerance})"
        return MetricCheck(label, baseline, candidate, ok, detail, informational=informational)
    ratio = abs(candidate - baseline) / abs(baseline)
    ok = ratio <= tolerance
    detail = f"relative delta {ratio * 100:.3f}% (limit {tolerance * 100:.1f}%)"
    return MetricCheck(label, baseline, candidate, ok, detail, informational=informational)


def _compare_distribution_fields(
    label_prefix: str,
    baseline: Distribution,
    candidate: Distribution,
    *,
    tolerance: float,
    informational_fields: frozenset[str] = frozenset(),
) -> list[MetricCheck]:
    checks: list[MetricCheck] = []
    for field in GLOBAL_INTERVAL_FIELDS:
        left = getattr(baseline, field)
        right = getattr(candidate, field)
        checks.append(
            _within_ratio(
                f"{label_prefix}.{field}",
                left,
                right,
                tolerance=tolerance,
                informational=field in informational_fields,
            )
        )
    return checks


_P99_INFORMATIONAL = frozenset({"p99"})


def compare_replay_stability(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    tolerance_ratio: float = 0.05,
) -> StabilityReport:
    """Validate two replay runs against repeatability criteria."""

    if tolerance_ratio <= 0:
        raise ValueError("tolerance_ratio must be positive")

    baseline_summary = load_summary(baseline_dir)
    candidate_summary = load_summary(candidate_dir)

    checks: list[MetricCheck] = [
        _cold_start_check("baseline", baseline_dir),
        _cold_start_check("candidate", candidate_dir),
        _process_start_check(baseline_dir, candidate_dir),
        _exact_int("requests.requests", baseline_summary, candidate_summary, ("requests", "requests")),
        _exact_int("tasks.failed", baseline_summary, candidate_summary, ("tasks", "failed")),
        _exact_int("requests.input_tokens", baseline_summary, candidate_summary, ("requests", "input_tokens")),
        _exact_int("requests.output_tokens", baseline_summary, candidate_summary, ("requests", "output_tokens")),
    ]

    for stat in ("mean", "p50", "p95", "p99"):
        key = ("tasks", "duration_seconds", stat)
        checks.append(
            _within_ratio(
                f"tasks.duration_seconds.{stat}",
                _extract(baseline_summary, *key),
                _extract(candidate_summary, *key),
                tolerance=tolerance_ratio,
                informational=stat in _P99_INFORMATIONAL,
            )
        )

    for stat in ("mean", "p50", "p95", "p99"):
        for section, prefix in (
            ("latency_seconds", "requests.latency_seconds"),
            ("ttft_seconds", "requests.ttft_seconds"),
        ):
            key = ("requests", section, stat)
            checks.append(
                _within_ratio(
                    f"{prefix}.{stat}",
                    _extract(baseline_summary, *key),
                    _extract(candidate_summary, *key),
                    tolerance=tolerance_ratio,
                    informational=stat in _P99_INFORMATIONAL,
                )
            )

    prefix_left = _extract(baseline_summary, "requests", "prefix_cache_hit_rate")
    prefix_right = _extract(candidate_summary, "requests", "prefix_cache_hit_rate")
    if not isinstance(prefix_left, (int, float)) or not isinstance(prefix_right, (int, float)):
        prefix_left = _extract(baseline_summary, "vllm", "prefix_cache_hit_rate")
        prefix_right = _extract(candidate_summary, "vllm", "prefix_cache_hit_rate")
        prefix_label = "vllm.prefix_cache_hit_rate"
    else:
        prefix_label = "requests.prefix_cache_hit_rate"
    if isinstance(prefix_left, (int, float)) and isinstance(prefix_right, (int, float)):
        checks.append(
            _within_ratio(
                prefix_label,
                float(prefix_left),
                float(prefix_right),
                tolerance=tolerance_ratio,
            )
        )
    else:
        checks.append(MetricCheck(prefix_label, prefix_left, prefix_right, False, "prefix cache hit rate unavailable"))

    wall_left = _extract(baseline_summary, "run_wall_time_seconds")
    wall_right = _extract(candidate_summary, "run_wall_time_seconds")
    if isinstance(wall_left, (int, float)) and isinstance(wall_right, (int, float)):
        checks.append(
            _within_ratio(
                "run_wall_time_seconds",
                float(wall_left),
                float(wall_right),
                tolerance=tolerance_ratio,
            )
        )
    else:
        checks.append(MetricCheck("run_wall_time_seconds", wall_left, wall_right, False, "run wall time unavailable"))

    baseline_requests = baseline_dir / "requests.jsonl"
    candidate_requests = candidate_dir / "requests.jsonl"
    if baseline_requests.is_file() and candidate_requests.is_file():
        baseline_dist = global_interval_distribution(baseline_requests)
        candidate_dist = global_interval_distribution(candidate_requests)
        checks.extend(
            _compare_distribution_fields(
                "request_interval",
                baseline_dist,
                candidate_dist,
                tolerance=tolerance_ratio,
                informational_fields=_P99_INFORMATIONAL,
            )
        )
    else:
        checks.append(
            MetricCheck(
                "request_interval",
                None,
                None,
                False,
                "requests.jsonl missing in one or both run directories",
            )
        )

    return StabilityReport(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        tolerance_ratio=tolerance_ratio,
        checks=tuple(checks),
    )


def format_stability_report(report: StabilityReport) -> str:
    lines = [
        f"Replay stability: baseline={report.baseline_dir} candidate={report.candidate_dir} "
        f"tolerance={report.tolerance_ratio * 100:.1f}%",
    ]
    for check in report.checks:
        if check.informational:
            status = "INFO" if check.ok else "WARN"
        else:
            status = "OK" if check.ok else "FAIL"
        lines.append(f"  [{status}] {check.label}: {check.baseline} vs {check.candidate} — {check.detail}")
    lines.append("PASS" if report.passed else "FAIL")
    return "\n".join(lines)
