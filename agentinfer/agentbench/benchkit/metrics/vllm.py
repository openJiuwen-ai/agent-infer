# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Aggregate raw vLLM Prometheus evidence without writing artifacts."""

import math
from collections.abc import Mapping

from prometheus_client.parser import text_string_to_metric_families
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

LatencyValue = float | int | None
SeriesLabels = tuple[tuple[str, str], ...]
SeriesKey = tuple[str, SeriesLabels]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PrometheusSnapshot:
    """Store Prometheus samples keyed by metric name and complete label set."""

    samples: Mapping[SeriesKey, float]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class VllmMetrics:
    """Store instance-wide vLLM metric deltas for a valid snapshot window."""

    available: bool
    reason: str | None
    counter_reset_detected: bool
    lifecycle_verified: bool
    latency_breakdown_seconds: Mapping[str, Mapping[str, LatencyValue]]
    counters: Mapping[str, float]
    request_finished_by_reason: Mapping[str, int]
    prefix_cache_hit_rate: float | None
    prompt_token_hit_rate: float | None


_METRIC_COMPONENTS: dict[str, str] = {
    "queue_time": "vllm:request_queue_time_seconds",
    "prefill_time": "vllm:request_prefill_time_seconds",
    "decode_time": "vllm:request_decode_time_seconds",
    "inference_time": "vllm:request_inference_time_seconds",
    "ttft": "vllm:time_to_first_token_seconds",
    "e2e_latency": "vllm:e2e_request_latency_seconds",
}
_COUNTER_METRICS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
)
_REQUEST_FINISHED_METRIC = "vllm:request_success_total"
_LIFECYCLE_METRIC = "process_start_time_seconds"


def parse_prometheus(text: str) -> PrometheusSnapshot:
    """Parse Prometheus text while preserving each sample's complete labels."""

    samples: dict[SeriesKey, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if not (sample.name.startswith("vllm:") or sample.name == _LIFECYCLE_METRIC):
                continue
            key = (sample.name, tuple(sorted(sample.labels.items())))
            if key in samples:
                raise ValueError(f"duplicate Prometheus series: {sample.name}")
            samples[key] = float(sample.value)
    return PrometheusSnapshot(samples)


def aggregate_vllm_metrics(start_text: str | None, end_text: str | None) -> VllmMetrics:
    """Compute instance-wide vLLM deltas from comparable Prometheus snapshots."""

    if not start_text or not end_text:
        return _unavailable("missing_metrics_snapshot")
    try:
        before = parse_prometheus(start_text)
        after = parse_prometheus(end_text)
    except ValueError:
        return _unavailable("invalid_prometheus_snapshot")
    if not before.samples or not after.samples:
        return _unavailable("empty_parsed_metrics")

    lifecycle_values = (
        *_series_values(before.samples, _LIFECYCLE_METRIC).values(),
        *_series_values(after.samples, _LIFECYCLE_METRIC).values(),
    )
    if any(not math.isfinite(value) for value in lifecycle_values):
        return _unavailable("non_finite_metric_value")

    lifecycle_verified, lifecycle_changed, lifecycle_incomplete = _check_lifecycle(before, after)
    if lifecycle_incomplete:
        return _unavailable("lifecycle_series_mismatch")
    if lifecycle_changed:
        return _unavailable("counter_reset_detected", reset=True, lifecycle_verified=True)

    consumed_names = {
        *(name + suffix for name in _METRIC_COMPONENTS.values() for suffix in ("_sum", "_count")),
        *_COUNTER_METRICS,
        _REQUEST_FINISHED_METRIC,
    }
    before_series = {key: value for key, value in before.samples.items() if key[0] in consumed_names}
    after_series = {key: value for key, value in after.samples.items() if key[0] in consumed_names}
    if not before_series or not after_series:
        return _unavailable("no_supported_metrics", lifecycle_verified=lifecycle_verified)
    if before_series.keys() != after_series.keys():
        return _unavailable("metric_series_mismatch", lifecycle_verified=lifecycle_verified)
    if any(not math.isfinite(value) for value in (*before_series.values(), *after_series.values())):
        return _unavailable("non_finite_metric_value", lifecycle_verified=lifecycle_verified)

    deltas = {key: after_series[key] - value for key, value in before_series.items()}
    if any(value < 0 for value in deltas.values()):
        return _unavailable("counter_reset_detected", reset=True, lifecycle_verified=lifecycle_verified)

    breakdown: dict[str, dict[str, LatencyValue]] = {}
    for component, metric_name in _METRIC_COMPONENTS.items():
        sums = _series_values(deltas, metric_name + "_sum")
        counts = _series_values(deltas, metric_name + "_count")
        if bool(sums) != bool(counts) or (sums and sums.keys() != counts.keys()):
            return _unavailable("metric_series_mismatch", lifecycle_verified=lifecycle_verified)
        if not sums:
            continue
        if any(not value.is_integer() for value in counts.values()):
            return _unavailable("invalid_histogram_count", lifecycle_verified=lifecycle_verified)
        total = sum(sums.values())
        count = int(sum(counts.values()))
        breakdown[component] = {"count": count, "sum": total, "mean": total / count if count else None}

    counters = {}
    for name in _COUNTER_METRICS:
        series = _series_values(deltas, name)
        if series:
            counters[name] = sum(series.values())

    request_finished_by_reason: dict[str, int] = {}
    for labels, value in _series_values(deltas, _REQUEST_FINISHED_METRIC).items():
        finished_reason = dict(labels).get("finished_reason")
        if finished_reason is None:
            return _unavailable("missing_finished_reason_label", lifecycle_verified=lifecycle_verified)
        if not value.is_integer():
            return _unavailable("invalid_request_finished_count", lifecycle_verified=lifecycle_verified)
        request_finished_by_reason[finished_reason] = request_finished_by_reason.get(finished_reason, 0) + int(value)

    available = (
        any(component["count"] for component in breakdown.values())
        or any(counters.values())
        or any(request_finished_by_reason.values())
    )
    prefix_hits = counters.get("vllm:prefix_cache_hits_total")
    prefix_queries = counters.get("vllm:prefix_cache_queries_total")
    cached_tokens = counters.get("vllm:prompt_tokens_cached_total")
    prompt_tokens = counters.get("vllm:prompt_tokens_total")
    return VllmMetrics(
        available,
        None if available else "no_run_window_samples",
        False,
        lifecycle_verified,
        breakdown,
        counters,
        request_finished_by_reason,
        prefix_hits / prefix_queries if prefix_hits is not None and prefix_queries else None,
        cached_tokens / prompt_tokens if cached_tokens is not None and prompt_tokens else None,
    )


def _series_values(samples: Mapping[SeriesKey, float], name: str) -> dict[SeriesLabels, float]:
    return {labels: value for (sample_name, labels), value in samples.items() if sample_name == name}


def _check_lifecycle(before: PrometheusSnapshot, after: PrometheusSnapshot) -> tuple[bool, bool, bool]:
    before_values = _series_values(before.samples, _LIFECYCLE_METRIC)
    after_values = _series_values(after.samples, _LIFECYCLE_METRIC)
    if not before_values and not after_values:
        return False, False, False
    if not before_values or not after_values or before_values.keys() != after_values.keys():
        return False, False, True
    return True, before_values != after_values, False


def _unavailable(
    reason: str,
    *,
    reset: bool = False,
    lifecycle_verified: bool = False,
) -> VllmMetrics:
    return VllmMetrics(False, reason, reset, lifecycle_verified, {}, {}, {}, None, None)
