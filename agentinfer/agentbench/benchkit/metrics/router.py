# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Aggregate workflow-prediction metrics from Router Prometheus snapshots."""

from __future__ import annotations

import math
from collections import Counter

from prometheus_client.parser import text_string_to_metric_families

SeriesKey = tuple[str, tuple[tuple[str, str], ...]]

_PREFIX = "vllm_router_workflow_"


def aggregate_router_prediction_metrics(start_text: str | None, end_text: str | None) -> dict[str, object]:
    """Compute prediction counters and histogram summaries for one replay window."""

    if not start_text or not end_text:
        return {"available": False, "reason": "missing_metrics_snapshot"}
    try:
        before = _parse(start_text)
        after = _parse(end_text)
    except ValueError:
        return {"available": False, "reason": "invalid_prometheus_snapshot"}
    if not after:
        return {"available": False, "reason": "no_workflow_prediction_metrics"}

    deltas = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in set(before) | set(after)}
    if any(not math.isfinite(value) for value in deltas.values()):
        return {"available": False, "reason": "non_finite_metric_value"}
    if any(value < 0 for (name, _), value in deltas.items() if name != "vllm_router_workflow_prediction_profile_keys"):
        return {"available": False, "reason": "counter_reset_detected"}

    observations = int(_sum(deltas, "vllm_router_workflow_output_actual_tokens_count"))
    predictions = int(_sum(deltas, "vllm_router_workflow_output_prediction_total"))
    sources: Counter[str] = Counter()
    backoff: Counter[str] = Counter()
    for labels, value in _series(deltas, "vllm_router_workflow_output_prediction_total"):
        label_map = dict(labels)
        sources[label_map.get("source", "unknown")] += int(value)
        backoff[label_map.get("backoff_level", "unknown")] += int(value)

    coverage: dict[str, float | None] = {}
    for quantile in ("p95", "p99"):
        covered = _labeled_sum(
            deltas,
            "vllm_router_workflow_output_coverage_total",
            {"quantile": quantile, "covered": "true"},
        )
        missed = _labeled_sum(
            deltas,
            "vllm_router_workflow_output_coverage_total",
            {"quantile": quantile, "covered": "false"},
        )
        coverage[quantile] = covered / (covered + missed) if covered + missed else None

    reservation_error: dict[str, dict[str, float | int | None]] = {}
    for direction in ("over_reserved", "under_reserved"):
        count = _labeled_sum(
            deltas,
            "vllm_router_workflow_output_reservation_error_tokens_count",
            {"direction": direction},
        )
        total = _labeled_sum(
            deltas,
            "vllm_router_workflow_output_reservation_error_tokens_sum",
            {"direction": direction},
        )
        reservation_error[direction] = {
            "count": int(count),
            "tokens": total,
            "mean_tokens": total / count if count else None,
        }

    return {
        "available": True,
        "reason": None,
        "output_length": {
            "observations": observations,
            "predictions": predictions,
            "prediction_availability": predictions / observations if observations else None,
            "coverage": coverage,
            "prediction_sources": dict(sorted(sources.items())),
            "backoff_levels": dict(sorted(backoff.items())),
            "reservation_error": reservation_error,
        },
        "return_time": {
            "predictions": int(_sum(deltas, "vllm_router_workflow_return_prediction_total")),
            "observations": int(_sum(deltas, "vllm_router_workflow_return_actual_seconds_count")),
            "brier_score_mean": _histogram_mean(deltas, "vllm_router_workflow_return_brier_score"),
            "nll_mean": _histogram_mean(deltas, "vllm_router_workflow_return_nll"),
        },
        "profile_keys": _end_gauges(after, "vllm_router_workflow_prediction_profile_keys", "predictor"),
    }


def _parse(text: str) -> dict[SeriesKey, float]:
    samples: dict[SeriesKey, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if not sample.name.startswith(_PREFIX):
                continue
            key = (sample.name, tuple(sorted(sample.labels.items())))
            if key in samples:
                raise ValueError(f"duplicate Prometheus series: {sample.name}")
            samples[key] = float(sample.value)
    return samples


def _series(samples: dict[SeriesKey, float], name: str) -> list[tuple[tuple[tuple[str, str], ...], float]]:
    return [(labels, value) for (sample_name, labels), value in samples.items() if sample_name == name]


def _sum(samples: dict[SeriesKey, float], name: str) -> float:
    return sum(value for _, value in _series(samples, name))


def _labeled_sum(samples: dict[SeriesKey, float], name: str, required: dict[str, str]) -> float:
    return sum(
        value
        for labels, value in _series(samples, name)
        if all(dict(labels).get(key) == expected for key, expected in required.items())
    )


def _histogram_mean(samples: dict[SeriesKey, float], name: str) -> float | None:
    count = _sum(samples, name + "_count")
    total = _sum(samples, name + "_sum")
    return total / count if count else None


def _end_gauges(samples: dict[SeriesKey, float], name: str, label: str) -> dict[str, float]:
    return {dict(labels).get(label, "unknown"): value for labels, value in _series(samples, name)}
