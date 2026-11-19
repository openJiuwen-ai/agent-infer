# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Router workflow-prediction Prometheus aggregation tests."""

import pytest

from agentinfer.agentbench.benchkit.metrics.router import aggregate_router_prediction_metrics


def test_router_prediction_metrics_aggregate_window_deltas() -> None:
    start = """
vllm_router_workflow_output_actual_tokens_count{target="all"} 10
vllm_router_workflow_output_prediction_total{target="all",source="offline",backoff_level="0"} 4
vllm_router_workflow_output_coverage_total{target="all",quantile="p95",covered="true"} 4
vllm_router_workflow_output_reservation_error_tokens_count{target="all",quantile="p95",direction="over_reserved"} 4
vllm_router_workflow_output_reservation_error_tokens_sum{target="all",quantile="p95",direction="over_reserved"} 40
vllm_router_workflow_prediction_profile_keys{predictor="output_length"} 12
"""
    end = """
vllm_router_workflow_output_actual_tokens_count{target="all"} 20
vllm_router_workflow_output_prediction_total{target="all",source="offline",backoff_level="0"} 5
vllm_router_workflow_output_prediction_total{target="all",source="blended",backoff_level="0"} 8
vllm_router_workflow_output_coverage_total{target="all",quantile="p95",covered="true"} 11
vllm_router_workflow_output_coverage_total{target="all",quantile="p95",covered="false"} 3
vllm_router_workflow_output_reservation_error_tokens_count{target="all",quantile="p95",direction="over_reserved"} 10
vllm_router_workflow_output_reservation_error_tokens_sum{target="all",quantile="p95",direction="over_reserved"} 100
vllm_router_workflow_output_reservation_error_tokens_count{target="all",quantile="p95",direction="under_reserved"} 2
vllm_router_workflow_output_reservation_error_tokens_sum{target="all",quantile="p95",direction="under_reserved"} 30
vllm_router_workflow_output_reservation_error_tokens_count{target="all",quantile="p99",direction="over_reserved"} 2
vllm_router_workflow_output_reservation_error_tokens_sum{target="all",quantile="p99",direction="over_reserved"} 100
vllm_router_workflow_output_actual_tokens_count{target="worker"} 100
vllm_router_workflow_output_prediction_total{target="worker",source="other",backoff_level="1"} 100
vllm_router_workflow_output_coverage_total{target="worker",quantile="p95",covered="false"} 100
vllm_router_workflow_output_reservation_error_tokens_count{target="worker",quantile="p95",direction="over_reserved"} 1
vllm_router_workflow_output_reservation_error_tokens_sum{target="worker",quantile="p95",direction="over_reserved"} 500
vllm_router_workflow_return_prediction_total{target="all"} 4
vllm_router_workflow_return_prediction_total{target="worker"} 200
vllm_router_workflow_prediction_profile_keys{predictor="output_length"} 12
"""

    result = aggregate_router_prediction_metrics(start, end)
    output = result["output_length"]
    assert isinstance(output, dict)

    assert result["available"] is True
    assert output["observations"] == 10
    assert output["predictions"] == 9
    assert output["prediction_availability"] == 0.9
    assert output["coverage"]["p95"] == pytest.approx(0.7)
    assert output["prediction_sources"] == {"blended": 8, "offline": 1}
    assert output["backoff_levels"] == {"0": 9}
    assert output["reservation_error"] == {
        "p95": {
            "over_reserved": {"count": 6, "tokens": 60, "mean_tokens": 10},
            "under_reserved": {"count": 2, "tokens": 30, "mean_tokens": 15},
        },
        "p99": {
            "over_reserved": {"count": 2, "tokens": 100, "mean_tokens": 50},
            "under_reserved": {"count": 0, "tokens": 0, "mean_tokens": None},
        },
    }
    assert result["return_time"] == {
        "predictions": 4,
        "observations": 0,
        "brier_score_mean": None,
        "nll_mean": None,
    }
    assert result["profile_keys"] == {"output_length": 12}


def test_router_prediction_metrics_report_missing_snapshots() -> None:
    assert aggregate_router_prediction_metrics(None, None) == {
        "available": False,
        "reason": "missing_metrics_snapshot",
    }


@pytest.mark.parametrize(
    ("start_value", "end_snapshot", "reason"),
    [
        ("1", 'vllm_router_workflow_output_prediction_total{target="all"} invalid\n', "invalid_prometheus_snapshot"),
        ("1", "other_metric 2\n", "no_workflow_prediction_metrics"),
        ("1", 'vllm_router_workflow_output_prediction_total{target="all"} NaN\n', "non_finite_metric_value"),
        ("2", 'vllm_router_workflow_output_prediction_total{target="all"} 1\n', "counter_reset_detected"),
    ],
)
def test_router_prediction_metrics_report_unavailable_reason(start_value: str, end_snapshot: str, reason: str) -> None:
    start_snapshot = f'vllm_router_workflow_output_prediction_total{{target="all"}} {start_value}\n'

    assert aggregate_router_prediction_metrics(start_snapshot, end_snapshot) == {
        "available": False,
        "reason": reason,
    }
