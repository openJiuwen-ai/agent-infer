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
vllm_router_workflow_prediction_profile_keys{predictor="output_length"} 12
"""

    result = aggregate_router_prediction_metrics(start, end)
    output = result["output_length"]

    assert result["available"] is True
    assert output["observations"] == 10
    assert output["predictions"] == 9
    assert output["prediction_availability"] == 0.9
    assert output["coverage"]["p95"] == pytest.approx(0.7)
    assert output["prediction_sources"] == {"blended": 8, "offline": 1}
    assert output["backoff_levels"] == {"0": 9}
    assert output["reservation_error"]["under_reserved"]["mean_tokens"] == 15
    assert result["profile_keys"] == {"output_length": 12}


def test_router_prediction_metrics_report_missing_snapshots() -> None:
    assert aggregate_router_prediction_metrics(None, None) == {
        "available": False,
        "reason": "missing_metrics_snapshot",
    }
