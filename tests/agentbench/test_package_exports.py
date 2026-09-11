# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify collector and metric package re-export contracts."""


def test_collector_package_exports() -> None:
    from agentinfer.agentbench.benchkit.collectors import (
        capture_vllm_metrics,
        collect_environment,
        collect_source_control,
        load_correctness_artifact,
    )

    assert all(
        callable(symbol)
        for symbol in (
            capture_vllm_metrics,
            collect_environment,
            collect_source_control,
            load_correctness_artifact,
        )
    )


def test_metric_package_exports() -> None:
    from agentinfer.agentbench.benchkit.metrics import (
        EvidenceCapture,
        LatencyStats,
        ObservedTopology,
        RequestMetrics,
        SourceHealth,
        TaskMetrics,
        VllmMetrics,
        aggregate_request_metrics,
        aggregate_task_results,
        aggregate_vllm_metrics,
        derive_session_topology,
        evaluate_captures,
        parse_prometheus,
    )

    assert all(
        symbol is not None
        for symbol in (
            EvidenceCapture,
            LatencyStats,
            ObservedTopology,
            RequestMetrics,
            SourceHealth,
            TaskMetrics,
            VllmMetrics,
            aggregate_request_metrics,
            aggregate_task_results,
            aggregate_vllm_metrics,
            derive_session_topology,
            evaluate_captures,
            parse_prometheus,
        )
    )
