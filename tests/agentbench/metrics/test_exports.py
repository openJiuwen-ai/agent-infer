"""Verify collector and metric package exports."""


def test_collector_package_exports() -> None:
    from agentinfer.agentbench.benchkit.collectors import (
        capture_router_snapshot,
        capture_vllm_metrics,
        collect_environment,
        collect_source_control,
        load_correctness_artifact,
    )

    assert all(
        callable(symbol)
        for symbol in (
            capture_router_snapshot,
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
        RouterMetrics,
        SourceHealth,
        TaskMetrics,
        VllmMetrics,
        aggregate_request_metrics,
        aggregate_router_events,
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
            RouterMetrics,
            SourceHealth,
            TaskMetrics,
            VllmMetrics,
            aggregate_request_metrics,
            aggregate_router_events,
            aggregate_task_results,
            aggregate_vllm_metrics,
            derive_session_topology,
            evaluate_captures,
            parse_prometheus,
        )
    )
