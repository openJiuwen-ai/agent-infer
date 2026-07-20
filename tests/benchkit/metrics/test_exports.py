"""Verify collector and metric package exports."""


def test_collector_package_exports() -> None:
    from agentcache.benchmarks.benchkit.collectors import (
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
    from agentcache.benchmarks.benchkit.metrics import (
        EvidenceCapture,
        RouterMetrics,
        SourceHealth,
        TaskMetrics,
        VllmMetrics,
        aggregate_router_events,
        aggregate_task_results,
        aggregate_vllm_metrics,
        evaluate_captures,
        parse_prometheus,
    )

    assert all(
        symbol is not None
        for symbol in (
            EvidenceCapture,
            RouterMetrics,
            SourceHealth,
            TaskMetrics,
            VllmMetrics,
            aggregate_router_events,
            aggregate_task_results,
            aggregate_vllm_metrics,
            evaluate_captures,
            parse_prometheus,
        )
    )
