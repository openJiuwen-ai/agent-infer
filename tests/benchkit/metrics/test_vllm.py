from agentcache.benchmarks.benchkit.metrics.vllm import aggregate_vllm_metrics, parse_prometheus


def test_parse_prometheus_preserves_labels_and_timestamps() -> None:
    snapshot = parse_prometheus(
        """
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{model_name="a"} 2
vllm:prefix_cache_hits_total{model_name="b"} 3 1710000000
other_metric 99
"""
    )

    assert snapshot.samples == {
        ("vllm:prefix_cache_hits_total", (("model_name", "a"),)): 2.0,
        ("vllm:prefix_cache_hits_total", (("model_name", "b"),)): 3.0,
    }


def test_aggregate_vllm_metrics_retains_all_target_metrics() -> None:
    start = _snapshot(1.0, process_start=100.0)
    end = _snapshot(4.0, process_start=100.0)

    metrics = aggregate_vllm_metrics(start, end)

    assert metrics.available is True
    assert metrics.reason is None
    assert metrics.counter_reset_detected is False
    assert metrics.lifecycle_verified is True
    assert set(metrics.latency_breakdown_seconds) == {
        "queue_time",
        "prefill_time",
        "decode_time",
        "inference_time",
        "ttft",
        "e2e_latency",
    }
    assert metrics.latency_breakdown_seconds["queue_time"] == {"count": 3, "sum": 6.0, "mean": 2.0}
    assert metrics.counters == {
        "vllm:prefix_cache_hits_total": 6.0,
        "vllm:prefix_cache_queries_total": 6.0,
        "vllm:prompt_tokens_total": 6.0,
        "vllm:prompt_tokens_cached_total": 6.0,
    }
    assert metrics.request_finished_by_reason == {"stop": 6}


def test_aggregate_vllm_metrics_aggregates_after_series_delta() -> None:
    start = """
vllm:prefix_cache_hits_total{model_name="a"} 10
vllm:prefix_cache_hits_total{model_name="b"} 20
"""
    end = """
vllm:prefix_cache_hits_total{model_name="a"} 13
vllm:prefix_cache_hits_total{model_name="b"} 25
"""

    metrics = aggregate_vllm_metrics(start, end)

    assert metrics.available is True
    assert metrics.counters["vllm:prefix_cache_hits_total"] == 8.0


def test_aggregate_vllm_metrics_rejects_one_series_reset() -> None:
    start = """
vllm:prefix_cache_hits_total{model_name="a"} 10
vllm:prefix_cache_hits_total{model_name="b"} 20
"""
    end = """
vllm:prefix_cache_hits_total{model_name="a"} 2
vllm:prefix_cache_hits_total{model_name="b"} 40
"""

    metrics = aggregate_vllm_metrics(start, end)

    assert metrics.available is False
    assert metrics.reason == "counter_reset_detected"
    assert metrics.counter_reset_detected is True


def test_aggregate_vllm_metrics_rejects_series_mismatch() -> None:
    start = 'vllm:prefix_cache_hits_total{model_name="a"} 10'
    end = 'vllm:prefix_cache_hits_total{model_name="b"} 20'

    metrics = aggregate_vllm_metrics(start, end)

    assert metrics.available is False
    assert metrics.reason == "metric_series_mismatch"


def test_aggregate_vllm_metrics_rejects_non_finite_values() -> None:
    for value in ("NaN", "+Inf", "-Inf"):
        metrics = aggregate_vllm_metrics(
            "vllm:prefix_cache_hits_total 1",
            f"vllm:prefix_cache_hits_total {value}",
        )

        assert metrics.available is False
        assert metrics.reason == "non_finite_metric_value"


def test_aggregate_vllm_metrics_rejects_non_finite_lifecycle_values() -> None:
    for value in ("NaN", "+Inf", "-Inf"):
        metrics = aggregate_vllm_metrics(
            f"process_start_time_seconds {value}\nvllm:prefix_cache_hits_total 1",
            f"process_start_time_seconds {value}\nvllm:prefix_cache_hits_total 2",
        )

        assert metrics.available is False
        assert metrics.reason == "non_finite_metric_value"
        assert metrics.lifecycle_verified is False


def test_aggregate_vllm_metrics_detects_process_restart() -> None:
    metrics = aggregate_vllm_metrics(_snapshot(1.0, process_start=100.0), _snapshot(4.0, process_start=200.0))

    assert metrics.available is False
    assert metrics.reason == "counter_reset_detected"
    assert metrics.counter_reset_detected is True
    assert metrics.lifecycle_verified is True


def test_aggregate_vllm_metrics_reports_unverified_lifecycle_when_marker_missing() -> None:
    metrics = aggregate_vllm_metrics(_snapshot(1.0), _snapshot(4.0))

    assert metrics.available is True
    assert metrics.lifecycle_verified is False


def test_aggregate_vllm_metrics_rejects_one_sided_lifecycle_marker() -> None:
    metrics = aggregate_vllm_metrics(_snapshot(1.0, process_start=100.0), _snapshot(4.0))

    assert metrics.available is False
    assert metrics.reason == "lifecycle_series_mismatch"
    assert metrics.counter_reset_detected is False
    assert metrics.lifecycle_verified is False


def test_aggregate_vllm_metrics_requires_two_supported_snapshots() -> None:
    assert aggregate_vllm_metrics(None, "metric 1").reason == "missing_metrics_snapshot"
    assert aggregate_vllm_metrics("unrelated 1", "unrelated 2").reason == "empty_parsed_metrics"


def test_aggregate_vllm_metrics_rejects_invalid_histogram_count() -> None:
    start = """
vllm:request_queue_time_seconds_sum 1
vllm:request_queue_time_seconds_count 1
"""
    end = """
vllm:request_queue_time_seconds_sum 2
vllm:request_queue_time_seconds_count 2.5
"""

    metrics = aggregate_vllm_metrics(start, end)

    assert metrics.available is False
    assert metrics.reason == "invalid_histogram_count"


def test_aggregate_vllm_metrics_rejects_fractional_count_per_series() -> None:
    start = """
vllm:request_queue_time_seconds_sum{model_name="a"} 1
vllm:request_queue_time_seconds_count{model_name="a"} 1
vllm:request_queue_time_seconds_sum{model_name="b"} 1
vllm:request_queue_time_seconds_count{model_name="b"} 1
"""
    end = """
vllm:request_queue_time_seconds_sum{model_name="a"} 2
vllm:request_queue_time_seconds_count{model_name="a"} 1.5
vllm:request_queue_time_seconds_sum{model_name="b"} 2
vllm:request_queue_time_seconds_count{model_name="b"} 1.5
"""

    metrics = aggregate_vllm_metrics(start, end)

    assert metrics.available is False
    assert metrics.reason == "invalid_histogram_count"


def test_aggregate_vllm_metrics_preserves_finished_reason() -> None:
    start = """
vllm:request_success_total{finished_reason="stop",model_name="a"} 1
vllm:request_success_total{finished_reason="stop",model_name="b"} 2
vllm:request_success_total{finished_reason="length",model_name="a"} 3
vllm:request_success_total{finished_reason="abort",model_name="a"} 4
vllm:request_success_total{finished_reason="error",model_name="a"} 5
vllm:request_success_total{finished_reason="repetition",model_name="a"} 6
"""
    end = """
vllm:request_success_total{finished_reason="stop",model_name="a"} 3
vllm:request_success_total{finished_reason="stop",model_name="b"} 5
vllm:request_success_total{finished_reason="length",model_name="a"} 7
vllm:request_success_total{finished_reason="abort",model_name="a"} 9
vllm:request_success_total{finished_reason="error",model_name="a"} 11
vllm:request_success_total{finished_reason="repetition",model_name="a"} 13
"""

    metrics = aggregate_vllm_metrics(start, end)

    assert metrics.available is True
    assert metrics.request_finished_by_reason == {
        "stop": 5,
        "length": 4,
        "abort": 5,
        "error": 6,
        "repetition": 7,
    }


def test_aggregate_vllm_metrics_requires_finished_reason_label() -> None:
    metrics = aggregate_vllm_metrics(
        "vllm:request_success_total 1",
        "vllm:request_success_total 2",
    )

    assert metrics.available is False
    assert metrics.reason == "missing_finished_reason_label"


def test_aggregate_vllm_metrics_rejects_fractional_finished_count() -> None:
    metrics = aggregate_vllm_metrics(
        'vllm:request_success_total{finished_reason="stop"} 1',
        'vllm:request_success_total{finished_reason="stop"} 1.5',
    )

    assert metrics.available is False
    assert metrics.reason == "invalid_request_finished_count"


def _snapshot(value: float, *, process_start: float | None = None) -> str:
    lines = []
    if process_start is not None:
        lines.append(f"process_start_time_seconds {process_start}")
    for metric_name in (
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:request_inference_time_seconds",
        "vllm:time_to_first_token_seconds",
        "vllm:e2e_request_latency_seconds",
    ):
        lines.append(f'{metric_name}_sum{{model_name="test"}} {value * 2}')
        lines.append(f'{metric_name}_count{{model_name="test"}} {value}')
    for metric_name in (
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prompt_tokens_total",
        "vllm:prompt_tokens_cached_total",
    ):
        lines.append(f'{metric_name}{{model_name="test"}} {value * 2}')
    lines.append(f'vllm:request_success_total{{finished_reason="stop",model_name="test"}} {value * 2}')
    return "\n".join(lines)
