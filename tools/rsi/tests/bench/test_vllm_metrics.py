from __future__ import annotations

from vllm_evolve.bench.vllm_metrics import parse_prometheus_metrics


def test_parse_vllm_scheduler_metrics_across_colon_names():
    text = """
# HELP vllm:num_requests_running Number running.
vllm:num_requests_running{model_name="m"} 64
vllm:num_requests_waiting{model_name="m"} 23
vllm:kv_cache_usage_perc{model_name="m",cache_type="full"} 0.91
vllm:num_preemptions_total{model_name="m"} 7
vllm:iteration_tokens_total_count{model_name="m"} 123
"""
    metrics = parse_prometheus_metrics(text)
    assert metrics["running_requests"] == 64
    assert metrics["waiting_requests"] == 23
    assert metrics["kv_cache_occupancy"] == 0.91
    assert metrics["preemptions_total"] == 7
    assert metrics["scheduler_invocations_total"] == 123


def test_missing_metrics_stay_missing_instead_of_becoming_zero_pressure():
    metrics = parse_prometheus_metrics("# no samples\n")
    assert metrics["running_requests"] is None
    assert metrics["kv_cache_occupancy"] is None


def test_kv_groups_use_peak_but_request_workers_are_summed():
    text = """
vllm_num_requests_running{engine="0"} 20
vllm_num_requests_running{engine="1"} 21
vllm_gpu_cache_usage_perc{cache_type="a"} 0.7
vllm_gpu_cache_usage_perc{cache_type="b"} 0.9
"""
    metrics = parse_prometheus_metrics(text)
    assert metrics["running_requests"] == 41
    assert metrics["kv_cache_occupancy"] == 0.9
