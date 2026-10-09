"""frontier_sim._primary_value must resolve a latency-aggregate metric (ttft/e2e percentile|mean)
from Frontier's real *_statistics dicts — not silently fall through to throughput. Pure/offline."""
from __future__ import annotations

import pytest

from vllm_evolve.bench.config import build_bench_config
from vllm_evolve.bench.frontier_sim import _primary_value


def _pv(parsed: dict, metric: str):
    bc = build_bench_config(model="Llama-3.2-1B-Instruct")
    bc.statistical.primary_metric = metric
    return _primary_value(parsed, bc)


_PARSED = {
    "ttft_stats": {"p99": 42.0, "mean": 10.0, "p90": 20.0, "median": 9.0},
    "e2e_stats": {"p99": 100.0, "mean": 50.0},
    "requests_per_second": 5.0,
    "tokens_per_second": 500.0,
    "per_request": [],
    "columns": [],
}


def test_latency_aggregates_come_from_the_stats_dict():
    assert _pv(_PARSED, "ttft_p99_ms") == ("ttft_p99_ms", 42.0)     # was None before the fix
    assert _pv(_PARSED, "ttft_mean_ms") == ("ttft_mean_ms", 10.0)
    assert _pv(_PARSED, "e2e_p99_ms") == ("e2e_p99_ms", 100.0)


def test_throughput_metrics_unchanged():
    assert _pv(_PARSED, "throughput_req_s") == ("throughput_req_s", 5.0)
    assert _pv(_PARSED, "output_throughput_tok_s") == ("output_throughput_tok_s", 500.0)


def test_missing_latency_stat_raises_never_substitutes_throughput():
    # honesty: a requested p99 with no stats must RAISE, not silently return throughput_req_s
    blind = {**_PARSED, "ttft_stats": None}
    with pytest.raises(RuntimeError):
        _pv(blind, "ttft_p99_ms")
