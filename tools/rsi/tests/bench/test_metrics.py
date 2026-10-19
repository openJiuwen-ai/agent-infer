"""Unit tests for bench.metrics — exact-value assertions, no GPU."""
from __future__ import annotations

import math

from vllm_evolve.bench.metrics import BenchMetrics, RequestRecord, _percentile


def _rec(rid, ttft, tpot=10.0, e2e=None, prompt=100, out=50, success=True):
    return RequestRecord(
        request_id=rid,
        arrival_time_s=0.0,
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_ms=e2e if e2e is not None else ttft + tpot * out,
        num_prompt_tokens=prompt,
        num_output_tokens=out,
        success=success,
    )


def test_percentile_linear_interpolation():
    vals = [100.0, 200.0, 300.0, 400.0]
    assert math.isclose(_percentile(vals, 50), 250.0)
    assert math.isclose(_percentile(vals, 90), 370.0)
    assert math.isclose(_percentile(vals, 99), 397.0)


def test_percentile_edge_cases():
    assert _percentile([], 50) == 0.0
    assert _percentile([42.0], 99) == 42.0


def test_throughput_and_percentiles():
    records = [
        _rec("a", 100.0, out=50),
        _rec("b", 200.0, out=50),
        _rec("c", 300.0, out=50),
        _rec("d", 400.0, out=50),
    ]
    m = BenchMetrics.from_records(records, duration_s=4.0)

    assert m.num_requests == 4
    assert m.num_completed == 4
    assert m.num_failed == 0
    # 4 completed over 4 seconds -> 1 req/s
    assert math.isclose(m.request_throughput_req_s, 1.0)
    # 4 reqs * 50 output tokens = 200 tokens over 4 s -> 50 tok/s
    assert math.isclose(m.output_throughput_tok_s, 50.0)
    # (100 prompt + 50 out) * 4 = 600 over 4 s -> 150 tok/s
    assert math.isclose(m.total_token_throughput_tok_s, 150.0)
    assert math.isclose(m.ttft_ms["p50"], 250.0)
    assert math.isclose(m.ttft_ms["p99"], 397.0)
    assert math.isclose(m.mean_ttft_ms, 250.0)


def test_failed_requests_excluded_from_latency_and_throughput():
    records = [
        _rec("a", 100.0, out=50, success=True),
        _rec("b", 999.0, out=50, success=False),  # failed: ignored
    ]
    m = BenchMetrics.from_records(records, duration_s=1.0)
    assert m.num_completed == 1
    assert m.num_failed == 1
    # only the one completed request counts
    assert math.isclose(m.request_throughput_req_s, 1.0)
    assert math.isclose(m.ttft_ms["p50"], 100.0)


def test_tpot_excludes_short_requests():
    records = [
        _rec("a", 100.0, tpot=10.0, out=1),    # too short -> no TPOT
        _rec("b", 100.0, tpot=20.0, out=10),   # counts
    ]
    m = BenchMetrics.from_records(records, duration_s=1.0)
    # only the second request contributes to TPOT
    assert math.isclose(m.tpot_ms["p50"], 20.0)
    assert math.isclose(m.mean_tpot_ms, 20.0)


def test_empty_and_zero_duration():
    assert BenchMetrics.from_records([], duration_s=10.0).num_requests == 0
    m = BenchMetrics.from_records([_rec("a", 100.0)], duration_s=0.0)
    assert m.request_throughput_req_s == 0.0
    assert m.output_throughput_tok_s == 0.0


def test_to_dict_roundtrips_keys():
    m = BenchMetrics.from_records([_rec("a", 100.0)], duration_s=1.0)
    d = m.to_dict()
    assert d["num_completed"] == 1
    assert "p99" in d["ttft_ms"]
