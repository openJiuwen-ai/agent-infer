"""Unit tests for bench.slo — exact-value assertions, no GPU."""
from __future__ import annotations

import math

from vllm_evolve.bench.metrics import RequestRecord
from vllm_evolve.bench.slo import SLO, attains, evaluate_slo


def _rec(rid, ttft, tpot=10.0, e2e=500.0, out=50, success=True):
    return RequestRecord(
        request_id=rid, ttft_ms=ttft, tpot_ms=tpot, e2e_ms=e2e,
        num_prompt_tokens=100, num_output_tokens=out, success=success,
    )


def test_empty_slo_attains_on_success_only():
    slo = SLO()
    assert slo.is_empty()
    assert attains(_rec("a", 9999.0, success=True), slo) is True
    assert attains(_rec("b", 1.0, success=False), slo) is False


def test_ttft_threshold():
    slo = SLO(ttft_ms=250.0)
    assert attains(_rec("a", 200.0), slo) is True
    assert attains(_rec("b", 250.0), slo) is True   # boundary inclusive
    assert attains(_rec("c", 300.0), slo) is False


def test_tpot_threshold_ignored_for_short_requests():
    slo = SLO(tpot_ms=15.0)
    # 1 output token -> TPOT undefined -> not penalised
    assert attains(_rec("short", 100.0, tpot=99.0, out=1), slo) is True
    # long request with bad TPOT -> fails
    assert attains(_rec("long", 100.0, tpot=99.0, out=10), slo) is False


def test_failed_request_never_attains():
    slo = SLO(ttft_ms=10_000.0)
    assert attains(_rec("a", 1.0, success=False), slo) is False


def test_goodput_under_ttft_slo():
    records = [
        _rec("a", 100.0, out=50),
        _rec("b", 200.0, out=50),
        _rec("c", 300.0, out=50),
        _rec("d", 400.0, out=50),
    ]
    res = evaluate_slo(records, SLO(ttft_ms=250.0), duration_s=4.0)
    assert res.total == 4
    assert res.completed == 4
    assert res.attained == 2                       # a, b
    assert math.isclose(res.attainment_rate, 0.5)
    assert math.isclose(res.goodput_req_s, 0.5)    # 2 attained / 4 s
    assert math.isclose(res.goodput_tok_s, 25.0)   # 2*50 / 4 s


def test_goodput_counts_failures_against_attainment():
    records = [
        _rec("a", 100.0, success=True),
        _rec("b", 100.0, success=False),
    ]
    res = evaluate_slo(records, SLO(ttft_ms=250.0), duration_s=1.0)
    assert res.total == 2
    assert res.completed == 1
    assert res.attained == 1
    assert math.isclose(res.attainment_rate, 0.5)


def test_zero_duration_no_divide_by_zero():
    res = evaluate_slo([_rec("a", 100.0)], SLO(ttft_ms=250.0), duration_s=0.0)
    assert res.goodput_req_s == 0.0
    assert res.goodput_tok_s == 0.0
