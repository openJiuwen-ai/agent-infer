"""Unit tests for bench.compare — deterministic bootstrap, no GPU."""
from __future__ import annotations

from vllm_evolve.bench.compare import (
    Verdict,
    coefficient_of_variation,
    compare_metric,
)


def test_cv_basic():
    assert coefficient_of_variation([10.0]) == 0.0
    assert coefficient_of_variation([5.0, 5.0, 5.0]) == 0.0
    cv = coefficient_of_variation([10.0, 12.0, 8.0, 11.0, 9.0])
    assert 0.0 < cv < 0.3


def test_better_for_lower_is_better_latency():
    # latency: lower is better. Candidate halves it cleanly.
    baseline = [200.0, 210.0, 205.0, 195.0, 200.0]
    candidate = [100.0, 105.0, 95.0, 100.0, 98.0]
    c = compare_metric("ttft", baseline, candidate, higher_is_better=False, seed=1)
    assert c.verdict is Verdict.BETTER
    assert c.rel_improvement_pct > 40.0
    assert c.ci_low_pct > 2.0


def test_worse_for_lower_is_better_latency():
    baseline = [100.0, 105.0, 95.0, 100.0, 98.0]
    candidate = [200.0, 210.0, 205.0, 195.0, 200.0]
    c = compare_metric("ttft", baseline, candidate, higher_is_better=False, seed=1)
    assert c.verdict is Verdict.WORSE
    assert c.ci_high_pct < -2.0


def test_better_for_higher_is_better_throughput():
    baseline = [1000.0, 1010.0, 990.0, 1005.0, 995.0]
    candidate = [1400.0, 1420.0, 1380.0, 1405.0, 1395.0]
    c = compare_metric("throughput", baseline, candidate, higher_is_better=True, seed=2)
    assert c.verdict is Verdict.BETTER


def test_inconclusive_when_overlapping():
    baseline = [100.0, 110.0, 90.0, 105.0, 95.0]
    candidate = [101.0, 109.0, 92.0, 104.0, 96.0]
    c = compare_metric("ttft", baseline, candidate, higher_is_better=False, seed=3)
    assert c.verdict is Verdict.INCONCLUSIVE


def test_high_variance_overrides_to_third_state():
    baseline = [100.0, 100.0, 100.0]
    candidate = [10.0, 300.0, 50.0]  # huge spread
    c = compare_metric(
        "ttft", baseline, candidate, higher_is_better=False,
        cv_threshold=0.3, seed=4,
    )
    assert c.verdict is Verdict.HIGH_VARIANCE_INCONCLUSIVE


def test_bootstrap_is_deterministic():
    baseline = [200.0, 210.0, 205.0, 195.0, 200.0]
    candidate = [100.0, 105.0, 95.0, 100.0, 98.0]
    a = compare_metric("ttft", baseline, candidate, higher_is_better=False, seed=7)
    b = compare_metric("ttft", baseline, candidate, higher_is_better=False, seed=7)
    assert a.ci_low_pct == b.ci_low_pct
    assert a.ci_high_pct == b.ci_high_pct


def test_empty_input_raises():
    import pytest

    with pytest.raises(ValueError):
        compare_metric("x", [], [1.0], higher_is_better=True)
