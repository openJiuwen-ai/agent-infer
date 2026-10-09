"""Unit tests for the multi-seed runner: escalation + classification, no GPU."""
from __future__ import annotations

import math

from vllm_evolve.bench.metrics import BenchMetrics, RequestRecord
from vllm_evolve.bench.outcome import OutcomeClass
from vllm_evolve.bench.runner import aggregate, run_profile
from vllm_evolve.bench.slo import SLO, SLOResult

SEEDS = list(range(1, 11))  # 10 seeds available


def _make_run_one(count_by_seed):
    """Fake backend: seed -> N successful records (each 1 s wall clock)."""
    def run_one(seed):
        k = count_by_seed[seed]
        recs = [
            RequestRecord(
                request_id=f"{seed}-{i}", ttft_ms=100.0, tpot_ms=10.0,
                e2e_ms=600.0, num_prompt_tokens=100, num_output_tokens=50, success=True,
            )
            for i in range(k)
        ]
        return recs, 1.0
    return run_one


# primary metric = number of completed requests (controllable per seed)
def _primary(metrics: BenchMetrics, slo: SLOResult) -> float:
    return float(metrics.num_completed)


def test_aggregate_basic():
    agg = aggregate([10.0, 10.0, 10.0])
    assert agg.median == 10.0
    assert agg.cv == 0.0
    assert agg.n == 3


def test_stable_run_stops_at_first_tier_and_is_eval_result():
    run_one = _make_run_one({s: 10 for s in SEEDS})
    res = run_profile(
        "p", "num_completed", run_one, _primary, SLO(), SEEDS,
        seed_tiers=(3, 5, 10), cv_threshold=0.1,
    )
    assert res.seeds_used == [1, 2, 3]           # no escalation needed
    assert res.outcome_class is OutcomeClass.EVAL_RESULT
    assert math.isclose(res.primary_aggregate.median, 10.0)


def test_high_variance_escalates_then_marks_third_state():
    # alternating 1 / 20 keeps CV high through every tier
    counts = {s: (1 if s % 2 else 20) for s in SEEDS}
    run_one = _make_run_one(counts)
    res = run_profile(
        "p", "num_completed", run_one, _primary, SLO(), SEEDS,
        seed_tiers=(3, 5, 10), cv_threshold=0.1,
    )
    assert len(res.seeds_used) == 10             # escalated to the max tier
    assert res.outcome_class is OutcomeClass.HIGH_VARIANCE_INCONCLUSIVE


def test_zero_completed_is_invalid_metrics():
    run_one = _make_run_one({s: 0 for s in SEEDS})
    res = run_profile(
        "p", "num_completed", run_one, _primary, SLO(), SEEDS,
        seed_tiers=(3, 5, 10), cv_threshold=0.1,
    )
    assert res.outcome_class is OutcomeClass.INVALID_METRICS


def test_no_cv_threshold_runs_only_first_tier():
    run_one = _make_run_one({s: (1 if s % 2 else 20) for s in SEEDS})
    res = run_profile(
        "p", "num_completed", run_one, _primary, SLO(), SEEDS,
        seed_tiers=(3, 5, 10), cv_threshold=None,
    )
    assert res.seeds_used == [1, 2, 3]           # no escalation without a threshold
    assert res.outcome_class is OutcomeClass.EVAL_RESULT
