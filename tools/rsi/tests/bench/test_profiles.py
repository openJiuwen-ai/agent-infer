"""Unit tests for profile + decision-policy loading from the shipped configs."""
from __future__ import annotations

from pathlib import Path

from vllm_evolve.bench.metrics import BenchMetrics
from vllm_evolve.bench.profiles import (
    load_decision_policy,
    load_profile,
    primary_metric_fn,
)
from vllm_evolve.bench.slo import SLOResult

CONFIG = Path(__file__).resolve().parents[2] / "config" / "bench"


def test_load_throughput_profile():
    p = load_profile(CONFIG / "profiles" / "throughput.yaml")
    assert p.name == "throughput"
    assert p.regime == "throughput"
    assert p.slo.ttft_ms == 5000
    assert p.slo.tpot_ms == 100
    assert p.primary_metric == "goodput_req_s"
    assert p.seed_tiers == [3, 5, 10]
    assert p.cv_threshold == 0.10


def test_all_three_profiles_load():
    for n in ("throughput", "latency", "replay"):
        p = load_profile(CONFIG / "profiles" / f"{n}.yaml")
        assert p.regime == n
        assert p.primary_metric == "goodput_req_s"


def test_replay_profile_names_a_dataset():
    p = load_profile(CONFIG / "profiles" / "replay.yaml")
    assert p.workload["kind"] == "trace"
    assert p.workload["dataset"] == "burstgpt"


def test_load_decision_policy():
    dp = load_decision_policy(CONFIG / "decision.yaml")
    assert dp.require_improve == ["throughput"]
    assert dp.forbid_regress == ["latency", "replay"]
    assert dp.epsilon_pct == 2.0


def test_primary_metric_fn_resolves_goodput():
    fn = primary_metric_fn("goodput_req_s")
    metrics = BenchMetrics()
    slo = SLOResult(goodput_req_s=9.5)
    assert fn(metrics, slo) == 9.5


def test_primary_metric_fn_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        primary_metric_fn("nonsense")(BenchMetrics(), SLOResult())
