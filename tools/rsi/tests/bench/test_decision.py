"""Unit tests for bench.decision — precedence + manual policy, no GPU."""
from __future__ import annotations

from vllm_evolve.bench.compare import Verdict
from vllm_evolve.bench.decision import DecisionPolicy, Outcome, decide

POLICY = DecisionPolicy(
    require_improve=["throughput"],
    forbid_regress=["latency", "replay"],
)


def test_accept_when_required_improves_and_nothing_regresses():
    verdicts = {
        "throughput": Verdict.BETTER,
        "latency": Verdict.INCONCLUSIVE,
        "replay": Verdict.BETTER,
    }
    assert decide(verdicts, POLICY).outcome is Outcome.ACCEPT


def test_reject_when_forbidden_profile_regresses():
    verdicts = {
        "throughput": Verdict.BETTER,   # required improved...
        "latency": Verdict.WORSE,       # ...but a forbidden one regressed
        "replay": Verdict.BETTER,
    }
    assert decide(verdicts, POLICY).outcome is Outcome.REJECT


def test_reject_when_required_profile_regresses():
    verdicts = {"throughput": Verdict.WORSE, "latency": Verdict.BETTER, "replay": Verdict.BETTER}
    assert decide(verdicts, POLICY).outcome is Outcome.REJECT


def test_inconclusive_when_required_inconclusive():
    verdicts = {
        "throughput": Verdict.INCONCLUSIVE,
        "latency": Verdict.BETTER,
        "replay": Verdict.BETTER,
    }
    assert decide(verdicts, POLICY).outcome is Outcome.INCONCLUSIVE


def test_inconclusive_when_required_missing():
    verdicts = {"latency": Verdict.BETTER, "replay": Verdict.BETTER}
    assert decide(verdicts, POLICY).outcome is Outcome.INCONCLUSIVE


def test_high_variance_in_relevant_profile_is_inconclusive():
    verdicts = {
        "throughput": Verdict.BETTER,
        "latency": Verdict.HIGH_VARIANCE_INCONCLUSIVE,
        "replay": Verdict.BETTER,
    }
    assert decide(verdicts, POLICY).outcome is Outcome.INCONCLUSIVE


def test_forbidden_regression_beats_high_variance():
    # precedence: a forbidden WORSE rejects even if another relevant profile is high-variance
    verdicts = {
        "throughput": Verdict.HIGH_VARIANCE_INCONCLUSIVE,
        "latency": Verdict.WORSE,
        "replay": Verdict.BETTER,
    }
    assert decide(verdicts, POLICY).outcome is Outcome.REJECT


def test_policy_from_dict():
    p = DecisionPolicy.from_dict(
        {"require_improve": ["throughput"], "forbid_regress": ["latency"], "epsilon_pct": 3.0}
    )
    assert p.require_improve == ["throughput"]
    assert p.forbid_regress == ["latency"]
    assert p.epsilon_pct == 3.0
