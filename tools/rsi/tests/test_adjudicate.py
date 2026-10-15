"""R5/AC3: adjudicate is a deterministic pure function of (prediction, per-arm medians). No agent
channel; margin band and missing values -> inconclusive. Pure/offline."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.adjudicate import (  # noqa: E402
    FALSIFIED,
    INCONCLUSIVE,
    SUPPORTED,
    adjudicate,
)
from vllm_evolve.engine.experiment import metric_key  # noqa: E402

_M = {"column": "ttft", "agg": "p99", "group_by": "request_session_id"}
_K = metric_key(_M)


def _per_arm(a, b):
    return {"A": {_K: a}, "B": {_K: b}}


def _pred(comparator, arm_a, arm_b, margin=0.0):
    return {"metric": _M, "comparator": comparator,
            "arm_a": arm_a, "arm_b": arm_b, "margin": margin}


def test_supported_and_falsified_by_comparator():
    # H*: B (group admission) improves p99 -> B < A. With A=260, B=200, predicting B<A is supported.
    v = adjudicate(_pred("<", "B", "A", margin=10), _per_arm(a=260.0, b=200.0))
    assert v.verdict == SUPPORTED and v.measured["diff"] == 200.0 - 260.0
    # the opposite prediction (B>A) on the same numbers is falsified
    assert adjudicate(_pred(">", "B", "A", margin=10), _per_arm(260.0, 200.0)).verdict == FALSIFIED


def test_within_margin_is_inconclusive():
    # |B - A| = 5 <= margin 10 -> inconclusive (no forced call)
    v = adjudicate(_pred("<", "B", "A", margin=10), _per_arm(205.0, 200.0))
    assert v.verdict == INCONCLUSIVE


def test_missing_value_is_inconclusive_never_fabricated():
    assert adjudicate(_pred("<", "B", "A"), _per_arm(260.0, None)).verdict == INCONCLUSIVE
    assert adjudicate(_pred("<", "B", "A"), {"A": {_K: 260.0}}).verdict == INCONCLUSIVE  # B absent


def test_deterministic():
    p, pa = _pred("<", "B", "A", margin=5), _per_arm(260.0, 200.0)
    assert adjudicate(p, pa).to_dict() == adjudicate(p, pa).to_dict()


def test_no_agent_channel_cannot_inject_the_answer():
    # a producer adding a forged 'verdict'/'measured' to the prediction mapping cannot change the
    # computed result — adjudicate reads only metric/comparator/arms/margin + the per-arm numbers.
    honest = _pred("<", "B", "A", margin=10)
    forged = {**honest, "verdict": "supported", "measured": {"diff": -9999}, "note": "trust me"}
    pa = _per_arm(205.0, 200.0)                              # within margin -> inconclusive
    assert adjudicate(forged, pa).verdict == INCONCLUSIVE == adjudicate(honest, pa).verdict


def test_malformed_prediction_is_inconclusive_never_raises():
    pa = _per_arm(1.0, 2.0)
    # bad comparator
    assert adjudicate({"metric": _M, "comparator": "≈", "arm_a": "A", "arm_b": "B"},
                      pa).verdict == INCONCLUSIVE
    # NON-NUMERIC margin (Codex R2 blocker: must NOT raise float('bad'))
    assert adjudicate({"metric": _M, "comparator": "<", "arm_a": "A", "arm_b": "B",
                       "margin": "bad"}, pa).verdict == INCONCLUSIVE
    # missing metric
    assert adjudicate({"comparator": "<", "arm_a": "A", "arm_b": "B"}, pa).verdict == INCONCLUSIVE
    # malformed metric expr (bad agg)
    assert adjudicate({"metric": {"column": "ttft", "agg": "nope"}, "comparator": "<",
                       "arm_a": "A", "arm_b": "B"}, pa).verdict == INCONCLUSIVE
    # missing arm names
    assert adjudicate({"metric": _M, "comparator": "<", "arm_a": "", "arm_b": "B"},
                      pa).verdict == INCONCLUSIVE
    # NEGATIVE margin: the within-margin band is unreachable, so a confident verdict would be issued
    # for any tiny diff -> treat as malformed -> inconclusive (not supported/falsified).
    v = adjudicate(_pred("<", "B", "A", margin=-5), _per_arm(260.0, 200.0))
    assert v.verdict == INCONCLUSIVE and "malformed" in v.reason
