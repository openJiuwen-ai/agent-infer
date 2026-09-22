"""M-D4: strict acceptance gate vs strong baseline (paired bootstrap, zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from vllm_evolve.core.accept import (  # noqa: E402
    accept_vs_strong_baseline,
    paired_ci,
)

# +25% gain, low variance
_BASE = [100.0, 101.0, 99.0, 100.5]
_CAND = [125.0, 126.0, 124.0, 125.5]
_HB = [100.0, 100.0, 101.0]
_HC = [124.0, 125.0, 124.0]


def _accept(baseline, candidate, *, effective=True, quality=True, **kw):
    """Helper: by default supply proven effective+quality so we test the stats path."""
    return accept_vs_strong_baseline(baseline, candidate, candidate_effective=effective,
                                     quality_ok=quality, **kw)


def test_strong_gain_with_holdout_accepted():
    v = _accept(_BASE, _CAND, threshold_pct=20.0, holdout_baseline=_HB, holdout_candidate=_HC)
    assert v.accepted is True and v.point_gain_pct >= 20 and v.ci_low_pct >= 20
    assert v.holdout_ok is True


def test_missing_evidence_rejected_even_with_gain():
    # Codex HIGH#1: forgetting to pass effective/quality must NOT accept
    v = accept_vs_strong_baseline(_BASE, _CAND, threshold_pct=20.0,
                                  holdout_baseline=_HB, holdout_candidate=_HC)
    assert v.accepted is False
    assert any("effectiveness NOT proven" in r for r in v.reasons)
    assert any("quality NOT certified" in r for r in v.reasons)


def test_gain_below_threshold_rejected():
    cand = [108.0, 109.0, 107.0, 108.5]   # ~+8% < 20%
    v = _accept(_BASE, cand, threshold_pct=20.0, holdout_baseline=_HB, holdout_candidate=_HC)
    assert v.accepted is False and any("primary gate" in r for r in v.reasons)


def test_ineffective_candidate_rejected_despite_gain():
    v = _accept(_BASE, _CAND, effective=False, threshold_pct=20.0,
                holdout_baseline=_HB, holdout_candidate=_HC)
    assert v.accepted is False and any("effectiveness NOT proven" in r for r in v.reasons)


def test_quality_regression_rejected():
    v = _accept(_BASE, _CAND, quality=False, threshold_pct=20.0,
                holdout_baseline=_HB, holdout_candidate=_HC)
    assert v.accepted is False and any("quality NOT certified" in r for r in v.reasons)


def test_no_holdout_not_accepted():
    v = _accept(_BASE, _CAND, threshold_pct=20.0)
    assert v.accepted is False and v.holdout_ok is None
    assert any("needs bootstrap AND holdout" in r for r in v.reasons)


def test_holdout_collapse_rejected():
    hc = [100.0, 101.0, 100.0]   # ~0% at holdout config -> gain didn't generalize
    v = _accept(_BASE, _CAND, threshold_pct=20.0, holdout_baseline=_HB, holdout_candidate=hc)
    assert v.accepted is False and v.holdout_ok is False


def test_paired_ci_requires_matched_lengths():
    with pytest.raises(ValueError):
        paired_ci([1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_ci_rejects_empty_sample():
    with pytest.raises(ValueError):
        paired_ci([], [])


def test_accept_rejects_non_real_source_metadata_independently():
    # LOCK D at accept (Codex R0 step 2): a candidate whose eval-result metadata is non-real is
    # refused on metadata ALONE — even when effectiveness, quality, gain, and holdout would adopt.
    base = [5.0, 5.0, 5.0]
    cand = [10.0, 10.0, 10.0]   # +100%
    hb, hc = [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]
    adopts = accept_vs_strong_baseline(base, cand, candidate_effective=True, quality_ok=True,
                                       holdout_baseline=hb, holdout_candidate=hc)
    assert adopts.accepted is True   # sanity: without metadata this would adopt

    blocked = accept_vs_strong_baseline(
        base, cand, candidate_effective=True, quality_ok=True,
        holdout_baseline=hb, holdout_candidate=hc,
        candidate_meta={"source": "local_smoke", "outcome_class": "local_smoke_nonqualifying"})
    assert blocked.accepted is False
    assert any("non_real_source_blocked" in r for r in blocked.reasons)

    # a FORGED real source (lying source but smoke outcome_class) is also refused
    forged = accept_vs_strong_baseline(
        base, cand, candidate_effective=True, quality_ok=True,
        holdout_baseline=hb, holdout_candidate=hc,
        candidate_meta={"source": "real_vllm", "outcome_class": "local_smoke_nonqualifying"})
    assert forged.accepted is False


def test_accept_rejects_empty_or_unequal_samples_without_raising():
    # Codex review P2: empty / unequal-length value lists must yield a non-adoption verdict, not a
    # ValueError from the paired stats (which would abort ve verify-gain / ve autopt JSON output).
    from vllm_evolve.core.accept import accept_vs_strong_baseline
    empty = accept_vs_strong_baseline([], [], candidate_effective=True, quality_ok=True)
    assert empty.accepted is False
    assert any("insufficient paired samples" in r for r in empty.reasons)
    unequal = accept_vs_strong_baseline([5.0, 5.0], [7.0],
                                        candidate_effective=True, quality_ok=True)
    assert unequal.accepted is False
    assert any("insufficient paired samples" in r for r in unequal.reasons)


def test_accept_rejects_mismatched_holdout_without_raising():
    # Codex review P2: mismatched HOLDOUT per-seed counts must reject (holdout unusable), not raise
    # ValueError from the paired holdout stats (which would abort the verify-gain JSON).
    from vllm_evolve.core.accept import accept_vs_strong_baseline
    v = accept_vs_strong_baseline(
        [5.0, 5.0, 5.0], [7.0, 7.0, 7.0], candidate_effective=True, quality_ok=True,
        holdout_baseline=[5.0, 5.0], holdout_candidate=[7.0])   # mismatched holdout lengths
    assert v.accepted is False
    assert any("holdout samples mismatched" in r for r in v.reasons)
