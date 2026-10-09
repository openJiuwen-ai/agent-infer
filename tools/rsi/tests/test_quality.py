"""M-D3: quality gate — gain only counts if quality holds (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.quality import (  # noqa: E402
    QualityMeasurement,
    QualityThresholds,
    quality_ok,
)


def _m(ppl=None, em=None, agree=None):
    return QualityMeasurement(perplexity=ppl, task_em=em, output_agreement=agree)


def test_within_thresholds_ok():
    v = quality_ok(_m(10.0, 0.80, 0.995), _m(10.05, 0.798, 0.995))  # +0.5% ppl, -0.25% EM
    assert v.ok is True and v.reasons == []


def test_missing_em_when_gate_enabled_fails():
    # EM gate enabled by default -> a missing EM is FAIL, not a skip (Codex HIGH#4)
    v = quality_ok(_m(10.0, 0.80, 0.999), _m(10.0, None, 0.999))
    assert v.ok is False and any("EM not measured" in r for r in v.reasons)


def test_missing_agreement_when_gate_enabled_fails():
    v = quality_ok(_m(10.0, 0.80, 0.999), _m(10.0, 0.80, None))
    assert v.ok is False and any("agreement not measured" in r for r in v.reasons)


def test_disabling_gate_explicitly_allows_missing():
    from vllm_evolve.engine.quality import QualityThresholds
    t = QualityThresholds(max_em_drop_pct=None, min_output_agreement=None)  # only ppl gate
    v = quality_ok(_m(10.0, None, None), _m(10.05, None, None), t)
    assert v.ok is True


def test_perplexity_regression_fails():
    v = quality_ok(_m(10.0, 0.80), _m(11.0, 0.80, 0.999))   # +10% ppl
    assert v.ok is False and any("perplexity" in r for r in v.reasons)


def test_em_drop_fails():
    v = quality_ok(_m(10.0, 0.80), _m(10.0, 0.70, 0.999))   # -12.5% EM
    assert v.ok is False and any("EM" in r for r in v.reasons)


def test_low_output_agreement_fails():
    v = quality_ok(_m(10.0, 0.80), _m(10.0, 0.80, 0.90))
    assert v.ok is False and any("agreement" in r for r in v.reasons)


def test_missing_perplexity_cannot_certify():
    # honest: absent evidence is NOT "quality fine"
    v = quality_ok(_m(None, 0.80), _m(None, 0.80, 0.999))
    assert v.ok is False and any("perplexity not measured" in r for r in v.reasons)


def test_thresholds_are_pre_registered_and_respected():
    strict = QualityThresholds(max_perplexity_increase_pct=0.1)
    assert quality_ok(_m(10.0, 0.8), _m(10.05, 0.8, 0.999), strict).ok is False  # +0.5% > 0.1%
