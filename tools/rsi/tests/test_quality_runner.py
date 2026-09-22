"""AC5: fixed calibration-set quality measurement runner (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.quality_runner import (  # noqa: E402
    calibration_set,
    is_quality_certified,
    measure_quality,
    measured_quality_verdict,
)


def test_calibration_set_is_frozen_and_nonempty():
    cs = calibration_set()
    assert len(cs) >= 3 and all("prompt" in c and "expected" in c for c in cs)


def test_measure_quality_with_model():
    def fake_measure(prompts):
        assert len(prompts) == len(calibration_set())
        return {"perplexity": 10.0, "task_em": 0.8, "output_agreement": 0.999}
    m = measure_quality(fake_measure)
    assert m.perplexity == 10.0 and m.task_em == 0.8


def test_missing_model_is_not_measured():
    # box-gated: no measure_fn -> None (NOT measured), never an assumed pass
    assert measure_quality(None) is None
    assert is_quality_certified(None) is False


def test_measure_failure_is_not_measured():
    def boom(_prompts):
        raise RuntimeError("model unavailable")
    assert measure_quality(boom) is None


def test_measured_verdict_certifies_only_when_measured_and_passing():
    def base(_):
        return {"perplexity": 10.0, "task_em": 0.80, "output_agreement": 0.999}

    def good(_):
        return {"perplexity": 10.05, "task_em": 0.798, "output_agreement": 0.999}

    def bad(_):
        return {"perplexity": 13.0, "task_em": 0.80, "output_agreement": 0.999}   # +30% ppl

    assert is_quality_certified(measured_quality_verdict(base, good)) is True
    assert is_quality_certified(measured_quality_verdict(base, bad)) is False
    # either side unmeasured -> not certified (hard fail)
    assert measured_quality_verdict(base, None) is None
    assert is_quality_certified(measured_quality_verdict(None, good)) is False
