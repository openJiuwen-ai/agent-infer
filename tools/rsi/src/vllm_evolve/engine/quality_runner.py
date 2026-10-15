"""M-D3 (runner half) — fixed calibration-set quality MEASUREMENT.

``quality.py`` judges supplied measurements; this RUNS the measurement. Against a FROZEN
calibration set (pre-registered, never tuned to pass), it produces a ``QualityMeasurement``
(perplexity / task EM / deterministic output agreement) via an injected ``measure_fn`` —
real = the served model (box-gated), tests = a mock. If measurement is unavailable or fails,
it returns ``None``, which the caller MUST treat as NOT MEASURED — and the gate hard-fails.
Never assume quality_ok by construction (Codex: representation OR scheduling candidates alike
must be measured before adoption).
"""
from __future__ import annotations

from vllm_evolve.engine.quality import QualityMeasurement, QualityThresholds, quality_ok

# Frozen calibration set: a small fixed battery of prompts + expected continuations. It is
# pre-registered and never edited to make a candidate pass.
_CALIBRATION_SET: tuple[dict, ...] = (
    {"prompt": "The capital of France is", "expected": " Paris"},
    {"prompt": "Water is made of hydrogen and", "expected": " oxygen"},
    {"prompt": "2 + 2 =", "expected": " 4"},
    {"prompt": "The opposite of hot is", "expected": " cold"},
    {"prompt": "The first president of the United States was", "expected": " George Washington"},
)


def calibration_set() -> tuple[dict, ...]:
    return _CALIBRATION_SET


def measure_quality(measure_fn, calibration=None) -> QualityMeasurement | None:
    """Measure one config's quality on the fixed set.

    ``measure_fn(prompts) -> {perplexity, task_em, output_agreement}`` (real = served model,
    box-gated). Returns a QualityMeasurement, or ``None`` when measurement is unavailable or
    fails — the caller MUST treat ``None`` as NOT MEASURED (a hard fail), never an assumed pass.
    """
    cal = calibration or _CALIBRATION_SET
    if measure_fn is None:
        return None                       # no model to measure (box-gated) -> not measured
    try:
        m = measure_fn([c["prompt"] for c in cal])
    except Exception:                     # noqa: BLE001 - honest measurement failure
        return None
    if not m:
        return None
    return QualityMeasurement(perplexity=m.get("perplexity"), task_em=m.get("task_em"),
                              output_agreement=m.get("output_agreement"))


def measured_quality_verdict(baseline_fn, candidate_fn, thresholds: QualityThresholds | None = None,
                             calibration=None):
    """Measure baseline + candidate on the FROZEN set and judge. Returns a QualityVerdict, or
    ``None`` when either side could not be measured (NOT MEASURED -> the gate hard-fails)."""
    base = measure_quality(baseline_fn, calibration)
    cand = measure_quality(candidate_fn, calibration)
    if base is None or cand is None:
        return None
    return quality_ok(base, cand, thresholds)


def is_quality_certified(verdict) -> bool:
    """True ONLY if a measured verdict exists AND it passed. None (not measured) -> False."""
    return bool(verdict is not None and verdict.ok)
