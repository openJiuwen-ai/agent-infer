"""Candidate-vs-baseline comparison with a three-state verdict.

Given a metric measured across several seeds for both the baseline (seed
policy) and a candidate policy, decide whether the candidate is **better**,
**worse**, **inconclusive**, or **high-variance-inconclusive** on that metric.

Design choices that keep this honest:

* The point estimate is the **median** across seeds (robust to a single bad
  run), not the mean.
* Significance uses a **bootstrap confidence interval** of the
  direction-adjusted relative improvement, so a difference only counts when it
  is larger than seed-to-seed noise. The bootstrap is seeded, so the verdict is
  deterministic and reproducible.
* ``epsilon_pct`` is a dead-band: improvements smaller than it are treated as a
  tie (INCONCLUSIVE), so we never claim victory on noise.
* If the candidate's coefficient of variation exceeds ``cv_threshold`` the
  verdict is ``HIGH_VARIANCE_INCONCLUSIVE`` — a distinct third state, never
  folded into pass or fail.
"""
from __future__ import annotations

import enum
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from vllm_evolve.bench.metrics import _percentile

_EPS = 1e-12


class Verdict(str, enum.Enum):
    BETTER = "better"
    WORSE = "worse"
    INCONCLUSIVE = "inconclusive"
    HIGH_VARIANCE_INCONCLUSIVE = "high_variance_inconclusive"


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Sample CV = stdev / |mean|. Zero for < 2 samples or zero mean."""
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if abs(mean) < _EPS:
        return 0.0
    return abs(statistics.stdev(values) / mean)


def _paired_seed_deltas(
    baseline: Sequence[float],
    candidate: Sequence[float],
    higher_is_better: bool = True,
) -> list[float]:
    """Direction-adjusted percentage delta for each matched seed."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired stat needs equal-length, non-empty matched samples")
    deltas = []
    for base, cand in zip(baseline, candidate):
        denom = abs(base) if abs(base) > _EPS else _EPS
        improvement = (cand - base) if higher_is_better else (base - cand)
        deltas.append(100.0 * improvement / denom)
    return deltas


def paired_relative_improvement(
    baseline: Sequence[float],
    candidate: Sequence[float],
    higher_is_better: bool = True,
) -> float:
    """Median relative improvement across matched per-seed measurements."""
    return statistics.median(
        _paired_seed_deltas(baseline, candidate, higher_is_better)
    )


def paired_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    higher_is_better: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 95.0,
) -> tuple[float, float]:
    """Deterministic paired-bootstrap CI over matched per-seed deltas."""
    deltas = _paired_seed_deltas(baseline, candidate, higher_is_better)
    rng = random.Random(seed)
    count = len(deltas)
    bootstraps = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(count)] for _ in range(count)]
        bootstraps.append(statistics.median(sample))
    bootstraps.sort()
    lo_q = (100.0 - ci) / 2.0
    return _percentile(bootstraps, lo_q), _percentile(bootstraps, 100.0 - lo_q)


def _bootstrap_improvement_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    higher_is_better: bool,
    n_boot: int,
    seed: int,
    ci: float = 95.0,
) -> tuple[float, float]:
    """Bootstrap CI (percent) of the direction-adjusted relative improvement.

    Improvement is expressed as a percentage of the baseline median magnitude.
    Positive = candidate better, regardless of metric direction.
    """
    rng = random.Random(seed)
    nb, nc = len(baseline), len(candidate)
    deltas: list[float] = []
    for _ in range(n_boot):
        b = [baseline[rng.randrange(nb)] for _ in range(nb)]
        c = [candidate[rng.randrange(nc)] for _ in range(nc)]
        bm = statistics.median(b)
        cm = statistics.median(c)
        improvement = (cm - bm) if higher_is_better else (bm - cm)
        denom = abs(bm) if abs(bm) > _EPS else _EPS
        deltas.append(100.0 * improvement / denom)
    deltas.sort()
    lo_q = (100.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    return _percentile(deltas, lo_q), _percentile(deltas, hi_q)


@dataclass
class MetricComparison:
    metric: str
    higher_is_better: bool
    baseline_median: float
    candidate_median: float
    rel_improvement_pct: float   # direction-adjusted, vs baseline median
    ci_low_pct: float
    ci_high_pct: float
    candidate_cv: float
    verdict: Verdict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


def compare_metric(
    metric: str,
    baseline: Sequence[float],
    candidate: Sequence[float],
    higher_is_better: bool,
    *,
    epsilon_pct: float = 2.0,
    cv_threshold: float | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> MetricComparison:
    """Compare one metric across seeds and return a three-state verdict.

    ``baseline`` and ``candidate`` are the per-seed values of the same metric.
    """
    if not baseline or not candidate:
        raise ValueError("baseline and candidate must each have >= 1 sample")

    base_med = statistics.median(baseline)
    cand_med = statistics.median(candidate)
    denom = abs(base_med) if abs(base_med) > _EPS else _EPS
    improvement = (cand_med - base_med) if higher_is_better else (base_med - cand_med)
    rel = 100.0 * improvement / denom

    lo, hi = _bootstrap_improvement_ci(baseline, candidate, higher_is_better, n_boot, seed)
    ccv = coefficient_of_variation(candidate)

    if cv_threshold is not None and ccv > cv_threshold:
        verdict = Verdict.HIGH_VARIANCE_INCONCLUSIVE
    elif lo > epsilon_pct:
        verdict = Verdict.BETTER
    elif hi < -epsilon_pct:
        verdict = Verdict.WORSE
    else:
        verdict = Verdict.INCONCLUSIVE

    return MetricComparison(
        metric=metric,
        higher_is_better=higher_is_better,
        baseline_median=base_med,
        candidate_median=cand_med,
        rel_improvement_pct=rel,
        ci_low_pct=lo,
        ci_high_pct=hi,
        candidate_cv=ccv,
        verdict=verdict,
    )
