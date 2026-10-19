"""Multi-seed runner: median fitness + variance-driven seed escalation.

A single serving run is noisy, so a profile is run across several seeds and the
**median** is the reported fitness. If the per-seed spread (coefficient of
variation) of the primary metric is too high at the current seed tier, the
runner escalates to the next tier (e.g. 3 -> 5 -> 10). If the spread is still
too high at the maximum tier, the outcome is ``high_variance_inconclusive`` —
an honest "we can't tell", never a fabricated pass.

The actual per-seed execution is injected as a ``run_one`` callable
``seed -> (records, duration_s)``. In production this is the real vLLM backend
(GPU-gated); in tests it is a deterministic fake. This keeps all the loop /
escalation / classification logic unit-testable without a GPU.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from vllm_evolve.bench.metrics import BenchMetrics, RequestRecord
from vllm_evolve.bench.outcome import OutcomeClass
from vllm_evolve.bench.slo import SLO, SLOResult, evaluate_slo

# seed -> (per-request records, wall-clock duration seconds)
RunOne = Callable[[int], tuple[Sequence[RequestRecord], float]]
# (metrics, slo_result) -> the single scalar this profile is judged on
PrimaryMetricFn = Callable[[BenchMetrics, SLOResult], float]


@dataclass
class SeedRun:
    seed: int
    metrics: BenchMetrics
    slo_result: SLOResult
    primary_value: float

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "slo_result": self.slo_result.to_dict(),
            "primary_value": self.primary_value,
        }


@dataclass
class Aggregate:
    median: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    cv: float = 0.0
    n: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(values: Sequence[float]) -> Aggregate:
    vals = list(values)
    if not vals:
        return Aggregate()
    if len(vals) == 1:
        return Aggregate(median=vals[0], mean=vals[0], std=0.0, cv=0.0, n=1)
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals)
    cv = abs(std / mean) if abs(mean) > 1e-12 else 0.0
    return Aggregate(median=statistics.median(vals), mean=mean, std=std, cv=cv, n=len(vals))


@dataclass
class BenchResult:
    profile: str
    primary_metric: str
    seed_runs: list[SeedRun] = field(default_factory=list)
    primary_aggregate: Aggregate = field(default_factory=Aggregate)
    outcome_class: OutcomeClass = OutcomeClass.EVAL_RESULT
    seeds_used: list[int] = field(default_factory=list)
    cv_threshold: float | None = None

    @property
    def primary_values(self) -> list[float]:
        return [r.primary_value for r in self.seed_runs]

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "primary_metric": self.primary_metric,
            "seed_runs": [r.to_dict() for r in self.seed_runs],
            "primary_aggregate": self.primary_aggregate.to_dict(),
            "outcome_class": self.outcome_class.value,
            "seeds_used": list(self.seeds_used),
            "cv_threshold": self.cv_threshold,
        }


def run_profile(
    profile: str,
    primary_metric: str,
    run_one: RunOne,
    primary_metric_fn: PrimaryMetricFn,
    slo: SLO,
    seeds: Sequence[int],
    *,
    seed_tiers: Sequence[int] = (3, 5, 10),
    cv_threshold: float | None = None,
) -> BenchResult:
    """Run ``profile`` across seeds, escalating tiers while variance is high.

    ``seeds`` must contain at least ``max(seed_tiers)`` distinct seeds. The
    runner takes the first N for each tier. Escalation stops as soon as the CV
    of the primary metric is within ``cv_threshold`` (or ``cv_threshold`` is
    ``None``), or the largest tier is reached.
    """
    tiers = sorted(t for t in seed_tiers if t <= len(seeds))
    if not tiers:
        tiers = [len(seeds)]

    seed_runs: list[SeedRun] = []
    ran = 0
    agg = Aggregate()
    for tier in tiers:
        # run any additional seeds needed to reach this tier
        for seed in seeds[ran:tier]:
            records, duration = run_one(seed)
            metrics = BenchMetrics.from_records(records, duration)
            slo_result = evaluate_slo(records, slo, duration)
            primary = primary_metric_fn(metrics, slo_result)
            seed_runs.append(SeedRun(seed, metrics, slo_result, primary))
        ran = tier
        agg = aggregate([r.primary_value for r in seed_runs])
        if cv_threshold is None or agg.cv <= cv_threshold:
            break

    outcome = _classify(seed_runs, agg, cv_threshold)
    return BenchResult(
        profile=profile,
        primary_metric=primary_metric,
        seed_runs=seed_runs,
        primary_aggregate=agg,
        outcome_class=outcome,
        seeds_used=[r.seed for r in seed_runs],
        cv_threshold=cv_threshold,
    )


def _classify(
    seed_runs: Sequence[SeedRun],
    agg: Aggregate,
    cv_threshold: float | None,
) -> OutcomeClass:
    """Map a finished multi-seed run to an outcome class.

    Backend-level failures (crash / timeout / plugin load) are classified by the
    backend before reaching here; this function only distinguishes a usable
    result from invalid metrics or unresolved high variance.
    """
    total_completed = sum(r.metrics.num_completed for r in seed_runs)
    if not seed_runs or total_completed == 0:
        return OutcomeClass.INVALID_METRICS
    if cv_threshold is not None and agg.cv > cv_threshold:
        return OutcomeClass.HIGH_VARIANCE_INCONCLUSIVE
    return OutcomeClass.EVAL_RESULT


# Convenience extractors for the common primary metrics.
def goodput_req_s(metrics: BenchMetrics, slo: SLOResult) -> float:
    return slo.goodput_req_s


def output_throughput(metrics: BenchMetrics, slo: SLOResult) -> float:
    return metrics.output_throughput_tok_s
