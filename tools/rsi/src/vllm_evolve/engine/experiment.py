"""R1 — experiments as declarative data (P1).

An ``ExperimentSpec`` declares: a workload (a trace, the primitive), A/B ``arms`` (each a policy +
knob delta), the ``MetricExpr`` metrics to measure, ``seeds``, range guardrails, and a ``budget``
(a hard cap on total arm×seed simulations). ``run_experiment`` validates by RANGE (never an enum of
shapes), materializes the trace once, then runs each arm×seed through an injected ``eval_fn`` with
the BUDGET CHARGED AT THE EVAL-CALL BOUNDARY (mirrors ``evolve_loop``): once the cap is reached it
records ``budget_exhausted`` instead of silently continuing. It evaluates each metric per arm per
seed with the SHARED catalog evaluator and reduces across seeds by MEDIAN — the exact per-arm values
``adjudicate`` (R5) consumes. It NEVER adjudicates and never fabricates a missing value.

``run_experiment`` is agent-free and deterministic given ``eval_fn``. engine layer (not a frozen
seed); imports ``bench`` (allowed) but no ``core``.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vllm_evolve.bench.frontier_catalog import FrontierCatalog, MetricExpr, evaluate
from vllm_evolve.engine.workload_synth import (
    TraceRanges,
    WorkloadSpec,
    validate_trace,
    write_trace_csv,
)

# Hard ceiling on total arm×seed simulations per experiment (mirrors evolve_loop's eval budget).
_DEFAULT_BUDGET = 24


@dataclass
class Arm:
    """One arm = a named condition: a policy (None == baseline vllm_v1) + a knob delta."""

    name: str
    policy_path: str | None = None
    knobs: dict = field(default_factory=dict)


@dataclass
class ExperimentSpec:
    """Declarative experiment. ``knob_ranges`` is a RANGE guardrail ``{knob: (lo, hi)}`` (never an
    enum of allowed values). ``budget`` caps total arm×seed sims."""

    workload: WorkloadSpec
    arms: list
    metrics: list                                  # list[MetricExpr | mapping]
    seeds: list = field(default_factory=lambda: [0])
    knobs: dict = field(default_factory=dict)
    budget: int = _DEFAULT_BUDGET
    ranges: TraceRanges = field(default_factory=TraceRanges)
    knob_ranges: dict = field(default_factory=dict)

    def planned_evals(self) -> int:
        return len(self.arms) * len(self.seeds)


@dataclass
class ArmSeedRecord:
    arm: str
    seed: int
    metrics: dict = field(default_factory=dict)    # {metric_key: value|None}
    missing: dict = field(default_factory=dict)     # {metric_key: [missing cols]}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    experiment_id: str
    out_dir: str
    spec_sha: str
    metric_keys: list = field(default_factory=list)
    records: list = field(default_factory=list)     # [ArmSeedRecord]
    per_arm: dict = field(default_factory=dict)      # {arm: {metric_key: median|None}}
    per_arm_missing: dict = field(default_factory=dict)  # {arm: {metric_key: [missing seed...]}}
    evals_used: int = 0
    terminated: str = ""    # complete | budget_exhausted | invalid_spec
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["records"] = [r if isinstance(r, dict) else r.to_dict() for r in self.records]
        return d


def metric_key(m) -> str:
    """A stable key for a MetricExpr (shared by experiment records and adjudication lookups)."""
    e = MetricExpr.from_obj(m)
    return f"{e.column}|{e.agg}|{e.group_by}|{e.group_reduce}"


def _spec_sha(spec: ExperimentSpec, trace: list) -> str:
    payload = {
        "trace": trace,
        "arms": [asdict(a) for a in spec.arms],
        "metrics": [MetricExpr.from_obj(m).to_dict() for m in spec.metrics],
        "seeds": list(spec.seeds),
        "knobs": spec.knobs,
        "budget": spec.budget,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def validate_spec(spec: ExperimentSpec, trace: list) -> list:
    """Range/structure validation (never an enum). Returns errors (empty == admissible)."""
    errors: list[str] = []
    if not spec.arms:
        errors.append("no arms")
    if not spec.metrics:
        errors.append("no metrics")
    if not spec.seeds:
        errors.append("no seeds")
    elif len(set(spec.seeds)) != len(spec.seeds):
        # duplicate seeds would median the SAME seed multiple times and report it as N independent
        # seeds (overstating support) — reject rather than silently dedup.
        errors.append(f"duplicate seeds {spec.seeds}")
    if spec.budget < 1:
        errors.append(f"budget {spec.budget} < 1")
    errors += [f"workload: {e}" for e in validate_trace(trace, spec.ranges)]
    for m in spec.metrics:
        if not MetricExpr.from_obj(m).is_valid():
            errors.append(f"bad metric expr: {m}")
    for k, v in spec.knobs.items():
        if k in spec.knob_ranges:
            lo, hi = spec.knob_ranges[k]
            if not lo <= v <= hi:
                errors.append(f"knob {k}={v} out of range ({lo}, {hi})")
    return errors


def run_experiment(spec: ExperimentSpec, eval_fn, *, base_out_dir: str,
                   experiment_id: str) -> ExperimentResult:
    """Run each arm×seed through the injected ``eval_fn`` -> ``FrontierCatalog``, charging the
    budget at the eval-call boundary. Returns per-arm cross-seed MEDIAN metric values.

    Honesty: a metric the catalog cannot compute stays ``None`` (and its missing-list is recorded);
    medians are over the seeds that produced a value, or ``None`` if none did. Never adjudicates.
    """
    out_dir = Path(base_out_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        trace = spec.workload.build()
    except ValueError as exc:
        return ExperimentResult(experiment_id=experiment_id, out_dir=str(out_dir), spec_sha="",
                                terminated="invalid_spec", errors=[f"workload: {exc}"])

    errors = validate_spec(spec, trace)
    spec_sha = _spec_sha(spec, trace)
    if errors:
        return ExperimentResult(
            experiment_id=experiment_id, out_dir=str(out_dir), spec_sha=spec_sha,
            terminated="invalid_spec", errors=errors)

    trace_path = write_trace_csv(trace, out_dir / "trace.csv")
    keys = [metric_key(m) for m in spec.metrics]
    exprs = [MetricExpr.from_obj(m) for m in spec.metrics]

    records: list[ArmSeedRecord] = []
    # per_arm_raw[arm][key] = [values across seeds that produced a value];
    # per_arm_missing[arm][key] = set of seeds with NO value (evaluate -> None, OR budget-skipped)
    per_arm_raw: dict = {a.name: {k: [] for k in keys} for a in spec.arms}
    per_arm_missing: dict = {a.name: {k: set() for k in keys} for a in spec.arms}
    evaluated: set = set()                          # (arm.name, seed) pairs actually run
    evals_used = 0
    terminated = "complete"

    for arm in spec.arms:
        for seed in spec.seeds:
            if evals_used >= spec.budget:                       # BUDGET at the eval-call boundary
                terminated = "budget_exhausted"
                break
            catalog: FrontierCatalog = eval_fn(arm, seed, spec, str(trace_path), str(out_dir))
            evals_used += 1
            evaluated.add((arm.name, seed))
            rec = ArmSeedRecord(arm=arm.name, seed=seed)
            for key, expr in zip(keys, exprs):
                cv = evaluate(expr, catalog)
                rec.metrics[key] = cv.value
                if cv.value is None:
                    rec.missing[key] = cv.missing
                    per_arm_missing[arm.name][key].add(seed)
                else:
                    per_arm_raw[arm.name][key].append(cv.value)
            records.append(rec)
        if terminated == "budget_exhausted":
            break

    # A planned arm/seed never evaluated (budget exhaustion skipped it) is ALSO missing for every
    # requested metric — not just the evaluate()->None case. Without this a budget-skipped seed
    # would leave an empty missing-list and the reducer would median the surviving seed (Codex R3).
    for arm in spec.arms:
        for seed in spec.seeds:
            if (arm.name, seed) not in evaluated:
                for key in keys:
                    per_arm_missing[arm.name][key].add(seed)

    # Reduce an arm/metric to a number ONLY when every spec seed produced a value (count == seeds,
    # no missing/skipped seed); else None, so adjudicate sees incomplete data as inconclusive.
    n_seeds = len(spec.seeds)
    per_arm = {
        arm: {k: (statistics.median(vs)
                  if (len(vs) == n_seeds and not per_arm_missing[arm][k]) else None)
              for k, vs in by_key.items()}
        for arm, by_key in per_arm_raw.items()
    }
    # persist only real gaps, as sorted seed lists
    per_arm_missing = {arm: {k: sorted(ss) for k, ss in by_key.items() if ss}
                       for arm, by_key in per_arm_missing.items()}
    result = ExperimentResult(
        experiment_id=experiment_id, out_dir=str(out_dir), spec_sha=spec_sha, metric_keys=keys,
        records=records, per_arm=per_arm, per_arm_missing=per_arm_missing,
        evals_used=evals_used, terminated=terminated)
    (out_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return result
