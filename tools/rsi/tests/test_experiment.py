"""R1/AC4: ExperimentSpec + run_experiment. Budget charged at the eval-call boundary; per-arm
cross-seed median via the shared evaluator; range (not enum) validation; missing metric stays None.
Pure/offline — eval_fn is injected and returns constructed catalogs (no Frontier, no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.frontier_catalog import FrontierCatalog  # noqa: E402
from vllm_evolve.engine.experiment import (  # noqa: E402
    Arm,
    ExperimentSpec,
    metric_key,
    run_experiment,
)
from vllm_evolve.engine.workload_synth import WorkloadSpec  # noqa: E402

_TTFT = {"column": "ttft", "agg": "p50"}


def _catalog_with_ttft(values: list) -> FrontierCatalog:
    rows = [{"ttft": str(v)} for v in values]
    return FrontierCatalog(out_dir="x", run_id="r", dir=Path("x"), system={},
                           rows=rows, columns=["ttft"])


def _arm_eval(per_arm_value: dict):
    """eval_fn whose ttft depends on the arm (so per-arm medians differ); records calls."""
    calls = []

    def _fn(arm, seed, spec, trace_path, out_dir):
        calls.append((arm.name, seed))
        return _catalog_with_ttft([per_arm_value[arm.name]])
    return _fn, calls


def _spec(**kw) -> ExperimentSpec:
    base = dict(workload=WorkloadSpec(template="steady", params={"n_requests": 4}),
                arms=[Arm("A"), Arm("B")], metrics=[_TTFT], seeds=[0, 1])
    base.update(kw)
    return ExperimentSpec(**base)


def test_per_arm_cross_seed_median_via_shared_evaluator(tmp_path):
    fn, calls = _arm_eval({"A": 100.0, "B": 50.0})
    res = run_experiment(_spec(), fn, base_out_dir=str(tmp_path), experiment_id="e1")
    assert res.terminated == "complete" and res.evals_used == 4        # 2 arms x 2 seeds
    k = metric_key(_TTFT)
    assert res.per_arm["A"][k] == 100.0 and res.per_arm["B"][k] == 50.0
    assert len(calls) == 4
    assert (tmp_path / "e1" / "trace.csv").exists()                    # trace materialized once
    assert (tmp_path / "e1" / "result.json").exists()                  # experiment_ref artifact


def test_budget_charged_at_eval_boundary(tmp_path):
    # 2 arms x 2 seeds = 4 planned, but budget=2 -> stop after 2 evals, never silently continue
    fn, calls = _arm_eval({"A": 100.0, "B": 50.0})
    res = run_experiment(_spec(budget=2), fn, base_out_dir=str(tmp_path), experiment_id="e2")
    assert res.terminated == "budget_exhausted" and res.evals_used == 2
    assert len(calls) == 2                                  # eval_fn called exactly budget times


def test_invalid_spec_is_rejected_by_range_not_crash(tmp_path):
    # an out-of-range workload (token count beyond TraceRanges) -> invalid_spec with explicit errors
    fn, _ = _arm_eval({"A": 1.0, "B": 1.0})
    bad = _spec(workload=WorkloadSpec(template="steady",
                                      params={"n_requests": 2, "prefill_tokens": 10_000_000}))
    res = run_experiment(bad, fn, base_out_dir=str(tmp_path), experiment_id="e3")
    assert res.terminated == "invalid_spec" and any("workload" in e for e in res.errors)


def test_duplicate_seeds_are_rejected_not_medianed_twice(tmp_path):
    # seeds=[0,0] would run seed 0 twice and report a "2-seed" median of one seed -> overstates
    # support. Reject as invalid_spec instead of silently deduping.
    fn, calls = _arm_eval({"A": 100.0, "B": 50.0})
    res = run_experiment(_spec(seeds=[0, 0]), fn, base_out_dir=str(tmp_path), experiment_id="ed")
    assert res.terminated == "invalid_spec" and any("duplicate seeds" in e for e in res.errors)
    assert calls == []                                      # never ran a degenerate experiment
    assert res.evals_used == 0                              # no eval_fn calls on invalid spec


def test_missing_metric_stays_none_never_invented(tmp_path):
    # catalog lacks the requested column -> per-arm value None + missing recorded (no fabrication)
    def fn(arm, seed, spec, trace_path, out_dir):
        return FrontierCatalog(out_dir="x", run_id="r", dir=Path("x"), system={},
                               rows=[{"other": "1"}], columns=["other"])
    res = run_experiment(_spec(seeds=[0]), fn, base_out_dir=str(tmp_path), experiment_id="e4")
    k = metric_key(_TTFT)
    assert res.per_arm["A"][k] is None and res.per_arm["B"][k] is None
    assert any("ttft" in rec.missing.get(k, []) for rec in res.records)


def test_partial_missing_seed_makes_arm_metric_none_not_a_partial_median(tmp_path):
    # A: seed 0 has ttft, seed 1 is missing it; B: both seeds present. The plan requires that any
    # missing required seed makes the per-arm value None (never a median over the surviving seed),
    # so downstream adjudication sees inconclusive instead of comparing incomplete data.
    def fn(arm, seed, spec, trace_path, out_dir):
        if arm.name == "A" and seed == 1:
            return FrontierCatalog(out_dir="x", run_id="r", dir=Path("x"), system={},
                                   rows=[{"other": "1"}], columns=["other"])   # ttft absent
        return _catalog_with_ttft([100.0 if arm.name == "A" else 50.0])
    res = run_experiment(_spec(), fn, base_out_dir=str(tmp_path), experiment_id="em")
    k = metric_key(_TTFT)
    assert res.per_arm["A"][k] is None                       # NOT 100.0 (no partial-seed median)
    assert res.per_arm["B"][k] == 50.0                       # B complete -> real median
    assert res.per_arm_missing["A"][k] == [1]                # the missing seed is preserved
    # the persisted result.json keeps the missing-seed evidence
    import json as _json
    saved = _json.loads((tmp_path / "em" / "result.json").read_text(encoding="utf-8"))
    assert saved["per_arm_missing"]["A"][k] == [1]


def test_budget_skipped_seed_is_missing_not_a_partial_median(tmp_path):
    # Codex R3 blocker: budget=1 with seeds [0,1] runs only A/seed0. A/seed1 (and all of B) are
    # PLANNED but never evaluated. The skipped seed must count as missing, so per_arm[A] is None
    # (not the seed0 value) and adjudication is inconclusive — never compares partial data.
    from vllm_evolve.engine.adjudicate import INCONCLUSIVE, adjudicate
    fn, calls = _arm_eval({"A": 100.0, "B": 50.0})
    res = run_experiment(_spec(budget=1), fn, base_out_dir=str(tmp_path), experiment_id="eb")
    k = metric_key(_TTFT)
    assert len(calls) == 1 and res.terminated == "budget_exhausted"
    assert res.per_arm["A"][k] is None and res.per_arm["B"][k] is None   # no partial-seed median
    assert 1 in res.per_arm_missing["A"][k]                              # A/seed1 recorded skipped
    import json as _json
    saved = _json.loads((tmp_path / "eb" / "result.json").read_text(encoding="utf-8"))
    assert 1 in saved["per_arm_missing"]["A"][k]                         # persisted evidence
    pred = {"metric": _TTFT, "comparator": "<", "arm_a": "B", "arm_b": "A", "margin": 1}
    assert adjudicate(pred, res.per_arm).verdict == INCONCLUSIVE


def test_run_experiment_is_deterministic(tmp_path):
    fn1, _ = _arm_eval({"A": 7.0, "B": 3.0})
    fn2, _ = _arm_eval({"A": 7.0, "B": 3.0})
    r1 = run_experiment(_spec(), fn1, base_out_dir=str(tmp_path / "a"), experiment_id="e")
    r2 = run_experiment(_spec(), fn2, base_out_dir=str(tmp_path / "b"), experiment_id="e")
    assert r1.per_arm == r2.per_arm and r1.spec_sha == r2.spec_sha


def test_knob_range_guardrail(tmp_path):
    fn, _ = _arm_eval({"A": 1.0, "B": 1.0})
    res = run_experiment(_spec(knobs={"qps": 999}, knob_ranges={"qps": (1, 50)}),
                         fn, base_out_dir=str(tmp_path), experiment_id="e5")
    assert res.terminated == "invalid_spec" and any("qps" in e for e in res.errors)
