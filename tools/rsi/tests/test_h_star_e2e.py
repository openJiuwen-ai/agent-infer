"""AC8 — the H* acceptance loop on REAL CPU Frontier (express -> measure -> execute -> adjudicate ->
ledger -> research reuse).

This is the plan's acceptance anchor. It is **Frontier-gated**: it runs the REAL Frontier simulator
(no stub) when a Frontier checkout + interpreter are configured (VE_FRONTIER_REPO /
VE_FRONTIER_PYTHON with the ve_policy bridge applied), and SKIPS cleanly otherwise — it never
fabricates a number or a pass off the simulator. The offline chain wiring is already covered by
tests/test_experiment_chain.py (R6) with a deterministic eval; this file is the real run only.

Pass criterion (per the plan): LOOP CLOSURE, not hypothesis truth — a real deterministic verdict of
`supported`, `falsified`, OR `inconclusive` all pass, provided the full chain ran, is cited, and is
DoD-B honest (PROPOSAL-layer ledger; never the adoption gate).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_evolve.engine.experiment import Arm, ExperimentSpec
from vllm_evolve.engine.experiment_chain import frontier_catalog_eval, run_hypothesis
from vllm_evolve.engine.workload_synth import WorkloadSpec
from vllm_evolve.knowledge import research
from vllm_evolve.ops.inspect import verify_citations
from vllm_evolve.store.db import Store

REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "policy_group_admission.py"

# H*: "突发多租户负载下,prefix-cache 抖动是 TTFT 尾延迟主因;按租户分组接纳改善 p99 TTFT。"
H_STAR_STATEMENT = ("bursty multi-tenant load: prefix-cache thrash drives the TTFT tail; "
                    "grouping admission by tenant improves p99 TTFT")
# worst-tenant p99 TTFT: p99 of ttft grouped by tenant (request_session_id), reduced by the MAX
# group (the tail tenant). Prediction: B (grouped admission) < A (FCFS) by a margin.
H_STAR_METRIC = {"column": "ttft", "agg": "p99",
                 "group_by": "request_session_id", "group_reduce": "max"}
H_STAR_PREDICTION = {"metric": H_STAR_METRIC, "comparator": "<",
                     "arm_a": "B", "arm_b": "A", "margin": 1.0}


def _frontier_ready() -> bool:
    """True only when a real Frontier checkout + interpreter are configured AND the ve_policy
    bridge is applied (so the B arm runs the grouped-admission policy). Any failure -> skip."""
    from vllm_evolve.bench.frontier_sim import resolve_frontier_env
    try:
        repo, py = resolve_frontier_env()
    except Exception:
        return False
    repo_p, py_p = Path(repo), Path(py)
    if not (repo_p / "frontier").is_dir() or not py_p.exists():
        return False
    bridge = (repo_p / "frontier" / "scheduler" / "replica_scheduler"
              / "ve_policy_replica_scheduler.py")
    return bridge.is_file()


def _h_star_spec() -> ExperimentSpec:
    return ExperimentSpec(
        workload=WorkloadSpec(template="bursty_multi_tenant",
                              params={"n_tenants": 2, "bursts_per_tenant": 2, "burst_size": 4}),
        # A = FCFS baseline (vllm_v1); B = the hand-written grouped-admission policy fixture.
        arms=[Arm("A"), Arm("B", policy_path=str(_FIXTURE))],
        metrics=[H_STAR_METRIC], seeds=[0], budget=4)


@pytest.mark.skipif(not _frontier_ready(),
                    reason="real Frontier not configured (VE_FRONTIER_REPO/VE_FRONTIER_PYTHON + "
                           "ve_policy bridge); AC8 runs the real simulator, never a stub")
def test_h_star_closed_loop_on_real_cpu_frontier(tmp_path):
    db = str(tmp_path / "ledger.db")
    store = Store(db)
    res = run_hypothesis(
        store, hypothesis_id="h_star", statement=H_STAR_STATEMENT,
        prediction=H_STAR_PREDICTION, spec=_h_star_spec(), eval_fn=frontier_catalog_eval,
        base_out_dir=str(tmp_path / "exp"), run_id="h_star_e2e", source="frontier_sim")

    # --- LOOP CLOSURE: a real deterministic verdict, the full prediction->execution->adjudication
    #     chain in order, recorded honestly.
    assert res["verdict"] in {"supported", "falsified", "inconclusive"}
    assert res["prediction_event_id"] < res["execution_event_id"] < res["adjudication_event_id"]
    assert res["terminated"] == "complete"          # 2 arms x 1 seed <= budget 4

    # --- REAL MEASUREMENT (not a hollow loop): both arms must have produced the grouped p99 TTFT
    #     from the real Frontier catalog. This guards the empty-catalog class of bug (a wrong CLI
    #     flag once made Frontier exit non-zero -> empty catalog -> inconclusive-from-missing, which
    #     would otherwise sneak past as a "pass"). With real numbers the verdict can be any of the
    #     three, but the measurement leg must be genuine.
    mkey = "ttft|p99|request_session_id|max"
    assert res["per_arm"]["A"][mkey] is not None and res["per_arm"]["B"][mkey] is not None
    # per_arm_missing keeps an (empty) entry per arm; "no gaps" means every inner dict is empty.
    assert all(not miss for miss in res["per_arm_missing"].values())

    # --- MECHANISM PROOF: the B arm must have actually run the grouped-admission policy AND used
    #     the defer channel — not silently fallen back to FCFS (which would still yield real metrics
    #     and pass the checks above). The ve_policy honesty marker is written by the bridge at exit.
    marker_path = (tmp_path / "exp" / "h_star" / "arms" / "B_0" / "ve_policy_marker.json")
    assert marker_path.is_file(), f"B-arm ve_policy marker missing: {marker_path}"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["invocations"] > 0, marker        # the policy actually drove scheduling
    assert marker["fallbacks"] == 0, marker          # never fell back to stock ordering
    assert marker["defers"] > 0, marker              # the explicit defer channel genuinely fired

    view = store.get_hypothesis("h_star")
    assert view["verdict"] == res["verdict"]
    assert view["adjudicated_execution_event_id"] == res["execution_event_id"]

    # --- CITATION verifies (the result is real, not fabricated)
    assert verify_citations(store, res["citations"])["all_real"] is True
    store.close()

    # --- RESEARCH REUSE: the NEXT round's research can find and cite this hypothesis as
    #     hypothesis:<event_id>, and that citation verifies.
    gathered = research.gather("scheduling", db=db)
    assert any(h["hypothesis_id"] == "h_star" for h in gathered["hypotheses"])
    assert f"hypothesis:{res['adjudication_event_id']}" in gathered["citations"]
    store2 = Store(db)
    try:
        assert verify_citations(store2, gathered["citations"])["all_real"] is True
    finally:
        store2.close()
