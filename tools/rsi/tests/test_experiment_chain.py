"""R6/AC5: the research chain (register -> run -> adjudicate -> ledger) and the `ve experiment`
admin CLI. eval_fn is stubbed (no Frontier), so the full chain runs offline. Pure/offline."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.frontier_catalog import FrontierCatalog  # noqa: E402
from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.engine import experiment_chain  # noqa: E402
from vllm_evolve.engine.experiment import Arm, ExperimentSpec  # noqa: E402
from vllm_evolve.engine.workload_synth import WorkloadSpec  # noqa: E402
from vllm_evolve.ops.inspect import verify_citations  # noqa: E402
from vllm_evolve.store.db import Store  # noqa: E402

_METRIC = {"column": "ttft", "agg": "p99", "group_by": "request_session_id"}
_PRED = {"metric": _METRIC, "comparator": "<", "arm_a": "B", "arm_b": "A", "margin": 5}


def _arm_catalog(ttft_by_session: dict) -> FrontierCatalog:
    rows = [{"ttft": str(v), "request_session_id": str(s)}
            for s, vs in ttft_by_session.items() for v in vs]
    return FrontierCatalog(out_dir="x", run_id="r", dir=Path("x"), system={},
                           rows=rows, columns=["ttft", "request_session_id"])


def _stub_eval(arm, seed, spec, trace_path, out_dir):
    # A (FCFS) has a worse per-tenant tail than B (group admission) -> prediction B<A supported
    if arm.name == "A":
        return _arm_catalog({0: [240, 260], 1: [250, 270]})
    return _arm_catalog({0: [190, 200], 1: [195, 205]})


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        workload=WorkloadSpec(template="bursty_multi_tenant",
                              params={"n_tenants": 2, "bursts_per_tenant": 1, "burst_size": 2}),
        arms=[Arm("A"), Arm("B")], metrics=[_METRIC], seeds=[0], budget=4)


def test_frontier_catalog_eval_uses_the_real_trace_replay_flag(tmp_path, monkeypatch):
    # The production eval must hand the trace to Frontier via its REAL flag
    # (--trace_request_generator_config_trace_file: the TRACE_REPLAY generator's config class is
    # TraceRequestGeneratorConfig). The wrong prefix (trace_replay_*) made Frontier exit non-zero ->
    # empty catalog -> a hollow inconclusive. Capture the argv offline (no real Frontier needed).
    from vllm_evolve.bench import frontier_sim

    captured = {}

    def fake_invoke(argv, repo, out_dir, extra_env=None):
        captured["argv"] = list(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frontier_sim, "resolve_frontier_env", lambda: (str(tmp_path), "python"))
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", fake_invoke)

    arm = SimpleNamespace(name="A", policy_path=None)
    spec = SimpleNamespace(knobs={})
    experiment_chain.frontier_catalog_eval(arm, 0, spec, str(tmp_path / "t.csv"), str(tmp_path))

    argv = captured["argv"]
    assert "--trace_request_generator_config_trace_file" in argv
    assert "--trace_replay_request_generator_config_trace_file" not in argv
    i = argv.index("--request_generator_config_type")
    assert argv[i + 1] == "trace_replay"


def test_frontier_catalog_eval_resolves_policy_root_from_policy_path(tmp_path, monkeypatch):
    # VE_POLICY_ROOT must be the repo root (the ancestor holding integrations/frontier/
    # ve_policy_api.py), resolved from the POLICY PATH — NEVER the caller cwd. A wrong cwd is the
    # fallback where the B fixture can't import ve_policy_api and the bridge degrades to FCFS. Run
    # from a non-repo cwd to prove the resolution is cwd-independent.
    from vllm_evolve.bench import frontier_sim

    captured = {}

    def fake_invoke(argv, repo, out_dir, extra_env=None):
        captured["extra_env"] = extra_env
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frontier_sim, "resolve_frontier_env", lambda: (str(tmp_path), "python"))
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", fake_invoke)
    monkeypatch.chdir(tmp_path)                       # a NON-repo cwd

    fixture = REPO_ROOT / "tests" / "fixtures" / "policy_group_admission.py"
    arm = SimpleNamespace(name="B", policy_path=str(fixture))
    spec = SimpleNamespace(knobs={})
    experiment_chain.frontier_catalog_eval(arm, 0, spec, str(tmp_path / "t.csv"), str(tmp_path))

    root = Path(captured["extra_env"]["VE_POLICY_ROOT"]).resolve()
    assert (root / "integrations" / "frontier" / "ve_policy_api.py").is_file()
    assert root == REPO_ROOT.resolve()


def test_policy_root_is_cwd_independent_for_an_out_of_tree_policy(tmp_path, monkeypatch):
    # An out-of-tree (copied/generated) policy whose ancestors do NOT contain ve_policy_api.py must
    # STILL resolve VE_POLICY_ROOT to the vllm-evolve checkout (derived from the module) — NEVER the
    # caller cwd. Copy the fixture outside the repo + chdir to a different non-repo dir to prove it.
    import shutil

    from vllm_evolve.bench import frontier_sim

    captured = {}

    def fake_invoke(argv, repo, out_dir, extra_env=None):
        captured["extra_env"] = extra_env
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frontier_sim, "resolve_frontier_env", lambda: (str(tmp_path), "python"))
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", fake_invoke)

    outside = tmp_path / "outside"
    outside.mkdir()
    copied = outside / "policy_group_admission.py"
    shutil.copy(REPO_ROOT / "tests" / "fixtures" / "policy_group_admission.py", copied)
    cwd_dir = tmp_path / "elsewhere"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)                        # a non-repo cwd, distinct from the policy

    arm = SimpleNamespace(name="B", policy_path=str(copied))
    spec = SimpleNamespace(knobs={})
    experiment_chain.frontier_catalog_eval(arm, 0, spec, str(tmp_path / "t.csv"), str(tmp_path))

    root = Path(captured["extra_env"]["VE_POLICY_ROOT"]).resolve()
    assert root == REPO_ROOT.resolve()               # the checkout, NOT cwd_dir nor the policy dir
    assert (root / "integrations" / "frontier" / "ve_policy_api.py").is_file()


def test_policy_root_raises_when_api_is_unreachable(tmp_path, monkeypatch):
    # If neither the policy ancestors nor the module-derived checkout hold the API, fail LOUDLY
    # rather than launch Frontier with a wrong import root. Force the checkout lookup to miss.
    from vllm_evolve.engine import experiment_chain as ec

    monkeypatch.setattr(ec, "_API_REL", ("does_not_exist", "ve_policy_api.py"))
    copied = tmp_path / "p.py"
    copied.write_text("def schedule_batch(*a, **k):\n    return None\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="VE_POLICY_ROOT"):
        ec._policy_root_for(str(copied))


def test_run_hypothesis_full_chain(tmp_path):
    s = Store(tmp_path / "h.db")
    res = experiment_chain.run_hypothesis(
        s, hypothesis_id="h1", statement="group admission improves p99 ttft", prediction=_PRED,
        spec=_spec(), eval_fn=_stub_eval, base_out_dir=str(tmp_path / "exp"))
    assert res["verdict"] == "supported"                   # B's per-tenant tail < A's by > margin
    # the ledger holds the full chain, in order, and the cited event is real
    view = s.get_hypothesis("h1")
    assert view["prediction_event_id"] < res["execution_event_id"] < res["adjudication_event_id"]
    assert view["verdict"] == "supported" and view["adjudicated_execution_event_id"] \
        == res["execution_event_id"]
    assert verify_citations(s, res["citations"])["all_real"] is True
    s.close()


def test_run_hypothesis_records_inconclusive_on_missing(tmp_path):
    # an eval that never emits the metric column -> per_arm None -> inconclusive, recorded honestly
    def blind(arm, seed, spec, trace_path, out_dir):
        return FrontierCatalog(out_dir="x", run_id="r", dir=Path("x"), system={},
                               rows=[{"other": "1"}], columns=["other"])
    s = Store(tmp_path / "h.db")
    res = experiment_chain.run_hypothesis(
        s, hypothesis_id="h2", statement="H", prediction=_PRED, spec=_spec(),
        eval_fn=blind, base_out_dir=str(tmp_path / "exp"))
    assert res["verdict"] == "inconclusive"
    assert s.get_hypothesis("h2")["verdict"] == "inconclusive"
    s.close()


def test_run_hypothesis_reuses_a_pre_registered_prediction(tmp_path):
    # the documented two-step flow: `ve experiment register <spec>` then `ve experiment run <spec>`.
    # run_hypothesis must REUSE the existing prediction (the store enforces one per hypothesis_id),
    # not re-register and raise.
    s = Store(tmp_path / "h.db")
    pred_id = s.register_prediction(hypothesis_id="h1", run_id="", statement="S",
                                    prediction=_PRED, source="frontier_sim")
    res = experiment_chain.run_hypothesis(
        s, hypothesis_id="h1", statement="S", prediction=_PRED, spec=_spec(),
        eval_fn=_stub_eval, base_out_dir=str(tmp_path / "exp"))
    assert res["prediction_event_id"] == pred_id        # reused, NOT a second registration
    assert res["verdict"] == "supported"                # still runs + adjudicates the existing pred
    assert s.get_hypothesis("h1")["prediction_event_id"] == pred_id
    s.close()


def test_run_hypothesis_rejects_a_conflicting_prediction(tmp_path):
    # reusing an id with a DIFFERENT prediction is a real conflict (would adjudicate the wrong
    # prediction) -> fail clearly rather than silently.
    s = Store(tmp_path / "h.db")
    s.register_prediction(hypothesis_id="h1", run_id="", statement="S",
                          prediction=_PRED, source="frontier_sim")
    conflicting = {**_PRED, "margin": 999}
    with pytest.raises(ValueError, match="already registered with a different prediction"):
        experiment_chain.run_hypothesis(
            s, hypothesis_id="h1", statement="S", prediction=conflicting, spec=_spec(),
            eval_fn=_stub_eval, base_out_dir=str(tmp_path / "exp"))
    s.close()


def test_prefix_cache_flag_uses_the_active_scheduler_namespace(tmp_path, monkeypatch):
    # The bridge is vllm_v1-compatible and Frontier's prefix-cache allow-list includes VE_POLICY.
    # The flag must target the active scheduler; the unused vllm_v1 slot silently disabled caching
    # on candidate arms and made prefix-stress comparisons different-caliber.
    from vllm_evolve.bench import frontier_sim

    captured = []

    def fake_invoke(argv, repo, out_dir, extra_env=None):
        captured.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frontier_sim, "resolve_frontier_env", lambda: (str(tmp_path), "python"))
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", fake_invoke)

    fixture = REPO_ROOT / "tests" / "fixtures" / "policy_group_admission.py"
    spec = SimpleNamespace(knobs={})
    experiment_chain.frontier_catalog_eval(                       # a ve_policy (policy) arm
        SimpleNamespace(name="B", policy_path=str(fixture), knobs={}), 0, spec,
        str(tmp_path / "t.csv"), str(tmp_path))
    b_argv = captured[0]
    assert "--ve_policy_scheduler_config_enable_prefix_caching" in b_argv
    assert "--vllm_v1_scheduler_config_enable_prefix_caching" not in b_argv


def test_per_arm_knobs_reach_the_frontier_command(tmp_path, monkeypatch):
    # an A/B that varies arms[*].knobs must produce DIFFERENT Frontier configs, not identical ones
    # it then adjudicates as a meaningless delta.
    from vllm_evolve.bench import frontier_sim

    captured = []

    def fake_invoke(argv, repo, out_dir, extra_env=None):
        captured.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frontier_sim, "resolve_frontier_env", lambda: (str(tmp_path), "python"))
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", fake_invoke)

    spec = SimpleNamespace(knobs={})
    a = SimpleNamespace(name="A", policy_path=None, knobs={"max_num_seqs": 4})
    b = SimpleNamespace(name="B", policy_path=None, knobs={"max_num_seqs": 16})
    experiment_chain.frontier_catalog_eval(a, 0, spec, str(tmp_path / "t.csv"), str(tmp_path))
    experiment_chain.frontier_catalog_eval(b, 0, spec, str(tmp_path / "t.csv"), str(tmp_path))

    flag = "--vllm_v1_scheduler_config_batch_size_cap"
    assert captured[0][captured[0].index(flag) + 1] == "4"
    assert captured[1][captured[1].index(flag) + 1] == "16"


def test_ve_experiment_run_register_list_cli(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(experiment_chain, "frontier_catalog_eval", _stub_eval)   # no real Frontier
    db = str(tmp_path / "cli.db")
    spec_doc = {
        "hypothesis_id": "hcli", "statement": "group admission improves p99 ttft",
        "prediction": _PRED,
        "workload": {"template": "bursty_multi_tenant",
                     "params": {"n_tenants": 2, "bursts_per_tenant": 1, "burst_size": 2}},
        "arms": [{"name": "A"}, {"name": "B"}], "metrics": [_METRIC], "seeds": [0], "budget": 4,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_doc), encoding="utf-8")

    rc = cli_main.main(["experiment", "run", str(spec_path), "--db", db,
                        "--out", str(tmp_path / "art")])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True and out["verdict"] == "supported"
    assert out["citations"] == [f"hypothesis:{out['adjudication_event_id']}"]

    # list surfaces it (verdict filter works)
    rc = cli_main.main(["experiment", "list", "--db", db, "--verdict", "supported"])
    listed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and any(h["hypothesis_id"] == "hcli" for h in listed["hypotheses"])
