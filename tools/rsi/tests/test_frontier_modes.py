"""Phase B: each agent mode (autopt / tune / port) + the intent router works end-to-end under
``--backend frontier_sim`` — off-box, no GPU, the Frontier subprocess stubbed by a fixture. Every
mode can only ever conclude DoD-B (frontier_sim is non-real even when ve_policy executes). Zero GPU.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench import frontier_sim  # noqa: E402
from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.tools import phase_guard  # noqa: E402

SEED = str(REPO_ROOT / "targets" / "scheduling" / "seed.py")

_SYSTEM_METRICS = """{
  "simulation_metadata": {"total_requests": 4, "completed_requests": 4},
  "throughput_metrics": {"requests_per_second": 3.5, "tokens_per_second": 1280.0},
  "memory_utilization_percent": {"MONOLITHIC": 15.0},
  "preemption_statistics": {"total_preemption_events": 0, "total_preempted_requests": 0}
}"""
_REQUEST_METRICS_CSV = (
    "Request Id,ttft,request_e2e_time,tpot,"
    "request_inter_arrival_delay,request_first_scheduling_delay\n"
    "0,30.0,120.0,0.9,,0.0\n1,31.0,121.0,0.9,0.5,0.0\n"
    "2,29.0,119.0,0.9,0.5,0.0\n3,30.0,120.0,0.9,0.5,0.0\n"
)
# idle engine (duty 0.3) + empty queue -> a REAL under_saturated signature for the loop to act on
_BATCH_LEDGER = (
    '{"batch_id": 0, "stage_start_ts": 0.0, "stage_end_ts": 2.0, "request_ids": [0, 1]}\n'
    '{"batch_id": 1, "stage_start_ts": 9.0, "stage_end_ts": 10.0, "request_ids": [2]}\n'
)


_MARKER = ('{"scheduler": "ve_policy", "policy_sha256": "stub-sha",'
           ' "invocations": 42, "fallbacks": 0}')


def _stub_invoke(argv, repo, out_dir, extra_env=None):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "system_metrics.json").write_text(_SYSTEM_METRICS, encoding="utf-8")
    (d / "request_metrics.csv").write_text(_REQUEST_METRICS_CSV, encoding="utf-8")
    (d / "frontier_stage_batch_ledger.jsonl").write_text(_BATCH_LEDGER, encoding="utf-8")
    (d / "ve_policy_marker.json").write_text(_MARKER, encoding="utf-8")
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture(autouse=True)
def _reset_phase():
    phase_guard.reset_state()
    yield
    phase_guard.reset_state()


@pytest.fixture(autouse=True)
def _frontier_backend(monkeypatch):
    monkeypatch.setenv("VE_FRONTIER_REPO", "/fake/frontier")
    monkeypatch.setenv("VE_FRONTIER_PYTHON", "/fake/python")
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke)


# ── C3: different load -> different REAL signature -> different optimization target ──

# saturated engine + deep queue + KV pressure + preemptions (all real-shaped Frontier outputs)
_SYSTEM_METRICS_HIGH = """{
  "simulation_metadata": {"total_requests": 6, "completed_requests": 6},
  "throughput_metrics": {"requests_per_second": 0.4, "tokens_per_second": 160.0},
  "memory_utilization_percent": {"MONOLITHIC": 92.0},
  "preemption_statistics": {"total_preemption_events": 3, "total_preempted_requests": 2}
}"""
_REQUEST_METRICS_CSV_HIGH = (
    "Request Id,ttft,request_e2e_time,tpot,"
    "request_inter_arrival_delay,request_first_scheduling_delay\n"
    + "".join(f"{i},900.0,15000.0,40.0,{'' if i == 0 else '0.01'},500.0\n" for i in range(6))
)
_BATCH_LEDGER_HIGH = (
    '{"batch_id": 0, "stage_start_ts": 0.0, "stage_end_ts": 9.7, "request_ids": [0,1,2,3]}\n'
    '{"batch_id": 1, "stage_start_ts": 9.7, "stage_end_ts": 20.0, "request_ids": [4,5]}\n'
)


def _stub_invoke_high(argv, repo, out_dir, extra_env=None):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "system_metrics.json").write_text(_SYSTEM_METRICS_HIGH, encoding="utf-8")
    (d / "request_metrics.csv").write_text(_REQUEST_METRICS_CSV_HIGH, encoding="utf-8")
    (d / "frontier_stage_batch_ledger.jsonl").write_text(_BATCH_LEDGER_HIGH, encoding="utf-8")
    (d / "ve_policy_marker.json").write_text(_MARKER, encoding="utf-8")
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_c3_load_flips_diagnosis_and_targets(monkeypatch):
    """The localization chain works in-sim: low load -> under_saturated -> pressure knobs;
    high load -> kv_capacity -> KV knobs. Same code path, only the (real-shaped) signals differ."""
    from vllm_evolve.engine.diagnose import diagnose
    from vllm_evolve.engine.profile import collect_profile_frontier
    from vllm_evolve.engine.targets import select_targets

    cfg = {"target": "scheduling", "runner_kind": "strong_baseline", "n_requests": 4,
           "max_seeds": 1, "model": "facebook/opt-125m"}

    # low load (default autouse stub): idle engine (duty 0.3), empty queue, low memory
    prof_low = collect_profile_frontier(cfg)
    diag_low = diagnose(prof_low)
    assert diag_low.bottleneck == "under_saturated"
    targets_low = [t["target"] for t in select_targets(diag_low).candidates]
    assert "config:max_num_seqs" in targets_low

    # high load: saturated engine, waiting depth 6, kv 0.92, preemptions
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke_high)
    prof_high = collect_profile_frontier(cfg)
    assert prof_high.vllm["waiting"] > 4 and prof_high.vllm["kv_util"] >= 0.9
    diag_high = diagnose(prof_high)
    assert diag_high.bottleneck == "kv_capacity"
    targets_high = [t["target"] for t in select_targets(diag_high).candidates]
    assert "config:gpu_memory_utilization" in targets_high
    assert targets_high != targets_low          # the flip reaches the optimization points


# ── D3: sim evolution — variants ranked by sim score, search-only, never adopted ──

def test_d3_sim_evolve_ranks_variants_and_never_adopts(tmp_path):
    from vllm_evolve.core.schemas import Profile, Spec
    from vllm_evolve.engine.evolve_target import sim_evolve_fn

    # fake profile_fn: score depends on which variant file is being evaluated (sim behavior stub);
    # marker_verified is False exactly like a real frontier_sim profile. The author now emits a
    # STRUCTURAL catalogue (sort families + cache/KV/anti-starvation/load variants); score each
    # distinctly with sjf the winner.
    scores_by_stem = {"variant_fcfs": 12.0, "variant_sjf": 30.0, "variant_ljf": 10.0,
                      "variant_lifo": 20.0, "variant_cache_first": 25.0, "variant_kv_reserve": 22.0,
                      "variant_prefill_cap": 18.0, "variant_starvation": 15.0,
                      "variant_load_adaptive": 28.0}

    def fake_profile_fn(config):
        score = scores_by_stem.get(Path(config["policy"]).stem, 5.0)
        return Profile(config=config, metrics={"tok_s": score}, vllm={}, gpu={},
                       marker_verified=False, source="frontier_sim",
                       outcome_class="simulator_nonqualifying")

    evolve = sim_evolve_fn(fake_profile_fn, variants_dir=str(tmp_path / "variants"),
                           winner_dir=str(tmp_path / "winner"))
    cand = evolve({"model": "m"}, Spec(metric="tok_s", direction="max"))

    scores = [t["sim_score"] for t in cand.trials]
    assert len(set(scores)) >= 5                      # the structural catalogue is differentiated
    assert cand.score == 30.0 and "sjf" in cand.value["policy"]   # correct ranking (max)
    assert cand.marker_verified is False              # search-only: can never adopt
    assert (tmp_path / "winner" / "work_variant.py").is_file()
    assert (tmp_path / "winner" / "REQUIRES_REAL_VLLM_VERIFICATION").is_file()
    ev = json.loads((tmp_path / "winner" / "evidence.json").read_text(encoding="utf-8"))
    assert ev["variant"] == "sjf" and ev["source"] == "frontier_sim"

    # the orchestrator's adoption guard: an unverified candidate is recorded, never adopted
    from vllm_evolve.engine.orchestrate import run_autopt
    res = run_autopt(Spec(metric="tok_s", direction="max"),
                     eval_fn=fake_profile_fn_for_loop(tmp_path), evolve_fn=evolve,
                     base_config={"policy": "x"}, max_rounds=1)
    assert res["adopted"] is None


def fake_profile_fn_for_loop(tmp_path):
    # base-profile eval for the loop: a scheduling_queue signature so code:schedule_batch ranks in
    from vllm_evolve.core.schemas import Profile

    def fn(config):
        return Profile(config=config, metrics={"tok_s": 15.0},
                       vllm={"kv_util": 0.2, "waiting": 8, "preempt": 0},
                       gpu={"duty_cycle": 0.7}, marker_verified=False,
                       source="frontier_sim", outcome_class="simulator_nonqualifying")
    return fn


# ── B1: autopt (goal-driven auto-research) ────────────────────────────────────

def test_b1_autopt_frontier_concludes_dod_b():
    # the full L1->L4 loop runs off-box on frontier_sim and can ONLY conclude DoD-B (never adopted).
    from vllm_evolve.core.schemas import Spec
    from vllm_evolve.engine.orchestrate import run_autopt
    from vllm_evolve.engine.profile import collect_profile_frontier
    spec = Spec(metric="output_throughput_tok_s", direction="max")
    res = run_autopt(spec, eval_fn=collect_profile_frontier,
                     base_config={"n_requests": 100, "model": "facebook/opt-125m"}, max_rounds=3)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    searched = [r.get("searched_target") for r in res["rounds"] if r.get("searched_target")]
    assert searched, "the loop must actually exercise optimize (search a target)"


def test_b1_autopt_cli_frontier_concludes_dod_b(capsys):
    # the same through the REAL CLI: parser + cmd_autopt seam select --backend frontier_sim.
    rc = cli_main.main(["autopt", "maximize throughput", "--n-requests", "100",
                        "--max-rounds", "2", "--model", "facebook/opt-125m",
                        "--backend", "frontier_sim"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert out["backend"] == "frontier_sim" and out["outcome"] == "dod_b"
    assert out["adopted"] is None


# ── B2: tune (optimize a GIVEN policy's serving config) ───────────────────────

def test_b2_tune_frontier_runs_and_cannot_gain(tmp_path, capsys):
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(running, state):\n    return None\n", encoding="utf-8")
    rc = cli_main.main(["tune", str(pol), "scheduling", "--backend", "frontier_sim",
                        "--max-rounds", "1", "--max-evals", "1", "--no-holdout"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert out["mode"] == "tune" and out["backend"] == "frontier_sim"
    assert not out.get("adopted")               # frontier_sim is non-real -> never adoption
    assert "kept" not in out and "committed" not in out


# ── B3: port (version x hardware migration/adaptation matrix) ──────────────────

def test_b3_port_frontier_matrix_plumbing(tmp_path, capsys):
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(running, state):\n    return None\n", encoding="utf-8")
    report = tmp_path / "port_report.json"
    rc = cli_main.main(["port", str(pol), "scheduling", "--versions", "0.21.0,0.22.0",
                        "--hardware", "h100", "--backend", "frontier_sim", "--out", str(report)])
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and res["ok"] is True and res["mode"] == "port"
    assert len(res["matrix"]) == 2                       # 2 versions x 1 hardware
    for cell in res["matrix"]:
        assert cell["source"] == "frontier_sim" and cell["source"] != "real_vllm"
        assert cell["hardware_id"] == "h100"
    assert res["summary"]["real_cells"] == 0             # no cell can ever be a real gain
    assert report.is_file()


# ── B4: intent router (free-text requirement -> correct agent mode) ────────────

@pytest.mark.parametrize("text, expected_mode", [
    ("maximize throughput", "autopt"),                       # goal-driven -> autopt
    ("调优这个 policy 的 max_num_seqs", "tune"),              # tuning verb + given policy -> tune
    ("tune this given policy for latency", "tune"),
    ("把这个调度算法适配到新版本 vLLM", "port"),               # 适配 + 版本 -> port
    ("port the scheduler across hardware h100", "port"),     # port / across / hardware -> port
])
def test_b4_intent_router_picks_correct_mode(text, expected_mode):
    from vllm_evolve.intent.router import route
    r = route(text)
    assert r["mode"] == expected_mode, (text, r["mode"])
    # the default target stays the one supported target; the spec is always parsed
    assert r["target"] == "scheduling" and r["target_supported"] is True
    assert r["spec"]["metric"] and r["spec"]["direction"] in ("max", "min")
