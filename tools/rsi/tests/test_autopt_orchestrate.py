"""M5: top orchestration — L1->L4 loop, adoption vs honest DoD-B (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.config import build_bench_config  # noqa: E402
from vllm_evolve.core.schemas import Candidate, Profile, Spec  # noqa: E402
from vllm_evolve.engine.orchestrate import run_autopt  # noqa: E402

_SPEC = Spec(metric="goodput_req_s", direction="max")
# same-caliber bench_config provenance carried by every profile (so the autopt-loop
# same_caliber check passes; A/B differs only by the thing under test).
_BC = build_bench_config(runner_kind="candidate", model="facebook/opt-125m").to_dict()


def _quality(config):
    # injected measured-quality runner: a frozen-set measure_fn that certifies quality
    def measure(_prompts):
        return {"perplexity": 10.0, "task_em": 0.80, "output_agreement": 0.999}
    return measure

# diagnosis signal presets
_UNDERSAT = {"gpu": {"sm_util_max": 35, "duty_cycle": 0.1},
             "vllm": {"kv_util": 0.2, "waiting": 0}}
_COMPUTE = {"gpu": {"sm_util_max": 95, "duty_cycle": 0.95},
            "vllm": {"kv_util": 0.5, "waiting": 1}}
# scheduling_queue: free SM capacity + standing queue + KV not full + no preempt
_SCHEDQ = {"gpu": {"sm_util_max": 70, "duty_cycle": 0.7},
           "vllm": {"kv_util": 0.4, "waiting": 20, "preempt": 0}}


def _prof(signals, vals, verified=True, eval_result=None):
    # these stand in for REAL bench profiles, so they carry a real-run source/outcome_class
    # (the LOCK D accept guard rejects non-real metadata). A code candidate's effectiveness is read
    # from eval_result["effective"] (native_bench records it), so code-adoption tests pass it here.
    return Profile(gpu=dict(signals["gpu"]), vllm=dict(signals["vllm"]),
                   metrics={"goodput_req_s": sorted(vals)[len(vals) // 2]},
                   per_seed={"goodput_req_s": list(vals)}, marker_verified=verified,
                   source="real_vllm", outcome_class="eval_result",
                   bench_config=dict(_BC), eval_result=eval_result)


# a wired knob is needed in the base config so the honest holdout can vary it
_BASE = {"n_requests": 100}


def test_happy_path_adopts_a_real_gain():
    # baseline is under-saturated; raising max_num_seqs lifts goodput; bottleneck
    # then shifts to compute -> adopted (gain generalizes on the holdout config).
    def eval_fn(config):
        seqs = config.get("max_num_seqs")
        if seqs is None:
            return _prof(_UNDERSAT, [5.0, 5.0, 5.1])
        gp = {128: 6.0, 256: 7.1, 512: 6.5}.get(seqs, 5.0)
        return _prof(_COMPUTE, [gp, gp + 0.1, gp - 0.1])

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=3,
                     quality_measure_fn=_quality)
    assert res["outcome"] == "adopted"
    assert res["adopted"]["target"] == "config:max_num_seqs"
    assert res["adopted"]["config"]["max_num_seqs"] == 256
    assert res["conclusion"]["result"] == "gain"


def test_no_gain_is_honest_dod_b():
    # nothing beats the baseline -> DoD-B with next directions, never a fake win.
    def eval_fn(config):
        return _prof(_UNDERSAT, [5.0, 5.0, 5.1])

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=3)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    assert res["conclusion"]["result"] == "dod_b"
    assert res["conclusion"]["next_directions"]
    # it must have actually tried more than one wired target
    searched = [r.get("searched_target") for r in res["rounds"] if r.get("searched_target")]
    assert len(set(searched)) >= 2


def test_unverified_candidate_never_adopted():
    # a config that looks great but isn't marker-verified can't be accepted.
    def eval_fn(config):
        if not (config.get("max_num_seqs") or config.get("concurrency")):
            return _prof(_UNDERSAT, [5.0, 5.0, 5.1])
        return _prof(_COMPUTE, [99.0, 99.0, 99.0], verified=False)

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=2)
    assert res["outcome"] == "dod_b" and res["adopted"] is None


def test_code_target_evolves_when_scheduling_bound():
    # config search doesn't help; the evolved schedule_batch policy does -> adopted. The real bench
    # marks the candidate EFFECTIVE (invoked + reordered + no fallback) via eval_result.
    def eval_fn(config):
        if config.get("policy"):
            return _prof(_SCHEDQ, [7.0, 7.1, 6.9], eval_result={"effective": True})
        return _prof(_SCHEDQ, [5.0, 5.0, 5.1])

    def evolve_fn(base_config, spec):
        return Candidate(target="code:schedule_batch", kind="code",
                         value={"policy": "runs_real/evolved.py"}, marker_verified=True,
                         metrics={"goodput_req_s": 7.0})

    res = run_autopt(_SPEC, eval_fn=eval_fn, evolve_fn=evolve_fn, base_config=_BASE,
                     max_rounds=3, quality_measure_fn=_quality)
    assert res["outcome"] == "adopted"
    assert res["adopted"]["target"] == "code:schedule_batch"
    assert res["adopted"]["config"]["policy"] == "runs_real/evolved.py"


def test_ineffective_code_candidate_not_adopted():
    # a code candidate that LOADED (marker_verified) and posts better numbers but the REAL bench
    # reports effective=False (it fell back / did not reorder) must NOT adopt — Codex review P1.
    def eval_fn(config):
        if config.get("policy"):
            return _prof(_SCHEDQ, [7.0, 7.1, 6.9], eval_result={"effective": False})
        return _prof(_SCHEDQ, [5.0, 5.0, 5.1])

    def evolve_fn(base_config, spec):
        return Candidate(target="code:schedule_batch", kind="code",
                         value={"policy": "runs_real/evolved.py"}, marker_verified=True,
                         metrics={"goodput_req_s": 7.0})

    res = run_autopt(_SPEC, eval_fn=eval_fn, evolve_fn=evolve_fn, base_config=_BASE,
                     max_rounds=3, quality_measure_fn=_quality)
    assert res["outcome"] == "dod_b" and res["adopted"] is None


def test_code_target_without_evolve_fn_stops_honestly():
    # scheduling-bound, config can't help, and no evolve_fn wired -> honest DoD-B.
    def eval_fn(config):
        return _prof(_SCHEDQ, [5.0, 5.0, 5.1])

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=3)
    assert res["outcome"] == "dod_b"
    notes = [r.get("note", "") for r in res["rounds"]]
    assert any("schedule_batch" in n or "evolve" in n for n in notes)


def test_unverified_evolution_candidate_not_adopted():
    def eval_fn(config):
        return _prof(_SCHEDQ, [7.0, 7.1, 6.9] if config.get("policy") else [5.0, 5.0, 5.1])

    def evolve_fn(base_config, spec):
        return Candidate(target="code:schedule_batch", kind="code",
                         value={"policy": "runs_real/evolved.py"}, marker_verified=False)

    res = run_autopt(_SPEC, eval_fn=eval_fn, evolve_fn=evolve_fn, base_config=_BASE,
                     max_rounds=2)
    assert res["outcome"] == "dod_b" and res["adopted"] is None


def test_max_rounds_is_respected():
    calls = {"n": 0}

    def eval_fn(config):
        calls["n"] += 1
        return _prof(_UNDERSAT, [5.0, 5.0, 5.1])

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=1)
    assert len(res["rounds"]) == 1


def test_cmd_autopt_cli(monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    from vllm_evolve.cli import main as cli_main
    from vllm_evolve.engine import profile as prof_mod
    from vllm_evolve.engine import quality_remote as qr

    def fake_collect(config):
        seqs = config.get("max_num_seqs")
        if seqs is None:
            return _prof(_UNDERSAT, [5.0, 5.0, 5.1])
        gp = {128: 6.0, 256: 7.1, 512: 6.5}.get(seqs, 5.0)
        return _prof(_COMPUTE, [gp, gp + 0.1, gp - 0.1])

    monkeypatch.setattr(prof_mod, "collect_profile", fake_collect)
    # default backend is remote -> cmd_autopt now builds the remote quality hook; stub the ssh probe
    # so quality is NOT MEASURED here (no box) -> the run stays an honest DoD-B (offline).
    monkeypatch.setattr(qr, "_ssh_run",
                        lambda remote, cmd, payload: SimpleNamespace(returncode=1, stdout="",
                                                                     stderr="no box"))
    rc = cli_main.main(["autopt", "maximize goodput", "--n-requests", "100",
                      "--max-rounds", "3"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    # The CLI has no measured-quality runner wired (box-gated) -> honest DoD-B, never a fake
    # adopt: a real throughput gain cannot be adopted without MEASURED quality (Codex R5).
    assert rc == 0 and out["ok"] is True and out["outcome"] == "dod_b"
    assert any(r.get("searched_target") for r in out["rounds"])   # it DID run the loop


def test_cmd_autopt_remote_without_model_pins_default_for_quality(monkeypatch, capsys):
    # ve autopt --backend remote without --model: the bench serves a default model, so the quality
    # hook must get that SAME default — else it returns None for every measurement and a real remote
    # gain can never adopt through the direct autopt entry (Codex review P2; same fix as `ve run`).
    from vllm_evolve.cli import main as cli_main
    captured = {}

    def fake_run(spec, **kw):
        captured["qmf"] = kw.get("quality_measure_fn")
        return {"outcome": "dod_b", "adopted": None, "backend": kw.get("backend")}

    monkeypatch.setattr("vllm_evolve.modes.autopt.run", fake_run)
    rc = cli_main.main(["autopt", "maximize goodput"])      # remote default, NO --model
    assert rc == 0
    qmf = captured["qmf"]
    assert qmf is not None and qmf({}) is not None          # quality hook got the served model


def test_autopt_requires_measured_quality_to_adopt():
    # real gain + same caliber but NO quality_measure_fn -> NOT MEASURED -> never adopt
    def eval_fn(config):
        seqs = config.get("max_num_seqs")
        if seqs is None:
            return _prof(_UNDERSAT, [5.0, 5.0, 5.1])
        gp = {128: 6.0, 256: 7.1, 512: 6.5}.get(seqs, 5.0)
        return _prof(_COMPUTE, [gp, gp + 0.1, gp - 0.1])

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=3)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    strict = [r["strict_accept"] for r in res["rounds"] if r.get("strict_accept")]
    assert any(any("quality NOT certified" in x for x in s["reasons"]) for s in strict)


def test_autopt_rejects_caliber_mismatch():
    # candidate ran a DIFFERENT model caliber -> same_caliber_mismatch -> never adopt
    other = build_bench_config(runner_kind="candidate", model="other/model-7b").to_dict()

    def eval_fn(config):
        seqs = config.get("max_num_seqs")
        if seqs is None:
            return _prof(_UNDERSAT, [5.0, 5.0, 5.1])
        p = _prof(_COMPUTE, [7.1, 7.2, 7.0])
        p.bench_config = other
        return p

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=2,
                     quality_measure_fn=_quality)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    assert any(r.get("same_caliber") == "same_caliber_mismatch" for r in res["rounds"])


def _happy_eval(config):
    # under-saturated baseline; raising max_num_seqs lifts goodput (primary A/B is same-caliber)
    seqs = config.get("max_num_seqs")
    if seqs is None:
        return _prof(_UNDERSAT, [5.0, 5.0, 5.1])
    gp = {128: 6.0, 256: 7.1, 512: 6.5}.get(seqs, 5.0)
    return _prof(_COMPUTE, [gp, gp + 0.1, gp - 0.1])


def test_autopt_rejects_holdout_caliber_mismatch():
    # primary A/B same-caliber, but the HOLDOUT candidate ran a different model -> reject (R6)
    other = build_bench_config(runner_kind="candidate", model="other/model-7b").to_dict()

    def eval_fn(config):
        p = _happy_eval(config)
        if config.get("n_requests") == 125 and config.get("max_num_seqs") is not None:
            p.bench_config = other          # holdout candidate ran a different caliber
        return p

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=2,
                     quality_measure_fn=_quality)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    assert any(r.get("holdout_same_caliber") == "holdout_same_caliber_mismatch"
               for r in res["rounds"])


def test_autopt_rejects_holdout_missing_provenance():
    def eval_fn(config):
        p = _happy_eval(config)
        if config.get("n_requests") == 125:     # holdout configs lack provenance
            p.bench_config = None
        return p

    res = run_autopt(_SPEC, eval_fn=eval_fn, base_config=_BASE, max_rounds=2,
                     quality_measure_fn=_quality)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    assert any(r.get("holdout_same_caliber") == "holdout_same_caliber_unverifiable"
               for r in res["rounds"])


def test_holdout_config_actually_differs_for_small_workloads():
    # Codex review P2: +25% truncates to the same int for small concurrency/n_requests (1-3), which
    # would record a holdout at an UNCHANGED operating point. The holdout must genuinely differ.
    from vllm_evolve.engine.orchestrate import _holdout_config
    for base in (1, 2, 3):
        h = _holdout_config({"concurrency": base, "n_requests": 50}, "")
        assert h is not None and h["concurrency"] != base
    # larger values still scale up by +25%
    assert _holdout_config({"concurrency": 8, "n_requests": 50}, "")["concurrency"] == 10
    # nothing to vary -> None (honest: no holdout rather than a no-op rerun)
    assert _holdout_config({"model": "m"}, "") is None
