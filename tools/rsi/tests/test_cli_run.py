"""VE_RUN: the natural-language front door (`ve run`). Pure/offline — the hardware->backend,
metric->profile, and method->mode mappings are unit-tested, method routing is monkeypatched, and
one un-mocked end-to-end smoke runs through autopt + local_smoke (no GPU, no real Frontier).

The load-bearing honesty test is `test_dod_stamp_is_decided_by_backend_not_the_result`: an off-box
backend stays DoD-B EVEN IF the underlying result claims an adoption — the stamp is decided at the
quarantine boundary (the backend), never from a result a synthetic run could shape.
"""
from __future__ import annotations

import json

from vllm_evolve.cli import main as cli_main
from vllm_evolve.cli.main import _run_dod_stamp, _run_resolve_backend, _run_resolve_profile


def test_hw_maps_to_backend():
    # no GPU -> the off-box simulator; a real GPU spec -> remote real vLLM
    assert _run_resolve_backend("none", None) == "frontier_sim"
    assert _run_resolve_backend("cpu", None) == "frontier_sim"
    assert _run_resolve_backend("", None) == "frontier_sim"
    assert _run_resolve_backend("1xA100", None) == "remote"
    assert _run_resolve_backend("2xH100-80GB", None) == "remote"
    # an explicit --backend overrides the hardware mapping (both directions)
    assert _run_resolve_backend("none", "local_smoke") == "local_smoke"
    assert _run_resolve_backend("1xA100", "frontier_sim") == "frontier_sim"


def test_metric_maps_to_profile():
    assert _run_resolve_profile("minimize p99 TTFT") == "latency"
    assert _run_resolve_profile("tighten tail latency") == "latency"
    assert _run_resolve_profile("replay the azure trace") == "replay"
    assert _run_resolve_profile("maximize goodput") == "throughput"
    assert _run_resolve_profile("maximize throughput") == "throughput"


def test_dod_stamp_is_decided_by_backend_not_the_result():
    adopted = {"outcome": "adopted", "adopted": {"target": "scheduling"}}
    # off-box backends are DoD-B even when the underlying result claims an adoption
    for be in ("local_smoke", "frontier_sim"):
        level, note = _run_dod_stamp(be, adopted)
        assert level == "DoD-B"
        assert "box-gated" in note and "No adoption gate" in note
    # only a real-vLLM (remote) holdout-confirmed adoption is DoD-A
    assert _run_dod_stamp("remote", adopted)[0] == "DoD-A"
    assert _run_dod_stamp("remote", {"outcome": "dod_b", "adopted": None})[0] == "DoD-B"


def _last_json(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_method_routes_to_the_right_mode(monkeypatch, capsys):
    calls: dict = {}

    def fake_autopt_run(spec, **kw):
        calls["autopt"] = kw
        return {"outcome": "dod_b", "adopted": None, "backend": kw.get("backend")}

    def fake_tune_run(policy, spec, **kw):
        calls["tune"] = {"policy": policy, **kw}
        return {"outcome": "dod_b", "adopted": None, "mode": "tune", "backend": kw.get("backend")}

    monkeypatch.setattr("vllm_evolve.modes.autopt.run", fake_autopt_run)
    monkeypatch.setattr("vllm_evolve.modes.tune.run", fake_tune_run)

    # evolve -> autopt_mode.run ; declaration + resolved are echoed; off-box -> DoD-B
    rc = cli_main.main(["run", "--model", "demo", "--hw", "none",
                        "--metric", "maximize", "throughput", "--method", "evolve",
                        "--backend", "local_smoke"])
    out = _last_json(capsys)
    assert rc == 0 and "autopt" in calls and "tune" not in calls
    assert out["resolved"] == {"backend": "local_smoke", "profile": "throughput",
                               "goal": out["resolved"]["goal"]}
    assert out["declaration"] == {"model": "demo", "hardware": "none",
                                  "metric": "maximize throughput", "method": "evolve"}
    assert out["dod_level"] == "DoD-B"

    # tune -> tune_mode.run with the default seed policy ; latency metric -> latency profile
    calls.clear()
    rc = cli_main.main(["run", "--hw", "cpu", "--metric", "minimize", "p99", "TTFT",
                        "--method", "tune", "--backend", "local_smoke"])
    out = _last_json(capsys)
    assert rc == 0 and "tune" in calls and "autopt" not in calls
    assert out["resolved"]["profile"] == "latency"
    assert calls["tune"]["policy"].endswith("seed.py")


def test_remote_evolve_wires_the_real_engine_not_the_noop(monkeypatch, capsys):
    # P1 regression: method=evolve must drive the generational template-author engine on EVERY
    # backend (incl. the GPU 'remote' mapping) — never the honest no-op default_evolve_fn.
    import vllm_evolve.engine.evolve_target as et

    def _boom(*a, **k):
        raise AssertionError("default_evolve_fn (no-op) must not be used by `ve run` evolve")

    monkeypatch.setattr(et, "default_evolve_fn", _boom)
    captured: dict = {}

    def fake_autopt_run(spec, **kw):
        captured.update(kw)
        return {"outcome": "dod_b", "adopted": None, "backend": kw.get("backend")}

    monkeypatch.setattr("vllm_evolve.modes.autopt.run", fake_autopt_run)

    rc = cli_main.main(["run", "--hw", "1xA100", "--metric", "maximize", "throughput",
                        "--method", "evolve", "--backend", "remote"])
    out = _last_json(capsys)
    assert rc == 0 and out["resolved"]["backend"] == "remote"
    assert captured.get("author_fn") is not None          # the real evolution surface ...
    assert captured.get("evolve_fn") is None              # ... not the legacy single-shot / no-op
    assert isinstance(captured.get("evolve_params"), dict)
    assert {"generations", "population"} <= captured["evolve_params"].keys()


def test_default_hw_none_maps_to_frontier_sim_and_is_dod_b(monkeypatch, capsys):
    # The MAPPED default for --hw none is frontier_sim (not local_smoke). Prove it reaches a DoD-B
    # verdict WITHOUT external Frontier creds by mocking the underlying run (no --backend override).
    monkeypatch.setattr("vllm_evolve.modes.autopt.run",
                        lambda spec, **kw: {"outcome": "dod_b", "adopted": None,
                                            "backend": kw.get("backend")})
    rc = cli_main.main(["run", "--hw", "none", "--metric", "maximize", "throughput",
                        "--method", "evolve"])
    out = _last_json(capsys)
    assert rc == 0 and out["resolved"]["backend"] == "frontier_sim"
    assert out["dod_level"] == "DoD-B" and "box-gated" in out["honesty_note"]


def test_quality_hook_is_remote_only(monkeypatch, capsys):
    # The measured-quality hook the adoption gate needs is built ONLY for remote (real box-gated
    # measure). Off-box backends pass None -> quality NOT measured -> cannot adopt -> DoD-B.
    seen: dict = {}

    def fake_autopt(spec, **kw):
        seen[kw.get("backend")] = kw.get("quality_measure_fn")
        return {"outcome": "dod_b", "adopted": None, "backend": kw.get("backend")}

    monkeypatch.setattr("vllm_evolve.modes.autopt.run", fake_autopt)
    for be in ("remote", "frontier_sim", "local_smoke"):
        hw = "1xA100" if be == "remote" else "none"
        cli_main.main(["run", "--hw", hw, "--metric", "maximize", "throughput",
                       "--method", "evolve", "--backend", be, "--model", "m"])
        capsys.readouterr()
    assert seen["remote"] is not None                         # real box-gated measure wired
    assert seen["frontier_sim"] is None and seen["local_smoke"] is None   # off-box NOT measured


def test_remote_without_model_pins_the_default_model_for_quality(monkeypatch, capsys):
    # --model omitted: the bench serves a default model, so the quality probe must get that SAME
    # default — else make_remote_quality_measure_fn returns None for every measurement and remote
    # DoD-A is unreachable. Prove the quality hook can actually measure (model present).
    captured: dict = {}

    def fake_autopt(spec, **kw):
        captured["qmf"] = kw.get("quality_measure_fn")
        return {"outcome": "dod_b", "adopted": None, "backend": kw.get("backend")}

    monkeypatch.setattr("vllm_evolve.modes.autopt.run", fake_autopt)
    rc = cli_main.main(["run", "--hw", "1xA100", "--metric", "maximize", "throughput",
                        "--method", "evolve", "--backend", "remote"])       # NO --model
    out = _last_json(capsys)
    assert rc == 0
    qmf = captured["qmf"]
    assert qmf is not None and qmf({}) is not None        # a real measure_fn, not the no-model None
    assert out["declaration"]["model"]                    # the verdict surfaces the effective model


def _adopting_collect_and_metric():
    """A real-vLLM-metadata eval where raising max_num_seqs lifts goodput (same fixture shape the
    orchestrator adoption tests use), plus the metric `parse_goal('maximize goodput')` produces."""
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.core.schemas import Profile
    from vllm_evolve.intent.spec import parse_goal
    metric = parse_goal("maximize goodput").metric
    bc = build_bench_config(runner_kind="candidate", model="facebook/opt-125m").to_dict()
    undersat = {"gpu": {"sm_util_max": 35, "duty_cycle": 0.1},
                "vllm": {"kv_util": 0.2, "waiting": 0}}
    compute = {"gpu": {"sm_util_max": 95, "duty_cycle": 0.95},
               "vllm": {"kv_util": 0.5, "waiting": 1}}

    def prof(sig, vals):
        return Profile(gpu=dict(sig["gpu"]), vllm=dict(sig["vllm"]),
                       metrics={metric: sorted(vals)[len(vals) // 2]},
                       per_seed={metric: list(vals)}, marker_verified=True,
                       source="real_vllm", outcome_class="eval_result", bench_config=dict(bc))

    def fake_collect(config):
        seqs = config.get("max_num_seqs")
        if seqs is None:
            return prof(undersat, [5.0, 5.0, 5.1])
        gp = {128: 6.0, 256: 6.5, 512: 7.1}.get(seqs, 5.0)   # winner = max_num_seqs 512
        return prof(compute, [gp, gp + 0.1, gp - 0.1])

    return fake_collect


def test_ve_run_remote_reaches_dod_a_end_to_end(monkeypatch, capsys):
    # No GPU: mock the box-gated bits (real-vLLM eval + served-model quality), then prove a real
    # config win flows through the UNCHANGED accept gate to an adopted DoD-A verdict.
    monkeypatch.setattr("vllm_evolve.engine.profile.collect_profile",
                        _adopting_collect_and_metric())
    monkeypatch.setattr("vllm_evolve.cli.main._run_quality_measure_fn",
                        lambda backend, base: (lambda config: (lambda _p: {
                            "perplexity": 10.0, "task_em": 0.8, "output_agreement": 0.999})))
    rc = cli_main.main(["run", "--hw", "1xA100", "--metric", "maximize", "goodput",
                        "--method", "evolve", "--backend", "remote", "--model", "facebook/opt-125m",
                        "--n-requests", "100"])
    out = _last_json(capsys)
    assert rc == 0
    assert out["result"]["outcome"] == "adopted"
    assert out["dod_level"] == "DoD-A"


def test_remote_quality_serves_the_winner_config(monkeypatch, capsys):
    # End-to-end: the candidate quality measure runs ON the remote box (over ssh) and is built from
    # the WINNER config, not just base. The winner raises max_num_seqs to 512; the real quality hook
    # must ship that to the box's probe (the ssh payload carries max_num_seqs == 512). Uses the real
    # _run_quality_measure_fn (not mocked); only the ssh boundary + the bench eval are stubbed.
    import json as _json
    from types import SimpleNamespace

    import vllm_evolve.engine.quality_remote as qr
    payloads: list = []

    def _fake_ssh(remote, remote_cmd, payload_json):
        p = _json.loads(payload_json)
        payloads.append(p)
        body = {"perplexity": 10.0, "task_em": 0.8, "continuations": [" x"] * len(p["prompts"])}
        return SimpleNamespace(returncode=0, stdout=_json.dumps(body), stderr="")

    monkeypatch.setattr(qr, "_ssh_run", _fake_ssh)
    monkeypatch.setattr("vllm_evolve.engine.profile.collect_profile",
                        _adopting_collect_and_metric())
    rc = cli_main.main(["run", "--hw", "1xA100", "--metric", "maximize", "goodput",
                        "--method", "evolve", "--backend", "remote", "--model", "facebook/opt-125m",
                        "--n-requests", "100"])
    out = _last_json(capsys)
    assert rc == 0 and out["result"]["outcome"] == "adopted"
    assert any(p["config"].get("max_num_seqs") == 512 for p in payloads)   # winner sent to the box


def test_ve_run_remote_without_measured_quality_stays_dod_b(monkeypatch, capsys):
    # Same real gain, but quality NOT measured (factory returns None) -> the gate refuses adoption.
    monkeypatch.setattr("vllm_evolve.engine.profile.collect_profile",
                        _adopting_collect_and_metric())
    monkeypatch.setattr("vllm_evolve.cli.main._run_quality_measure_fn", lambda backend, base: None)
    rc = cli_main.main(["run", "--hw", "1xA100", "--metric", "maximize", "goodput",
                        "--method", "evolve", "--backend", "remote", "--model", "facebook/opt-125m",
                        "--n-requests", "100"])
    out = _last_json(capsys)
    assert rc == 0 and out["result"]["outcome"] == "dod_b" and out["dod_level"] == "DoD-B"


def test_remote_run_failure_returns_structured_json_not_traceback(monkeypatch, capsys):
    # a box-gated failure (remote unreachable / serve fails) before a Profile must still emit ONE
    # JSON verdict, never a traceback — the front-door contract (Codex review P2).
    def boom(spec, **kw):
        raise RuntimeError("ssh: connect to host gpu-box port 22: Network is unreachable")

    monkeypatch.setattr("vllm_evolve.modes.autopt.run", boom)
    rc = cli_main.main(["run", "--hw", "1xA100", "--metric", "maximize", "throughput",
                        "--method", "evolve", "--backend", "remote", "--model", "m"])
    out = _last_json(capsys)
    assert rc == 1        # box-gated/runtime failure -> exit 1 (like cmd_bench), not 2 (rejection)
    assert out["ok"] is False and out["outcome_class"] == "run_failed"
    assert out["resolved"]["backend"] == "remote" and out["declaration"]["method"] == "evolve"
    assert out["dod_level"] == "DoD-B" and "unreachable" in out["error"]


def test_empty_metric_and_bad_method_are_rejected(capsys):
    rc = cli_main.main(["run", "--hw", "none", "--metric", "   ", "--backend", "local_smoke"])
    assert rc == 2 and _last_json(capsys)["outcome_class"] == "empty_metric"


def test_offline_end_to_end_smoke_is_dod_b(capsys):
    # the plan's headline no-GPU one-liner: a real (un-mocked) offline run via autopt + local_smoke
    rc = cli_main.main(["run", "--model", "demo", "--hw", "none",
                        "--metric", "maximize", "throughput", "--method", "evolve",
                        "--backend", "local_smoke"])
    out = _last_json(capsys)
    assert rc == 0 and out["ok"] is True
    assert out["resolved"]["backend"] == "local_smoke"
    assert out["dod_level"] == "DoD-B"
    assert "box-gated" in out["honesty_note"]
    # the underlying result is a real dict (not fabricated) and carries the backend provenance
    assert out["result"].get("backend") == "local_smoke"
