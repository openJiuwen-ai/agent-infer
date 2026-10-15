"""AC2-AC5: the WHOLE ar round + a full autopt loop run end-to-end off-box via --backend
local_smoke — and the synthetic result is quarantined so it can NEVER become a gain/keep/AC6.

Zero GPU. Drives the real CLI through the real phase machine; the only "backend" is the in-process
synthetic local_smoke one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.tools import phase_guard  # noqa: E402

SEED = str(REPO_ROOT / "targets" / "scheduling" / "seed.py")


@pytest.fixture(autouse=True)
def _reset_phase():
    phase_guard.reset_state()
    yield
    phase_guard.reset_state()


def _run(capsys, *argv):
    """Run one `ar` verb; return (rc, last-json-line-parsed-or-None)."""
    rc = cli_main.main(list(argv))
    out = capsys.readouterr().out.strip()
    parsed = None
    for line in reversed(out.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    return rc, parsed


def _bench_eval(parsed) -> dict:
    """Pull the eval_result out of an `ve bench` payload."""
    return parsed["result"]["eval_result"]


def test_full_round_runs_off_box_and_compare_refuses_smoke(tmp_path, capsys, monkeypatch):
    # AC2: every phase advances off-box with local_smoke; AC3: compare REFUSES the smoke A/B.
    monkeypatch.setenv("VE_BENCH_BACKEND", "local_smoke")
    work = tmp_path / "work.py"
    work.write_text(Path(SEED).read_text(encoding="utf-8") + "\n# candidate variant\n",
                    encoding="utf-8")

    # READ_CONTEXT -> DESIGN -> GENERATE -> VERIFY
    assert _run(capsys, "phase", "set", "READ_CONTEXT")[0] == 0
    assert _run(capsys, "context", "scheduling")[0] == 0
    assert _run(capsys, "phase", "set", "DESIGN")[0] == 0
    assert _run(capsys, "design", "--note", "smoke plumbing dry-run", "scheduling")[0] == 0
    assert _run(capsys, "phase", "set", "GENERATE")[0] == 0
    assert _run(capsys, "phase", "set", "VERIFY")[0] == 0
    rc, _ = _run(capsys, "verify", str(work), "scheduling")
    assert rc == 0  # -> VERIFY_PASSED

    # BENCHMARK (bench runs in VERIFY_PASSED): candidate + baseline, both synthetic
    rc, cand_out = _run(capsys, "bench", str(work), "scheduling", "--runner", "candidate")
    assert rc == 0 and cand_out["backend"] == "local_smoke"
    cand_ev = _bench_eval(cand_out)
    assert cand_ev["source"] == "local_smoke"
    assert cand_ev["outcome_class"] == "local_smoke_nonqualifying"

    rc, base_out = _run(capsys, "bench", SEED, "scheduling", "--runner", "strong_baseline")
    assert rc == 0
    base_ev = _bench_eval(base_out)

    base_json = tmp_path / "base.json"
    cand_json = tmp_path / "cand.json"
    base_json.write_text(json.dumps(base_ev), encoding="utf-8")
    cand_json.write_text(json.dumps(cand_ev), encoding="utf-8")

    # advance to KEEP_OR_DISCARD and compare -> AC3: HARD-REFUSE the synthetic A/B
    assert _run(capsys, "phase", "set", "BENCHMARK")[0] == 0
    assert _run(capsys, "phase", "set", "KEEP_OR_DISCARD")[0] == 0
    rc, cmp_out = _run(capsys, "compare", str(base_json), str(cand_json))
    assert rc == 2 and cmp_out["outcome_class"] == "non_real_source_blocked"

    # the round can only end in a discard (a smoke result can never be kept)
    assert _run(capsys, "phase", "set", "COMMIT_OR_ROLLBACK")[0] == 0
    rc, _ = _run(capsys, "discard", str(work), "--reason", "local_smoke plumbing only")
    assert rc == 0


def test_quarantine_verify_gain_and_keep_block_smoke(tmp_path, capsys):
    # AC3: verify-gain and keep over a smoke artifact also HARD-REFUSE.
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.bench.local_smoke import run_local_smoke
    from vllm_evolve.tools.phase_guard import Phase
    bc = build_bench_config(runner_kind="strong_baseline", model="facebook/opt-125m", remote="box")
    smoke = run_local_smoke(bc).eval_result
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(smoke), encoding="utf-8")

    # verify-gain (needs its own phase): smoke base+cand -> blocked
    for ph in (Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY,
               Phase.VERIFY_PASSED, Phase.BENCHMARK, Phase.KEEP_OR_DISCARD):
        phase_guard.transition(ph)
    rc, vg = _run(capsys, "verify-gain", str(p), str(p), "--metric", "output_throughput_tok_s")
    assert rc == 2 and vg["outcome_class"] == "non_real_source_blocked" and vg["adopt"] is False

    # keep over a smoke eval_result -> blocked
    phase_guard.transition(Phase.COMMIT_OR_ROLLBACK)
    policy = tmp_path / "w.py"
    policy.write_text("def schedule_batch(rs, st):\n    return None\n", encoding="utf-8")
    rc, kp = _run(capsys, "keep", str(policy), "--eval-result", str(p),
                  "--archive-root", str(tmp_path / "arch"))
    assert rc == 2 and kp["outcome_class"] == "non_real_source_blocked"


def test_lock_d_rejects_forged_real_source(tmp_path, capsys):
    # LOCK D requires source=='real_vllm' AND outcome_class=='eval_result': a forged artifact that
    # LIES about its source but keeps a non-eval outcome_class is still HARD-REFUSED everywhere.
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.tools.phase_guard import Phase
    bc = build_bench_config(runner_kind="strong_baseline", model="facebook/opt-125m",
                            remote="box").to_dict()
    forged = {"source": "real_vllm", "outcome_class": "local_smoke_nonqualifying",
              "primary_metric": "output_throughput_tok_s",
              "raw_per_seed_metrics": [{"primary_value": 999.0}], "values": [999.0],
              "bench_config": bc}
    p = tmp_path / "forged.json"
    p.write_text(json.dumps(forged), encoding="utf-8")

    for ph in (Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY,
               Phase.VERIFY_PASSED, Phase.BENCHMARK, Phase.KEEP_OR_DISCARD):
        phase_guard.transition(ph)
    rc, cmp_out = _run(capsys, "compare", str(p), str(p))
    assert rc == 2 and cmp_out["outcome_class"] == "non_real_source_blocked"
    rc, vg = _run(capsys, "verify-gain", str(p), str(p), "--metric", "output_throughput_tok_s")
    assert rc == 2 and vg["outcome_class"] == "non_real_source_blocked"
    phase_guard.transition(Phase.COMMIT_OR_ROLLBACK)
    pol = tmp_path / "w.py"
    pol.write_text("def schedule_batch(rs, st):\n    return None\n", encoding="utf-8")
    rc, kp = _run(capsys, "keep", str(pol), "--eval-result", str(p),
                  "--archive-root", str(tmp_path / "arch"))
    assert rc == 2 and kp["outcome_class"] == "non_real_source_blocked"


def test_smoke_eval_result_is_structurally_nonpromotable():
    # AC4: by construction a smoke result can never be a clean success / real source.
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.bench.eval_result import validate_eval_result
    from vllm_evolve.bench.local_smoke import run_local_smoke
    from vllm_evolve.bench.outcome import OutcomeClass, is_success
    bc = build_bench_config(runner_kind="strong_baseline", model="facebook/opt-125m", remote="box")
    res = run_local_smoke(bc)
    ev = res.eval_result
    validate_eval_result(ev)                                  # schema-valid (plumbing exercises)
    assert ev["source"] == "local_smoke" and ev["source"] != "real_vllm"
    assert ev["outcome_class"] == OutcomeClass.LOCAL_SMOKE_NONQUALIFYING.value
    assert is_success(OutcomeClass(ev["outcome_class"])) is False
    assert res.to_dict()["effective"] is False and res.to_dict()["quality_ok"] is None
    assert "vllm-evolve:" not in json.dumps(ev)               # never the plugin marker namespace


def test_autopt_local_smoke_concludes_dod_b(capsys):
    # AC5: a full autopt L1->L4 loop runs off-box and can ONLY conclude DoD-B (never adopted).
    from vllm_evolve.core.schemas import Spec
    from vllm_evolve.engine.orchestrate import run_autopt
    from vllm_evolve.engine.profile import collect_profile_local_smoke
    spec = Spec(metric="output_throughput_tok_s", direction="max")
    res = run_autopt(spec, eval_fn=collect_profile_local_smoke,
                     base_config={"n_requests": 100, "model": "facebook/opt-125m"}, max_rounds=3)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    searched = [r.get("searched_target") for r in res["rounds"] if r.get("searched_target")]
    assert searched, "the loop must actually exercise optimize (search a target)"


def test_ar_autopt_cli_local_smoke_concludes_dod_b(capsys, monkeypatch):
    # AC5 through the REAL CLI: exercises the parser + cmd_autopt backend seam (not run_autopt).
    monkeypatch.setenv("VE_BENCH_BACKEND", "local_smoke")
    rc = cli_main.main(["autopt", "maximize throughput", "--n-requests", "100",
                      "--max-rounds", "2", "--model", "facebook/opt-125m"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert out["backend"] == "local_smoke" and out["outcome"] == "dod_b"
    assert out["adopted"] is None


def test_collect_profile_local_smoke_threads_max_seeds():
    # Codex review P2: max_seeds (a StatisticalConfig field) must survive collect_profile's override
    # filter and reach the BenchConfig, else budgeted profile/optimize/autopt runs all seeds.
    from vllm_evolve.engine.profile import collect_profile_local_smoke
    prof = collect_profile_local_smoke({"target": "scheduling", "runner_kind": "strong_baseline",
                                        "max_seeds": 2})
    assert prof.bench_config["statistical"]["max_seeds"] == 2
