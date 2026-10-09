"""A-F3: the WHOLE ve round runs end-to-end off-box via --backend frontier_sim, and the Frontier
result is quarantined so it can NEVER become a gain/keep/AC6. The Frontier subprocess is stubbed by
a fixture (the goal is to validate the end-to-end CAPABILITY, not to run Frontier). Zero GPU.

Drives the REAL CLI through the REAL phase machine; the only thing faked is Frontier's own process.
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
  "request_e2e_time_statistics": {"p50": 120.0, "unit": "ms"},
  "memory_utilization_percent": {"MONOLITHIC": 15.0},
  "preemption_statistics": {"total_preemption_events": 0, "total_preempted_requests": 0}
}"""
_REQUEST_METRICS_CSV = (
    "Request Id,ttft,request_e2e_time,tpot,"
    "request_inter_arrival_delay,request_first_scheduling_delay\n"
    "0,30.0,120.0,0.9,,0.0\n1,31.0,121.0,0.9,0.5,0.0\n"
    "2,29.0,119.0,0.9,0.5,0.0\n3,30.0,120.0,0.9,0.5,0.0\n"
)
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
    # select frontier_sim + point at a dummy Frontier + stub the subprocess with a fixture
    monkeypatch.setenv("VE_BENCH_BACKEND", "frontier_sim")
    monkeypatch.setenv("VE_FRONTIER_REPO", "/fake/frontier")
    monkeypatch.setenv("VE_FRONTIER_PYTHON", "/fake/python")
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke)


def _run(capsys, *argv):
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
    return parsed["result"]["eval_result"]


def test_full_round_runs_off_box_and_compare_refuses_frontier(tmp_path, capsys):
    # the whole round advances off-box on frontier_sim; compare HARD-REFUSES the non-real A/B.
    work = tmp_path / "work.py"
    work.write_text(Path(SEED).read_text(encoding="utf-8") + "\n# candidate variant\n",
                    encoding="utf-8")

    assert _run(capsys, "phase", "set", "READ_CONTEXT")[0] == 0
    assert _run(capsys, "context", "scheduling")[0] == 0
    assert _run(capsys, "phase", "set", "DESIGN")[0] == 0
    assert _run(capsys, "design", "--note", "frontier plumbing dry-run", "scheduling")[0] == 0
    assert _run(capsys, "phase", "set", "GENERATE")[0] == 0
    assert _run(capsys, "phase", "set", "VERIFY")[0] == 0
    assert _run(capsys, "verify", str(work), "scheduling")[0] == 0  # -> VERIFY_PASSED

    rc, cand_out = _run(capsys, "bench", str(work), "scheduling", "--runner", "candidate")
    assert rc == 0 and cand_out["backend"] == "frontier_sim"
    cand_ev = _bench_eval(cand_out)
    assert cand_ev["source"] == "frontier_sim"
    assert cand_ev["outcome_class"] == "simulator_nonqualifying"

    rc, base_out = _run(capsys, "bench", SEED, "scheduling", "--runner", "strong_baseline")
    assert rc == 0
    base_ev = _bench_eval(base_out)

    base_json = tmp_path / "base.json"
    cand_json = tmp_path / "cand.json"
    base_json.write_text(json.dumps(base_ev), encoding="utf-8")
    cand_json.write_text(json.dumps(cand_ev), encoding="utf-8")

    assert _run(capsys, "phase", "set", "BENCHMARK")[0] == 0
    assert _run(capsys, "phase", "set", "KEEP_OR_DISCARD")[0] == 0
    rc, cmp_out = _run(capsys, "compare", str(base_json), str(cand_json))
    assert rc == 2 and cmp_out["outcome_class"] == "non_real_source_blocked"

    # a frontier_sim result can only ever end in a discard (DoD-B)
    assert _run(capsys, "phase", "set", "COMMIT_OR_ROLLBACK")[0] == 0
    rc, _ = _run(capsys, "discard", str(work), "--reason", "frontier_sim plumbing only")
    assert rc == 0


def test_verify_gain_and_keep_block_frontier(tmp_path, capsys):
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.tools.phase_guard import Phase
    bc = build_bench_config(runner_kind="strong_baseline", model="facebook/opt-125m", remote="box")
    ev = frontier_sim.run_frontier_sim(bc).eval_result
    p = tmp_path / "frontier.json"
    p.write_text(json.dumps(ev), encoding="utf-8")

    for ph in (Phase.READ_CONTEXT, Phase.DESIGN, Phase.GENERATE, Phase.VERIFY,
               Phase.VERIFY_PASSED, Phase.BENCHMARK, Phase.KEEP_OR_DISCARD):
        phase_guard.transition(ph)
    rc, vg = _run(capsys, "verify-gain", str(p), str(p), "--metric", "output_throughput_tok_s")
    assert rc == 2 and vg["outcome_class"] == "non_real_source_blocked" and vg["adopt"] is False

    phase_guard.transition(Phase.COMMIT_OR_ROLLBACK)
    policy = tmp_path / "w.py"
    policy.write_text("def schedule_batch(rs, st):\n    return None\n", encoding="utf-8")
    rc, kp = _run(capsys, "keep", str(policy), "--eval-result", str(p),
                  "--archive-root", str(tmp_path / "arch"))
    assert rc == 2 and kp["outcome_class"] == "non_real_source_blocked"


def test_frontier_eval_result_is_structurally_nonpromotable():
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.bench.eval_result import validate_eval_result
    from vllm_evolve.bench.outcome import OutcomeClass, is_success
    bc = build_bench_config(runner_kind="strong_baseline", model="facebook/opt-125m", remote="box")
    res = frontier_sim.run_frontier_sim(bc)
    ev = res.eval_result
    validate_eval_result(ev)
    assert ev["source"] == "frontier_sim" and ev["source"] != "real_vllm"
    assert ev["outcome_class"] == OutcomeClass.SIMULATOR_NONQUALIFYING.value
    assert is_success(OutcomeClass(ev["outcome_class"])) is False
    assert res.to_dict()["effective"] is False and res.to_dict()["quality_ok"] is None
    assert "vllm-evolve:" not in json.dumps(ev)   # never the plugin marker namespace


def test_frontier_sim_is_not_in_the_frozen_core_closure():
    # the new backend must never enter the frozen judgment closure (it is reached only from
    # cli/main.py + engine/profile.py, like local_smoke). Reuse the real frozen-core walker.
    import test_frozen_core as fc
    closure = fc._closure(fc._FROZEN_SEEDS)
    assert "bench/frontier_sim.py" not in closure, sorted(closure)
