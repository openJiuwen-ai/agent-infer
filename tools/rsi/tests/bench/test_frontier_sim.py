"""A-F1: the out-of-process ``frontier_sim`` adapter is quarantined like ``local_smoke`` — it
can NEVER be a real gain/keep/AC6. Offline: the Frontier subprocess is stubbed by a fixture
(the goal is to validate the end-to-end CAPABILITY, not to actually run Frontier). Zero GPU.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench import frontier_sim  # noqa: E402
from vllm_evolve.bench.config import build_bench_config  # noqa: E402
from vllm_evolve.bench.eval_result import real_source_block, validate_eval_result  # noqa: E402

# A representative Frontier output bundle (real-shaped, latencies in MILLISECONDS like the real
# CSV): system_metrics.json + request_metrics.csv + the per-batch stage ledger (signal source).
_SYSTEM_METRICS = """{
  "simulation_metadata": {"total_requests": 4, "completed_requests": 4},
  "throughput_metrics": {"requests_per_second": 3.5, "tokens_per_second": 1280.0,
                         "total_duration_seconds": 1.14},
  "request_e2e_time_statistics": {"p50": 120.0, "unit": "ms"},
  "ttft_statistics": {"p50": 30.0, "unit": "ms"},
  "memory_utilization_percent": {"MONOLITHIC": 15.0},
  "preemption_statistics": {"total_preemption_events": 0, "total_preempted_requests": 0}
}"""
_REQUEST_METRICS_CSV = (
    "Request Id,ttft,request_e2e_time,tpot,"
    "request_inter_arrival_delay,request_first_scheduling_delay\n"
    "0,30.0,120.0,0.9,,0.0\n"
    "1,31.0,121.0,0.9,0.5,0.0\n"
    "2,29.0,119.0,0.9,0.5,0.0\n"
    "3,30.0,120.0,0.9,0.5,0.0\n"
)
# two busy windows over a 10s span -> engine_duty 0.3 (a genuinely idle engine)
_BATCH_LEDGER = (
    '{"batch_id": 0, "stage_start_ts": 0.0, "stage_end_ts": 2.0, "request_ids": [0, 1]}\n'
    '{"batch_id": 1, "stage_start_ts": 9.0, "stage_end_ts": 10.0, "request_ids": [2]}\n'
)


_MARKER = ('{"scheduler": "ve_policy", "policy_sha256": "stub-sha",'
           ' "invocations": 42, "fallbacks": 0}')


def _stub_invoke(write_files=True, marker=_MARKER):
    """Return a fake _invoke_frontier that drops the fixture bundle into out_dir, rc=0."""
    def _fake(argv, repo, out_dir, extra_env=None):
        if write_files:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "system_metrics.json").write_text(_SYSTEM_METRICS, encoding="utf-8")
            (d / "request_metrics.csv").write_text(_REQUEST_METRICS_CSV, encoding="utf-8")
            (d / "frontier_stage_batch_ledger.jsonl").write_text(_BATCH_LEDGER, encoding="utf-8")
            if marker is not None:
                (d / "ve_policy_marker.json").write_text(marker, encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return _fake


@pytest.fixture
def _frontier_env(monkeypatch):
    monkeypatch.setenv("VE_FRONTIER_REPO", "/fake/frontier")
    monkeypatch.setenv("VE_FRONTIER_PYTHON", "/fake/python")


def test_resolve_env_raises_when_unset(monkeypatch):
    monkeypatch.delenv("VE_FRONTIER_REPO", raising=False)
    monkeypatch.delenv("VE_FRONTIER_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="VE_FRONTIER_REPO"):
        frontier_sim.resolve_frontier_env()


def test_frontier_sim_result_is_quarantined_and_schema_valid(_frontier_env, monkeypatch):
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke())
    cfg = build_bench_config(model="facebook/opt-125m", profile="throughput",
                             n_requests=4, max_seeds=2)
    res = frontier_sim.run_frontier_sim(cfg)
    ev = res.eval_result

    # LOCK A / B
    assert ev["source"] == "frontier_sim"
    assert ev["outcome_class"] == "simulator_nonqualifying"
    # LOCK C on the result object
    d = res.to_dict()
    assert d["effective"] is False and d["quality_ok"] is None and d["marker_verified"] is False
    assert "FRONTIER_SIM" in ev["command_line"]  # no plugin-marker namespace

    # default primary metric is throughput → goodput stays None (honest), real tokens/s used
    assert ev["primary_metric"] == "output_throughput_tok_s"
    assert ev["goodput"] is None
    assert ev["raw_per_seed_metrics"] and all(
        p["source"] == "frontier_sim" for p in ev["raw_per_seed_metrics"])
    assert [p["primary_value"] for p in ev["raw_per_seed_metrics"]] == [1280.0, 1280.0]  # 2 seeds

    # schema-valid + LOCK D refuses it at the gate boundary
    validate_eval_result(ev)
    block = real_source_block(ev)
    assert block is not None and block["outcome_class"] == "non_real_source_blocked"


def test_goodput_path_uses_real_per_request_latencies(_frontier_env, monkeypatch):
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke())
    # ask for goodput with an SLO the fixture latencies all satisfy (e2e<=200ms, ttft<=50ms)
    cfg = build_bench_config(model="facebook/opt-125m", profile="throughput",
                             n_requests=4, max_seeds=1)
    cfg.statistical.primary_metric = "goodput_req_s"
    cfg.statistical.slo = {"e2e_ms": 200.0, "ttft_ms": 50.0}
    ev = frontier_sim.run_frontier_sim(cfg).eval_result
    assert ev["primary_metric"] == "goodput_req_s"
    # all 4 requests meet the SLO → goodput rate == requests_per_second (3.5), from real numbers.
    # The top-level goodput field is an object (schema), from the aggregate of per-seed values.
    assert ev["goodput"] == {"median_req_s": 3.5}
    assert ev["raw_per_seed_metrics"][0]["primary_value"] == 3.5
    validate_eval_result(ev)


def test_knobs_change_the_frontier_invocation(_frontier_env):
    # gap-2: searched knobs genuinely change the simulation — different values, different argv.
    def _argv(**kw):
        cfg = build_bench_config(model="facebook/opt-125m", profile="throughput", **kw)
        return frontier_sim.build_frontier_argv("py", cfg, out_dir="o", run_id="r", seed=0)

    a = _argv(max_num_seqs=128, concurrency=8)
    b = _argv(max_num_seqs=256, concurrency=64)
    assert a != b
    assert a[a.index("--vllm_v1_scheduler_config_batch_size_cap") + 1] == "128"
    assert b[b.index("--vllm_v1_scheduler_config_batch_size_cap") + 1] == "256"
    qps_flag = "--poisson_request_interval_generator_config_qps"
    assert a[a.index(qps_flag) + 1] == "8.0"      # concurrency -> open-loop qps approximation
    assert b[b.index(qps_flag) + 1] == "64.0"
    # arrival_rate_qps wins over concurrency when both are set
    c = _argv(arrival_rate_qps=2.5, concurrency=64)
    assert c[c.index(qps_flag) + 1] == "2.5"
    # max_num_batched_tokens maps to Frontier's max_tokens_in_batch
    d = _argv(max_num_batched_tokens=4096)
    assert d[d.index("--vllm_v1_scheduler_config_max_tokens_in_batch") + 1] == "4096"


def test_sim_signals_are_extracted_from_real_outputs(_frontier_env, monkeypatch):
    # gap-1: bottleneck signals come from Frontier's OWN files — memory util (system_metrics),
    # preemption stats, queue depth reconstructed from per-request columns, ledger engine duty.
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke())
    cfg = build_bench_config(model="facebook/opt-125m", profile="throughput",
                             n_requests=4, max_seeds=1)
    ev = frontier_sim.run_frontier_sim(cfg).eval_result
    sig = ev["sim_signals"]
    assert sig["kv_util"] == 0.15          # memory_utilization_percent / 100
    assert sig["preempt"] == 0             # preemption_statistics.total_preemption_events
    assert sig["waiting"] == 0             # all scheduling delays are 0 -> no standing queue
    assert sig["running"] == 2             # max request_ids per ledger batch
    assert sig["engine_duty"] == 0.3       # 3s busy over a 10s span


def test_policy_mode_selects_ve_policy_and_marker_guard_fires(_frontier_env, monkeypatch, tmp_path):
    # Phase D: a candidate WITH a policy file runs the ve_policy scheduler; a missing/dishonest
    # marker turns the trial into a FAILED eval_result (never scores baseline behavior).
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(*a):\n    return None\n", encoding="utf-8")
    cfg = build_bench_config(runner_kind="candidate", policy_path=str(pol),
                             model="facebook/opt-125m", profile="throughput", max_seeds=1)
    argv = frontier_sim.build_frontier_argv("py", cfg, out_dir="o", run_id="r", seed=0)
    assert argv[argv.index("--replica_scheduler_config_type") + 1] == "ve_policy"
    # baseline never selects ve_policy
    base = build_bench_config(runner_kind="strong_baseline", model="facebook/opt-125m")
    argv_b = frontier_sim.build_frontier_argv("py", base, out_dir="o", run_id="r", seed=0)
    assert argv_b[argv_b.index("--replica_scheduler_config_type") + 1] == "vllm_v1"

    # marker missing -> failed trial
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke(marker=None))
    ev = frontier_sim.run_frontier_sim(cfg).eval_result
    assert "marker violation" in (ev["error_text"] or "")
    # all-fallback marker -> failed trial
    bad = '{"policy_sha256": "x", "invocations": 5, "fallbacks": 5}'
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke(marker=bad))
    ev = frontier_sim.run_frontier_sim(cfg).eval_result
    assert "fell back on every invocation" in (ev["error_text"] or "")
    # honest marker -> sim_policy_sha surfaced (informational), locks untouched
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _stub_invoke())
    ev = frontier_sim.run_frontier_sim(cfg).eval_result
    assert ev["sim_policy_sha"] == "stub-sha" and ev["sim_marker"]["invocations"] == 42
    assert ev["source"] == "frontier_sim"
    assert ev["outcome_class"] == "simulator_nonqualifying"
    validate_eval_result(ev)


def test_clean_nonzero_exit_becomes_failed_eval_result_not_crash(_frontier_env, monkeypatch):
    def _fail(argv, repo, out_dir, extra_env=None):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="Frontier release guard error")
    monkeypatch.setattr(frontier_sim, "_invoke_frontier", _fail)
    cfg = build_bench_config(model="facebook/opt-125m", profile="throughput", n_requests=4)
    ev = frontier_sim.run_frontier_sim(cfg).eval_result
    assert ev["source"] == "frontier_sim"
    assert ev["outcome_class"] == "simulator_nonqualifying"
    assert "Frontier exit 1" in (ev["error_text"] or "")
    validate_eval_result(ev)
    assert real_source_block(ev) is not None  # still refused → DoD-B
