"""M-B1: layered BenchConfig + vanilla/candidate rendering (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.config import (  # noqa: E402
    CANDIDATE,
    VANILLA,
    BenchConfig,
    EngineConfig,
    EnvironmentConfig,
    RunnerConfig,
    same_caliber,
)


def _cfg(runner_kind, **engine):
    return BenchConfig(engine=EngineConfig(scheduler_cls="generated_scheduler.EvolvedScheduler",
                                           **engine),
                       runner=RunnerConfig(runner_kind=runner_kind))


def test_same_caliber_catches_environment_drift():
    # vanilla vs candidate on a DIFFERENT GPU/version is not a comparable A/B (Codex HIGH#2)
    v = BenchConfig(runner=RunnerConfig(runner_kind=VANILLA),
                    environment=EnvironmentConfig(gpus="0", vllm_version="0.14.0"))
    c = BenchConfig(runner=RunnerConfig(runner_kind=CANDIDATE),
                    environment=EnvironmentConfig(gpus="1", vllm_version="0.13.0"))
    ok, diffs = same_caliber(v, c)
    assert not ok and any("environment" in d for d in diffs)


def test_same_caliber_catches_remote_drift():
    v = BenchConfig(runner=RunnerConfig(runner_kind=VANILLA, remote="box-a"))
    c = BenchConfig(runner=RunnerConfig(runner_kind=CANDIDATE, remote="box-b"))
    ok, diffs = same_caliber(v, c)
    assert not ok and any("runner.remote" in d for d in diffs)


def test_vanilla_omits_scheduler_cls():
    args = _cfg(VANILLA).to_serve_args()
    assert "--scheduler-cls" not in args   # GAP-B: vanilla = vLLM default scheduler, no plugin


def test_candidate_attaches_scheduler_cls():
    args = _cfg(CANDIDATE).to_serve_args()
    assert "--scheduler-cls" in args
    assert "generated_scheduler.EvolvedScheduler" in args


def test_same_caliber_vanilla_vs_candidate_ok():
    # identical engine/workload, only runner differs -> A/B is fair (no knob drift)
    v = _cfg(VANILLA, gpu_memory_utilization=0.9, max_num_seqs=256)
    c = _cfg(CANDIDATE, gpu_memory_utilization=0.9, max_num_seqs=256)
    ok, diffs = same_caliber(v, c)
    assert ok and diffs == []


def test_same_caliber_catches_drift():
    v = _cfg(VANILLA, gpu_memory_utilization=0.9)
    c = _cfg(CANDIDATE, gpu_memory_utilization=0.85)  # accidental drift
    ok, diffs = same_caliber(v, c)
    assert not ok and any("gpu_memory_utilization" in d for d in diffs)


def test_same_caliber_exempts_the_searched_knob():
    # Codex review P1: an autopt config win changes the SEARCHED knob (e.g. max_num_seqs); that is
    # the thing under test, so exempting it must read as same caliber (else no config win can ever
    # adopt). Any OTHER engine/workload drift is still caught.
    base = _cfg(VANILLA, max_num_seqs=128)
    cand = _cfg(CANDIDATE, max_num_seqs=256)              # differs only by the searched knob
    assert same_caliber(base, cand)[0] is False                          # no exempt -> mismatch
    assert same_caliber(base, cand, exempt={"max_num_seqs"})[0] is True  # exempt -> same caliber
    # a non-searched drift is STILL a mismatch even when the searched knob is exempt
    base2 = _cfg(VANILLA, max_num_seqs=128, gpu_memory_utilization=0.9)
    cand2 = _cfg(CANDIDATE, max_num_seqs=256, gpu_memory_utilization=0.5)
    ok, diffs = same_caliber(base2, cand2, exempt={"max_num_seqs"})
    assert not ok and any("gpu_memory_utilization" in d for d in diffs)


def test_real_levers_render():
    args = BenchConfig(engine=EngineConfig(quantization="fp8", kv_cache_dtype="fp8",
                                           max_num_batched_tokens=4096,
                                           enable_chunked_prefill=True,
                                           async_scheduling=True)).to_serve_args()
    assert "--quantization" in args and "fp8" in args
    assert "--kv-cache-dtype" in args
    assert "--max-num-batched-tokens" in args and "4096" in args
    assert "--enable-chunked-prefill" in args
    assert "--async-scheduling" in args


def test_provenance_records_resolved_scheduler():
    v = _cfg(VANILLA)
    p = v.provenance()
    assert p["runner_kind"] == "vanilla" and p["scheduler_cls_requested"] is None
    assert "--scheduler-cls" not in p["rendered_serve_args"]
    c = _cfg(CANDIDATE)
    assert c.provenance()["scheduler_cls_requested"] == "generated_scheduler.EvolvedScheduler"


def test_roundtrip():
    c = _cfg(CANDIDATE, quantization="fp8")
    assert BenchConfig.from_dict(c.to_dict()).engine.quantization == "fp8"
