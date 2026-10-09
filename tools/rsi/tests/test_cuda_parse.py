"""M-B4: CUDA chrome-trace parser -> KernelBreakdown (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.cuda_parse import (  # noqa: E402
    classify_kernel,
    parse_torch_trace,
)
from vllm_evolve.engine.kernel_map import match_levers  # noqa: E402


def _ev(name, dur, cat="kernel"):
    return {"ph": "X", "name": name, "dur": dur, "cat": cat}


def test_classify_kernel():
    assert classify_kernel("flash_fwd_kernel") == "attention"
    assert classify_kernel("cutlass_sgemm_128x128") == "gemm"
    assert classify_kernel("reshape_and_cache_kernel") == "kv"
    assert classify_kernel("ncclAllReduceKernel") == "comm"
    assert classify_kernel("elementwise_add") == "other"


def test_parse_trace_fractions_sum_to_one():
    trace = {"traceEvents": [
        _ev("cutlass_hgemm", 600), _ev("cutlass_hgemm", 400),   # 1000 gemm
        _ev("flash_fwd_kernel", 400),                            # 400 attention
        _ev("reshape_and_cache", 100),                           # 100 kv
        _ev("some_cpu_op", 9999, cat="cpu_op"),                  # ignored (not kernel)
    ]}
    bd = parse_torch_trace(trace)
    assert bd.gemm_time_frac == 0.6667 or abs(bd.gemm_time_frac - 1000 / 1500) < 1e-3
    assert abs(bd.attention_time_frac - 400 / 1500) < 1e-3
    assert abs(bd.kv_time_frac - 100 / 1500) < 1e-3
    assert bd.total_kernel_ms == 1.5    # 1500 us


def test_breakdown_feeds_kernel_lever_map():
    # a GEMM-dominated trace -> to_signals -> KernelToLeverMap fires gemm rule
    # (needs tc_util too, which a plain trace lacks -> add it as if from ncu)
    trace = {"traceEvents": [_ev("cutlass_hgemm", 600), _ev("flash_fwd", 100)]}
    bd = parse_torch_trace(trace)
    bd.tc_util = 0.30                     # supplied by ncu in a real Full run
    matched = match_levers(bd.to_signals())
    assert any(r["name"] == "gemm_compute_bound" for r in matched)


def test_empty_trace_degrades_honestly():
    bd = parse_torch_trace({"traceEvents": []})
    assert bd.gemm_time_frac is None and bd.to_signals() == {}


def test_only_kernel_events_counted():
    trace = {"traceEvents": [_ev("cutlass_gemm", 100),
                             {"ph": "X", "name": "launch", "dur": 9999, "cat": "cpu_op"}]}
    assert parse_torch_trace(trace).gemm_time_frac == 1.0
