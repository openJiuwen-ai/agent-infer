"""M-D1: KernelToLeverMap — KernelBreakdown -> deterministic lever set (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.kernel_map import load_lever_map, match_levers  # noqa: E402


def _names(matched):
    return {m["name"] for m in matched}


def _knobs(matched, name):
    return next(m["allowed_knobs"] for m in matched if m["name"] == name)


def test_loads_default_map():
    rules = load_lever_map()
    assert len(rules) >= 4 and all("evidence" in r and "allowed_knobs" in r for r in rules)


def test_gemm_compute_bound_maps_to_quantization():
    bd = {"gemm_time_frac": 0.60, "tc_util": 0.30}
    m = match_levers(bd)
    assert "gemm_compute_bound" in _names(m)
    knobs = _knobs(m, "gemm_compute_bound")
    assert "quantization_fp8" in knobs
    rule = next(r for r in m if r["name"] == "gemm_compute_bound")
    assert rule["status"] == "confirmed" and "no_quant_checkpoint" in rule["disqualifiers"]


def test_kv_bandwidth_bound_range_signal():
    assert "kv_bandwidth_bound" in _names(match_levers({"dram_bw_util": 0.85,
                                                        "sm_occupancy": 0.65}))
    # sm_occupancy outside [0.5,0.8] -> no match (range op honored)
    assert "kv_bandwidth_bound" not in _names(match_levers({"dram_bw_util": 0.85,
                                                            "sm_occupancy": 0.95}))


def test_launch_overhead_and_attention():
    assert "launch_overhead_bound" in _names(match_levers({"launch_gap_frac": 0.25}))
    att = match_levers({"attention_time_frac": 0.45})
    assert "attention_bound" in _names(att)
    assert next(r for r in att if r["name"] == "attention_bound")["status"] == "suspected"


def test_missing_signal_never_fabricates():
    # gemm rule needs BOTH gemm_time_frac AND tc_util; only one present -> no match
    assert match_levers({"gemm_time_frac": 0.60}) == []
    assert match_levers({}) == []


def test_deterministic():
    bd = {"gemm_time_frac": 0.6, "tc_util": 0.3, "launch_gap_frac": 0.25}
    assert match_levers(bd) == match_levers(bd)
