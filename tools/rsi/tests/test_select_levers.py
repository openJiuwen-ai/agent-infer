"""M-D2b: diagnosis-driven lever selection (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.select_levers import select_actionable_levers  # noqa: E402


def test_gemm_bound_selects_quantization_when_capable():
    bd = {"gemm_time_frac": 0.6, "tc_util": 0.3}
    caps = {"gpu_supports_fp8": True, "has_quant_checkpoint": True}
    sel = select_actionable_levers(bd, capabilities=caps)
    fp8 = next(s for s in sel if s["lever"] == "quantization" and s["value"] == "fp8")
    assert fp8["applicable"] is True and fp8["from_rule"] == "gemm_compute_bound"
    assert "perplexity_delta" in fp8["quality_gate"]


def test_disqualified_lever_kept_but_ranked_last():
    bd = {"gemm_time_frac": 0.6, "tc_util": 0.3}
    sel = select_actionable_levers(bd, capabilities={"gpu_supports_fp8": False,
                                                     "has_quant_checkpoint": False})
    # both fp8 and awq disqualified here -> all applicable=False, with reasons
    assert all(s["applicable"] is False for s in sel)
    assert all(s["reason"] for s in sel)


def test_applicable_ranked_before_disqualified():
    bd = {"gemm_time_frac": 0.6, "tc_util": 0.3}        # fp8 ok, awq needs ckpt
    sel = select_actionable_levers(bd, capabilities={"gpu_supports_fp8": True,
                                                     "has_quant_checkpoint": False})
    assert sel[0]["applicable"] is True                  # applicable first
    assert any(not s["applicable"] for s in sel)


def test_no_kernel_match_no_levers():
    assert select_actionable_levers({}) == []


def test_unwired_lever_marked_inapplicable():
    # attention_bound -> spec_decode -> speculative_decoding, which is NOT wired to the bench
    sel = select_actionable_levers({"attention_time_frac": 0.45})
    spec = [s for s in sel if s["lever"] == "speculative_decoding"]
    assert spec and all(s["applicable"] is False and "not wired" in s["reason"] for s in spec)


def test_launch_bound_maps_to_cuda_graph():
    sel = select_actionable_levers({"launch_gap_frac": 0.25})
    assert any(s["lever"] == "enforce_eager" and s["value"] is False for s in sel)
