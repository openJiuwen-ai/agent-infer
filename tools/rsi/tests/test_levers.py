"""M-D2a: real levers + capability/disqualifier gate (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from vllm_evolve.bench.config import BenchConfig  # noqa: E402
from vllm_evolve.engine.levers import (  # noqa: E402
    WIRED_LEVERS,
    apply_lever,
    check_lever_applicable,
)


def test_real_levers_are_wired():
    for k in ("quantization", "kv_cache_dtype", "max_num_batched_tokens"):
        assert k in WIRED_LEVERS


def test_apply_lever_sets_engine_field():
    cfg = apply_lever(BenchConfig(), "quantization", "fp8")
    assert cfg.engine.quantization == "fp8"
    assert "--quantization" in cfg.to_serve_args()


def test_apply_lever_refuses_unwired():
    with pytest.raises(ValueError):
        apply_lever(BenchConfig(), "totally_made_up_knob", 1)


def test_quantization_is_an_artifact_not_a_knob():
    ok, why = check_lever_applicable("quantization", "fp8", gpu_supports_fp8=False)
    assert ok is False and "fp8" in why
    assert check_lever_applicable("quantization", "fp8", gpu_supports_fp8=True)[0] is True
    ok2, why2 = check_lever_applicable("quantization", "awq", has_quant_checkpoint=False)
    assert ok2 is False and "checkpoint" in why2


def test_kv_fp8_needs_support():
    assert check_lever_applicable("kv_cache_dtype", "fp8", gpu_supports_fp8=False)[0] is False
    assert check_lever_applicable("kv_cache_dtype", "fp8", gpu_supports_fp8=True)[0] is True


def test_enabling_default_on_chunked_prefill_is_disqualified():
    ok, why = check_lever_applicable("enable_chunked_prefill", True)
    assert ok is False and "default-on" in why
