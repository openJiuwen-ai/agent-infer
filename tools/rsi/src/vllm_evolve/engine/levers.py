"""M-D2a — real throughput levers: apply to a BenchConfig + capability/disqualifier gate.

Quantization is an *artifact*, not a plain knob (Codex#7): fp8/awq weight quant
usually needs a quantized checkpoint and hardware/backend support; KV fp8 is a
separate switch. ``check_lever_applicable`` encodes the KernelToLeverMap
disqualifiers so the optimizer honestly skips a lever the model/GPU can't take,
instead of "searching" a no-op. The vanilla/strong baseline is always the same
model in fp16/bf16; only the candidate changes representation (and must pass the
quality gate, M-D3).
"""
from __future__ import annotations

from dataclasses import replace

# The executable lever set is single-sourced from BenchConfig (Codex R1): a knob not
# here would be a no-op search — the optimizer/selector refuse it.
from vllm_evolve.bench.config import WIRED_LEVERS, BenchConfig  # noqa: F401 (re-exported)


def apply_lever(cfg: BenchConfig, lever: str, value) -> BenchConfig:
    """Return a new BenchConfig with ``lever`` set to ``value`` on its EngineConfig."""
    if lever not in WIRED_LEVERS:
        raise ValueError(f"lever {lever!r} is not wired to the bench (would be a no-op)")
    if lever not in BenchConfig.__dataclass_fields__ and \
            lever not in cfg.engine.__dataclass_fields__:
        raise ValueError(f"lever {lever!r} is not an EngineConfig field")
    return replace(cfg, engine=replace(cfg.engine, **{lever: value}))


def check_lever_applicable(
    lever: str, value, *, model_artifact_kind: str = "hf-fp16",
    has_quant_checkpoint: bool = False, gpu_supports_fp8: bool = False,
    cuda_graph_already_on: bool = False, has_draft_model: bool = False,
) -> tuple[bool, str]:
    """Encode KernelToLeverMap disqualifiers. Returns (applicable, reason)."""
    if lever == "quantization":
        if value in ("fp8",) and not gpu_supports_fp8:
            return False, "fp8 weight quant: GPU/backend does not support fp8"
        if value in ("awq", "gptq") and not has_quant_checkpoint:
            return False, f"{value} weight quant needs a quantized checkpoint (artifact)"
        return True, "ok"
    if lever == "kv_cache_dtype":
        if value == "fp8" and not gpu_supports_fp8:
            return False, "fp8 KV cache: GPU/backend does not support fp8"
        return True, "ok"
    if lever == "enable_chunked_prefill" and value is True:
        # vLLM 0.14 defaults chunked-prefill on; enabling it is a likely no-op
        return False, ("chunked-prefill is default-on in vLLM 0.14 — "
                       "tune max_num_batched_tokens instead")
    if lever == "tensor_parallel_size" and value and value > 1:
        return True, "ok (multi-card; out of scope for single-card max-throughput)"
    if lever == "enforce_eager" and value is False:    # value False == enable cuda graph
        if cuda_graph_already_on:
            return False, "cuda graph already on (enforce_eager already False)"
        return True, "ok"
    return True, "ok"
