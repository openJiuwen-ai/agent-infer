"""L2 select-targets — bottleneck -> ranked candidate interventions.

The knowledge base, structured: each bottleneck maps to a list of candidate
targets {target, type(config|code|deploy|profile), leverage, cost, search_space?,
knowledge_refs, rationale}. ``select_targets`` ranks them by leverage minus cost
so cheap high-leverage CONFIG knobs come first and the expensive ``code`` /
``deploy`` targets come last — the session's hard lesson: don't evolve
``schedule_batch`` until profiling shows scheduling is actually the bottleneck.

A suspected/unknown diagnosis never gets a confident plan: it prepends a
``profile:add_signals`` step and a caveat (validate small, don't over-invest).
"""
from __future__ import annotations

from vllm_evolve.core.schemas import (
    COMM,
    COMPUTE,
    KV_CAPACITY,
    MEM_BANDWIDTH,
    PREFILL,
    SCHEDULING_QUEUE,
    UNDER_SATURATED,
    UNKNOWN,
    Diagnosis,
    TargetPlan,
)

_LEVERAGE = {"high": 3.0, "med": 2.0, "low_med": 1.5, "low": 1.0}
_COST = {"trivial": 0.0, "low": 0.3, "med": 0.6, "high": 1.0}

_ADD_SIGNALS = {
    "target": "profile:add_signals", "type": "profile", "leverage": "high",
    "cost": "trivial",
    "knowledge_refs": ["L1: missing duty-cycle/bandwidth/KV signals"],
    "rationale": "collect a fuller profile (GPU duty-cycle + bandwidth, KV util, "
                 "queue) before committing to a target",
}

# bottleneck -> candidate interventions (design §3 L2 table + session lessons).
_TARGETS: dict[str, list[dict]] = {
    COMPUTE: [
        {"target": "config:quantization", "type": "config", "leverage": "high",
         "cost": "low", "search_space": ["fp8", "awq", "gptq"],
         "knowledge_refs": ["compute-bound: cut FLOPs/bytes"],
         "rationale": "GPU FLOPS bound -> quantize to do less compute per token"},
        {"target": "config:speculative_decoding", "type": "config", "leverage": "med",
         "cost": "med", "knowledge_refs": ["draft model accepts multiple tokens/step"],
         "rationale": "amortize decode compute with a draft model"},
        {"target": "deploy:smaller_model", "type": "deploy", "leverage": "high",
         "cost": "high", "rationale": "a distilled/smaller model if quality allows"},
        {"target": "config:max_num_seqs", "type": "config", "leverage": "low",
         "cost": "trivial", "search_space": [64, 128, 256],
         "rationale": "larger batch amortizes weight load (only if under-batched)"},
    ],
    KV_CAPACITY: [
        {"target": "config:gpu_memory_utilization", "type": "config", "leverage": "high",
         "cost": "trivial", "search_space": [0.85, 0.90, 0.95],
         "knowledge_refs": ["KV pressure: more blocks"],
         "rationale": "raise KV headroom so decode doesn't preempt"},
        {"target": "config:max_num_batched_tokens", "type": "config", "leverage": "med",
         "cost": "trivial", "search_space": [2048, 4096, 8192],
         "rationale": "smaller per-step token budget -> finer prefill chunks -> smaller "
                      "transient KV spikes (chunked-prefill is already on by default)"},
        {"target": "config:max_model_len", "type": "config", "leverage": "med",
         "cost": "trivial", "rationale": "shorter context -> less KV per sequence"},
        {"target": "config:kv_cache_dtype", "type": "config", "leverage": "med",
         "cost": "low", "search_space": ["fp8"],
         "rationale": "fp8 KV halves KV bytes -> more sequences fit"},
        {"target": "deploy:tensor_parallel_size", "type": "deploy", "leverage": "high",
         "cost": "high", "search_space": [2, 4],
         "rationale": "shard weights -> free more KV per card"},
        {"target": "code:schedule_batch", "type": "code", "leverage": "low_med",
         "cost": "high",
         "rationale": "admission/eviction policy — ONLY if vLLM's KV-aware admission "
                      "proves insufficient (it usually is not)"},
    ],
    MEM_BANDWIDTH: [
        {"target": "config:quantization", "type": "config", "leverage": "high",
         "cost": "low", "search_space": ["fp8", "awq"],
         "rationale": "fewer bytes moved per token (weights + activations)"},
        {"target": "config:kv_cache_dtype", "type": "config", "leverage": "med",
         "cost": "low", "search_space": ["fp8"],
         "rationale": "fp8 KV halves KV byte movement during decode"},
        {"target": "config:max_num_seqs", "type": "config", "leverage": "med",
         "cost": "trivial", "search_space": [128, 256],
         "rationale": "larger batch amortizes weight reads across more tokens"},
    ],
    PREFILL: [
        {"target": "config:max_num_batched_tokens", "type": "config", "leverage": "high",
         "cost": "trivial", "search_space": [2048, 4096, 8192],
         "knowledge_refs": ["chunked-prefill is default-ON in vLLM 0.14; tune the budget"],
         "rationale": "smaller per-step token budget -> finer prefill chunks interleave "
                      "with decode -> lower TTFT (do NOT just 'enable' chunked-prefill; "
                      "it is already on)"},
        {"target": "config:enable_prefix_caching", "type": "config", "leverage": "med",
         "cost": "trivial",
         "rationale": "default-ON in vLLM 0.14 — only helps if the workload shares "
                      "prefixes; verify the prefix-cache hit rate before relying on it"},
        {"target": "deploy:prefill_decode_disagg", "type": "deploy", "leverage": "med",
         "cost": "high", "rationale": "disaggregate prefill from decode (heavy lift)"},
    ],
    SCHEDULING_QUEUE: [
        {"target": "config:max_num_seqs", "type": "config", "leverage": "med",
         "cost": "trivial", "search_space": [128, 256, 384],
         "rationale": "raise the running-batch cap while capacity is free"},
        {"target": "code:schedule_batch", "type": "code", "leverage": "med",
         "cost": "high",
         "knowledge_refs": ["this is the regime where admission/order genuinely matters"],
         "rationale": "evolve the admission/ordering policy — THIS is the one regime "
                      "where schedule_batch can move goodput"},
    ],
    UNDER_SATURATED: [
        {"target": "config:max_num_seqs", "type": "config", "leverage": "high",
         "cost": "trivial", "search_space": [128, 256, 512],
         "rationale": "raise concurrency cap to fill the idle GPU"},
        {"target": "config:concurrency", "type": "config", "leverage": "high",
         "cost": "trivial", "search_space": [128, 256, 512],
         "rationale": "drive more in-flight client requests"},
        {"target": "code:schedule_batch", "type": "code", "leverage": "low",
         "cost": "high", "rationale": "only if admission is too conservative"},
    ],
    COMM: [
        {"target": "deploy:tensor_parallel_size", "type": "deploy", "leverage": "high",
         "cost": "high", "search_space": [1, 2],
         "rationale": "right-size TP degree; lower it if comm dominates"},
        {"target": "deploy:pipeline_parallel_size", "type": "deploy", "leverage": "med",
         "cost": "high", "rationale": "pipeline instead of tensor parallel to cut all-reduce"},
    ],
    UNKNOWN: [],
}


def _rank(c: dict) -> float:
    return _LEVERAGE.get(c.get("leverage", "low"), 1.0) - _COST.get(c.get("cost", "high"), 1.0)


def select_targets(diagnosis: Diagnosis | dict) -> TargetPlan:
    if isinstance(diagnosis, dict):
        diagnosis = Diagnosis.from_dict(diagnosis)

    base = list(_TARGETS.get(diagnosis.bottleneck, []))
    note = ""

    # status=="unknown" means we don't trust the bottleneck at all (even if a label
    # is present) -> the only honest move is to gather more signal, never emit
    # concrete optimization candidates.
    if diagnosis.status == "unknown" or diagnosis.bottleneck == UNKNOWN or not base:
        return TargetPlan(
            bottleneck=diagnosis.bottleneck, status=diagnosis.status,
            candidates=[dict(_ADD_SIGNALS)],
            note="bottleneck not localized / diagnosis unknown -> profile more before "
                 "choosing a target",
        )

    candidates = sorted((dict(c) for c in base), key=_rank, reverse=True)

    if diagnosis.status != "confirmed":
        # a suspected diagnosis must not trigger an expensive commitment blindly
        candidates.insert(0, dict(_ADD_SIGNALS))
        note = (f"diagnosis is {diagnosis.status} — validate with the cheapest target "
                f"(or collect more signals) before any high-cost code/deploy change")

    return TargetPlan(bottleneck=diagnosis.bottleneck, status=diagnosis.status,
                      candidates=candidates, note=note)
