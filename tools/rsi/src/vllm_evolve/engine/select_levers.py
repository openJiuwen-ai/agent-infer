"""M-D2b — diagnosis-driven target selection: KernelBreakdown -> ranked levers.

Ties M-D1 (KernelToLeverMap: which bottleneck) to M-D2a (capability/disqualifier
gate): a profiled ``KernelBreakdown`` yields the matched bottleneck rules, whose
allowed knobs are expanded into concrete (lever, value) candidates, filtered by
hardware/artifact capability, and ranked confirmed-before-suspected /
applicable-before-disqualified. This is the Full-Autopt selection step (only runs
once profiling exists); the MVP path uses pre-registered candidates instead.
"""
from __future__ import annotations

from vllm_evolve.engine.kernel_map import match_levers
from vllm_evolve.engine.levers import WIRED_LEVERS, check_lever_applicable

# Map a KernelToLeverMap knob name -> a concrete (lever, value) on the BenchConfig.
# "_up" knobs are tuning bumps the optimizer will then search; cuda_graph turns OFF
# enforce_eager (cuda graphs require non-eager mode).
_KNOB_TO_LEVER: dict[str, tuple[str, object]] = {
    "quantization_fp8": ("quantization", "fp8"),
    "quantization_awq": ("quantization", "awq"),
    "kv_cache_dtype_fp8": ("kv_cache_dtype", "fp8"),
    "max_num_batched_tokens_up": ("max_num_batched_tokens", "search"),
    "cuda_graph": ("enforce_eager", False),
    "chunked_prefill_budget": ("max_num_batched_tokens", "search"),
    "spec_decode": ("speculative_decoding", "search"),
}


def select_actionable_levers(breakdown: dict, *, capabilities: dict | None = None) -> list[dict]:
    """Return ranked, capability-filtered (lever, value) candidates for a breakdown.

    Each item: {lever, value, from_rule, status, applicable, reason, quality_gate}.
    Disqualified candidates are KEPT (with applicable=False + reason) for an honest
    trace, but ranked last. Unmapped knobs are skipped (recorded nowhere = no fake
    action).
    """
    caps = capabilities or {}
    out: list[dict] = []
    seen: set[tuple] = set()
    for rule in match_levers(breakdown):
        for knob in rule["allowed_knobs"]:
            mapped = _KNOB_TO_LEVER.get(knob)
            if mapped is None:
                continue                       # unknown knob -> no fabricated action
            lever, value = mapped
            key = (lever, str(value), rule["name"])
            if key in seen:
                continue
            seen.add(key)
            if lever not in WIRED_LEVERS:
                # mapped but not executable by the bench -> honest applicable=False,
                # never presented as an actionable lever (Codex MED#7)
                ok, reason = False, f"lever {lever!r} not wired to the bench (not executable)"
            else:
                ok, reason = check_lever_applicable(lever, value, **caps)
            out.append({
                "lever": lever, "value": value, "from_rule": rule["name"],
                "status": rule["status"], "applicable": ok, "reason": reason,
                "quality_gate": rule["quality_gate"],
            })
    # rank: confirmed+applicable first, suspected/disqualified later (stable)
    out.sort(key=lambda x: (x["status"] != "confirmed", not x["applicable"]))
    return out
