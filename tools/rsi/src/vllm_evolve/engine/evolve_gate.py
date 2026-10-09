"""M-D3-full (gate) — decide whether evolving a CODE surface is the right move.

The hard-won lesson of this project: on a saturated single card the bottleneck is
usually compute/memory (lever = fp8 quantization / cuda graph), and evolving
``schedule_batch`` there moves nothing. Evolution of a code surface is warranted
ONLY when profiling points at scheduling/admission (free GPU capacity, a standing
queue, ordering matters). This gate refuses to "use evolve" on the wrong target —
it returns an honest reason to quantize-not-evolve instead.
"""
from __future__ import annotations

# Bottlenecks whose lever is a CONFIG knob, not a code change.
_CONFIG_LEVER_RULES = {"gemm_compute_bound", "kv_bandwidth_bound", "launch_overhead_bound"}


def should_evolve_code(matched_levers: list[dict] | None = None, *,
                       diagnosis_bottleneck: str | None = None) -> tuple[bool, str]:
    """Return (evolve_code, reason).

    True only when the diagnosis is scheduling/admission-bound. If profiling points
    at a config-lever bottleneck (compute/memory/launch), returns False with the
    config lever to use instead — never evolves ``schedule_batch`` just to run evolve.
    """
    rule_names = {m.get("from_rule") for m in (matched_levers or [])}
    has_config_bottleneck = bool(rule_names & _CONFIG_LEVER_RULES)

    if diagnosis_bottleneck == "scheduling_queue":
        # Conflicting evidence must NOT be overridden by the diagnosis string (Codex LOW#8):
        # if profiling also shows a compute/memory/launch bottleneck, resolve first.
        if has_config_bottleneck:
            return False, ("conflict: diagnosis says scheduling_queue but profiling shows a "
                           "compute/memory/launch bottleneck -> re-profile to resolve before "
                           "evolving code")
        return True, ("scheduling_queue: free capacity + admission order is the lever "
                      "-> evolving schedule_batch is justified")
    if has_config_bottleneck:
        return False, ("profiling points at compute/memory/launch -> the lever is a "
                       "config knob (quantization / fp8 KV / cuda graph), NOT "
                       "schedule_batch evolution")
    if not rule_names and diagnosis_bottleneck is None:
        return False, "no bottleneck localized -> profile more before evolving code"
    return False, "no code-surface (scheduling) bottleneck identified -> do not evolve code"
