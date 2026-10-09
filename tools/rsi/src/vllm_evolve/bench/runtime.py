"""Pure helpers for the real-vLLM bench.

The actual serving run lives in ``bench.native`` (native ``vllm serve``, no
Docker). This module holds the GPU-free pieces shared by it and the tests:

* ``StreamTiming`` + ``record_from_timing`` — per-request stream timings -> RequestRecord
* ``PLUGIN_*_MARKER`` + ``assert_plugin_invoked`` / ``assert_plugin_effective``
  — confirm the scheduler plugin loaded (and actually reordered)
* ``classify_failure`` — map (exit code, server log) -> OutcomeClass
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm_evolve.bench.metrics import RequestRecord
from vllm_evolve.bench.outcome import OutcomeClass

# Markers + the dot-form scheduler qualname are single-sourced from the template.
try:
    from targets.scheduling.plugin_template import (
        PLUGIN_ACTIVE_MARKER,
        PLUGIN_DEFERRED_MARKER,
        PLUGIN_FALLBACK_MARKER,
        PLUGIN_FORCED_ADMIT_MARKER,
        PLUGIN_INACTIVE_MARKER,
        PLUGIN_INVOKED_MARKER,
        PLUGIN_POLICY_CALL_MARKER,
        PLUGIN_PREEMPTED_MARKER,
        PLUGIN_REORDERED_MARKER,
        SCHEDULER_QUALNAME,
    )
except Exception:  # pragma: no cover - targets path not importable in some contexts
    PLUGIN_INVOKED_MARKER = "vllm-evolve: scheduler plugin invoked"
    PLUGIN_REORDERED_MARKER = "vllm-evolve: waiting reordered"
    PLUGIN_FALLBACK_MARKER = "vllm-evolve: policy fallback"
    PLUGIN_ACTIVE_MARKER = "vllm-evolve: policy mechanism active"
    PLUGIN_INACTIVE_MARKER = "vllm-evolve: policy mechanism inactive"
    PLUGIN_POLICY_CALL_MARKER = "vllm-evolve: policy decision evaluated"
    PLUGIN_DEFERRED_MARKER = "vllm-evolve: waiting request deferred"
    PLUGIN_FORCED_ADMIT_MARKER = "vllm-evolve: starvation bound forced admission"
    PLUGIN_PREEMPTED_MARKER = "vllm-evolve: running request preempted"
    SCHEDULER_QUALNAME = "generated_scheduler.EvolvedScheduler"

# vLLM resolves --scheduler-cls via rsplit('.', 1): it MUST be a dotted qualname,
# never the colon (entry-point) form, which crashes at startup.
DEFAULT_SCHEDULER_CLS = SCHEDULER_QUALNAME


@dataclass
class StreamTiming:
    """Raw timing of one streamed request (seconds, monotonic clock)."""

    request_id: str
    submit_time_s: float
    first_token_time_s: float | None
    end_time_s: float | None
    num_prompt_tokens: int
    num_output_tokens: int
    success: bool = True
    error: str | None = None


def record_from_timing(t: StreamTiming) -> RequestRecord:
    """Convert raw stream timings into a RequestRecord with ms latencies."""
    ok = t.success and t.first_token_time_s is not None and t.end_time_s is not None
    if not ok:
        return RequestRecord(
            request_id=t.request_id,
            arrival_time_s=t.submit_time_s,
            num_prompt_tokens=t.num_prompt_tokens,
            num_output_tokens=t.num_output_tokens,
            success=False,
            error=t.error or "incomplete stream",
        )
    ttft_ms = (t.first_token_time_s - t.submit_time_s) * 1000.0
    e2e_ms = (t.end_time_s - t.submit_time_s) * 1000.0
    if t.num_output_tokens > 1:
        tpot_ms = (t.end_time_s - t.first_token_time_s) / (t.num_output_tokens - 1) * 1000.0
    else:
        tpot_ms = 0.0
    return RequestRecord(
        request_id=t.request_id,
        arrival_time_s=t.submit_time_s,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        num_prompt_tokens=t.num_prompt_tokens,
        num_output_tokens=t.num_output_tokens,
        success=True,
    )


def _marker_present(log: str, marker: str, nonce: str = "") -> bool:
    """Substring match for ``marker`` (optionally suffixed by the per-run ``nonce``).

    The wrapper emits ``"<marker> <nonce>"``; with a nonce the gate requires it, so a
    candidate that merely prints the static marker text cannot satisfy the gate (it
    can't know the per-run nonce). Substring (not line-anchored) survives vLLM's
    ``(EngineCore pid=...)`` log prefix.
    """
    needle = f"{marker} {nonce}" if nonce else marker
    return needle in (log or "")


def assert_plugin_invoked(container_log: str, nonce: str = "") -> bool:
    """True iff the server log shows the scheduler plugin loaded + ran (nonce-aware)."""
    return _marker_present(container_log or "", PLUGIN_INVOKED_MARKER, nonce)


def assert_plugin_effective(container_log: str, nonce: str = "") -> bool:
    """True iff the evolved policy actually changed admission AND never fell back.

    M-B2 (anti-fabrication, NOT a gain proof): a candidate is *effective* only when
    it loaded (``invoked``), genuinely changed the waiting order (``reordered``),
    and did NOT raise into the stock-ordering fallback (``fallback`` absent). With a
    per-run ``nonce`` the markers cannot be forged by printing static text. Even so,
    effectiveness says NOTHING about throughput — the unforgeable gain proof is the
    harness-measured paired A/B in M-D4, which a policy cannot fake.
    """
    log = container_log or ""
    return (_marker_present(log, PLUGIN_INVOKED_MARKER, nonce)
            and (
                _marker_present(log, PLUGIN_REORDERED_MARKER, nonce)
                or _marker_present(log, PLUGIN_PREEMPTED_MARKER, nonce)
                or _marker_present(log, PLUGIN_DEFERRED_MARKER, nonce)
            )
            and not _marker_present(log, PLUGIN_FALLBACK_MARKER, nonce))


def plugin_provenance(container_log: str, nonce: str = "") -> dict:
    """Structured candidate provenance for eval_result (M-B2), nonce-aware."""
    log = container_log or ""
    invoked = _marker_present(log, PLUGIN_INVOKED_MARKER, nonce)
    reordered = _marker_present(log, PLUGIN_REORDERED_MARKER, nonce)
    fallback = _marker_present(log, PLUGIN_FALLBACK_MARKER, nonce)
    active = _marker_present(log, PLUGIN_ACTIVE_MARKER, nonce)
    inactive = _marker_present(log, PLUGIN_INACTIVE_MARKER, nonce)
    preempted = _marker_present(log, PLUGIN_PREEMPTED_MARKER, nonce)
    deferred = _marker_present(log, PLUGIN_DEFERRED_MARKER, nonce)
    policy_call_needle = (
        f"{PLUGIN_POLICY_CALL_MARKER} {nonce}"
        if nonce else PLUGIN_POLICY_CALL_MARKER
    )
    reorder_needle = (
        f"{PLUGIN_REORDERED_MARKER} {nonce}"
        if nonce else PLUGIN_REORDERED_MARKER
    )
    deferred_needle = (
        f"{PLUGIN_DEFERRED_MARKER} {nonce}"
        if nonce else PLUGIN_DEFERRED_MARKER
    )
    forced_needle = (
        f"{PLUGIN_FORCED_ADMIT_MARKER} {nonce}"
        if nonce else PLUGIN_FORCED_ADMIT_MARKER
    )
    fb_needle = f"{PLUGIN_FALLBACK_MARKER} {nonce}" if nonce else PLUGIN_FALLBACK_MARKER
    preempt_needle = (
        f"{PLUGIN_PREEMPTED_MARKER} {nonce}"
        if nonce else PLUGIN_PREEMPTED_MARKER
    )
    return {
        "invoked": invoked,
        "policy_invocation_count": log.count(policy_call_needle),
        "reordered": reordered,
        "reorder_action_count": log.count(reorder_needle),
        "deferred": deferred,
        "deferred_request_actions": log.count(deferred_needle),
        "starvation_forced_admissions": log.count(forced_needle),
        "preempted": preempted,
        "preemption_count": log.count(preempt_needle),
        "fallback": fallback,
        "fallback_count": log.count(fb_needle),
        "effective": invoked and (reordered or preempted or deferred) and not fallback,
        "mechanism_applicable": (
            True if active else False if inactive else None
        ),
        "execution_valid": (
            invoked
            and not fallback
            and (reordered or preempted or deferred or active or inactive)
        ),
    }


def vanilla_provenance(container_log: str, *, model: str = "", vllm_version: str = "") -> dict:
    """Positive provenance for a VANILLA run (M-B2): prove it's real vLLM-default,
    NOT our plugin. Requires NO plugin marker; instead asserts the absence of any
    plugin marker / custom-scheduler load, plus the model/version it served.
    """
    log = container_log or ""
    plugin_seen = any(m in log for m in
                      (
                          PLUGIN_INVOKED_MARKER,
                          PLUGIN_REORDERED_MARKER,
                          PLUGIN_FALLBACK_MARKER,
                          PLUGIN_ACTIVE_MARKER,
                          PLUGIN_INACTIVE_MARKER,
                          PLUGIN_POLICY_CALL_MARKER,
                          PLUGIN_DEFERRED_MARKER,
                          PLUGIN_FORCED_ADMIT_MARKER,
                          PLUGIN_PREEMPTED_MARKER,
                      ))
    custom_sched = "using custom scheduler" in log.lower() or "--scheduler-cls" in log
    return {
        "runner_kind": "vanilla",
        "no_plugin_marker": not plugin_seen,
        "no_custom_scheduler": not custom_sched,
        "trusted_baseline": (not plugin_seen) and (not custom_sched),
        "model": model,
        "vllm_version": vllm_version,
    }


# Shared prefix of every provenance marker, and the reserved identifiers a candidate
# could use to reference/import/print a marker or the runtime nonce.
MARKER_NAMESPACE = "vllm-evolve:"
_RESERVED_IDENTIFIERS = (
    "PLUGIN_INVOKED_MARKER", "PLUGIN_REORDERED_MARKER", "PLUGIN_FALLBACK_MARKER",
    "PLUGIN_ACTIVE_MARKER", "PLUGIN_INACTIVE_MARKER",
    "PLUGIN_POLICY_CALL_MARKER", "PLUGIN_DEFERRED_MARKER",
    "PLUGIN_FORCED_ADMIT_MARKER",
    "PLUGIN_PREEMPTED_MARKER",
    "_PLUGIN_INVOKED_MARKER", "_PLUGIN_REORDERED_MARKER", "_PLUGIN_FALLBACK_MARKER",
    "_PLUGIN_ACTIVE_MARKER", "_PLUGIN_INACTIVE_MARKER",
    "_PLUGIN_POLICY_CALL_MARKER", "_PLUGIN_DEFERRED_MARKER",
    "_PLUGIN_FORCED_ADMIT_MARKER",
    "_PLUGIN_PREEMPTED_MARKER",
    "VE_MARKER_NONCE",
)


def contains_marker_forgery(policy_source: str) -> bool:
    """True if a candidate's source references the provenance-marker namespace at all.

    A scheduling policy has NO legitimate reason to mention the reserved marker text
    (the ``vllm-evolve:`` namespace), the marker constants, or the runtime nonce env
    var — so ANY reference is treated as a forgery attempt and rejected before render
    (Codex HIGH#3). This closes static printing, variable storage, string concatenation
    of the prefix, import-and-print, and nonce introspection. Combined with the per-run
    nonce on the real markers (which a static source cannot know), forging the effective
    gate is blocked. The ultimate backstop is that effectiveness is only an
    anti-fabrication signal: gain is proven by harness-measured throughput, unforgeable
    by the policy.
    """
    src = policy_source or ""
    if MARKER_NAMESPACE in src:
        return True
    return any(ident in src for ident in _RESERVED_IDENTIFIERS)


def classify_failure(exit_code: int, container_log: str) -> OutcomeClass | None:
    """Map a finished run to a failure class, or ``None`` on clean success.

    Precedence is most-specific-first; the runner/outcome layer applies the
    global precedence across classes.
    """
    log = (container_log or "").lower()
    hardware_markers = (
        "could not select device driver",
        "no cuda-capable device",
        "failed to initialize nvml",
        "no gpu",
        "nvidia-container-cli",
    )
    plugin_markers = (
        "no module named 'generated_scheduler'",
        "no module named \"generated_scheduler\"",
        "evolvedscheduler",
        "scheduler_cls",
        "--scheduler-cls",
        "failed to import scheduler",
    )
    if any(m in log for m in hardware_markers):
        return OutcomeClass.HARDWARE_UNAVAILABLE
    if exit_code == 124:  # GNU timeout / wall-clock kill
        return OutcomeClass.BENCH_TIMEOUT
    if any(m in log for m in plugin_markers):
        return OutcomeClass.PLUGIN_LOAD_FAILURE
    if exit_code != 0:
        return OutcomeClass.VLLM_CRASH
    return None
