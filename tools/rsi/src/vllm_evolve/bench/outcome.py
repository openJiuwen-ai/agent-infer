"""Single-label outcome taxonomy with documented precedence (AC-9).

Every evaluation resolves to exactly one ``OutcomeClass``. When more than one
condition is true, the highest-precedence class wins (a safety rejection always
beats a crash, which beats a timeout, etc.). ``high_variance_inconclusive`` is a
first-class third state — never folded into "pass" (``eval_result``) or any
failure bucket. ``unclassified_failure`` is the explicit fallback for outcomes
that match nothing else; callers preserve the raw error text alongside it.
"""
from __future__ import annotations

import enum


class OutcomeClass(str, enum.Enum):
    SAFETY_REJECTION = "safety_rejection"
    PLUGIN_LOAD_FAILURE = "plugin_load_failure"
    # candidate plugin LOADED but never became effective (no reorder / fell back): the metrics
    # reflect the DEFAULT order, so it is NOT a promotable candidate result — a FAILURE class.
    PLUGIN_INEFFECTIVE = "plugin_ineffective"
    VLLM_CRASH = "vllm_crash"
    BENCH_TIMEOUT = "bench_timeout"
    INVALID_METRICS = "invalid_metrics"
    L3_OVERFIT_REJECTION = "l3_overfit_rejection"
    HARDWARE_UNAVAILABLE = "hardware_unavailable"
    HIGH_VARIANCE_INCONCLUSIVE = "high_variance_inconclusive"
    EVAL_RESULT = "eval_result"
    UNCLASSIFIED_FAILURE = "unclassified_failure"
    # Synthetic in-process `local_smoke` backend result — plumbing only, NEVER a real run.
    # A FAILURE class (so is_success() is False); stamped directly by the smoke backend, so it is
    # intentionally absent from PRECEDENCE (like UNCLASSIFIED_FAILURE).
    LOCAL_SMOKE_NONQUALIFYING = "local_smoke_nonqualifying"
    # Out-of-process Frontier-simulator backend result — non-real plumbing only, NEVER a real run.
    # A FAILURE class (is_success() False); like LOCAL_SMOKE_NONQUALIFYING it is stamped directly by
    # the backend and intentionally absent from PRECEDENCE.
    SIMULATOR_NONQUALIFYING = "simulator_nonqualifying"


# Highest priority first. ``UNCLASSIFIED_FAILURE`` is intentionally absent: it is
# the fallback returned when none of these are present.
PRECEDENCE: tuple[OutcomeClass, ...] = (
    OutcomeClass.SAFETY_REJECTION,
    OutcomeClass.PLUGIN_LOAD_FAILURE,
    OutcomeClass.VLLM_CRASH,
    OutcomeClass.BENCH_TIMEOUT,
    OutcomeClass.INVALID_METRICS,
    OutcomeClass.L3_OVERFIT_REJECTION,
    OutcomeClass.HARDWARE_UNAVAILABLE,
    OutcomeClass.HIGH_VARIANCE_INCONCLUSIVE,
    OutcomeClass.EVAL_RESULT,
)

# Neither a clean success nor "we can't tell" — the genuine failure buckets.
_NON_FAILURE = (OutcomeClass.EVAL_RESULT, OutcomeClass.HIGH_VARIANCE_INCONCLUSIVE)
FAILURE_CLASSES: frozenset[OutcomeClass] = frozenset(
    c for c in OutcomeClass if c not in _NON_FAILURE
)


def resolve(present: set[OutcomeClass]) -> OutcomeClass:
    """Return the single highest-precedence class among those present.

    An empty set, or a set containing only classes outside ``PRECEDENCE``,
    resolves to ``UNCLASSIFIED_FAILURE``.
    """
    for c in PRECEDENCE:
        if c in present:
            return c
    return OutcomeClass.UNCLASSIFIED_FAILURE


def is_success(outcome: OutcomeClass) -> bool:
    """Only ``eval_result`` is a clean success. High-variance is NOT a success."""
    return outcome is OutcomeClass.EVAL_RESULT
