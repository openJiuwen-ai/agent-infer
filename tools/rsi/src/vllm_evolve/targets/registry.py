"""Central registry of supported optimization targets.

Single source of truth for the single-target invariant (CLAUDE.md: only
``scheduling`` is wired to the real backend). CLI verbs, the ``modes`` layer, and
the intent router all consult this — no scattered per-verb checks. Pure; no I/O.
"""
from __future__ import annotations

SUPPORTED_TARGETS: tuple[str, ...] = ("scheduling",)
DEFAULT_TARGET = "scheduling"


def is_supported(target: str) -> bool:
    return target in SUPPORTED_TARGETS


def rejection_payload(target: str) -> dict | None:
    """Structured rejection for an unsupported target, else ``None``.

    A falsy target (``None`` / ``""``) is treated as "use the verb's default" and is
    NOT rejected here — callers pass the resolved target. Any explicit non-supported
    target is refused so a round verb never silently runs against ``scheduling``."""
    if target and target not in SUPPORTED_TARGETS:
        supported = ", ".join(repr(t) for t in SUPPORTED_TARGETS)
        return {
            "ok": False,
            "outcome_class": "unsupported_target",
            "target": target,
            "reason": (f"only {supported} is wired to the real backend; "
                       f"{target!r} is experimental / not runnable."),
        }
    return None
