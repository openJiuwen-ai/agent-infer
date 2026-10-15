"""Intent router (L3, UNTRUSTED): natural-language request -> a routing PROPOSAL.

Classifies a free-text request into one of the three top-level modes
(``autopt`` / ``tune`` / ``port``), resolves the optimization target (validated
against :mod:`vllm_evolve.targets.registry`), and parses the objective into a
:class:`Spec` (reusing :func:`vllm_evolve.intent.spec.parse_goal`).

This is a **PROPOSAL only** — it never measures, never judges, never runs a verb.
It has no ``Bash``/``Write`` surface and MUST NOT appear in the frozen
measure/judge closure (``tests/test_frozen_core.py`` forbids ``intent/``). A
human/CLI decides whether to act on the proposal; the modes layer enforces every
gate. ``research`` is NOT a mode — it is a step *inside* the ``autopt`` flow.
"""
from __future__ import annotations

import re
from dataclasses import asdict

from vllm_evolve.intent.spec import goal_notes, parse_goal
from vllm_evolve.targets import registry

MODES = ("autopt", "tune", "port")

# port: adapt/tune one algorithm across vLLM versions / hardware (debug + perf).
_PORT_KW = ("port", "版本", "version", "硬件", "hardware", "适配", "迁移", "migrate", "across ")
# tune: optimize/tune a GIVEN policy.
_TUNE_KW = ("tune", "调优", "调参", "tuning", "optimize this", "given policy", "指定 policy",
            "这个 policy", "该 policy")
_PY_PATH = re.compile(r"(?<!\w)([\w./\\-]+\.py)(?!\w)")


def _detect_policy_path(text: str) -> str | None:
    m = _PY_PATH.search(text)
    return m.group(1) if m else None


def detect_mode(text: str, *, has_policy: bool) -> str:
    """Pick a top-level mode. port keywords win (cross-version/hardware), then tune
    (explicit tuning verbs or a concrete policy file), else the default autopt flow."""
    t = text.lower()
    if any(k in t for k in _PORT_KW):
        return "port"
    if has_policy or any(k in t for k in _TUNE_KW):
        return "tune"
    return "autopt"


def route(text: str, *, target: str | None = None) -> dict:
    """Return a routing PROPOSAL dict for ``text``. Pure; no side effects.

    ``target`` defaults to the registry default; an explicit unsupported target is
    flagged (``target_supported=False`` + ``target_rejection``) rather than raised —
    the caller/mode decides. ``spec`` is the parsed objective; ``policy`` is a
    detected ``*.py`` path (used by tune/port) if any."""
    policy = _detect_policy_path(text)
    mode = detect_mode(text, has_policy=policy is not None)
    tgt = target or registry.DEFAULT_TARGET
    rejection = registry.rejection_payload(tgt)
    return {
        "mode": mode,
        "target": tgt,
        "target_supported": rejection is None,
        "target_rejection": rejection,
        "policy": policy,
        "spec": asdict(parse_goal(text)),
        "notes": list(goal_notes(text)),
        "raw_intent": text,
    }
