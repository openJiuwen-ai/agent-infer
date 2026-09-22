"""M-D1 KernelToLeverMap — map a CUDA ``KernelBreakdown`` to allowed levers.

Driven by ``config/autopt/kernel_lever_map.yaml`` (concrete thresholds, no
hand-waving). ``match_levers`` is DETERMINISTIC: given a breakdown it returns the
exact matched rules + their allowed knobs / quality gates / disqualifiers. A
missing signal makes its rule NOT match (honest — never fabricated). This is the
executable core of "read CUDA profiling -> know what to change".
"""
from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_MAP = Path(__file__).resolve().parents[2].parent / "config" / "autopt" / \
    "kernel_lever_map.yaml"

# Recognized KernelBreakdown signals (0-1 fractions / utilizations). All Optional.
KERNEL_SIGNALS = (
    "gemm_time_frac", "tc_util", "dram_bw_util", "sm_occupancy",
    "launch_gap_frac", "attention_time_frac",
)


def load_lever_map(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else _DEFAULT_MAP
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return list(data.get("rules") or [])


def _check(op: str, value, threshold) -> bool:
    if value is None:
        return False                       # missing signal -> rule cannot fire (no guessing)
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == "in":
        return threshold[0] <= value <= threshold[1]
    return False


def match_levers(breakdown: dict, rules: list[dict] | None = None) -> list[dict]:
    """Return matched rules (deterministic) for a KernelBreakdown dict.

    Each result: {name, status, allowed_knobs, expected_effect, quality_gate,
    disqualifiers, matched_evidence}. A rule fires only if ALL its evidence
    conditions hold against present signals.
    """
    rules = rules if rules is not None else load_lever_map()
    out: list[dict] = []
    for r in rules:
        ev = r.get("evidence") or {}
        hits = {sig: breakdown.get(sig) for sig, c in ev.items()
                if _check(c["op"], breakdown.get(sig), c["threshold"])}
        if len(hits) == len(ev) and ev:
            out.append({
                "name": r["name"], "status": r.get("status", "suspected"),
                "allowed_knobs": list(r.get("allowed_knobs") or []),
                "expected_effect": r.get("expected_effect", ""),
                "quality_gate": list(r.get("quality_gate") or []),
                "disqualifiers": list(r.get("disqualifiers") or []),
                "matched_evidence": hits,
            })
    return out
