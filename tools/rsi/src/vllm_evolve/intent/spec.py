"""L0 intent — parse a natural-language goal into a formal :class:`Spec`.

v1 is deliberately a transparent keyword/regex parser (no LLM): it extracts the
objective metric, direction, a numeric target, and a few constraints (SLO / QPS /
concurrency). Anything it cannot read is left at an honest default and reported as
*inferred* via :func:`goal_notes` — it never invents a target or pretends a guess
was stated. The original text is always preserved in ``Spec.raw_intent``.
"""
from __future__ import annotations

import re

from vllm_evolve.core.schemas import Spec

# keyword -> canonical metric. Order matters (more specific first).
_METRIC_RULES: list[tuple[tuple[str, ...], str]] = [
    (("goodput",), "goodput_req_s"),
    (("ttft", "first token", "first-token", "首 token", "首token", "time to first"),
     "ttft_p99_ms"),
    (("tpot", "time per output", "per-token", "每 token", "每个 token", "decode latency"),
     "tpot_ms"),
    (("throughput", "吞吐", "tok/s", "tokens/s", "tokens per second"), "tok_s"),
    (("p99", "latency", "延迟", "时延"), "ttft_p99_ms"),   # generic latency -> TTFT p99
]

# Objective-direction verbs. NB: SLO phrases ("under/below/低于/不超过 <N>ms") are
# NOT here — they are constraints, parsed separately, and must not flip the
# objective direction (e.g. "maximize goodput under 500ms" is a MAX goal).
_MIN_KW = ("降低", "压到", "压低", "降到", "减少", "minimi", "reduce", "cut ", "lower",
           "decrease")
_MAX_KW = ("最大化", "提升", "提高", "maximi", "increase", "raise", "improve", "higher")
_LATENCY_METRICS = ("ttft_p99_ms", "tpot_ms")


def detect_metric(text: str) -> tuple[str, bool]:
    """Return (metric, found). ``found=False`` means the default was used."""
    t = text.lower()
    for kws, metric in _METRIC_RULES:
        if any(k in t for k in kws):
            return metric, True
    return "goodput_req_s", False                  # honest default headline metric


def detect_direction(text: str, metric: str) -> tuple[str, bool]:
    """Return (direction, explicit). Falls back to the metric's natural direction.

    When both a min and a max verb appear (e.g. "maximize goodput, cut latency"),
    trust the metric's natural sense rather than whichever keyword came first.
    """
    t = text.lower()
    natural = "min" if metric in _LATENCY_METRICS else "max"
    has_min = any(k in t for k in _MIN_KW)
    has_max = any(k in t for k in _MAX_KW)
    if has_min and not has_max:
        return "min", True
    if has_max and not has_min:
        return "max", True
    if has_min and has_max:
        return natural, True            # conflicting verbs -> metric's natural sense
    return natural, False


_NUM_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ms|毫秒|qps|req/?s|tok/?s|tokens/?s|s\b|秒)?", re.I)


def detect_target_and_constraints(text: str, metric: str) -> tuple[float | None, dict]:
    """Pull a numeric target (for latency metrics) + QPS/concurrency constraints."""
    target: float | None = None
    constraints: dict = {}
    for num, unit in _NUM_UNIT_RE.findall(text):
        val = float(num)
        u = (unit or "").lower()
        if u in ("ms", "毫秒"):
            if metric in _LATENCY_METRICS and target is None:
                target = val
        elif u in ("qps", "req/s", "reqs", "req/s"):
            constraints.setdefault("qps", val)
        elif u in ("s", "秒") and metric in _LATENCY_METRICS and target is None:
            target = val * 1000.0                   # seconds -> ms
    m = re.search(r"(并发|concurrency)\D{0,4}(\d+)", text, re.I)
    if m:
        constraints["concurrency"] = int(m.group(2))
    # SLO latency bound ("under/below/低于/不超过 <N>ms") -> a constraint, not the objective.
    slo = re.search(r"(?:under|below|低于|不超过|≤|<=|<)\s*(\d+(?:\.\d+)?)\s*(ms|毫秒|s|秒)",
                    text, re.I)
    if slo:
        val = float(slo.group(1))
        constraints["slo_latency_ms"] = val * 1000.0 if slo.group(2) in ("s", "秒") else val
    return target, constraints


def goal_notes(text: str) -> list[str]:
    """Honest notes about what had to be inferred (not stated)."""
    notes: list[str] = []
    metric, metric_found = detect_metric(text)
    if not metric_found:
        notes.append("metric not stated -> defaulted to goodput_req_s (verify this is "
                     "the objective you mean)")
    _, dir_explicit = detect_direction(text, metric)
    if not dir_explicit:
        notes.append("direction not stated -> inferred from the metric's natural sense")
    return notes


def parse_goal(text: str) -> Spec:
    metric, _ = detect_metric(text)
    direction, _ = detect_direction(text, metric)
    target, constraints = detect_target_and_constraints(text, metric)
    return Spec(metric=metric, target=target, direction=direction,
                constraints=constraints, raw_intent=text)
