"""R5 — deterministic adjudication (P3).

``adjudicate(prediction, per_arm_results) -> Verdict`` is a PURE FUNCTION. Its only inputs are a
structured ``Prediction`` and the per-arm cross-seed MEDIAN values produced by the shared catalog
evaluator (``ExperimentResult.per_arm``). There is NO agent channel: an untrusted producer cannot
pass free text, a pre-cooked verdict, or anything else that changes the outcome — the verdict is a
function of the numbers alone. No randomness. A missing required value, or a difference within the
prediction's ``margin`` band, yields ``inconclusive`` (honesty over a forced call).

engine layer; reuses ``experiment.metric_key`` so the lookup key matches exactly what the experiment
recorded. Not a frozen seed; imports no ``core``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from vllm_evolve.bench.frontier_catalog import MetricExpr
from vllm_evolve.engine.experiment import metric_key

SUPPORTED = "supported"
FALSIFIED = "falsified"
INCONCLUSIVE = "inconclusive"


@dataclass
class Prediction:
    """A machine-checkable prediction: ``value(arm_a) <comparator> value(arm_b)`` by at least
    ``margin``, where the value is the shared ``MetricExpr`` reduced per arm across seeds."""

    metric: dict                       # MetricExpr-shaped
    comparator: str                    # ">" or "<"
    arm_a: str
    arm_b: str
    margin: float | None = 0.0         # None == provided but non-numeric (malformed)

    @classmethod
    def from_obj(cls, obj) -> Prediction:
        """Total normalization — NEVER raises on producer-supplied data. A non-numeric ``margin``
        becomes ``None`` (a malformed marker), not an exception."""
        if isinstance(obj, Prediction):
            return obj
        m = obj or {}
        try:
            margin = float(m.get("margin", 0.0))
        except (TypeError, ValueError):
            margin = None
        return cls(metric=m.get("metric"), comparator=m.get("comparator"),
                   arm_a=m.get("arm_a"), arm_b=m.get("arm_b"), margin=margin)

    def is_well_formed(self) -> bool:
        # margin must be a NON-NEGATIVE number: a negative margin makes the within-margin band
        # (abs(diff) <= margin) unreachable, so a confident supported/falsified would be issued for
        # an arbitrarily tiny diff — malformed -> inconclusive instead.
        return (self.comparator in (">", "<") and bool(self.arm_a) and bool(self.arm_b)
                and self.margin is not None and self.margin >= 0
                and MetricExpr.from_obj(self.metric).is_valid())


@dataclass
class Verdict:
    verdict: str
    measured: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def adjudicate(prediction, per_arm_results: dict) -> Verdict:
    """Deterministically decide supported / falsified / inconclusive from ONLY the structured
    prediction + the per-arm median values. Inputs beyond these two arguments do not exist, so a
    producer cannot influence the call."""
    p = Prediction.from_obj(prediction)
    if not p.is_well_formed():
        return Verdict(INCONCLUSIVE, {},
                       f"malformed prediction (comparator={p.comparator!r}, margin={p.margin!r}, "
                       "metric/arms must be valid) -> inconclusive, never raised")

    key = metric_key(p.metric)
    a = (per_arm_results.get(p.arm_a) or {}).get(key)
    b = (per_arm_results.get(p.arm_b) or {}).get(key)
    measured = {"arm_a": p.arm_a, "arm_b": p.arm_b, "value_a": a, "value_b": b,
                "comparator": p.comparator, "margin": p.margin, "metric_key": key}

    if a is None or b is None:
        return Verdict(INCONCLUSIVE, measured,
                       "missing measurement(s) -> cannot adjudicate (never fabricated)")

    diff = a - b
    measured["diff"] = diff
    if abs(diff) <= p.margin:
        return Verdict(INCONCLUSIVE, measured,
                       f"|value_a - value_b|={abs(diff)} within margin {p.margin}")

    a_greater = diff > 0
    # comparator ">": supported when a exceeds b beyond margin; "<": supported when a is below b.
    is_supported = a_greater if p.comparator == ">" else (not a_greater)
    verdict = SUPPORTED if is_supported else FALSIFIED
    return Verdict(verdict, measured,
                   f"value_a={a} {p.comparator} value_b={b} by {abs(diff)} (margin {p.margin})")
