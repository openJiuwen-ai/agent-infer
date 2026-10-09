"""Decision-policy roll-up: turn per-profile verdicts into one outcome.

The verdict on each benchmark profile (throughput / latency / replay) is
produced by :mod:`vllm_evolve.bench.compare`. This module applies a
**human-specified** decision policy to roll those up into a single
ACCEPT / REJECT / INCONCLUSIVE outcome. The policy is the place where the user
says what "better" means for their deployment — which profiles must improve and
which must not regress — so the engine never decides that for them.

Precedence (highest first):

1. Any ``forbid_regress`` profile is WORSE  -> REJECT.
2. Any decision-relevant profile is HIGH_VARIANCE_INCONCLUSIVE -> INCONCLUSIVE
   (we cannot trust the measurement enough to decide).
3. Any ``require_improve`` profile is WORSE -> REJECT.
4. Any ``require_improve`` profile is INCONCLUSIVE or missing -> INCONCLUSIVE.
5. Otherwise (all required improved, nothing forbidden regressed) -> ACCEPT.
"""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field

from vllm_evolve.bench.compare import Verdict


class Outcome(str, enum.Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


@dataclass
class DecisionPolicy:
    require_improve: list[str] = field(default_factory=list)
    forbid_regress: list[str] = field(default_factory=list)
    epsilon_pct: float = 2.0

    @classmethod
    def from_dict(cls, data: dict) -> DecisionPolicy:
        return cls(
            require_improve=list(data.get("require_improve", [])),
            forbid_regress=list(data.get("forbid_regress", [])),
            epsilon_pct=float(data.get("epsilon_pct", 2.0)),
        )


@dataclass
class DecisionResult:
    outcome: Outcome
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


def decide(
    profile_verdicts: dict[str, Verdict],
    policy: DecisionPolicy,
) -> DecisionResult:
    """Roll up per-profile verdicts into one outcome per the precedence above."""
    reasons: list[str] = []

    # 1. forbidden regression dominates everything.
    for p in policy.forbid_regress:
        if profile_verdicts.get(p) == Verdict.WORSE:
            reasons.append(f"{p}: regressed but is in forbid_regress -> reject")
            return DecisionResult(Outcome.REJECT, reasons)

    # 2. high variance in any decision-relevant profile -> cannot decide.
    relevant = list(dict.fromkeys([*policy.require_improve, *policy.forbid_regress]))
    for p in relevant:
        if profile_verdicts.get(p) == Verdict.HIGH_VARIANCE_INCONCLUSIVE:
            reasons.append(f"{p}: high-variance inconclusive -> cannot decide")
            return DecisionResult(Outcome.INCONCLUSIVE, reasons)

    # 3 & 4. required improvements.
    pending_inconclusive = False
    for p in policy.require_improve:
        v = profile_verdicts.get(p)
        if v == Verdict.BETTER:
            continue
        if v == Verdict.WORSE:
            reasons.append(f"{p}: required to improve but regressed -> reject")
            return DecisionResult(Outcome.REJECT, reasons)
        # INCONCLUSIVE or missing
        reasons.append(f"{p}: required to improve but verdict={v.value if v else 'missing'}")
        pending_inconclusive = True

    if pending_inconclusive:
        return DecisionResult(Outcome.INCONCLUSIVE, reasons)

    reasons.append("all required improvements met; no forbidden regressions")
    return DecisionResult(Outcome.ACCEPT, reasons)
