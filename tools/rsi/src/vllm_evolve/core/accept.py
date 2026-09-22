"""M-D4 acceptance gate — candidate vs the STRONG baseline, no self-deception.

A candidate is accepted ONLY when ALL hold:
  1. it is **effective** (M-B2: really ran + changed scheduling, no fallback) —
     anti-fabrication, NOT a gain proof;
  2. **quality** did not regress (for model-representation changes like fp8);
  3. **paired bootstrap on per-seed deltas** vs the strong baseline clears a HARD
     threshold: ``point_gain ≥ X%`` AND ``CI_low ≥ X%`` (not just > epsilon);
  4. the **holdout** (a different operating point) clears the same gate — and the
     holdout is judged once (frozen), never tuned against.

All trials are recorded (no reporting only the winner -> no winner's curse). With
no holdout the verdict is NOT accepted (a positive result needs both gates).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from vllm_evolve.bench.compare import paired_ci, paired_relative_improvement

# The hard X% accept gate vs the strong baseline. A CONFIG/CLI default that lives in the FROZEN
# judgment core — the gate threshold is a human decision and is NEVER taken from a (possibly
# sub-agent-produced) Spec. Adoption paths (ve verify-gain, run_autopt) read it from here.
_DEFAULT_ACCEPT_THRESHOLD_PCT = 20.0


@dataclass
class AcceptVerdict:
    accepted: bool
    point_gain_pct: float
    ci_low_pct: float
    ci_high_pct: float
    holdout_ok: bool | None
    holdout_point_pct: float | None
    reasons: list = field(default_factory=list)
    trials: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def accept_vs_strong_baseline(
    baseline, candidate, *, threshold_pct: float = 20.0, higher_is_better: bool = True,
    candidate_effective: bool | None = None, quality_ok: bool | None = None,
    holdout_baseline=None, holdout_candidate=None, n_boot: int = 2000, seed: int = 0,
    candidate_meta: dict | None = None, baseline_meta: dict | None = None,
) -> AcceptVerdict:
    # Evidence must be EXPLICITLY proven True. Defaults are None (missing evidence),
    # which rejects — a caller that forgets to pass a real effective/quality verdict
    # cannot accidentally get an acceptance (Codex HIGH#1).
    reasons: list[str] = []

    # LOCK D at the accept boundary, INDEPENDENT of the effective/quality booleans: if eval-result
    # metadata (source/outcome_class) is supplied, anything that is not a clean real run
    # (source='real_vllm' AND outcome_class='eval_result') is HARD-REJECTED — so a synthetic
    # local_smoke candidate cannot be accepted even if its numbers/booleans looked acceptable.
    from vllm_evolve.bench.eval_result import real_source_block
    non_real_source = False
    for label, meta in (("candidate", candidate_meta), ("baseline", baseline_meta)):
        if meta is not None and real_source_block(meta) is not None:
            non_real_source = True
            reasons.append(
                f"{label} is not a real-vLLM eval_result (source={meta.get('source')!r}, "
                f"outcome_class={meta.get('outcome_class')!r}) — non_real_source_blocked, reject")

    if candidate_effective is not True:
        reasons.append("candidate effectiveness NOT proven (pass candidate_effective=True only "
                       "from a real effective-gate pass) — anti-fabrication, reject")
    if quality_ok is not True:
        reasons.append("quality NOT certified (pass quality_ok=True only from a real "
                       "quality-gate pass) — reject")

    # Paired stats need non-empty, equal-length samples. Empty (no metric values) or unequal seed
    # counts (e.g. tiers escalated differently) are an honest non-adoption verdict, NOT a crash:
    # paired_relative_improvement / paired_ci would raise ValueError and abort the JSON output that
    # ve verify-gain / ve autopt promise (Codex review P2).
    if not baseline or not candidate or len(baseline) != len(candidate):
        reasons.append(
            f"insufficient paired samples to compute a gain (baseline_n={len(baseline)}, "
            f"candidate_n={len(candidate)}) — reject")
        return AcceptVerdict(
            accepted=False, point_gain_pct=0.0, ci_low_pct=0.0, ci_high_pct=0.0,
            holdout_ok=None, holdout_point_pct=None, reasons=reasons,
            trials={"baseline_n": len(baseline), "candidate_n": len(candidate),
                    "threshold_pct": threshold_pct, "effective": candidate_effective,
                    "quality_ok": quality_ok})

    point = paired_relative_improvement(baseline, candidate, higher_is_better)
    lo, hi = paired_ci(baseline, candidate, higher_is_better, n_boot, seed)
    primary_pass = point >= threshold_pct and lo >= threshold_pct
    if not primary_pass:
        reasons.append(f"primary gate: point {point:.2f}% / CI_low {lo:.2f}% "
                       f"< threshold {threshold_pct}%")

    holdout_ok: bool | None = None
    holdout_point: float | None = None
    if holdout_baseline and holdout_candidate and len(holdout_baseline) == len(holdout_candidate):
        holdout_point = paired_relative_improvement(holdout_baseline, holdout_candidate,
                                                    higher_is_better)
        h_lo, _ = paired_ci(holdout_baseline, holdout_candidate, higher_is_better,
                            n_boot, seed + 1)
        holdout_ok = holdout_point >= threshold_pct and h_lo >= threshold_pct
        if not holdout_ok:
            reasons.append(f"holdout gate: point {holdout_point:.2f}% / CI_low {h_lo:.2f}% "
                           f"< threshold {threshold_pct}%")
    elif holdout_baseline and holdout_candidate:
        # provided but mismatched per-seed counts -> unusable holdout. Reject (holdout_ok stays
        # falsy) instead of letting the paired stats raise ValueError (Codex review P2).
        holdout_ok = False
        reasons.append(f"holdout samples mismatched (baseline_n={len(holdout_baseline)}, "
                       f"candidate_n={len(holdout_candidate)}) — unusable, reject")
    else:
        reasons.append("no holdout run — a positive verdict needs bootstrap AND holdout")

    accepted = bool(candidate_effective is True and quality_ok is True
                    and primary_pass and holdout_ok is True and not non_real_source)
    return AcceptVerdict(
        accepted=accepted, point_gain_pct=round(point, 3), ci_low_pct=round(lo, 3),
        ci_high_pct=round(hi, 3), holdout_ok=holdout_ok,
        holdout_point_pct=None if holdout_point is None else round(holdout_point, 3),
        reasons=reasons,
        trials={"baseline_n": len(baseline), "candidate_n": len(candidate),
                "threshold_pct": threshold_pct, "effective": candidate_effective,
                "quality_ok": quality_ok},
    )
