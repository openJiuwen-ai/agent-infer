"""L4 verify-gain — is a candidate a real, robust win?

Three gates, in order:
  1. **anti-fabrication** — an unverified candidate (plugin never confirmed loaded)
     is rejected outright; it can't be accepted on a run we can't trust.
  2. **bootstrap significance** — ``compare_metric`` over per-seed values: BETTER
     only when the CI lower bound clears ``epsilon_pct`` and variance is under the
     CV gate (else inconclusive / high-variance / worse).
  3. **holdout** — the candidate re-measured on a DIFFERENT config must not
     collapse below ``holdout_floor`` x baseline; if a BETTER candidate fails this,
     it is flagged ``overfit`` and rejected.

Then a re-profile compares the candidate's bottleneck to the baseline's: if it
moved, ``next_action`` says to loop back to L1 (the bottleneck shifted). The core
is a pure function over value lists so it is fully unit-tested off-GPU.
"""
from __future__ import annotations

import statistics

from vllm_evolve.bench.compare import compare_metric
from vllm_evolve.core.schemas import GainVerdict, Spec


def extract_values(obj: dict, metric: str) -> list[float]:
    """Pull per-seed metric values from an eval_result / Candidate / {"values": []}.

    Tries ``values`` -> per-seed ``metrics[metric]`` -> ``slo_result[metric]`` ->
    ``primary_value`` (only when ``primary_metric`` matches), so a normal
    eval_result whose headline lives in ``primary_value`` is not read as empty.
    """
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("values"), list):
        return [float(x) for x in obj["values"] if isinstance(x, (int, float))]
    primary_metric = obj.get("primary_metric")
    out: list[float] = []
    for s in obj.get("raw_per_seed_metrics") or []:
        if not isinstance(s, dict):
            continue
        md = s.get("metrics") if isinstance(s.get("metrics"), dict) else {}
        slo = s.get("slo_result") if isinstance(s.get("slo_result"), dict) else {}
        v = md.get(metric)
        if not isinstance(v, (int, float)):
            v = slo.get(metric)
        if not isinstance(v, (int, float)) and primary_metric == metric:
            v = s.get("primary_value")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _next_action(final: str, overfit: bool, shifted: bool | None,
                 cand_bottleneck: str | None) -> str:
    if final == "rejected_unverified":
        return ("candidate run was not marker-verified — re-run with the plugin "
                "confirmed loaded before it can be judged")
    if final == "needs_holdout":
        return ("bootstrap shows BETTER, but no holdout was run — re-measure on a "
                "different config (seed/concurrency) to confirm before adopting")
    if overfit:
        return ("overfit: holdout collapsed -> reject this candidate, go back to L2 "
                "for the next target")
    if final == "better":  # holdout confirmed
        if shifted:
            where = f" to {cand_bottleneck}" if cand_bottleneck else ""
            return f"adopt; bottleneck moved{where} -> re-profile and loop L1"
        return "adopt the candidate (provable, holds up on holdout)"
    if final == "high_variance_inconclusive":
        return "too noisy -> add seeds / stabilize the regime, then re-verify"
    return "no provable gain -> back to L2 for the next target"


def verify_gain(baseline_values: list[float], candidate_values: list[float], spec: Spec, *,
                candidate_verified: bool = True,
                holdout_baseline_values: list[float] | None = None,
                holdout_candidate_values: list[float] | None = None,
                holdout_floor: float = 0.95, baseline_bottleneck: str | None = None,
                candidate_bottleneck: str | None = None, epsilon_pct: float = 2.0,
                cv_threshold: float | None = 0.10) -> GainVerdict:
    higher = spec.direction != "min"

    if not candidate_verified:
        return GainVerdict(
            verdict="rejected_unverified",
            next_action=_next_action("rejected_unverified", False, None, None),
            detail="candidate Profile was not marker-verified",
        )
    if not baseline_values or not candidate_values:
        return GainVerdict(
            verdict="inconclusive",
            next_action="insufficient per-seed values to bootstrap -> collect more runs",
            detail=f"baseline n={len(baseline_values)}, candidate n={len(candidate_values)}",
        )

    cmp = compare_metric(spec.metric, baseline_values, candidate_values, higher,
                         epsilon_pct=epsilon_pct, cv_threshold=cv_threshold)
    verdict = cmp.verdict.value

    # Honest holdout: candidate vs baseline measured at the SAME *different*
    # operating point — proves the gain GENERALIZES, not just that the candidate
    # didn't collapse vs the original baseline.
    holdout_ok: bool | None = None
    if holdout_baseline_values and holdout_candidate_values:
        hb = statistics.median(holdout_baseline_values)
        hc = statistics.median(holdout_candidate_values)
        if higher:
            holdout_ok = hc >= hb * holdout_floor      # still >= baseline at the new point
        else:  # min metric: candidate must not be worse (larger) than baseline there
            holdout_ok = hb == 0 or hc <= hb / holdout_floor

    # A positive verdict requires BOTH bootstrap AND holdout. better-on-bootstrap
    # with no holdout is NOT adoptable -> needs_holdout; with a failed holdout it
    # is overfit -> worse. Only holdout_ok is True yields a final "better".
    overfit = (verdict == "better") and (holdout_ok is False)
    if verdict == "better":
        if holdout_ok is True:
            final = "better"
        elif holdout_ok is False:
            final = "worse"
        else:
            final = "needs_holdout"
    else:
        final = verdict

    shifted: bool | None = None
    if baseline_bottleneck and candidate_bottleneck:
        shifted = baseline_bottleneck != candidate_bottleneck

    return GainVerdict(
        verdict=final,
        improvement_pct=round(cmp.rel_improvement_pct, 3),
        ci_low_pct=round(cmp.ci_low_pct, 3), ci_high_pct=round(cmp.ci_high_pct, 3),
        candidate_cv=round(cmp.candidate_cv, 4),
        holdout_ok=holdout_ok, overfit=overfit, bottleneck_shifted=shifted,
        next_action=_next_action(final, overfit, shifted, candidate_bottleneck),
        detail=f"bootstrap={verdict}; holdout_ok={holdout_ok}; shifted={shifted}",
    )
