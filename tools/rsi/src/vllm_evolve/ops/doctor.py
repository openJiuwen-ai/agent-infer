"""artifact-doctor — deterministic failure triage over an eval_result + log.

``diagnose`` is a pure function: it takes an eval_result dict (+ optional server
log + bundle completeness) and returns a structured report. It never guesses —
every finding carries a status:

* ``confirmed``  — directly supported by the data (e.g. the recorded
  ``outcome_class``, or a log marker that is present/absent),
* ``suspected``  — indirectly indicated (a heuristic on the metrics),
* ``unknown``    — the evidence needed to decide is absent (e.g. no serve.log).

Suggestions live in each finding's ``next_actions`` — they are recommendations,
never asserted as conclusions. ``limitations`` records what could not be checked
(a metrics-only bundle has no log, so log-dependent rules degrade to unknown).

``outcome_class`` is treated as a recorded fact; when re-deriving from raw
signals we defer to :func:`vllm_evolve.bench.outcome.resolve`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

_PLUGIN_MARKER = "vllm-evolve: scheduler plugin invoked"

# Per-outcome explanation + recommended next actions (suggestions, not verdicts).
_GUIDANCE: dict[str, tuple[str, list[str]]] = {
    "eval_result": ("clean successful evaluation", []),
    "plugin_load_failure": (
        "the scheduler plugin failed to load on the box",
        ["confirm --scheduler-cls uses the DOT form generated_scheduler.EvolvedScheduler",
         "run `ve verify <policy> scheduling` — the policy may not import",
         "read serve.log for the import traceback"],
    ),
    "vllm_crash": (
        "the vLLM server crashed during the run",
        ["read serve.log tail / eval_result.error_text",
         "try fewer concurrent seqs (--max-num-seqs) or a smaller model",
         "run `ar doctor --infra` to check GPU/driver health"],
    ),
    "bench_timeout": (
        "the benchmark exceeded its wall-time budget",
        ["raise the dispatch timeout",
         "reduce --n-requests / load",
         "check the box is not overloaded (ar doctor --infra)"],
    ),
    "hardware_unavailable": (
        "GPU/hardware was unavailable",
        ["run `ar doctor --infra --remote <box>`",
         "free the cards or check nvidia-smi for stuck processes"],
    ),
    "invalid_metrics": (
        "the measured metrics failed validation",
        ["inspect raw_per_seed_metrics",
         "verify the workload actually issued requests (num_completed > 0)"],
    ),
    "high_variance_inconclusive": (
        "the result is too noisy across seeds to call",
        ["add seeds (--max-seeds) to shrink the CI",
         "increase load (--max-num-seqs) so policies separate",
         "tighten the SLO, or switch to the latency profile"],
    ),
    "safety_rejection": (
        "the policy failed L1/L2 static checks (never reached the GPU)",
        ["run `ve verify <policy> scheduling` to see the exact issues",
         "remove forbidden patterns / fix the signature"],
    ),
    "l3_overfit_rejection": (
        "the policy looked overfit to one scenario",
        ["compare per-scenario fitness spread"],
    ),
    "unclassified_failure": (
        "the failure did not match a known class",
        ["read eval_result.error_text and serve.log"],
    ),
}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    status: str            # confirmed | suspected | unknown
    detail: str
    evidence_refs: list[str]
    next_actions: list[str]


def _evidence(er: dict, log: str | None) -> list[str]:
    """Neutral observed facts (no conclusions)."""
    ev: list[str] = []
    agg = er.get("aggregate_metrics") or {}
    if agg:
        ev.append(
            f"primary={er.get('primary_metric')} median={agg.get('median')} "
            f"cv={agg.get('cv')} n={agg.get('n')}"
        )
    if er.get("error_text"):
        ev.append(f"error_text present: {str(er['error_text'])[:120]}")
    per_seed = er.get("raw_per_seed_metrics") or []
    completed = [
        (s.get("metrics") or {}).get("num_completed")
        for s in per_seed
        if isinstance(s, dict)
    ]
    completed = [c for c in completed if isinstance(c, int)]
    if completed:
        ev.append(f"num_completed per seed: {completed}")
    if log is not None:
        ev.append(f"plugin_marker={'present' if _PLUGIN_MARKER in log else 'absent'}")
    return ev


def _completions(er: dict) -> list[int]:
    per_seed = er.get("raw_per_seed_metrics") or []
    return [
        (s.get("metrics") or {}).get("num_completed")
        for s in per_seed
        if isinstance(s, dict) and isinstance((s.get("metrics") or {}).get("num_completed"), int)
    ]


def _success_like(er: dict) -> bool:
    """Looks like a successful run: outcome_class eval_result, OR no outcome_class
    but a real median (a default-scheduler 'success' often has no class set)."""
    oc = er.get("outcome_class")
    if oc == "eval_result":
        return True
    return oc is None and (er.get("aggregate_metrics") or {}).get("median") is not None


def _rule_recorded_outcome(er: dict, log: str | None, recorded_only: bool) -> Finding:
    oc = er.get("outcome_class")
    if oc not in _GUIDANCE:
        return Finding(
            "unrecognized_outcome", "unknown",
            f"no recorded/known outcome_class (got {oc!r})", [],
            ["check the eval_result is a real bench artifact"],
        )
    detail, actions = _GUIDANCE[oc]
    # Recorded-only (synthesized from an artifact row, no real eval_result): we
    # have a label but nothing to corroborate it — never assert it as confirmed.
    if recorded_only:
        return Finding(
            oc, "unknown",
            f"{detail} — recorded outcome_class only; no eval_result to corroborate",
            ["outcome_class (recorded, uncorroborated)"], list(actions),
        )
    # Success is the dangerous claim (a forged eval_result must NOT read as clean):
    # only confirm it when corroborated by the log marker + non-zero completions.
    if oc == "eval_result":
        comps = _completions(er)
        if log is None:
            return Finding(
                oc, "suspected",
                "recorded eval_result but no serve.log to corroborate the plugin ran",
                ["outcome_class"],
                ["re-run with provenance (ve bench) to capture serve.log"],
            )
        if _PLUGIN_MARKER not in log or (comps and all(c == 0 for c in comps)):
            return Finding(
                oc, "suspected",
                "recorded eval_result but evidence contradicts it "
                "(see plugin_not_invoked / no_requests_completed)",
                ["outcome_class"], [],
            )
        return Finding(
            oc, "confirmed",
            "clean successful evaluation (plugin marker present, requests completed)",
            ["outcome_class", "serve.log: marker present"], [],
        )
    # Failure classes: trust the box's classification (conservative — it says it
    # failed) and explain it.
    return Finding(oc, "confirmed", f"{detail} (recorded outcome_class)",
                   ["outcome_class"], list(actions))


def _rule_plugin_marker(er: dict, log: str | None, completeness: str) -> Finding | None:
    """Was the candidate's plugin proven to run? (independent corroboration)."""
    if not _success_like(er):
        return None  # only meaningful for runs that claim to have succeeded
    if log is None:
        return Finding(
            "plugin_not_invoked", "unknown",
            "no serve.log available — cannot verify the plugin actually loaded",
            [], ["re-run via `ve bench` to capture serve.log + a complete bundle"],
        )
    if _PLUGIN_MARKER not in log:
        return Finding(
            "plugin_not_invoked", "confirmed",
            "serve.log lacks the plugin marker despite a success-like result — the "
            "reported metrics are the DEFAULT scheduler's, not the candidate's",
            ["serve.log: marker absent"],
            ["confirm --scheduler-cls is passed (DOT form) and the policy imports",
             "discard this result; it does not measure the candidate"],
        )
    return None  # marker present


def _rule_no_completions(er: dict) -> Finding | None:
    counts = _completions(er)
    if counts and all(c == 0 for c in counts):
        return Finding(
            "no_requests_completed", "suspected",
            "every seed completed 0 requests — the workload likely never ran",
            ["raw_per_seed_metrics: num_completed all 0"],
            ["check the server started and the client could connect",
             "read serve.log for startup errors"],
        )
    return None


def diagnose(
    eval_result: dict | None,
    log: str | None = None,
    completeness: str = "complete",
    recorded_only: bool = False,
) -> dict:
    """Return a structured triage report for one evaluation.

    ``recorded_only=True`` marks input synthesized from an artifact row (only the
    recorded outcome_class is known, no real eval_result) — confirmations are
    downgraded to ``unknown`` so a label can never read as a corroborated result.
    """
    er = eval_result or {}
    outcome_class = er.get("outcome_class") or "unknown_no_outcome"

    findings = [_rule_recorded_outcome(er, log, recorded_only)]
    plugin = _rule_plugin_marker(er, log, completeness)
    if plugin:
        findings.append(plugin)
    nocomp = _rule_no_completions(er)
    if nocomp:
        findings.append(nocomp)

    next_actions: list[str] = []
    for f in findings:
        for a in f.next_actions:
            if a not in next_actions:
                next_actions.append(a)

    limitations: list[str] = []
    if recorded_only:
        limitations.append(
            "recorded outcome_class only — no eval_result to corroborate"
        )
    if log is None:
        limitations.append("no serve.log — log-dependent rules ran as unknown")
    if not er:
        limitations.append("no eval_result available — only the outcome_class is known")

    return {
        "outcome_class": outcome_class,
        "completeness": completeness,
        "evidence": _evidence(er, log),
        "findings": [asdict(f) for f in findings],
        "next_actions": next_actions,
        "limitations": limitations,
    }
