"""ar repro — assess/attempt reproduction of a past evaluation.

Two phases, both honest about what they cannot establish:

* ``assess_repro`` (``--plan``, local): from the recorded provenance, decide the
  best achievable state and list the gaps. It NEVER claims a result is
  reproducible when the policy source / config / run params / framework version
  are missing — it degrades to ``unreproducible`` with explicit gaps.
* ``compare_rerun`` (``--run``, after a real re-bench): compare the new median to
  the recorded one and classify exact / equivalent / drifted, distinguishing
  environment drift (versions changed) from statistical variance / regression.

States: ``exact_reproducible`` | ``equivalent_rerun`` | ``unreproducible`` |
``drifted``. Nothing is fabricated — missing provenance is reported, not filled.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable

EXACT = "exact_reproducible"
EQUIVALENT = "equivalent_rerun"
UNREPRODUCIBLE = "unreproducible"
DRIFTED = "drifted"


def default_git_check(sha: str) -> bool:
    """True iff ``sha`` resolves to a commit in the local repo."""
    if not sha:
        return False
    try:
        r = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def assess_repro(prov: dict, *, git_check: Callable[[str], bool] | None = None) -> dict:
    """Plan-level reproducibility assessment from a provenance dict.

    ``prov`` keys: policy_source_available (bool), git_sha (str), config_available
    (bool), profile, seeds, model, vllm_version, recorded_median.
    """
    git_check = git_check or default_git_check
    git_sha = prov.get("git_sha") or ""
    git_resolvable = bool(git_sha) and git_check(git_sha)

    available = {
        "policy_source": bool(prov.get("policy_source_available")),
        "config": bool(prov.get("config_available")),
        "run_params": bool(prov.get("profile")),
        "git_sha": bool(git_sha),
        "git_resolvable": git_resolvable,
    }

    gaps: list[str] = []
    if not available["policy_source"]:
        gaps.append("policy source not in store/bundle — cannot re-run the candidate")
    if not available["config"]:
        gaps.append("no config snapshot — cannot restore the requested configuration")
    if not available["run_params"]:
        gaps.append("no profile recorded — unknown which workload to run")
    if not git_sha:
        gaps.append("no git_sha — cannot pin the framework version")
    elif not git_resolvable:
        gaps.append(
            f"git_sha {git_sha[:12]} not present locally — cannot check out the "
            "exact framework version"
        )

    # State estimate. Missing code/config/params -> cannot reproduce at all.
    if not (available["policy_source"] and available["config"] and available["run_params"]):
        state = UNREPRODUCIBLE
    elif git_resolvable:
        state = EXACT          # everything pinnable; --run confirms exact vs drifted
    else:
        state = EQUIVALENT     # code+config present, framework version not pinnable

    return {
        "state_estimate": state,
        "available": available,
        "gaps": gaps,
        "recorded": {
            "median": prov.get("recorded_median"),
            "profile": prov.get("profile"),
            "seeds": prov.get("seeds"),
            "git_sha": git_sha,
            "vllm_version": prov.get("vllm_version"),
            "model": prov.get("model"),
        },
        "limitations": [
            "--plan estimates from recorded provenance; exact-vs-drifted is only "
            "decided by an actual --run re-bench"
        ],
    }


def compare_rerun(
    recorded_median: float | None,
    new_median: float | None,
    *,
    tolerance: float = 0.05,
    env_changed: bool = False,
    exact_possible: bool = True,
) -> tuple[str, str]:
    """Classify a re-run against the recorded median. Returns (state, note).

    ``exact_possible`` must be True for an ``exact_reproducible`` verdict — set it
    False when seeds / framework sha / env could not be pinned (so a matching
    number is at best ``equivalent``, never claimed as bit-exact).
    """
    if recorded_median is None or new_median is None:
        return UNREPRODUCIBLE, "missing median(s) to compare"
    denom = max(abs(recorded_median), 1e-9)
    rel = abs(new_median - recorded_median) / denom
    if rel <= tolerance:
        if exact_possible and not env_changed:
            return EXACT, f"within tolerance (rel diff {rel:.1%} <= {tolerance:.0%})"
        reason = (
            "environment changed (vllm/cuda/driver/model)" if env_changed
            else "seeds / framework version not pinned"
        )
        return EQUIVALENT, f"within {tolerance:.0%} but {reason} — equivalent, not bit-exact"
    note = "metrics drifted beyond tolerance; " + (
        "likely environment drift (recorded vs current versions differ)"
        if env_changed
        else "likely statistical variance or a real regression — add seeds to disambiguate"
    )
    return DRIFTED, note
