"""Migration helpers for eval_result contracts, vLLM compatibility, and policies.

Pure assessment functions (testable, no I/O of their own beyond what the caller
passes). The honesty rules carry over: missing provenance is *flagged*, never
filled with empty strings; vLLM migration only *reports* (no auto-edit); the
policy scan only *marks* incompatibilities (never rewrites protected files).
"""
from __future__ import annotations

from collections.abc import Callable

from vllm_evolve.bench.eval_result import SCHEMA_VERSION, validate_eval_result

_PROVENANCE_FIELDS = ("policy_sha256", "git_sha", "config_sha256")

# Future schema upgrades register transforms here ("0.9" -> fn). Empty for now:
# the only released contract is SCHEMA_VERSION, so older/unknown versions are
# reported as unmigratable rather than silently "upgraded".
_EVAL_TRANSFORMS: dict[str, Callable[[dict], dict]] = {}


def assess_eval_result(er: dict) -> dict:
    """Validate one eval_result dict against the current contract + flag gaps.

    status:
      ``ok_current``                 valid, current schema, full provenance
      ``partial_provenance``         valid + current but some provenance missing
      ``needs_upgrade``              older version with a known transform
      ``unmigratable_unknown_version`` older/unknown version, no transform
      ``invalid``                    fails the current schema
    """
    if not isinstance(er, dict):
        return {"status": "invalid", "schema_version": None,
                "validation_error": "not a JSON object"}

    sv = er.get("schema_version")
    missing = [f for f in _PROVENANCE_FIELDS if not er.get(f)]

    valid = True
    verr = None
    try:
        validate_eval_result(er)
    except Exception as exc:  # noqa: BLE001 - jsonschema.ValidationError etc.
        valid = False
        verr = str(exc).splitlines()[0][:200]

    if not valid:
        status = "invalid"
    elif sv == SCHEMA_VERSION:
        status = "ok_current" if not missing else "partial_provenance"
    elif sv in _EVAL_TRANSFORMS:
        status = "needs_upgrade"
    else:
        status = "unmigratable_unknown_version"

    return {
        "status": status,
        "schema_version": sv,
        "target_version": SCHEMA_VERSION,
        "valid": valid,
        "validation_error": verr,
        "missing_provenance": missing,
    }


def upgrade_eval_result(er: dict) -> tuple[dict | None, str]:
    """Return (upgraded_er | None, note). Never fabricates missing provenance."""
    sv = er.get("schema_version")
    if sv == SCHEMA_VERSION:
        return None, "already current"
    transform = _EVAL_TRANSFORMS.get(sv)
    if transform is None:
        return None, f"no transform for schema_version {sv!r}"
    return transform(dict(er)), f"upgraded {sv} -> {SCHEMA_VERSION}"


def vllm_compat_report(constraint: str, given_version: str | None = None) -> dict:
    """Static vLLM compatibility advisory (report only — never auto-edits)."""
    potential_breaks = [
        "--scheduler-cls must use the DOT form (generated_scheduler.EvolvedScheduler), "
        "not a colon",
        "V1 scheduler plugin contract: vllm.v1.core.sched.scheduler.Scheduler "
        "signature may change across versions (scheduler_api_drift)",
        "`vllm serve` flags (e.g. --scheduler-cls, --max-num-seqs) may be "
        "renamed/removed",
        "metrics endpoint / eval_result field names may shift",
    ]
    return {
        "constraint": constraint,
        "given_version": given_version,
        "satisfies_constraint": None if given_version is None
        else _satisfies(constraint, given_version),
        "potential_breaks": potential_breaks,
        "note": "advisory only; adapter/plugin-template changes are not automatic",
    }


def _parse_version(s: str) -> list[int] | None:
    """Parse a dotted numeric version (tolerant of a leading 'v'). None if not numeric."""
    s = (s or "").strip().lstrip("vV")
    if not s:
        return None
    try:
        return [int(p) for p in s.split(".")]
    except ValueError:
        return None  # non-numeric segment (e.g. rc/dev) -> undecidable


def _satisfies(constraint: str, version: str) -> bool | None:
    """Check a single ``>=X.Y.Z`` constraint, padding unequal lengths. None if unparseable."""
    c = constraint.strip()
    if not c.startswith(">="):
        return None
    want = _parse_version(c[2:])
    got = _parse_version(version)
    if want is None or got is None:
        return None
    n = max(len(want), len(got))
    want += [0] * (n - len(want))
    got += [0] * (n - len(got))
    return got >= want


def scan_policies(
    files: list,
    *,
    verify_fn: Callable[[str, str], list[str]],
    target: str = "scheduling",
) -> dict:
    """Scan policy files for skeleton/signature compatibility — mark, never edit."""
    incompatible = []
    scanned = 0
    for f in files:
        try:
            code = f.read_text(encoding="utf-8")
        except OSError:
            incompatible.append({"path": str(f), "issues": ["unreadable"]})
            continue
        scanned += 1
        issues = verify_fn(code, target)
        if issues:
            incompatible.append({"path": str(f), "issues": issues})
    return {
        "scanned": scanned,
        "compatible": scanned - len(incompatible),
        "incompatible": incompatible,
    }
