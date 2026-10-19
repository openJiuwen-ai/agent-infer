"""Serialize a multi-seed ``BenchResult`` into the archived eval_result contract.

``runner.run_profile`` produces a ``BenchResult`` (per-seed ``primary_value`` +
aggregate + outcome). This turns it into the JSON that
``schemas/eval_result.schema.json`` defines and that ``ve compare`` consumes
(``raw_per_seed_metrics[].primary_value`` + ``primary_metric``). No fabrication:
every number comes from the BenchResult's real per-seed runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from vllm_evolve.bench.quality import QualityMeasurement, quality_ok
from vllm_evolve.bench.real_acceptance import evaluate_suite

if TYPE_CHECKING:
    from vllm_evolve.bench.runner import BenchResult

SCHEMA_VERSION = "1.0"
SCORE_AGG_FORMULA_VERSION = "median-v1"
REAL_SOURCE = "real_vllm"
REAL_ACCEPTANCE_OUTCOME = "real_acceptance_suite"
FORMAL_SCENARIOS = frozenset({
    "burstgpt_saturated",
    "burstgpt_high_pressure",
    "burstgpt_severe_pressure",
})
FORMAL_SEEDS = [0, 1, 2]


def acceptance_evidence_sha256(value: dict) -> str:
    """Canonical self-fingerprint for a formal-suite result (excluding the fingerprint field)."""
    payload = {key: item for key, item in value.items() if key != "evidence_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_acceptance_artifact_manifest(
    root: str | Path,
    *,
    eval_result_paths: list[str],
    quality_evidence_dirs: list[str],
) -> dict:
    """Hash the run-local raw files that a formal acceptance result references."""
    bundle_root = Path(root).resolve()
    if not bundle_root.is_dir():
        raise ValueError(f"acceptance artifact root does not exist: {bundle_root}")

    def relative(path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(bundle_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"acceptance artifact escapes immutable suite root: {resolved}"
            ) from exc

    eval_rel = [relative(path) for path in eval_result_paths]
    quality_rel = [relative(path) for path in quality_evidence_dirs]
    files = []
    for path in sorted(item for item in bundle_root.rglob("*") if item.is_file()):
        files.append({
            "path": path.relative_to(bundle_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        })
    return {
        "root": str(bundle_root),
        "eval_result_paths": eval_rel,
        "quality_evidence_dirs": quality_rel,
        "files": files,
    }


def real_source_block(ev: dict) -> dict | None:
    """LOCK D — a gain/keep boundary may consume ONLY a real-vLLM eval_result.

    Returns a ``non_real_source_blocked`` payload unless ``ev`` is BOTH ``source='real_vllm'`` AND
    ``outcome_class='eval_result'`` (a clean real run), else ``None``. Requiring both fields means a
    synthetic ``local_smoke`` artifact is refused, and so is a FORGED one that lies about its source
    (``source='real_vllm'`` with a non-``eval_result`` outcome_class) or vice-versa. This is the
    load-bearing guard that makes ``ve compare`` / ``verify-gain`` / ``keep`` / accept REFUSE
    instead of a verdict — so no synthetic run can ever become a gain / keep / AC6 result.
    """
    src = ev.get("source")
    oc = ev.get("outcome_class")
    if src != REAL_SOURCE or oc != "eval_result":
        return {
            "ok": False,
            "outcome_class": "non_real_source_blocked",
            "source": src,
            "observed_outcome_class": oc,
            "reason": (f"refusing a non-real result (source={src!r}, outcome_class={oc!r}); only a "
                       "clean real run (source='real_vllm' AND outcome_class='eval_result') may be "
                       "compared / verified / kept / accepted. Synthetic local_smoke runs are "
                       "plumbing only and can never be a gain/keep/AC6."),
        }
    return None


def real_adoption_block(
    ev: dict,
    acceptance_evidence: dict | None,
    *,
    policy_sha256: str,
    eval_result_path: str | Path | None = None,
) -> dict | None:
    """Fail closed unless a real eval is bound to a passing formal suite.

    ``real_source_block`` is the source-class quarantine.  Adoption needs a
    second, stronger boundary: a policy-hash-bound three-scenario result,
    including paired seeds, completion, execution, measured quality, and
    mechanism-control evidence. The recomputation stays inside the frozen
    ``bench`` judgment layer rather than trusting the orchestration layer.
    """
    source_block = real_source_block(ev)
    if source_block is not None:
        return source_block

    def blocked(reason: str) -> dict:
        return {
            "ok": False,
            "outcome_class": "real_acceptance_evidence_blocked",
            "source": ev.get("source"),
            "reason": reason,
        }

    suite = acceptance_evidence
    if not isinstance(suite, dict):
        return blocked(
            "non-manual keep requires --acceptance-evidence from the formal real-vLLM suite"
        )
    if suite.get("source") != REAL_SOURCE or suite.get("outcome_class") != REAL_ACCEPTANCE_OUTCOME:
        return blocked(
            "acceptance evidence must be source='real_vllm' and "
            "outcome_class='real_acceptance_suite'"
        )
    if suite.get("policy_sha256") != policy_sha256:
        return blocked("acceptance evidence policy SHA does not match the policy being kept")
    if ev.get("policy_sha256") != policy_sha256:
        return blocked("the primary eval_result policy SHA does not match the policy being kept")
    if suite.get("evidence_sha256") != acceptance_evidence_sha256(suite):
        return blocked("acceptance evidence canonical fingerprint is missing or mismatched")

    artifact_manifest = suite.get("artifact_manifest") or {}
    try:
        artifact_root = Path(artifact_manifest["root"]).resolve()
        entries = {
            row["path"]: row
            for row in artifact_manifest["files"]
            if isinstance(row, dict) and row.get("path")
        }
    except (KeyError, TypeError):
        return blocked("formal suite is missing its raw artifact hash manifest")
    if not artifact_root.is_dir() or not entries:
        return blocked("formal suite raw artifact bundle is missing or empty")
    for relative, entry in entries.items():
        path = (artifact_root / relative).resolve()
        try:
            path.relative_to(artifact_root)
        except ValueError:
            return blocked(f"artifact manifest path escapes suite root: {relative}")
        if not path.is_file():
            return blocked(f"artifact manifest file is missing: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            return blocked(f"artifact manifest SHA mismatch: {relative}")

    def verified_eval_file(path_value: str | Path | None, expected: dict, label: str) -> str | None:
        if not path_value:
            return f"{label} is missing its local eval_result path"
        path = Path(path_value).resolve()
        try:
            relative = path.relative_to(artifact_root).as_posix()
        except ValueError:
            return f"{label} eval_result escapes the formal suite root"
        if (
            relative not in entries
            or relative not in artifact_manifest.get("eval_result_paths", [])
        ):
            return f"{label} eval_result is not bound by the artifact manifest"
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return f"{label} eval_result file is unreadable"
        if observed != expected:
            return f"{label} embedded eval_result differs from its hashed raw file"
        return None

    main_file_error = verified_eval_file(eval_result_path, ev, "keep input")
    if main_file_error:
        return blocked(main_file_error)

    acceptance = suite.get("acceptance")
    if not isinstance(acceptance, dict):
        return blocked("acceptance evidence is missing the suite acceptance object")
    required_true = ("accepted", "performance_ok", "quality_ok", "mechanism_ok")
    failed = [key for key in required_true if acceptance.get(key) is not True]
    if failed:
        return blocked(f"formal acceptance gates did not pass: {failed}")
    thresholds = acceptance.get("thresholds") or {}
    if thresholds.get("paired_seeds") != FORMAL_SEEDS:
        return blocked("formal acceptance must use paired seeds [0, 1, 2]")

    rows = acceptance.get("rows") or []
    if {row.get("scenario") for row in rows} != FORMAL_SCENARIOS:
        return blocked("formal acceptance must contain exactly the three required scenarios")
    if not all(
        row.get("completion_ok") is True and row.get("execution_ok") is True
        for row in rows
    ):
        return blocked("formal acceptance contains incomplete or invalid execution evidence")

    mechanism = acceptance.get("mechanism_evidence") or {}
    if mechanism.get("required") is not True or mechanism.get("supported") is not True:
        return blocked("formal acceptance requires a supported mechanism-control ablation")
    control_rows = mechanism.get("rows") or []
    if {row.get("scenario") for row in control_rows} != FORMAL_SCENARIOS:
        return blocked("mechanism evidence must contain exactly the three required scenarios")
    if not all(
        row.get("completion_ok") is True and row.get("execution_ok") is True
        for row in control_rows
    ):
        return blocked("mechanism-control evidence contains incomplete or invalid execution")

    pairs = suite.get("pairs") or {}
    controls = suite.get("controls") or {}
    raw_pairs: dict[str, dict] = {}
    raw_controls: dict[str, dict] = {}
    control_sha256 = suite.get("control_sha256")
    if not isinstance(control_sha256, str) or len(control_sha256) != 64:
        return blocked("formal suite is missing the mechanism-control policy SHA")
    required_action_counters = suite.get("required_action_counters") or []
    if not required_action_counters or not all(
        isinstance(counter, str) and counter.strip()
        for counter in required_action_counters
    ):
        return blocked("formal suite is missing declared mechanism action counters")

    def action_counter_fired(result: dict) -> bool:
        provenance = result.get("plugin_provenance") or {}
        if result.get("mechanism_applicable") is False:
            return True
        return any(
            isinstance(value, (int, float))
            and value > 0
            and (key.endswith("_count") or key.endswith("_actions"))
            for key, value in provenance.items()
        )

    candidate_action_scenarios = 0
    for scenario in FORMAL_SCENARIOS:
        pair = pairs.get(scenario) or {}
        for role in ("baseline", "candidate"):
            run = pair.get(role) or {}
            result = run.get("eval_result") or {}
            file_error = verified_eval_file(
                run.get("local_eval_path"), result, f"{scenario}/{role}"
            )
            if file_error:
                return blocked(file_error)
            block = real_source_block(result)
            if block is not None:
                return blocked(f"{scenario}/{role} is not a clean real-vLLM eval_result")
            validity = result.get("workload_validity") or {}
            if not (
                validity.get("required") is True
                and validity.get("valid") is True
                and validity.get("verdict") == "valid_saturated_real_vllm"
            ):
                return blocked(f"{scenario}/{role} workload is not valid saturated real vLLM")
        candidate_result = (pair.get("candidate") or {}).get("eval_result") or {}
        control_result = (controls.get(scenario) or {}).get("eval_result") or {}
        if candidate_result.get("policy_sha256") != policy_sha256:
            return blocked(f"{scenario}/candidate policy SHA does not match the kept policy")
        if control_result.get("policy_sha256") != control_sha256:
            return blocked(f"{scenario}/mechanism-control policy SHA mismatch")
        raw_pairs[scenario] = {
            "baseline": (pair.get("baseline") or {}).get("eval_result") or {},
            "candidate": candidate_result,
        }
        raw_controls[scenario] = control_result
        for role, run in (
            ("candidate", pair.get("candidate") or {}),
            ("mechanism-control", controls.get(scenario) or {}),
        ):
            result = run.get("eval_result") or {}
            file_error = verified_eval_file(
                run.get("local_eval_path"), result, f"{scenario}/{role}"
            )
            if file_error:
                return blocked(file_error)
            if real_source_block(result) is not None:
                return blocked(f"{scenario}/{role} is not a clean real-vLLM eval_result")
            validity = result.get("workload_validity") or {}
            if not (
                validity.get("required") is True
                and validity.get("valid") is True
                and validity.get("verdict") == "valid_saturated_real_vllm"
            ):
                return blocked(f"{scenario}/{role} workload is not valid saturated real vLLM")
            provenance = result.get("plugin_provenance") or {}
            if result.get("marker_verified") is not True or provenance.get("fallback") is True:
                return blocked(f"{scenario}/{role} marker/fallback evidence is invalid")
            if not (
                result.get("effective") is True
                or result.get("mechanism_applicable") is False
            ):
                return blocked(f"{scenario}/{role} mechanism did not execute effectively")
            if not action_counter_fired(result):
                return blocked(f"{scenario}/{role} has no positive declared action counter")
            if role == "candidate" and result.get("mechanism_applicable") is not False:
                missing = [
                    counter
                    for counter in required_action_counters
                    if not isinstance(provenance.get(counter), (int, float))
                    or provenance.get(counter) <= 0
                ]
                if missing:
                    return blocked(
                        f"{scenario}/candidate declared action counters did not fire: {missing}"
                    )
                candidate_action_scenarios += 1

    if candidate_action_scenarios < 2:
        return blocked("declared candidate actions must fire in at least two formal scenarios")

    quality = suite.get("quality") or {}
    measured = quality.get("measured") or {}
    evidence_dirs = quality.get("evidence_dirs") or []
    bound_quality_dirs = set(artifact_manifest.get("quality_evidence_dirs") or [])
    quality_dir_files = []
    for path_value in evidence_dirs:
        path = Path(path_value).resolve()
        try:
            relative = path.relative_to(artifact_root).as_posix()
        except ValueError:
            return blocked("measured quality evidence escapes the formal suite root")
        if relative not in bound_quality_dirs or not path.is_dir():
            return blocked("measured quality evidence directory is not hash-bound")
        quality_dir_files.extend(
            name for name in entries if name == relative or name.startswith(relative + "/")
        )
    if not evidence_dirs or not quality_dir_files:
        return blocked("measured quality evidence directories are missing")
    try:
        quality_verdict = quality_ok(
            QualityMeasurement(**dict(measured["baseline"])),
            QualityMeasurement(**dict(measured["candidate"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return blocked(f"measured quality evidence is invalid: {exc}")
    if quality_verdict.ok is not True:
        return blocked(f"quality recomputation failed: {quality_verdict.reasons}")

    recomputed = evaluate_suite(
        raw_pairs,
        quality_ok=True,
        controls=raw_controls,
        require_mechanism_control=True,
        mechanism_name=str(suite.get("mechanism_name") or "candidate mechanism vs control"),
    )
    if recomputed.get("accepted") is not True:
        return blocked("raw formal-suite evidence fails recomputed acceptance gates")
    return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_schema() -> dict:
    """Load the eval_result JSON Schema shipped inside the package (no repo-root
    crawl, so it works under both editable and wheel installs)."""
    from importlib.resources import files

    text = (files("vllm_evolve.bench") / "schemas" / "eval_result.schema.json").read_text(
        encoding="utf-8"
    )
    return json.loads(text)


def build_eval_result(
    result: BenchResult,
    *,
    regime: str,
    slo: dict,
    vllm_version: str,
    wall_time_s: float,
    policy_sha256: str = "",
    config_sha256: str = "",
    git_sha: str = "",
    hardware_profile: str = "",
    command_line: str = "",
    cuda_version: str | None = None,
    gpu_driver_version: str | None = None,
    hardware_sku: str | None = None,
    warmup_policy_version: str | None = None,
    error_text: str | None = None,
) -> dict:
    """Map a ``BenchResult`` + run provenance to the eval_result dict."""
    seed_runs = result.seed_runs
    agg = result.primary_aggregate
    is_goodput = result.primary_metric.startswith("goodput")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "real_vllm",
        "policy_sha256": policy_sha256,
        "config_sha256": config_sha256,
        "git_sha": git_sha,
        "vllm_version": vllm_version,
        "vllm_commit": None,
        "cuda_version": cuda_version,
        "gpu_driver_version": gpu_driver_version,
        "hardware_profile": hardware_profile,
        "hardware_sku": hardware_sku,
        "docker_image_digest": None,
        "profile": result.profile,
        "regime": regime,
        "primary_metric": result.primary_metric,
        "seeds": list(result.seeds_used),
        "sample_count": len(seed_runs),
        "raw_per_seed_metrics": [sr.to_dict() for sr in seed_runs],
        "aggregate_metrics": {
            "primary_metric": result.primary_metric,
            "median": agg.median,
            "mean": agg.mean,
            "std": agg.std,
            "cv": agg.cv,
            "n": agg.n,
        },
        "slo": dict(slo or {}),
        "goodput": {"median_req_s": agg.median} if is_goodput else None,
        "outcome_class": result.outcome_class.value,
        "score_aggregation_formula_version": SCORE_AGG_FORMULA_VERSION,
        "warmup_policy_version": warmup_policy_version,
        "command_line": command_line,
        "wall_time_s": float(wall_time_s),
        "error_text": error_text,
    }


def validate_eval_result(d: dict) -> None:
    """Raise ``jsonschema.ValidationError`` if ``d`` violates the contract."""
    import jsonschema

    jsonschema.validate(d, load_schema())


def write_eval_result(d: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return p
