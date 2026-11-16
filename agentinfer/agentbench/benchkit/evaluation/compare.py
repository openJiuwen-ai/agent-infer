# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Compare finalized, compatible correctness evidence by task ID."""

import json
from pathlib import Path
from typing import Any

from ..artifacts.correctness import EvaluationSummary, TaskCorrectness
from ..metrics.correctness import correctness_metrics
from .inputs import digest, json_digest, read_object

_COMPARISON_CONFIG_KEYS = ("max_workers", "timeout_seconds", "overall_timeout_seconds", "cleanup_timeout_seconds")


def _artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("artifact path escapes evaluation directory")
    return path


def load_evaluation(path: Path) -> tuple[EvaluationSummary, dict, dict[str, dict]]:
    path = path.resolve()
    summary = EvaluationSummary.model_validate_json(path.read_bytes())
    root = path.parent
    manifest = read_object(root / summary.manifest_path)
    if manifest.get("schema_version") != "1" or manifest.get("adapter_version") != "1":
        raise ValueError("unsupported evaluation schema/adapter")
    for field, expected_type in (
        ("config", dict),
        ("images", dict),
        ("errors", list),
        ("input_hashes", dict),
        ("output_hashes", dict),
    ):
        if not isinstance(manifest.get(field), expected_type):
            raise ValueError(f"invalid or missing evaluation manifest field: {field}")
    if missing := set(_COMPARISON_CONFIG_KEYS) - manifest["config"].keys():
        raise ValueError(f"missing evaluation config fields: {', '.join(sorted(missing))}")
    if any(not isinstance(value, dict) for value in manifest["images"].values()):
        raise ValueError("invalid evaluation manifest field: images entries must be objects")
    if manifest.get("status") not in {"completed", "failed"} or not manifest.get("finished_at"):
        raise ValueError("evaluation is not finalized")
    if (manifest["source_run_id"], manifest["evaluation_id"]) != (summary.source_run_id, summary.evaluation_id):
        raise ValueError("evaluation identity mismatch")
    inputs, outputs = manifest["input_hashes"], manifest["output_hashes"]
    if not {"correctness.json", "task_results.jsonl"} <= outputs.keys():
        raise ValueError("missing output hashes")
    if not {"inputs/dataset.json", "inputs/task_plan.json", "inputs/predictions.jsonl"} <= inputs.keys():
        raise ValueError("missing input hashes")
    for hashes in (inputs, outputs):
        for relative, expected in hashes.items():
            if digest(_artifact(root, relative).read_bytes()) != expected:
                raise ValueError(f"artifact hash mismatch: {relative}")
    ids = json.loads((root / "inputs/task_plan.json").read_bytes())
    if not ids or len(ids) != len(set(ids)) or ids != manifest["instance_ids"]:
        raise ValueError("task plan mismatch")
    dataset = json.loads((root / "inputs/dataset.json").read_bytes())
    if json_digest(sorted(dataset, key=lambda row: row["instance_id"])) != manifest["dataset_sha256"]:
        raise ValueError("dataset fingerprint mismatch")
    if sorted(row["instance_id"] for row in dataset) != sorted(ids):
        raise ValueError("dataset task set mismatch")
    results = [
        TaskCorrectness.model_validate_json(line).model_dump()
        for line in (root / summary.task_results_path).read_bytes().splitlines()
        if line.strip()
    ]
    metrics = correctness_metrics(results, ids)
    if metrics != {"counts": summary.counts.model_dump(), "resolved_rate": summary.resolved_rate}:
        raise ValueError("summary differs from per-task results")
    for result in results:
        if result["patch_sha256"] is not None:
            key = f"inputs/patches/{result['instance_id']}.patch"
            if inputs.get(key) != result["patch_sha256"]:
                raise ValueError("result patch fingerprint mismatch")
        for key in ("report_path", "log_path"):
            if result[key] is not None and result[key] not in outputs:
                raise ValueError("result references unhashed evidence")
        if result["status"] in {"resolved", "unresolved"}:
            if not result["report_path"] or not result["patch_sha256"]:
                raise ValueError("scored result lacks report or patch identity")
            directory = Path(result["report_path"]).parent
            patch = str(directory / "patch.diff")
            if not {patch, str(directory / "test_output.txt")} <= outputs.keys():
                raise ValueError("scored result lacks hashed patch or test output")
            if outputs[patch] != result["patch_sha256"]:
                raise ValueError("report patch does not match submitted patch")
    return summary, manifest, {r["instance_id"]: r for r in results}


def compare_correctness(baseline: Path, candidate: Path) -> dict:
    report: dict[str, Any] = {"status": "insufficient_evidence", "reasons": []}
    try:
        base, bm, bt = load_evaluation(baseline)
        cand, cm, ct = load_evaluation(candidate)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["reasons"].append(str(exc))
        return report
    report.update(baseline=base.model_dump(), candidate=cand.model_dump())
    if set(bt) != set(ct):
        report["reasons"].append("completed task sets differ")
    for key in ("dataset_sha256", "generation", "harness", "adapter_version"):
        if not bm.get(key) or not cm.get(key) or bm[key] != cm[key]:
            report["reasons"].append(f"incompatible or missing {key}")
    for key in _COMPARISON_CONFIG_KEYS:
        if bm["config"].get(key) != cm["config"].get(key):
            report["reasons"].append(f"evaluation configuration differs: {key}")
    for label, summary, manifest, tasks in (("baseline", base, bm, bt), ("candidate", cand, cm, ct)):
        if summary.counts.error or manifest["status"] != "completed" or manifest["errors"]:
            report["reasons"].append(f"{label} evaluation has errors")
        used = {iid for iid, task in tasks.items() if task["status"] != "empty_patch"}
        if any(not manifest["images"].get(iid, {}).get("id") for iid in used):
            report["reasons"].append(f"{label} image identity is incomplete")
        if any(t["environment_failure"] for t in tasks.values()):
            report["reasons"].append(f"{label} has environment failure evidence")
    for iid in set(bt) & set(ct):
        if bt[iid]["status"] != "empty_patch" and ct[iid]["status"] != "empty_patch":
            if bm["images"].get(iid) != cm["images"].get(iid):
                report["reasons"].append(f"image identity differs: {iid}")
    if report["reasons"]:
        return report
    regressions = sorted(i for i in bt if bt[i]["status"] == "resolved" and ct[i]["status"] != "resolved")
    improvements = sorted(i for i in bt if bt[i]["status"] != "resolved" and ct[i]["status"] == "resolved")
    report.update(
        status="failed" if regressions else "passed",
        regressions=regressions,
        improvements=improvements,
        unchanged_resolved=sorted(i for i in bt if bt[i]["status"] == ct[i]["status"] == "resolved"),
        unchanged_not_resolved=sorted(i for i in bt if bt[i]["status"] != "resolved" and ct[i]["status"] != "resolved"),
        delta_resolved_rate=cand.resolved_rate - base.resolved_rate,
    )
    return report
