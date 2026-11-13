# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Import external correctness evidence without writing artifacts."""

import hashlib
import json
from pathlib import Path

from ..metrics.schema import EvidenceCapture


def load_correctness_artifact(path: Path) -> EvidenceCapture:
    """Load an external evaluator JSON artifact and expose its raw metadata."""

    if not path.is_file():
        return EvidenceCapture("correctness", None, False, "artifact missing", {})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return EvidenceCapture("correctness", path, False, str(exc), {})
    if not isinstance(payload, dict):
        return EvidenceCapture("correctness", path, False, "artifact must contain a JSON object", {})
    return EvidenceCapture("correctness", path, True, None, payload)


def collect_swebench_reports(output_dir: Path, run_id: str, model: str, tasks: list[dict]) -> list[dict]:
    """Collect SWE-bench 4.1 reports; never infer success from process exit."""
    from ..artifacts.correctness import TaskCorrectness

    collected = []
    for task in tasks:
        result = dict(task)
        if task["reason"] != "not_evaluated":
            collected.append(TaskCorrectness.model_validate(result).model_dump())
            continue
        iid = task["instance_id"]
        directory = output_dir / "raw" / "logs" / "run_evaluation" / run_id / model.replace("/", "__") / iid
        report_path = directory / "report.json"
        log_path = directory / "run_instance.log"
        if log_path.is_file():
            result["log_path"] = str(log_path.relative_to(output_dir))
        capture = load_correctness_artifact(report_path)
        try:
            if not capture.available:
                raise ValueError(capture.reason)
            if set(capture.metadata) != {iid}:
                raise ValueError("report identifies a different task")
            report = capture.metadata[iid]
            if not isinstance(report, dict) or type(report.get("resolved")) is not bool:
                raise ValueError("report has no boolean resolved result")
            if report.get("infra_failure_reason") is not None and not isinstance(report["infra_failure_reason"], str):
                raise ValueError("report has invalid infrastructure failure reason")
            # A report alone cannot identify the submitted answer.
            if hashlib.sha256((directory / "patch.diff").read_bytes()).hexdigest() != task["patch_sha256"]:
                raise ValueError("report patch does not match submitted patch")
            if not (directory / "test_output.txt").is_file():
                raise ValueError("report has no test output evidence")
            if report["resolved"] and (
                report.get("patch_successfully_applied") is not True or not isinstance(report.get("tests_status"), dict)
            ):
                raise ValueError("resolved report lacks test evidence")
            result.update(
                status="resolved" if report["resolved"] else "unresolved",
                reason=report.get("infra_failure_reason"),
                environment_failure=bool(report.get("infra_failure") or report.get("infra_failure_reason")),
                report_path=str(report_path.relative_to(output_dir)),
            )
        except (OSError, ValueError, TypeError) as exc:
            result.update(status="error", reason=f"invalid_or_missing_report: {exc}")
        collected.append(TaskCorrectness.model_validate(result).model_dump())
    return collected
