# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Offline aggregation against a caller-owned, frozen list of required checks."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import Category, FeedbackRecord, Verdict, require_text, validate_scope
from .taxonomy import TAXONOMY, Backend


def load_records(path: str | Path) -> list[FeedbackRecord]:
    """Load a JSON array, or a ``records`` envelope with explicit demo provenance."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    synthetic = False
    if isinstance(data, dict):
        if set(data) - {"records", "synthetic", "description", "scope", "required_checks"}:
            raise ValueError("unknown feedback envelope fields")
        synthetic = data.get("synthetic", False)
        if not isinstance(synthetic, bool):
            raise ValueError("synthetic must be a boolean")
        data = data.get("records")
    if not isinstance(data, list):
        raise ValueError("feedback JSON must contain a records array")
    records = []
    for item in data:
        if synthetic and isinstance(item, dict):
            item = {**item, "synthetic": True}
        records.append(FeedbackRecord.from_dict(item))
    return records


def evaluate_feedback(
    records: Sequence[FeedbackRecord],
    required_checks: Sequence[str],
    scope: Mapping[str, str],
) -> dict[str, Any]:
    """Aggregate validator evidence without probing hardware or inventing scores.

    ``scope`` requires candidate_id, baseline_id, backend, engine_version,
    model_revision, and workload_id. Records must exactly match the remaining
    scope dimensions. A check needs exactly one record; repeated trials must
    be combined by their validator before ingestion. This function evaluates
    check completeness and provenance, not authenticity or benchmark formulas.
    """
    if not isinstance(required_checks, Sequence) or isinstance(required_checks, (str, bytes)) or not required_checks:
        raise ValueError("required_checks must be a nonempty sequence")
    checks = tuple(require_text(check, "required check") for check in required_checks)
    if len(set(checks)) != len(checks):
        raise ValueError("required_checks must be unique")
    expected = validate_scope(scope)
    for key in ("candidate_id", "baseline_id", "backend"):
        require_text(expected.get(key), f"scope.{key}")
    Backend(expected["backend"])
    dimensions = {
        key: value for key, value in expected.items() if key not in ("candidate_id", "baseline_id", "backend")
    }
    outcomes = []
    for check in checks:
        matches = [record for record in records if record.check_id == check]
        reason = ""
        status = "INCONCLUSIVE"
        next_test = f"Collect one externally validated record for frozen check {check}."
        record = matches[0] if len(matches) == 1 else None
        if not matches:
            reason = "required evidence unavailable"
        elif len(matches) > 1:
            reason = "duplicate check records; aggregate trials explicitly before evaluation"
        elif record is not None:
            next_test = record.next_test
            if (record.candidate_id, record.baseline_id, record.backend.value) != (
                expected["candidate_id"],
                expected["baseline_id"],
                expected["backend"],
            ) or dict(record.scope) != dimensions:
                reason = "candidate, baseline, backend, version, model, or workload scope mismatch"
            elif record.verdict in (Verdict.UNAVAILABLE, Verdict.INCONCLUSIVE):
                reason = f"validator verdict is {record.verdict.value}"
            elif record.synthetic:
                reason = "synthetic/demo evidence cannot establish acceptance"
            elif record.category is Category.PERFORMANCE and record.diagnostic:
                reason = "profiled/diagnostic measurements cannot establish final performance acceptance"
                next_test = "Repeat the identical workload without profiling or diagnostic instrumentation."
            elif not record.evidence_refs:
                reason = "validator result has no evidence references"
            else:
                status = "PASS" if record.verdict is Verdict.PASS else "FAIL"
                reason = "external validator evidence accepted for this frozen check"
        outcomes.append(
            {
                "check_id": check,
                "status": status,
                "reason": reason,
                "next_test": next_test,
                "record": record.to_dict() if record is not None else None,
                "suggested_validators": list(TAXONOMY[record.layer].validators) if record is not None else [],
            }
        )
    statuses = {item["status"] for item in outcomes}
    overall = "FAIL" if "FAIL" in statuses else "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "PASS"
    return {
        "status": overall,
        "scope": expected,
        "required_checks": list(checks),
        "checks": outcomes,
        "ignored_check_ids": sorted({record.check_id for record in records} - set(checks)),
        "limitation": "Offline evidence aggregation only; referenced artifacts and validator honesty are not verified.",
    }
