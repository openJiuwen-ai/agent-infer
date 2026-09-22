# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Append-only experiment evidence for RSI's reported outer loop.

The ledger records measurements and references. It never executes commands or
decides acceptance; the independent benchmark artifacts remain authoritative.
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

METRICS = (
    "throughput_output_tokens_per_s",
    "throughput_input_tokens_per_s",
    "throughput_requests_per_s",
    "ttft_p50_ms",
    "tpot_p50_ms",
    "gsm8k_accuracy",
    "readiness_seconds",
)
STATUSES = ("measured", "failed", "blocked", "not_run")
_FIELDS = {
    "round_id",
    "timestamp",
    "hypothesis",
    "change",
    "model",
    "backend",
    "component_version",
    "config",
    "commands",
    "metrics",
    "status",
    "evidence",
    "failure_reason",
    "next_test",
}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_METRIC_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("Nonfinite JSON values are not allowed")


def load_experiment_json(text: str) -> dict[str, object]:
    """Read strict JSON without duplicate keys or NaN/Infinity."""

    value = json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Experiment must be a JSON object")
    return value


def validate_experiment(record: object) -> dict[str, object]:
    """Validate one record and fill standard missing metrics with null."""

    try:
        value = load_experiment_json(json.dumps(record, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("Record must contain finite JSON values") from error
    _require(_FIELDS <= value.keys(), "Experiment is missing required fields")
    _require(not (value.keys() - _FIELDS - {"baseline_round_id"}), "Unknown experiment fields")
    for key in ("round_id", "timestamp", "hypothesis", "change", "model", "backend", "component_version", "status"):
        _require(_text(value[key]), f"{key} must be a nonempty string")
    _require(bool(_ID.fullmatch(str(value["round_id"]))), "Invalid round_id")
    baseline = value.get("baseline_round_id")
    _require(
        baseline is None or (isinstance(baseline, str) and bool(_ID.fullmatch(baseline))), "Invalid baseline_round_id"
    )
    _require(baseline != value["round_id"], "baseline_round_id must reference a different round")
    try:
        timestamp = datetime.fromisoformat(str(value["timestamp"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be an ISO 8601 datetime with timezone") from error
    _require(timestamp.utcoffset() is not None, "timestamp requires a timezone")
    _require(value["status"] in STATUSES, "Invalid experiment status")
    _require(isinstance(value["config"], dict), "config must be a JSON object")
    precision = value["config"].get("precision")
    _require(precision is None or _text(precision), "config.precision must be a string or null")
    _require(
        isinstance(value["commands"], list) and all(_text(item) for item in value["commands"]),
        "commands must be an array of nonempty strings",
    )
    for key in ("failure_reason", "next_test"):
        _require(value[key] is None or _text(value[key]), f"{key} must be a nonempty string or null")
    if value["status"] != "measured":
        _require(_text(value["failure_reason"]), "Non-measured records require failure_reason")

    metrics = value["metrics"]
    _require(isinstance(metrics, dict), "metrics must be a JSON object")
    for key, metric in metrics.items():
        _require(bool(_METRIC_KEY.fullmatch(str(key))), "Invalid metric key")
        if metric is None:
            continue
        _require(type(metric) in (int, float) and math.isfinite(metric), "Metrics must be finite numbers or null")
        if key in METRICS or key.endswith("_tokens_per_s") or key.endswith("_requests_per_s"):
            _require(metric >= 0, f"{key} must be nonnegative")
        if key == "gsm8k_accuracy":
            _require(0 <= metric <= 1, "gsm8k_accuracy must be a fraction between 0 and 1")
    for key in METRICS:
        metrics.setdefault(key, None)

    _require(isinstance(value["evidence"], list), "evidence must be an array of path objects")
    for evidence in value["evidence"]:
        _require(isinstance(evidence, dict), "Evidence must be an object with path and optional sha256")
        _require(not (evidence.keys() - {"path", "sha256"}), "Unknown evidence fields")
        _require(_text(evidence.get("path")), "Evidence path must be a nonempty string")
        digest = evidence.get("sha256")
        _require(
            digest is None or (isinstance(digest, str) and bool(_SHA256.fullmatch(digest))), "Invalid evidence sha256"
        )
    if value["status"] == "measured":
        _require(any(metric is not None for metric in metrics.values()), "Measured records require a measurement")
        _require(bool(value["evidence"]), "Measured records require evidence references")
    return value


def _read_records(stream: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    ids: set[str] = set()
    for number, line in enumerate(stream, 1):  # type: ignore[union-attr]
        try:
            _require(line.endswith("\n"), "Incomplete ledger line")
            record = validate_experiment(load_experiment_json(line))
            round_id = str(record["round_id"])
            _require(round_id not in ids, "Duplicate round_id in ledger")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"Invalid experiment ledger line {number}") from error
        ids.add(round_id)
        records.append(record)
    return records


def list_experiments(run_dir: str | Path) -> list[dict[str, object]]:
    """Read records in append order without creating state."""

    import fcntl

    try:
        stream = (Path(run_dir) / "experiments.jsonl").open(encoding="utf-8")
    except FileNotFoundError:
        return []
    with stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return _read_records(stream)


def append_experiment(run_dir: str | Path, record: object) -> dict[str, object]:
    """Validate and append a record under an exclusive advisory lock."""

    import fcntl

    value = validate_experiment(record)
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "experiments.jsonl").open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        _require(not any(item["round_id"] == value["round_id"] for item in _read_records(stream)), "Duplicate round_id")
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return value
