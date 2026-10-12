# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify the PR #28 evidence bundle using only the Python standard library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "validation.json").read_text())
    for relative, expected in manifest["artifact_sha256"].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact checksum mismatch: {relative}")
    repository = root.parents[2]
    for relative, expected in manifest["tested_source_sha256"].items():
        actual = hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"tested source checksum mismatch: {relative}; revalidate after changing source")

    run = root / "historical-live" / "live-run"
    rows = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines()]
    summary = json.loads((run / "summary.json").read_text())
    observed = {
        "requests": len(rows),
        "successful_requests": sum(row["status"] == "success" for row in rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
    }
    for key, count in observed.items():
        if count != summary["requests"][key] or count != manifest["historical_live"]["expected"][key]:
            raise ValueError(f"historical summary mismatch: {key}")
    if len({row["session_id"] for row in rows}) != 8:
        raise ValueError("expected 8 historical live sessions")
    execution = json.loads((run / "replay-execution.json").read_text())["summary"]
    if execution["completed_tasks"] != 8 or execution["failed_tasks"] != 0:
        raise ValueError("historical task totals do not match")
    if execution["zero_output_requests_observed_one_token"] != 1:
        raise ValueError("expected one zero-output source request observed as one token")
    print(json.dumps({"status": "passed", "artifacts_checked": len(manifest["artifact_sha256"]), **observed}, indent=2))


if __name__ == "__main__":
    main()
