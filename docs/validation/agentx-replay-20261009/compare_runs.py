# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Compare two AgentX runs by source request identity and token counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(run: Path) -> dict:
    plan = json.loads((run / "replay-plan.json").read_text())
    by_runtime = {
        node["runtime_request_id"]: (task["source_session_id"], node["source_key"], node)
        for task in plan["tasks"]
        for node in task["requests"]
        if node["node_type"] == "request"
    }
    requests = {}
    for line in (run / "requests.jsonl").read_text().splitlines():
        fact = json.loads(line)
        session, source, node = by_runtime[fact["request_id"]]
        key = (session, source)
        if key in requests:
            raise ValueError(f"duplicate source request: {key}")
        requests[key] = {
            "status": fact["status"],
            "input_tokens": fact["input_tokens"],
            "output_tokens": fact["output_tokens"],
            "planned_input_tokens": node["planned_input_tokens"],
            "planned_output_tokens": node["planned_output_tokens"],
            "ttft_seconds": fact["ttft_seconds"],
        }
    return {"plan": plan, "summary": json.loads((run / "summary.json").read_text()), "requests": requests}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first, second = read(args.first), read(args.second)
    assert first["plan"]["source_sha256"] == second["plan"]["source_sha256"]
    assert first["plan"]["workload_fingerprint"] == second["plan"]["workload_fingerprint"]
    keys = set(first["requests"]) | set(second["requests"])
    token_mismatches = []
    for key in sorted(keys):
        a, b = first["requests"].get(key), second["requests"].get(key)
        if (
            a is None
            or b is None
            or any(
                a[field] != b[field]
                for field in (
                    "status",
                    "input_tokens",
                    "output_tokens",
                    "planned_input_tokens",
                    "planned_output_tokens",
                )
            )
        ):
            token_mismatches.append({"source_session_id": key[0], "source_request_id": key[1], "first": a, "second": b})
    report = {
        "source_sha256": first["plan"]["source_sha256"],
        "workload_fingerprint": first["plan"]["workload_fingerprint"],
        "first_requests": len(first["requests"]),
        "second_requests": len(second["requests"]),
        "token_mismatch_count": len(token_mismatches),
        "token_mismatches": token_mismatches,
        "first_summary": first["summary"]["requests"],
        "second_summary": second["summary"]["requests"],
    }
    print(json.dumps(report, indent=2))
    if token_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
