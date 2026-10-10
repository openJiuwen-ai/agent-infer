# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Select eight complete AgentX sessions suitable for the dual-L20 test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cc-traces-weka-062126-256k" / "traces.jsonl"
DESTINATION = ROOT / "agentx-l20-8sessions.jsonl"


def requests(session: dict) -> list[dict]:
    return [
        request
        for group in session["requests"]
        for request in (group["requests"] if group.get("type") == "subagent" else [group])
    ]


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    selected = []
    with SOURCE.open(encoding="utf-8") as source, DESTINATION.open("x", encoding="utf-8") as output:
        for line in source:
            session = json.loads(line)
            model_requests = requests(session)
            duration = max(r["t"] + r["api_time"] for r in model_requests) - min(r["t"] for r in model_requests)
            if duration <= 3600 and all(r["out"] > 0 and r["in"] + r["out"] <= 100000 for r in model_requests):
                selected.append(session)
                output.write(line if line.endswith("\n") else line + "\n")
                if len(selected) == 8:
                    break
    if len(selected) != 8:
        raise ValueError(f"found only {len(selected)} eligible sessions")
    rows = [r for session in selected for r in requests(session)]
    totals = {
        "sessions": len(selected),
        "requests": len(rows),
        "input_tokens": sum(r["in"] for r in rows),
        "output_tokens": sum(r["out"] for r in rows),
        "max_context_tokens": max(r["in"] + r["out"] for r in rows),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "subset_sha256": hashlib.sha256(DESTINATION.read_bytes()).hexdigest(),
    }
    assert (totals["requests"], totals["input_tokens"], totals["output_tokens"]) == (254, 13957952, 162310)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
