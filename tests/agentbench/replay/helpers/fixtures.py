# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Small source files with independently specified token and dependency oracles."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCES = ("agentinfer", "inferact_codex_swebenchpro", "tracelab", "agentX")
# Golden source identities/order for this fixture and seed 9; no runtime planner calls.
SAMPLE_ORDER = {
    "agentinfer": ["source-0", "source-1"] * 4,
    "inferact_codex_swebenchpro": [f"codex-session-{i:04d}" for i in (1, 0, 1, 0, 0, 1, 0, 1)],
    "tracelab": [
        f"tracelab-{suffix}"
        for suffix in (
            "b0be4ff3a4fb943bbf927cab",
            "8bf2c5cec54f7abb586a2928",
            "8bf2c5cec54f7abb586a2928",
            "b0be4ff3a4fb943bbf927cab",
            "8bf2c5cec54f7abb586a2928",
            "b0be4ff3a4fb943bbf927cab",
            "b0be4ff3a4fb943bbf927cab",
            "8bf2c5cec54f7abb586a2928",
        )
    ],
    "agentX": [f"agentx-source-{i}" for i in (0, 1, 1, 0, 0, 1, 1, 0)],
}
INPUTS = {
    "agentinfer": [64, 96, 128],
    "inferact_codex_swebenchpro": [5, 17, 29],
    "tracelab": [30, 40, 55],
    "agentX": [64, 128, 130],
}
OUTPUTS = {"agentinfer": [8, 8, 8], "inferact_codex_swebenchpro": [2, 2, 2], "tracelab": [7, 8, 9], "agentX": [8, 8, 8]}


def timestamp(seconds):
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


def build_fixture(root: Path, source: str, *, title=False) -> tuple[Path, dict]:
    """Write real adapter input; the expected table never calls a production planner."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source.jsonl"
    rows = []
    expected = {
        "source": source,
        "inputs": INPUTS[source],
        "outputs": OUTPUTS[source],
        "title": title,
        "task_num": 8,
        "sample_order": SAMPLE_ORDER[source],
    }
    for session in range(2):
        name = f"source-{session}"
        if source == "inferact_codex_swebenchpro":
            rows.append(
                {
                    "conversations": [
                        message
                        for text in ("hello", "abcdefghij", "klmnopqrst")
                        for message in ({"from": "human", "value": text}, {"from": "gpt", "value": "xy"})
                    ]
                }
            )
        elif source == "agentX":
            rows.append(
                {
                    "id": name,
                    "models": ["source-model"],
                    "block_size": 64,
                    "hash_id_scope": "local",
                    "requests": [
                        {
                            "t": i * 0.08,
                            "api_time": 0.03,
                            "in": count,
                            "out": 8,
                            "hash_ids": list(range((count + 63) // 64)),
                            "model": "source-model",
                            "type": "n",
                        }
                        for i, count in enumerate(INPUTS[source])
                    ],
                }
            )
        else:
            for i, (count, output) in enumerate(zip(INPUTS[source], OUTPUTS[source], strict=True)):
                start, end = i * 0.08, i * 0.08 + 0.03
                if source == "agentinfer":
                    purpose = "lead_main" if i == 0 else "continuation"
                    if title:
                        start, end = [(0, 1), (0.2, 0.5), (1.1, 1.2)][i]
                        purpose = ["lead_main", "lead_title", "continuation"][i]
                    rows.append(
                        {
                            "request_id": f"{name}-{i}",
                            "session_id": name,
                            "actor_id": "lead",
                            "actor_role": "lead",
                            "request_purpose": purpose,
                            "started_at": timestamp(start),
                            "finished_at": timestamp(end),
                            "input_tokens": count,
                            "output_tokens": output,
                            "status": "success",
                        }
                    )
                else:
                    prefix = [20, 34, 52][i]
                    rows.append(
                        {
                            "provider": "codex",
                            "project": "fixture",
                            "session_id": name,
                            "round_index": i,
                            "round_id": f"round-{i}",
                            "model": "source-model",
                            "input_tokens_total": count,
                            "prefix_tokens": prefix,
                            "newly_append_tokens": count - prefix,
                            "output_tokens": output,
                            "reasoning_output_tokens": 0,
                            "timing_events": [
                                {"event_type": "user_message", "timestamp": timestamp(start)},
                                {"event_type": "text", "timestamp": timestamp(end)},
                            ],
                            "tools": [],
                        }
                    )
    path.write_text(
        json.dumps(rows) if source == "inferact_codex_swebenchpro" else "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (root / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    return path, expected


def config(source_path: Path, expected: dict, concurrency: int, base_url: str, result_dir: Path):
    source = expected["source"]
    replay = {
        "trace_type": source,
        "trace_path": str(source_path),
        "sample_seed": 9,
        "request_timeout_seconds": 5,
        "prompt_calibration_tolerance_tokens": 0,
    }
    if source == "inferact_codex_swebenchpro":
        replay.update(
            interval_mode="lognormal",
            interval_lognormal={"p50_seconds": 0.002, "p95_seconds": 0.03, "p99_seconds": 0.09},
        )
    if source == "agentinfer":
        replay.update(
            prompt_transport="completion_token_ids",
            lead_1st_sys_shared_prefix=4,
            lead_1st_sys_session_prefix=4,
            lead_1st_tool_shared_prefix=8,
            lead_1st_msg_shared_prefix=4,
            lead_1st_trailing_system_prefix=4,
            lead_title_sys_shared_prefix=4,
        )
    return {
        "experiment": {
            "task_num": expected["task_num"],
            "max_concurrency": concurrency,
            "result_dir": str(result_dir),
            "run_timeout_seconds": 30,
        },
        "backend": {
            "base_url": base_url,
            "model": "replay-ci-model",
            "endpoint": "/v1/completions" if source == "agentX" else "/v1/chat/completions",
        },
        "replay": replay,
    }
