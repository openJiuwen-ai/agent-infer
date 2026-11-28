# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Project native tool evidence onto Workspace-Bench's executionTrace format."""

import json
from pathlib import Path


def _rows(path: Path):
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def _arguments(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {"arguments": value}
    return value if isinstance(value, dict) else {"arguments": value}


def jiuwen_trace(path: Path, session_id: str | None = None) -> dict:
    events = []
    calls = {}
    last_text = ""
    for frame in _rows(path):
        event, payload = frame["event"], frame["payload"]
        if event == "chat.delta":
            content = payload.get("content")
            if isinstance(content, str) and content:
                events.append(
                    {"type": "text", "role": "assistant", "content": content, "timestamp": payload.get("timestamp")}
                )
                last_text += content
        elif event == "chat.final" and payload.get("event_type") == "chat.final":
            if (session_id is None or payload.get("session_id") == session_id) and isinstance(
                payload.get("content"), str
            ):
                last_text = payload["content"]
                if not events or events[-1].get("content") != last_text:
                    events.append(
                        {
                            "type": "text",
                            "role": "assistant",
                            "content": last_text,
                            "timestamp": payload.get("timestamp"),
                        }
                    )
        elif event in ("chat.tool_call", "chat.tool_update", "chat.tool_result"):
            detail = payload.get("tool_call") or payload
            key = (payload.get("execution_id"), detail.get("tool_call_id"))
            if not key[1]:
                continue
            if key not in calls:
                calls[key] = {
                    "type": "tool",
                    "tool": detail.get("name") or payload.get("tool_name"),
                    "input": {},
                    "output": {},
                    "timestamp": payload.get("timestamp"),
                }
                events.append(calls[key])
            if detail.get("arguments"):
                calls[key]["input"] = _arguments(detail["arguments"])
            if event == "chat.tool_result":
                calls[key]["output"] = {k: payload[k] for k in ("result", "rendered_result", "success") if k in payload}
    return {"executionTrace": events, "lastText": last_text}


def zcode_trace(raw: Path) -> dict:
    events = []
    calls = {}
    for row in _rows(raw / "zcode-hooks.jsonl"):
        item = row["event"]
        if item["hookEventName"] not in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
            continue
        key = (item["sessionId"], item["toolCallId"])
        if key not in calls:
            calls[key] = {
                "type": "tool",
                "tool": item.get("toolName"),
                "input": item.get("toolInput", {}),
                "output": {},
                "timestamp": item.get("timestamp"),
            }
            events.append(calls[key])
        if item["hookEventName"] != "PreToolUse":
            calls[key]["output"] = {
                "response": item.get("toolResponse"),
                "error": item.get("error"),
                "event": item["hookEventName"],
            }
    last_text = ""
    for row in _rows(raw / "transcript.jsonl"):
        if row.get("type") == "turn.completed":
            message = row["payload"].get("response")
            if isinstance(message, str) and message:
                events.append(
                    {"type": "text", "role": "assistant", "content": message, "timestamp": row.get("timestamp")}
                )
                if not str(row.get("sessionId", "")).startswith("sess_subagent_"):
                    last_text = message
    events.sort(key=lambda event: str(event.get("timestamp") or ""))
    return {"executionTrace": events, "lastText": last_text}
