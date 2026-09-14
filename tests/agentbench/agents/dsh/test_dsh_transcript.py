# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify DSH session-log decoding and normalization."""

import json
import os
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.dsh.transcript import (
    DshStuckDetector,
    DshTranscriptNormalizer,
    find_session_logs,
    is_turn_complete,
    load_session_log,
    project_key,
    session_is_root,
)


def _event(event_type: str, data: dict) -> dict:
    return {"seq": 0, "type": event_type, "data": data}


def _assistant(text: str) -> dict:
    return _event(
        "assistant/message",
        {"message": {"content": [{"type": "text", "text": text}]}},
    )


def _tool_call(name: str, arguments: str) -> dict:
    return _event("tool/call", {"name": name, "arguments": arguments})


def _tool_result(text: str) -> dict:
    return _event(
        "tool/result",
        {"message": {"content": [{"type": "tool-result", "content": text}]}},
    )


def _turn_end(kind: str) -> dict:
    return _event("turn/end", {"reason": {"kind": kind}})


def _write_log(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_project_key_matches_observed_dsh_layout() -> None:
    assert project_key("/Users/hsliu/code/AgentInfer") == "--Users-hsliu-code-AgentInfer--"
    assert project_key("/a/b") == "--a-b--"
    with pytest.raises(ValueError, match="empty"):
        project_key("")


def test_find_session_logs_prefers_workspace_project_key(tmp_path: Path) -> None:
    workspace = tmp_path / "w"
    other = tmp_path / "o"
    workspace.mkdir()
    other.mkdir()
    own_dir = tmp_path / "sessions" / project_key(str(workspace.resolve())) / "session-own"
    other_dir = tmp_path / "sessions" / project_key(str(other.resolve())) / "session-other"
    own_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    (own_dir / "session.jsonl").write_text("{}\n", encoding="utf-8")
    (other_dir / "session.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(other_dir / "session.jsonl", (3, 3))
    os.utime(own_dir / "session.jsonl", (1, 1))

    logs = find_session_logs(tmp_path, workspace)

    assert logs[0].parent.parent.name == project_key(str(workspace.resolve()))


def test_find_session_logs_orders_newest_first(tmp_path: Path) -> None:
    older = tmp_path / "sessions" / "project" / "session-old" / "session.jsonl.zstd"
    newer = tmp_path / "sessions" / "project" / "session-new" / "session.jsonl"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    logs = find_session_logs(tmp_path)

    assert [path.name for path in logs] == ["session.jsonl", "session.jsonl.zstd"]


def test_load_session_log_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_log(path, [_assistant("hello")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")

    events = load_session_log(path)

    assert [event["type"] for event in events] == ["assistant/message"]


def test_load_session_log_decodes_zstd(tmp_path: Path) -> None:
    import zstandard

    raw = b"".join(json.dumps(event).encode() + b"\n" for event in [_assistant("hello"), _turn_end("completed")])
    path = tmp_path / "session.jsonl.zstd"
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))

    events = load_session_log(path)

    assert [event["type"] for event in events] == ["assistant/message", "turn/end"]


def test_normalizer_maps_semantic_events() -> None:
    normalizer = DshTranscriptNormalizer()

    events = normalizer.normalize([_assistant("answer"), _tool_call("bash", '{"command":"ls"}'), _tool_result("files")])

    assert events[0].type == "assistant"
    assert events[0].text_content == "answer"
    assert events[1].type == "tool_call"
    assert events[1].tool_name == "bash"
    assert events[2].type == "tool_result"
    assert events[2].tool_result_content == "files"


def test_stuck_detector_ignores_short_logs() -> None:
    detector = DshStuckDetector()
    asks = [_tool_call("ask_user_question", '{"questions":[]}') for _ in range(3)]

    assert detector.detect(DshTranscriptNormalizer().normalize(asks)) is None


def test_is_turn_complete_reads_last_turn_end() -> None:
    assert is_turn_complete([_assistant("x"), _turn_end("completed")]) is True
    assert is_turn_complete([_turn_end("completed"), _turn_end("error")]) is False
    assert is_turn_complete([_assistant("x")]) is False


def test_session_is_root_reads_lead_header() -> None:
    root = [
        {"type": "session", "version": 0, "id": "session-lead", "delegationDepth": 0, "cwd": "/w"},
        _assistant("x"),
    ]

    assert session_is_root(root) is True


def test_session_is_root_rejects_parented_subagent_header() -> None:
    subagent = [
        {
            "type": "session",
            "version": 0,
            "id": "session-sub",
            "parentSession": "session-lead",
            "delegationDepth": 1,
            "cwd": "/w",
        },
        _assistant("x"),
    ]

    assert session_is_root(subagent) is False


def test_session_is_root_rejects_positive_delegation_depth_without_parent() -> None:
    subagent = [
        {"type": "session", "version": 0, "id": "session-sub", "delegationDepth": 2, "cwd": "/w"},
        _assistant("x"),
    ]

    assert session_is_root(subagent) is False


def test_session_is_root_treats_null_parent_session_as_root() -> None:
    root = [
        {
            "type": "session",
            "version": 0,
            "id": "session-lead",
            "parentSession": None,
            "delegationDepth": 0,
            "cwd": "/w",
        },
        _assistant("x"),
    ]

    assert session_is_root(root) is True


def test_session_is_root_falls_back_to_root_without_header() -> None:
    assert session_is_root([_assistant("x"), _turn_end("completed")]) is True
