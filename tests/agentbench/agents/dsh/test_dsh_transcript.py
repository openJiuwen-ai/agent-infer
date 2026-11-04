# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify DSH session-log decoding and normalization."""

import json
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.dsh.transcript import (
    DshStuckDetector,
    DshTranscriptNormalizer,
    find_session_logs,
    is_turn_complete,
    load_session_log,
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


def _v4_events() -> list[dict]:
    return [
        {"type": "session", "version": 4, "id": "root", "delegationDepth": 0},
        _assistant("answer"),
        _tool_call("read", '{"file_path":"calc.py"}'),
        _tool_result("file contents"),
        _turn_end("completed"),
    ]


def _write_log(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_find_session_logs_collects_all_sessions(tmp_path: Path) -> None:
    logs = [
        tmp_path / "sessions" / "project-a" / "child" / "session.jsonl",
        tmp_path / "sessions" / "project-b" / "root" / "session.v4.jsonl",
    ]
    for path in logs:
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")

    assert find_session_logs(tmp_path) == sorted(logs)


def test_load_session_log_rejects_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_log(path, [_assistant("hello")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")

    with pytest.raises(json.JSONDecodeError):
        load_session_log(path)


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


@pytest.mark.parametrize("is_child", [False, True])
def test_v4_session_normalization(tmp_path, is_child):
    events = _v4_events()
    if is_child:
        events[0].update(id="child", parentSession="root", origin="subagent", delegationDepth=1)
    path = tmp_path / "session.v4.jsonl"
    _write_log(path, events)
    loaded = load_session_log(path)
    assert loaded == events
    normalized = DshTranscriptNormalizer().normalize(loaded)
    assert any(row.type == "assistant" for row in normalized)
    assert any(row.type == "tool_call" for row in normalized)
    assert is_turn_complete(loaded)
    assert session_is_root(loaded) is (not is_child)


def test_current_generation_wins_over_legacy_and_refuses_future(tmp_path):
    folder = tmp_path / "sessions" / "project" / "root"
    folder.mkdir(parents=True)
    (folder / "session.jsonl").write_text("{}\n")
    current = folder / "session.v4.jsonl"
    _write_log(current, _v4_events())
    assert find_session_logs(tmp_path) == [current]
    (folder / "session.v5.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Unsupported"):
        find_session_logs(tmp_path)


def test_v4_header_version_must_match_filename(tmp_path):
    path = tmp_path / "session.v4.jsonl"
    path.write_text('{"type":"session","version":0,"id":"root"}\n')
    with pytest.raises(ValueError, match="header"):
        load_session_log(path)


def test_multiframe_zstd_and_interrupted_tail(tmp_path):
    import zstandard

    expected = _v4_events()
    rows = [(json.dumps(event) + "\n").encode() for event in expected]
    compressor = zstandard.ZstdCompressor()
    path = tmp_path / "session.v4.jsonl.zstd"
    path.write_bytes(compressor.compress(b"".join(rows[:3])) + compressor.compress(b"".join(rows[3:])))
    assert load_session_log(path) == expected
    # A trailing incomplete frame is diagnostic only after forced termination.
    path.write_bytes(path.read_bytes() + compressor.compress(b'{"incomplete":true}\n')[:-2])
    with pytest.raises(ValueError, match="Truncated"):
        load_session_log(path)
    assert load_session_log(path, allow_partial=True)[: len(expected)] == expected


def test_partial_json_tail_is_only_allowed_after_interruption(tmp_path):
    path = tmp_path / "session.v4.jsonl"
    _write_log(path, _v4_events())
    path.write_bytes(path.read_bytes() + b'{"type":')
    with pytest.raises(ValueError, match="Incomplete"):
        load_session_log(path)
    assert is_turn_complete(load_session_log(path, allow_partial=True))
    path.write_bytes(path.read_bytes() + b"\n{}\n")
    with pytest.raises(json.JSONDecodeError):
        load_session_log(path, allow_partial=True)
