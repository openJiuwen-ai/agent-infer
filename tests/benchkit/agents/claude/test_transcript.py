"""Verify Claude transcript loading, normalization, and classification."""

import json
from pathlib import Path

import pytest

from agentcache.benchmarks.agents.claude.transcript import (
    NormalizedEvent,
    StuckDetector,
    TranscriptLoader,
    TranscriptNormalizer,
    is_transcript_complete,
)
from agentcache.benchmarks.agents.outcomes import TerminationReason


def test_loader_reads_only_json_objects(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"type": "assistant", "message": {"content": []}}),
                "not json",
                json.dumps(["not", "an", "event"]),
                json.dumps({"type": "user", "message": {"content": []}}),
            )
        ),
        encoding="utf-8",
    )

    events = TranscriptLoader().load(path)

    assert [event["type"] for event in events] == ["assistant", "user"]


def test_loader_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert TranscriptLoader().load(tmp_path / "missing.jsonl") == []


def test_normalizer_extracts_assistant_and_tool_result_content() -> None:
    raw = [
        {
            "type": "assistant",
            "agent_id": "lead",
            "message": {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "The fix is applied."}],
            },
        },
        {
            "type": "user",
            "isSidechain": True,
            "agentId": "a46ee8281cfe3d128",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "no-op"},
                            {"type": "text", "text": "validation error"},
                        ],
                    }
                ]
            },
        },
    ]

    events = TranscriptNormalizer().normalize(raw)

    assert events == [
        NormalizedEvent(
            type="assistant",
            agent_id="lead",
            stop_reason="end_turn",
            text_content="The fix is applied.",
        ),
        NormalizedEvent(
            type="user",
            agent_id="a46ee8281cfe3d128",
            tool_result_content="no-op\nvalidation error",
        ),
    ]


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "system", "message": {"content": []}},
        {"type": "assistant", "message": "invalid"},
        {"type": "user", "message": {"content": "invalid"}},
    ],
)
def test_normalizer_skips_unsupported_or_malformed_events(raw: dict[str, object]) -> None:
    assert TranscriptNormalizer().normalize([raw]) == []


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("not in plan mode", TerminationReason.PLAN_EXIT_LOOP),
        ("validation error", TerminationReason.VALIDATION_ERROR_LOOP),
        ("unknown command", TerminationReason.CONFIRMATION_HANG),
    ],
)
def test_stuck_detector_classifies_repeated_harness_failures(
    marker: str,
    expected: TerminationReason,
) -> None:
    events = [NormalizedEvent(type="assistant") for _ in range(50)]
    events.extend(NormalizedEvent(type="user", tool_result_content=marker) for _ in range(3))

    assert StuckDetector().detect(events) is expected


def test_stuck_detector_ignores_insufficient_evidence() -> None:
    events = [NormalizedEvent(type="assistant") for _ in range(50)]
    events.append(NormalizedEvent(type="user", tool_result_content="not in plan mode"))

    assert StuckDetector().detect(events) is None


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
def test_transcript_is_complete_for_finished_assistant_turn(stop_reason: str) -> None:
    assert is_transcript_complete([{"type": "assistant", "message": {"stop_reason": stop_reason, "content": []}}])


def test_transcript_is_complete_with_trailing_metadata() -> None:
    assert is_transcript_complete(
        [
            {"type": "assistant", "message": {"stop_reason": "end_turn", "content": []}},
            {"type": "system", "subtype": "turn_duration"},
        ]
    )


def test_transcript_is_not_complete_without_finished_assistant_turn() -> None:
    assert not is_transcript_complete(
        [
            {"type": "assistant", "stop_reason": "end_turn"},
            {"type": "user", "message": {"content": []}},
        ]
    )
