# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Normalize Claude Code transcripts and classify terminal task states.

``TranscriptLoader`` and ``TranscriptNormalizer`` form the JSONL boundary;
``StuckDetector`` and ``is_transcript_complete`` classify normalized or raw
events. Process lifecycle and terminal capture are owned by the Claude runtime.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from ..contracts import TerminationReason

# JSON-compatible values accepted at the transcript file boundary.
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
RawEvent: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class NormalizedEvent:
    """Represent transcript content consumed by task-state classifiers."""

    type: Literal["assistant", "user"]
    agent_id: str = "lead"
    stop_reason: str | None = None
    text_content: str = ""
    tool_result_content: str = ""


class TranscriptLoader:
    """Read valid JSON object events from a transcript JSONL file."""

    def load(self, path: Path) -> list[RawEvent]:
        """Return valid JSON object events from ``path``, skipping malformed lines."""

        if not path.is_file():
            return []
        events: list[RawEvent] = []
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(cast(RawEvent, event))
        return events


class TranscriptNormalizer:
    """Convert Claude Code assistant and user events to a stable event model."""

    def normalize(self, raw_events: Sequence[Mapping[str, object]]) -> list[NormalizedEvent]:
        """Return supported assistant and user events in normalized form."""

        events: list[NormalizedEvent] = []
        for event in raw_events:
            normalized = self._normalize_one(event)
            if normalized is not None:
                events.append(normalized)
        return events

    def _normalize_one(self, event: Mapping[str, object]) -> NormalizedEvent | None:
        """Normalize one supported event or return ``None`` for other shapes."""

        event_type = event.get("type")
        if event_type not in ("assistant", "user"):
            return None
        message = event.get("message")
        if not isinstance(message, Mapping):
            return None
        content = message.get("content")
        if not isinstance(content, list):
            return None
        normalized_agent_id = _agent_id(event)
        if event_type == "assistant":
            stop_reason = message.get("stop_reason")
            return NormalizedEvent(
                type="assistant",
                agent_id=normalized_agent_id,
                stop_reason=stop_reason if isinstance(stop_reason, str) else None,
                text_content=_text_blocks(content),
            )
        return NormalizedEvent(
            type="user",
            agent_id=normalized_agent_id,
            tool_result_content=_tool_results(content),
        )


def _agent_id(event: Mapping[str, object]) -> str:
    """Return the agent identity from supported Claude event fields."""

    for field in ("agentId", "agent_id"):
        value = event.get(field)
        if isinstance(value, str) and value:
            return value
    return "lead"


def _text_blocks(content: list[object]) -> str:
    """Join text from valid transcript content blocks."""

    parts: list[str] = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _tool_results(content: list[object]) -> str:
    """Join string or text-block content from tool-result blocks."""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "tool_result":
            continue
        result = block.get("content")
        if isinstance(result, str):
            parts.append(result)
        elif isinstance(result, list):
            parts.append(_text_blocks(result))
    return "\n".join(part for part in parts if part)


STUCK_MARKER_CLASSIFICATION: dict[TerminationReason, tuple[str, ...]] = {
    TerminationReason.PLAN_EXIT_LOOP: ("not in plan mode",),
    TerminationReason.VALIDATION_ERROR_LOOP: (
        "no-op",
        "no op",
        "validation error",
    ),
    TerminationReason.CONFIRMATION_HANG: ("unknown command",),
}


class StuckDetector:
    """Classify repeated harness-level failures near the transcript tail."""

    def detect(
        self,
        events: Sequence[NormalizedEvent],
        *,
        min_events: int = 50,
        min_repeats: int = 3,
        tail: int = 400,
    ) -> TerminationReason | None:
        """Return a stuck reason when a marker repeats in a sufficiently long tail."""

        if len(events) < min_events:
            return None
        recent = events[-tail:]
        searchable_events = [f"{event.text_content}\n{event.tool_result_content}".lower() for event in recent]
        for reason, markers in STUCK_MARKER_CLASSIFICATION.items():
            matches = sum(any(marker in searchable for marker in markers) for searchable in searchable_events)
            if matches >= min_repeats:
                return reason
        return None


def is_transcript_complete(raw_events: Sequence[Mapping[str, object]]) -> bool:
    """Return whether the last conversation event is a completed assistant turn."""

    last_event = next(
        (event for event in reversed(raw_events) if event.get("type") in ("assistant", "user")),
        None,
    )
    if last_event is None or last_event.get("type") != "assistant":
        return False
    stop_reason = last_event.get("stop_reason")
    message = last_event.get("message")
    if not stop_reason and isinstance(message, Mapping):
        stop_reason = message.get("stop_reason")
    return stop_reason in ("end_turn", "stop_sequence")
