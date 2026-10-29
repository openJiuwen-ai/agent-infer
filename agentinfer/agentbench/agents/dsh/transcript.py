# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Decode and normalize DeepSeek Harness session logs for task-state classification.

'find_session_logs', 'load_session_log', and 'DshTranscriptNormalizer' form the
session-log boundary; 'DshStuckDetector' and 'is_turn_complete' classify
normalized or raw events. Process lifecycle and deadline enforcement are owned
by the DSH runner.
"""

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from ..contracts import TerminationReason

# JSON-compatible values accepted at the session-log file boundary.
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
RawEvent: TypeAlias = dict[str, JsonValue]

SESSION_ROOT = "sessions"
SESSION_LOG_NAMES = ("session.v4.jsonl.zstd", "session.v4.jsonl", "session.jsonl.zstd", "session.jsonl")

_ASSISTANT_MESSAGE_EVENT = "assistant/message"
_TOOL_CALL_EVENT = "tool/call"
_TOOL_RESULT_EVENT = "tool/result"
_TURN_END_EVENT = "turn/end"


@dataclass(frozen=True)
class NormalizedEvent:
    """Represent session content consumed by task-state classifiers."""

    type: Literal["assistant", "tool_call", "tool_result", "turn_end"]
    text_content: str = ""
    tool_name: str = ""
    tool_result_content: str = ""


def find_session_logs(home: Path) -> list[Path]:
    """Return every session log under one task-local DSH home."""

    root = home / SESSION_ROOT
    if not root.is_dir():
        return []
    generations: dict[Path, list[Path]] = {}
    for path in root.rglob("session*.jsonl*"):
        version = re.fullmatch(r"session\.v(\d+)\.jsonl(?:\.zstd)?", path.name)
        if version and int(version[1]) != 4:
            raise ValueError(f"Unsupported DSH session generation: {path}")
        if path.is_file() and path.name in SESSION_LOG_NAMES:
            generations.setdefault(path.parent, []).append(path)
    logs = []
    for paths in generations.values():
        current = [path for path in paths if path.name.startswith("session.v4.")]
        selected = current or paths
        if len(selected) != 1:
            raise ValueError(f"Ambiguous DSH session generation: {selected}")
        logs.extend(selected)
    return sorted(logs)


def load_session_log(path: Path, *, allow_partial: bool = False, deadline: float | None = None) -> list[RawEvent]:
    """Decode complete records; only an interrupted final record may be partial."""

    import zstandard

    events: list[RawEvent] = []
    pending = b""
    decoder = zstandard.ZstdDecompressor().decompressobj() if path.name.endswith(".zstd") else None
    frame_complete = True
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("DSH log collection exceeded cleanup budget")
            if decoder is not None:
                decoded = bytearray()
                while chunk:
                    decoded.extend(decoder.decompress(chunk))
                    frame_complete = decoder.eof
                    chunk = decoder.unused_data
                    if frame_complete:
                        decoder = zstandard.ZstdDecompressor().decompressobj()
                chunk = bytes(decoded)
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"DSH log record is not an object: {path}")
                events.append(cast(RawEvent, event))
    if not frame_complete and not allow_partial:
        raise ValueError(f"Truncated DSH zstd frame: {path}")
    if pending.strip():
        if not allow_partial:
            raise ValueError(f"Incomplete DSH log record: {path}")
    if path.name.startswith("session.v4."):
        if not events or events[0].get("type") != "session" or events[0].get("version") != 4:
            raise ValueError(f"Invalid DSH v4 session header: {path}")
    return events


def _text_blocks(content: list[object]) -> str:
    """Join string content from supported message content blocks."""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "tool-result":
            result = block.get("content")
            if isinstance(result, str):
                parts.append(result)
            elif isinstance(result, list):
                parts.append(_text_blocks(result))
    return "\n".join(part for part in parts if part)


def session_is_root(events: Sequence[Mapping[str, object]]) -> bool:
    """Return whether a session log's header marks it as the lead (root) session.

    DSH records the lead session with 'delegationDepth' 0 and no
    'parentSession'; subagent sessions carry a parent link and a positive
    depth. Logs without a recognizable session header are treated as root so
    single-session runs keep the previous behavior.
    """

    for event in events:
        if event.get("type") != "session":
            continue
        data = event.get("data")
        header = data if isinstance(data, Mapping) else event
        if header.get("parentSession") is not None:
            return False
        depth = header.get("delegationDepth")
        if isinstance(depth, int) and depth != 0:
            return False
        return True
    return True


class DshTranscriptNormalizer:
    """Convert DSH session events to a stable normalized event model."""

    def normalize(self, raw_events: Sequence[Mapping[str, object]]) -> list[NormalizedEvent]:
        """Return supported semantic events in normalized form."""

        events: list[NormalizedEvent] = []
        for event in raw_events:
            normalized = self._normalize_one(event)
            if normalized is not None:
                events.append(normalized)
        return events

    def _normalize_one(self, event: Mapping[str, object]) -> NormalizedEvent | None:
        """Normalize one supported event or return None for other shapes."""

        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(data, Mapping):
            data = {}
        if event_type == _ASSISTANT_MESSAGE_EVENT:
            message = data.get("message")
            if not isinstance(message, Mapping):
                return None
            content = message.get("content")
            if not isinstance(content, list):
                return None
            return NormalizedEvent(type="assistant", text_content=_text_blocks(content))
        if event_type == _TOOL_CALL_EVENT:
            name = data.get("name")
            arguments = data.get("arguments")
            return NormalizedEvent(
                type="tool_call",
                tool_name=name if isinstance(name, str) else "",
                text_content=arguments if isinstance(arguments, str) else "",
            )
        if event_type == _TOOL_RESULT_EVENT:
            message = data.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            return NormalizedEvent(
                type="tool_result",
                tool_result_content=_text_blocks(content) if isinstance(content, list) else "",
            )
        if event_type == _TURN_END_EVENT:
            return NormalizedEvent(type="turn_end")
        return None


STUCK_MARKER_CLASSIFICATION: dict[TerminationReason, tuple[str, ...]] = {
    TerminationReason.CONFIRMATION_HANG: ("ask_user_question", "unknown tool", "no provider"),
    TerminationReason.VALIDATION_ERROR_LOOP: ("no-op", "no op", "validation error", "invalid arguments"),
}


class DshStuckDetector:
    """Classify repeated harness-level failures near the session tail."""

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
        searchable = [
            f"{event.text_content}\n{event.tool_name}\n{event.tool_result_content}".lower() for event in recent
        ]
        for reason, markers in STUCK_MARKER_CLASSIFICATION.items():
            matches = sum(any(marker in text for marker in markers) for text in searchable)
            if matches >= min_repeats:
                return reason
        return None


def is_turn_complete(raw_events: Sequence[Mapping[str, object]]) -> bool:
    """Return whether the last turn end event completed normally."""

    for event in reversed(raw_events):
        if event.get("type") != _TURN_END_EVENT:
            continue
        data = event.get("data")
        reason = data.get("reason") if isinstance(data, Mapping) else None
        kind = reason.get("kind") if isinstance(reason, Mapping) else None
        return kind == "completed"
    return False
