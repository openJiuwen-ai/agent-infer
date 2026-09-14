# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Decode and normalize DeepSeek Harness session logs for task-state classification.

'find_session_logs', 'load_session_log', and 'DshTranscriptNormalizer' form the
session-log boundary; 'DshStuckDetector' and 'is_turn_complete' classify
normalized or raw events. Process lifecycle and deadline enforcement are owned
by the DSH runner.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from ..contracts import TerminationReason

# JSON-compatible values accepted at the session-log file boundary.
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
RawEvent: TypeAlias = dict[str, JsonValue]

SESSION_ROOT = "sessions"
SESSION_LOG_NAMES = ("session.jsonl.zstd", "session.jsonl")

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


def _decompress_zstd(data: bytes) -> bytes:
    """Decompress the default zstd-encoded session log, importing the codec lazily.

    DSH session frames are written without a content size, so a streaming
    reader is required; the one-shot decompress() API rejects such frames.
    """

    import io

    import zstandard

    return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()


def _utf16_units(value: str) -> list[int]:
    """Return JavaScript-compatible UTF-16 code units, preserving surrogates."""

    encoded = value.encode("utf-16-le", "surrogatepass")
    return [encoded[offset] | (encoded[offset + 1] << 8) for offset in range(0, len(encoded), 2)]


def project_key(cwd: str) -> str:
    """Match DSH's readable, bounded project-directory naming scheme."""

    if not cwd:
        raise ValueError("cannot encode an empty project path")
    result: list[str] = []
    separator_run = False
    for code in _utf16_units(cwd):
        character = chr(code)
        if character in "/\\:":
            if not separator_run:
                result.append("-")
            separator_run = True
        elif character != "~" and (character.isascii() and (character.isalnum() or character in "._-")):
            result.append(character)
            separator_run = False
        else:
            result.append(f"~{code:04X}")
            separator_run = False
    slug = "".join(result).lstrip("-") or "root"
    return f"--{slug[:251]}--"


def find_session_logs(home: Path, workspace: Path | None = None) -> list[Path]:
    """Return every session log under one DSH home, newest first.

    When a workspace is supplied, logs under that workspace's project key sort
    first, so shared homes cannot leak another task's sessions into this one.
    """

    root = home / SESSION_ROOT
    if not root.is_dir():
        return []
    logs = [path for path in root.rglob("*") if path.is_file() and path.name in SESSION_LOG_NAMES]
    preferred: str | None = None
    if workspace is not None:
        try:
            preferred = project_key(str(workspace.resolve()))
        except ValueError:
            preferred = None

    def sort_key(path: Path) -> tuple[int, int]:
        in_preferred = 1 if preferred is not None and path.parent.parent.name == preferred else 0
        return (-in_preferred, -path.stat().st_mtime_ns)

    return sorted(logs, key=sort_key)


def load_session_log(path: Path) -> list[RawEvent]:
    """Read valid JSON object events from a session log, decoding zstd when needed."""

    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if path.name.endswith(".zstd"):
        try:
            raw = _decompress_zstd(raw)
        except Exception:
            # A decoding failure degrades to an empty transcript; it never crashes the task.
            return []
    events: list[RawEvent] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(cast(RawEvent, event))
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
