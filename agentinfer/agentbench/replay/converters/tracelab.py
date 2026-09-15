# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Convert normalized TraceLab rounds into token-recipe Replay IR.

This module owns source validation, event-time proxies, and IR emission. It does
not select Runtime Sessions, render model tokens, execute tools, or send requests.
``TraceLabConverter`` is the reader entry point.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO, TypedDict

from ..unified_trace_ir import write_trace_ir_manifest
from .base import ConverterSummary, ReplayDatasetConverter

_INPUT_EVENTS = frozenset({"user_message", "tool_result"})
_OUTPUT_EVENTS = frozenset({"reasoning", "text", "tool_call"})


class _TimingProxy(TypedDict, total=False):
    valid: bool
    reason: str
    basis: str
    input_ready_at: str
    input_ready_event_index: int
    input_ready_event_type: str
    output_end_at: str
    output_end_event_index: int
    output_end_event_type: str


def _required_string(row: dict[str, object], field: str, source_line: int) -> str:
    """Read a required non-empty source string and identify an invalid line."""

    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TraceLab line {source_line} has invalid {field}")
    return value.strip()


def _integer(row: dict[str, object], field: str, source_line: int, *, positive: bool) -> int:
    """Read a strict integer token or sequence field with the requested lower bound."""

    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"TraceLab line {source_line} {field} must be a {qualifier} integer")
    return value


def _timestamp(value: object, source_line: int, event_index: int) -> datetime:
    """Parse one event timestamp as UTC and retain line/event error context."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"TraceLab line {source_line} event {event_index} has invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"TraceLab line {source_line} event {event_index} has invalid timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timing_proxy(row: dict[str, object], source_line: int) -> _TimingProxy:
    """Select auditable input-ready and output-end event proxies for one round."""

    raw_events = row.get("timing_events")
    if not isinstance(raw_events, list):
        raise ValueError(f"TraceLab line {source_line} timing_events must be a list")
    events: list[tuple[datetime, int, str]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise ValueError(f"TraceLab line {source_line} event {index} is not an object")
        event_type = raw.get("event_type")
        if not isinstance(event_type, str):
            raise ValueError(f"TraceLab line {source_line} event {index} has invalid event_type")
        events.append((_timestamp(raw.get("timestamp"), source_line, index), index, event_type))
    output_events = [event for event in events if event[2] in _OUTPUT_EVENTS]
    if not output_events:
        return {"valid": False, "reason": "missing_output_event"}
    first_output = min(output_events)
    input_events = [event for event in events if event[2] in _INPUT_EVENTS and event[:2] < first_output[:2]]
    if not input_events:
        return {"valid": False, "reason": "missing_input_event_before_output"}
    input_ready = max(input_events)
    output_end = max(output_events)
    if output_end[0] < input_ready[0]:
        return {"valid": False, "reason": "output_before_input"}
    return {
        "valid": True,
        "basis": "event_proxy",
        "input_ready_at": input_ready[0].isoformat(),
        "input_ready_event_index": input_ready[1],
        "input_ready_event_type": input_ready[2],
        "output_end_at": output_end[0].isoformat(),
        "output_end_event_index": output_end[1],
        "output_end_event_type": output_end[2],
    }


def _source_session_id(provider: str, session_id: str) -> str:
    """Return a stable path-safe identity for a provider-scoped source Session."""

    identity = json.dumps([provider, session_id], ensure_ascii=False, separators=(",", ":"))
    return f"tracelab-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


class TraceLabConverter(ReplayDatasetConverter):
    """Strictly convert one normalized TraceLab JSONL file."""

    name = "tracelab"

    @staticmethod
    def _open(source: Path) -> TextIO:
        """Open a plain TraceLab JSONL source."""

        if source.suffix != ".jsonl":
            raise ValueError("TraceLab source must end with .jsonl")
        return source.open(encoding="utf-8")

    def convert(self, source: Path, output_dir: Path) -> ConverterSummary:
        """Validate every source round and atomically inventory the emitted IR."""

        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"TraceLab source is not a file: {source}")
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        identities: set[tuple[str, str, int]] = set()
        with self._open(source) as handle:
            for source_line, line in enumerate(handle, start=1):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid TraceLab JSON at line {source_line}: {exc}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"TraceLab line {source_line} is not an object")
                row = {str(key): value for key, value in raw.items()}
                provider = _required_string(row, "provider", source_line)
                source_session = _required_string(row, "session_id", source_line)
                round_index = _integer(row, "round_index", source_line, positive=False)
                identity = (provider, source_session, round_index)
                if identity in identities:
                    raise ValueError(f"TraceLab line {source_line} duplicates provider/session_id/round_index")
                identities.add(identity)
                input_tokens = _integer(row, "input_tokens_total", source_line, positive=True)
                output_tokens = _integer(row, "output_tokens", source_line, positive=True)
                cached_tokens = _integer(row, "prefix_tokens", source_line, positive=False)
                appended_tokens = _integer(row, "newly_append_tokens", source_line, positive=False)
                timing = _timing_proxy(row, source_line)
                tools = row.get("tools")
                if not isinstance(tools, list):
                    raise ValueError(f"TraceLab line {source_line} tools must be a list")
                if any(not isinstance(tool, dict) for tool in tools):
                    raise ValueError(f"TraceLab line {source_line} contains a non-object tool")
                session_id = _source_session_id(provider, source_session)
                request_id = f"{session_id}-round-{round_index:08d}"
                grouped[session_id].append(
                    {
                        "source_line": source_line,
                        "request_id": request_id,
                        "session_id": session_id,
                        "task_id": row.get("project"),
                        "actor_id": "lead",
                        "actor_role": "lead",
                        "status": "unknown",
                        "request_purpose": "lead_main" if round_index == 0 else "continuation",
                        "round_index": round_index,
                        "source_provider": provider,
                        "source_session_id": source_session,
                        "source_round_id": row.get("round_id"),
                        "source_trace_key": row.get("trace_key"),
                        "source_model": row.get("model"),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_tokens": cached_tokens,
                        "newly_append_tokens": appended_tokens,
                        "reasoning_output_tokens": row.get("reasoning_output_tokens"),
                        "timing_proxy": timing,
                        "tool_count": len(tools),
                        "tool_error_count": sum(tool.get("is_error") is True for tool in tools),
                    }
                )

        output_dir.mkdir(parents=True, exist_ok=False)
        requests_path = output_dir / "requests.jsonl"
        request_count = 0
        with requests_path.open("w", encoding="utf-8") as output:
            for session_id in sorted(grouped):
                rows = sorted(grouped[session_id], key=lambda row: int(row["round_index"]))
                previous: dict[str, object] | None = None
                for sequence_index, row in enumerate(rows):
                    row["sequence_index"] = sequence_index
                    row["context_after"] = previous["request_id"] if previous is not None else None
                    row["send_after"] = previous["request_id"] if previous is not None else None
                    timing = row["timing_proxy"]
                    assert isinstance(timing, dict)
                    if timing.get("valid"):
                        row["started_at"] = timing["input_ready_at"]
                        row["finished_at"] = timing["output_end_at"]
                    else:
                        # The explicit order remains valid for lognormal mode. Trace mode
                        # rejects this row before planning because timing_valid is false.
                        instant = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=sequence_index)
                        row["started_at"] = instant.isoformat()
                        row["finished_at"] = instant.isoformat()
                    row["timing_valid"] = bool(timing.get("valid"))
                    if previous is None:
                        row["source_gap_seconds"] = 0.0
                    else:
                        previous_timing = previous["timing_proxy"]
                        assert isinstance(previous_timing, dict)
                        if timing.get("valid") and previous_timing.get("valid"):
                            current_start = datetime.fromisoformat(str(timing["input_ready_at"]))
                            previous_end = datetime.fromisoformat(str(previous_timing["output_end_at"]))
                            gap = (current_start - previous_end).total_seconds()
                            row["source_gap_seconds"] = gap if gap >= 0 else None
                            if gap < 0:
                                row["timing_valid"] = False
                                timing["valid"] = False
                                timing["reason"] = "negative_inter_round_gap"
                        else:
                            row["source_gap_seconds"] = None
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    request_count += 1
                    previous = row

        summary = ConverterSummary("tracelab", len(grouped), request_count, 0)
        write_trace_ir_manifest(
            output_dir,
            converter_name=self.name,
            source_path=source,
            summary=summary.to_dict(),
            prompt_source_kind="token_recipe",
        )
        return summary
