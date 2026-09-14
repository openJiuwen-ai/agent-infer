# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Analyze historical request traces into deterministic Replay sessions."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from .schema import PromptKind, ReplayAnalysis, ReplayRequest, ReplaySession, required_request_keys
from .unified_trace_ir import PromptReference, parse_prompt_reference

_ROW_ANALYSIS_COUNTERS = (
    "invalid_json",
    "not_an_object",
    "missing_session_id",
    "missing_agent_id",
    "invalid_timestamps",
    "retry_matched_failures",
    "rows_without_replayable_tokens",
    "sessions_without_replayable_requests",
    "inferred_actor_role",
    "invalid_token_counts",
    "duplicate_request_ids",
    "pruned_terminal_timing_dependency_rows",
    "inferred_late_title_requests",
    "inferred_lead_name_requests",
)


@dataclass(frozen=True)
class _SourceRow:
    key: str
    source_line: int
    request_id: str | None
    task_id: str | None
    session_id: str
    actor_id: str
    actor_role: Literal["lead", "subagent", "unknown"]
    started_at: datetime
    finished_at: datetime
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    status: str
    request_purpose: str | None
    prompt_ref: PromptReference | None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _nonempty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _source_key(
    *,
    session_id: str,
    request_id: str | None,
    source_line: int,
    row: dict[str, object],
) -> str:
    canonical = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        "\0".join(
            (
                session_id,
                request_id or "",
                str(source_line),
                canonical,
            )
        ).encode()
    ).hexdigest()
    return f"r{source_line:08d}-{digest[:16]}"


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_rows(
    path: Path,
) -> tuple[dict[str, list[_SourceRow]], dict[str, int], int, int]:
    rows_by_session: dict[str, list[_SourceRow]] = defaultdict(list)
    row_analysis_counts: dict[str, int] = defaultdict(int)
    request_ids: set[str] = set()
    total_rows = 0
    valid_rows = 0

    with path.open(encoding="utf-8", errors="replace") as handle:
        for source_line, line in enumerate(handle, start=1):
            total_rows += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                row_analysis_counts["invalid_json"] += 1
                continue
            if not isinstance(raw, dict):
                row_analysis_counts["not_an_object"] += 1
                continue

            row = {str(key): value for key, value in raw.items()}
            session_id = _nonempty_string(row.get("session_id"))
            if session_id is None:
                row_analysis_counts["missing_session_id"] += 1
                continue

            explicit_role = _nonempty_string(row.get("actor_role"))
            actor_id = _nonempty_string(row.get("actor_id")) or _nonempty_string(row.get("agent_id"))
            if actor_id is None and explicit_role == "lead":
                actor_id = "lead"
            if actor_id is None:
                row_analysis_counts["missing_agent_id"] += 1
                continue
            if explicit_role in {"lead", "subagent", "unknown"}:
                actor_role = explicit_role
            else:
                actor_role = "lead" if actor_id == "lead" else "subagent"
                row_analysis_counts["inferred_actor_role"] += 1

            started_at = _timestamp(row.get("started_at"))
            finished_at = _timestamp(row.get("finished_at"))
            if started_at is None or finished_at is None or finished_at < started_at:
                row_analysis_counts["invalid_timestamps"] += 1
                continue

            input_tokens = _positive_int(row.get("input_tokens"))
            output_tokens = _positive_int(row.get("output_tokens"))
            cached_tokens = _nonnegative_int(row.get("cached_tokens"))
            if (
                (row.get("input_tokens") is not None and input_tokens is None)
                or (row.get("output_tokens") is not None and output_tokens is None)
                or (row.get("cached_tokens") is not None and cached_tokens is None)
            ):
                row_analysis_counts["invalid_token_counts"] += 1

            request_id = _nonempty_string(row.get("request_id"))
            if request_id is not None:
                if request_id in request_ids:
                    row_analysis_counts["duplicate_request_ids"] += 1
                request_ids.add(request_id)
            task_id = _nonempty_string(row.get("task_id"))
            status = _nonempty_string(row.get("status")) or "failed"
            key = _source_key(
                session_id=session_id,
                request_id=request_id,
                source_line=source_line,
                row=row,
            )
            rows_by_session[session_id].append(
                _SourceRow(
                    key=key,
                    source_line=source_line,
                    request_id=request_id,
                    task_id=task_id,
                    session_id=session_id,
                    actor_id=actor_id,
                    actor_role=actor_role,
                    started_at=started_at,
                    finished_at=finished_at,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    status=status,
                    request_purpose=_nonempty_string(row.get("request_purpose")),
                    prompt_ref=parse_prompt_reference(row.get("prompt_ref")),
                )
            )
            valid_rows += 1
    return rows_by_session, row_analysis_counts, total_rows, valid_rows


_PromptClassification = tuple[PromptKind, Literal["explicit", "inferred"], str | None]


def _overlap(left: _SourceRow, right: _SourceRow) -> bool:
    return left.started_at < right.finished_at and right.started_at < left.finished_at


_LEAD_NAME_MAX_INPUT_TOKENS = 512
_LEAD_NAME_MAX_OUTPUT_TOKENS = 32
_CONTEXT_FREE_PROMPT_KINDS = frozenset({"lead_title", "lead_name"})


def _late_auxiliary_prompt_kinds(
    rows: list[_SourceRow],
    kinds: dict[str, _PromptClassification],
) -> dict[str, PromptKind]:
    """Infer conservative late title/name requests from same-actor overlap and token scale."""

    inferred: dict[str, PromptKind] = {}
    conversational: _SourceRow | None = None
    for row in rows:
        classification = kinds.get(row.key)
        if classification is not None and classification[0] in _CONTEXT_FREE_PROMPT_KINDS:
            continue
        token_usable = row.status == "success" and row.input_tokens is not None and row.output_tokens is not None
        if not token_usable:
            continue
        if classification is not None and classification[1] == "explicit":
            conversational = row
            continue
        if conversational is not None and row.input_tokens <= 4096 and row.output_tokens <= 128:
            overlapping = [
                candidate
                for candidate in rows
                if candidate.key != row.key
                and candidate.status == "success"
                and candidate.input_tokens is not None
                and candidate.output_tokens is not None
                and _overlap(row, candidate)
            ]
            if any(
                row.input_tokens <= candidate.input_tokens * 0.20
                and row.input_tokens <= conversational.input_tokens * 0.25
                and candidate.input_tokens >= conversational.input_tokens * 0.80
                for candidate in overlapping
            ):
                inferred[row.key] = (
                    "lead_name"
                    if row.input_tokens <= _LEAD_NAME_MAX_INPUT_TOKENS
                    and row.output_tokens <= _LEAD_NAME_MAX_OUTPUT_TOKENS
                    else "lead_title"
                )
                continue
        conversational = row
    return inferred


def _prompt_kinds(rows: list[_SourceRow]) -> tuple[dict[str, _PromptClassification], int, int]:
    kinds: dict[str, _PromptClassification] = {}
    rows_by_actor: dict[str, list[_SourceRow]] = defaultdict(list)
    for row in rows:
        rows_by_actor[row.actor_id].append(row)
        if row.request_purpose in {"lead_title", "lead_name", "lead_main", "subagent_first", "continuation"}:
            kinds[row.key] = (cast(PromptKind, row.request_purpose), "explicit", None)

    inferred_late_titles = 0
    inferred_lead_names = 0
    for actor_rows in rows_by_actor.values():
        actor_rows.sort(key=lambda row: (row.started_at, row.finished_at, row.key))
        first = actor_rows[0]
        if first.actor_role != "lead":
            established = False
            for row in actor_rows:
                kinds.setdefault(
                    row.key,
                    (
                        "continuation" if established else "subagent_first",
                        "inferred",
                        "actor_sequence",
                    ),
                )
                established |= (
                    row.status == "success" and row.input_tokens is not None and row.output_tokens is not None
                )
            continue

        initial = [
            row for row in actor_rows if row.started_at < first.finished_at and first.started_at < row.finished_at
        ]
        paired_first_requests = len(initial) == 2 and all(
            row.status == "success" and row.input_tokens is not None and row.output_tokens is not None
            for row in initial
        )
        if paired_first_requests:
            title = min(initial, key=lambda row: (row.input_tokens or 0, row.output_tokens or 0, row.key))
            main = max(initial, key=lambda row: (row.input_tokens or 0, row.output_tokens or 0, row.key))
            kinds.setdefault(title.key, ("lead_title", "inferred", "overlapping_initial_lead_pair"))
            kinds.setdefault(main.key, ("lead_main", "inferred", "overlapping_initial_lead_pair"))
            established = True
        else:
            established = False
        for row in actor_rows:
            kinds.setdefault(
                row.key,
                (
                    "continuation" if established else "lead_main",
                    "inferred",
                    "actor_sequence",
                ),
            )
            established |= row.status == "success" and row.input_tokens is not None and row.output_tokens is not None
        auxiliary_kinds = _late_auxiliary_prompt_kinds(actor_rows, kinds)
        for key, prompt_kind in auxiliary_kinds.items():
            if kinds[key][0] in _CONTEXT_FREE_PROMPT_KINDS:
                continue
            if prompt_kind == "lead_name":
                kinds[key] = ("lead_name", "inferred", "overlapping_lead_name")
                inferred_lead_names += 1
            else:
                kinds[key] = ("lead_title", "inferred", "overlapping_late_lead_title")
                inferred_late_titles += 1
    return kinds, inferred_late_titles, inferred_lead_names


def _retry_matches(
    rows: list[_SourceRow],
    prompt_kinds: dict[str, _PromptClassification],
) -> tuple[dict[str, _SourceRow], dict[str, tuple[int, int]]]:
    """Match failures to the first later token-usable success from their actor."""

    matches: dict[str, _SourceRow] = {}
    resolved_tokens: dict[str, tuple[int, int]] = {}
    pending_by_actor: dict[tuple[str, PromptKind], list[_SourceRow]] = defaultdict(list)
    for row in rows:
        actor_prompt = (row.actor_id, prompt_kinds[row.key][0])
        token_pair = (
            (row.input_tokens, row.output_tokens)
            if row.input_tokens is not None and row.output_tokens is not None
            else None
        )
        if token_pair is not None:
            resolved_tokens[row.key] = token_pair
        if row.status == "success":
            if token_pair is not None:
                for failed in pending_by_actor.pop(actor_prompt, []):
                    matches[failed.key] = row
                    resolved_tokens[failed.key] = token_pair
            continue
        pending_by_actor[actor_prompt].append(row)
    return matches, resolved_tokens


def _dependency_kind(
    predecessor: _SourceRow | None,
    current: _SourceRow,
    parents: dict[str, str],
) -> Literal[
    "session_root",
    "same_agent",
    "parent_to_subagent",
    "blocking_subagent_to_parent",
    "cross_agent_completion",
]:
    if predecessor is None:
        return "session_root"
    if predecessor.actor_id == current.actor_id:
        return "same_agent"
    if parents.get(current.actor_id) == predecessor.actor_id:
        return "parent_to_subagent"
    if parents.get(predecessor.actor_id) == current.actor_id:
        return "blocking_subagent_to_parent"
    return "cross_agent_completion"


def _parallel_keys(rows: list[_SourceRow]) -> dict[str, tuple[str, ...]]:
    """Return overlapping request keys with an output-sensitive sweep."""

    overlaps: dict[str, list[str]] = {row.key: [] for row in rows}
    active: dict[str, _SourceRow] = {}
    active_by_finish: list[tuple[datetime, str]] = []

    for row in rows:
        while active_by_finish and active_by_finish[0][0] <= row.started_at:
            _, completed_key = heapq.heappop(active_by_finish)
            active.pop(completed_key, None)

        for candidate in active.values():
            if candidate.started_at < row.finished_at:
                overlaps[candidate.key].append(row.key)
                overlaps[row.key].append(candidate.key)

        active[row.key] = row
        heapq.heappush(active_by_finish, (row.finished_at, row.key))

    return {key: tuple(keys) for key, keys in overlaps.items()}


def _build_session(
    session_id: str,
    source_rows: list[_SourceRow],
    row_analysis_counts: dict[str, int],
) -> ReplaySession:
    rows = sorted(
        source_rows,
        key=lambda row: (row.started_at, row.finished_at, row.key),
    )
    prompt_kinds, inferred_late_titles, inferred_lead_names = _prompt_kinds(rows)
    row_analysis_counts["inferred_late_title_requests"] += inferred_late_titles
    row_analysis_counts["inferred_lead_name_requests"] += inferred_lead_names
    retry_matches, resolved_tokens = _retry_matches(rows, prompt_kinds)
    row_analysis_counts["retry_matched_failures"] += len(retry_matches)
    row_analysis_counts["rows_without_replayable_tokens"] += sum(row.key not in resolved_tokens for row in rows)
    parallel_keys = _parallel_keys(rows)

    origin = rows[0].started_at
    lead_actor_id = next((row.actor_id for row in rows if row.actor_role == "lead"), "lead")
    parents: dict[str, str] = {}
    requests: list[ReplayRequest] = []
    pending_by_finish: list[tuple[datetime, str, _SourceRow]] = []
    predecessor: _SourceRow | None = None
    latest_attempt_by_actor: dict[str, _SourceRow] = {}
    latest_success_by_actor: dict[str, _SourceRow] = {}
    for row in rows:
        while pending_by_finish and pending_by_finish[0][0] <= row.started_at:
            _, _, completed = heapq.heappop(pending_by_finish)
            completed_order = (completed.finished_at, completed.key)
            if predecessor is None or completed_order > (predecessor.finished_at, predecessor.key):
                predecessor = completed

            latest_attempt = latest_attempt_by_actor.get(completed.actor_id)
            if latest_attempt is None or completed_order > (latest_attempt.finished_at, latest_attempt.key):
                latest_attempt_by_actor[completed.actor_id] = completed

            if (
                completed.status == "success"
                and completed.key in resolved_tokens
                and prompt_kinds[completed.key][0] not in _CONTEXT_FREE_PROMPT_KINDS
            ):
                latest_success = latest_success_by_actor.get(completed.actor_id)
                if latest_success is None or completed_order > (latest_success.finished_at, latest_success.key):
                    latest_success_by_actor[completed.actor_id] = completed

        same_actor_success = latest_success_by_actor.get(row.actor_id)
        same_actor_attempt = latest_attempt_by_actor.get(row.actor_id)

        if row.actor_role != "lead" and row.actor_id not in parents:
            parent = (
                predecessor.actor_id
                if predecessor is not None and predecessor.actor_id != row.actor_id
                else lead_actor_id
            )
            parents[row.actor_id] = parent

        token_pair = resolved_tokens.get(row.key)
        retry_match = retry_matches.get(row.key)
        prompt_kind, prompt_kind_source, prompt_kind_inference_rule = prompt_kinds[row.key]
        delay_origin = predecessor.finished_at if predecessor is not None else origin
        requests.append(
            ReplayRequest(
                key=row.key,
                source_line=row.source_line,
                source_request_id=row.request_id,
                actor_id=row.actor_id,
                actor_role=row.actor_role,
                parent_actor_id=parents.get(row.actor_id),
                started_at=row.started_at,
                finished_at=row.finished_at,
                historical_status=row.status,
                source_input_tokens=row.input_tokens,
                source_output_tokens=row.output_tokens,
                source_cached_tokens=row.cached_tokens,
                input_tokens=token_pair[0] if token_pair is not None else None,
                output_tokens=token_pair[1] if token_pair is not None else None,
                replay_kind=("request" if token_pair is not None else "timing_dependency"),
                prompt_kind=prompt_kind if token_pair is not None else "none",
                prompt_kind_source=prompt_kind_source,
                prompt_kind_inference_rule=prompt_kind_inference_rule,
                retry_match_key=retry_match.key if retry_match is not None else None,
                prompt_recipe_key=retry_match.key if retry_match is not None else row.key,
                prompt_ref=(retry_match.prompt_ref if retry_match is not None else row.prompt_ref),
                send_after=predecessor.key if predecessor is not None else None,
                context_after=(
                    same_actor_success.key
                    if same_actor_success is not None and prompt_kind not in _CONTEXT_FREE_PROMPT_KINDS
                    else None
                ),
                delay_seconds=max(
                    0.0,
                    (row.started_at - delay_origin).total_seconds(),
                ),
                dependency_kind=_dependency_kind(predecessor, row, parents),
                same_agent_gap_seconds=(
                    max(
                        0.0,
                        (row.started_at - same_actor_attempt.finished_at).total_seconds(),
                    )
                    if same_actor_attempt is not None
                    else None
                ),
                parallel_with=parallel_keys[row.key],
            )
        )
        heapq.heappush(pending_by_finish, (row.finished_at, row.key, row))

    return ReplaySession(
        source_session_id=session_id,
        source_task_id=next((row.task_id for row in rows if row.task_id), None),
        requests=tuple(requests),
    )


def _pruned_timing_dependency_nodes(session: ReplaySession) -> int:
    required = required_request_keys(session)
    return sum(
        request.replay_kind == "timing_dependency" and request.key not in required for request in session.requests
    )


def _complete_row_analysis(row_analysis_counts: dict[str, int]) -> dict[str, int]:
    """Return a stable audit schema while preserving future counters."""

    complete = dict.fromkeys(_ROW_ANALYSIS_COUNTERS, 0)
    complete.update(row_analysis_counts)
    return dict(sorted(complete.items()))


def analyze_replay_trace(path: Path) -> ReplayAnalysis:
    """Parse a historical JSONL trace and infer Replay relationships."""

    source_path = path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Replay source trace is not a file: {source_path}")
    rows_by_session, row_analysis_counts, total_rows, valid_rows = _read_source_rows(source_path)
    sorted_session_rows = sorted(rows_by_session.items())
    built_sessions: list[ReplaySession] = []
    for session_id, session_rows in sorted_session_rows:
        built_session = _build_session(
            session_id,
            session_rows,
            row_analysis_counts,
        )
        built_sessions.append(built_session)
    sessions = tuple(built_sessions)
    sessions_without_requests = sum(not session.replayable for session in sessions)
    if sessions_without_requests:
        row_analysis_counts["sessions_without_replayable_requests"] += sessions_without_requests
    row_analysis_counts["pruned_terminal_timing_dependency_rows"] += sum(
        _pruned_timing_dependency_nodes(session) for session in sessions
    )
    return ReplayAnalysis(
        source_path=str(source_path),
        source_sha256=_source_sha256(source_path),
        total_rows=total_rows,
        valid_rows=valid_rows,
        row_analysis=_complete_row_analysis(row_analysis_counts),
        sessions=sessions,
    )


def _agent_relationships(session: ReplaySession) -> dict[str, dict[str, object]]:
    by_key = {request.key: request for request in session.requests}
    relationships: dict[str, dict[str, object]] = {}
    for actor_id in sorted({request.actor_id for request in session.requests}):
        actor_requests = [request for request in session.requests if request.actor_id == actor_id]
        relationships[actor_id] = {
            "actor_role": actor_requests[0].actor_role,
            "parent_actor_id": next(
                (request.parent_actor_id for request in actor_requests if request.parent_actor_id is not None),
                None,
            ),
            "parent_inferred": any(request.parent_actor_id is not None for request in actor_requests),
            "overlaps_with_actors": sorted(
                {
                    by_key[key].actor_id
                    for request in actor_requests
                    for key in request.parallel_with
                    if by_key[key].actor_id != actor_id
                }
            ),
        }
    return relationships


def replay_analysis_to_dict(analysis: ReplayAnalysis) -> dict[str, object]:
    """Serialize source analysis as an auditable artifact."""

    return {
        "schema_version": "1",
        "source": analysis.source_path,
        "source_sha256": analysis.source_sha256,
        "total_rows": analysis.total_rows,
        "valid_rows": analysis.valid_rows,
        "row_analysis": analysis.row_analysis,
        "sessions": [
            {
                "source_session_id": session.source_session_id,
                "source_task_id": session.source_task_id,
                "replayable": session.replayable,
                "agent_relationships": _agent_relationships(session),
                "requests": [request.to_dict() for request in session.requests],
            }
            for session in analysis.sessions
        ],
    }
