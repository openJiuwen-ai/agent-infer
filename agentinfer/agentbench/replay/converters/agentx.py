# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Convert AgentX nested sessions to complete hash-snapshot Replay recipes."""

from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..schema import ReplayAnalysis, ReplayRequest, ReplaySession
from ..unified_trace_ir import UnifiedTraceIR, sha256_file, write_trace_ir_manifest
from .base import ConverterSummary, ReplayDatasetConverter


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < (1 if positive else 0):
        raise ValueError(f"AgentX {label} must be a finite {'positive' if positive else 'non-negative'} number")
    return float(value)


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"AgentX {label} must be a {'positive' if positive else 'non-negative'} integer")
    return value


class AgentXConverter(ReplayDatasetConverter):
    """Retain every model request and infer timing edges from completed calls."""

    name = "agentx-hash-snapshot/v1"

    def convert(self, source: Path, output_dir: Path) -> ConverterSummary:
        source = source.resolve()
        if not source.is_file() or source.suffix != ".jsonl":
            raise ValueError(f"AgentX source must be an existing JSONL file: {source}")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        output_dir.mkdir(parents=True)
        sessions = requests = zero_output = subagent_groups = 0
        identities: set[str] = set()
        with (
            source.open(encoding="utf-8") as handle,
            (output_dir / "requests.jsonl").open("w", encoding="utf-8") as output,
        ):
            for source_line, line in enumerate(handle, start=1):
                try:
                    session = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"AgentX line {source_line} has invalid JSON: {exc}") from exc
                if not isinstance(session, dict):
                    raise ValueError(f"AgentX line {source_line} must be an object")
                source_id = session.get("id")
                if not isinstance(source_id, str) or not source_id or source_id in identities:
                    raise ValueError(f"AgentX line {source_line} has invalid or duplicate id")
                identities.add(source_id)
                block_size = _integer(session.get("block_size"), f"line {source_line} block_size", positive=True)
                if session.get("hash_id_scope") != "local":
                    raise ValueError(f"AgentX line {source_line} requires hash_id_scope=local")
                groups = session.get("requests")
                if not isinstance(groups, list) or not groups:
                    raise ValueError(f"AgentX line {source_line} has no requests")
                session_id = f"agentx-{source_id}"
                flattened: list[dict[str, object]] = []
                actors: set[str] = set()
                for outer_index, outer in enumerate(groups):
                    if not isinstance(outer, dict):
                        raise ValueError(f"AgentX line {source_line} request {outer_index} is not an object")
                    if outer.get("type") == "subagent":
                        subagent_groups += 1
                        actor = outer.get("agent_id")
                        if not isinstance(actor, str) or not actor or actor in actors or actor == "lead":
                            raise ValueError(f"AgentX line {source_line} request {outer_index} has invalid agent_id")
                        actors.add(actor)
                        inner = outer.get("requests")
                        if not isinstance(inner, list) or not inner:
                            raise ValueError(f"AgentX line {source_line} subagent {actor} has no requests")
                        for inner_index, request in enumerate(inner):
                            flattened.append(
                                self._request(
                                    request,
                                    session_id,
                                    source_line,
                                    outer_index,
                                    inner_index,
                                    actor,
                                    "subagent",
                                    block_size,
                                )
                            )
                    else:
                        flattened.append(
                            self._request(
                                outer,
                                session_id,
                                source_line,
                                outer_index,
                                None,
                                "lead",
                                "lead",
                                block_size,
                            )
                        )
                flattened.sort(key=lambda row: (row["t"], row["source_request_index"], row["source_inner_index"]))
                origin = float(flattened[0]["t"])
                active: list[tuple[float, str]] = []
                latest_completed: tuple[float, str] | None = None
                for sequence, row in enumerate(flattened):
                    started = float(row["t"])
                    while active and active[0][0] <= started:
                        completed = heapq.heappop(active)
                        if latest_completed is None or completed > latest_completed:
                            latest_completed = completed
                    predecessor = latest_completed[1] if latest_completed else None
                    previous_finish = latest_completed[0] if latest_completed else origin
                    row["sequence_index"] = sequence
                    row["send_after"] = predecessor
                    row["source_gap_seconds"] = max(0.0, started - previous_finish)
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    heapq.heappush(active, (started + float(row["api_time"]), str(row["request_id"])))
                    zero_output += row["output_tokens"] == 0
                    requests += 1
                sessions += 1
        summary = ConverterSummary("agentx", sessions, requests, 0)
        write_trace_ir_manifest(
            output_dir,
            converter_name=self.name,
            source_path=source,
            summary={**summary.to_dict(), "zero_output_requests": zero_output, "subagent_groups": subagent_groups},
            prompt_source_kind="hash_snapshot",
        )
        return summary

    @staticmethod
    def _request(
        raw: object,
        session_id: str,
        source_line: int,
        outer_index: int,
        inner_index: int | None,
        actor: str,
        role: str,
        block_size: int,
    ) -> dict[str, object]:
        location = f"line {source_line} request {outer_index} inner {inner_index}"
        if not isinstance(raw, dict) or raw.get("type") == "subagent":
            raise ValueError(f"AgentX {location} must be a model request")
        started = _number(raw.get("t"), f"{location} t")
        duration = _number(raw.get("api_time"), f"{location} api_time")
        input_tokens = _integer(raw.get("in"), f"{location} in", positive=True)
        output_tokens = _integer(raw.get("out"), f"{location} out")
        model = raw.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"AgentX {location} has invalid model")
        hashes = raw.get("hash_ids")
        if (
            not isinstance(hashes, list)
            or not hashes
            or any(type(item) is not int or item < 0 for item in hashes)
            or len(hashes) != (input_tokens + block_size - 1) // block_size
        ):
            raise ValueError(f"AgentX {location} hash_ids do not match in/block_size")
        return {
            "request_id": f"{session_id}-{outer_index:08d}-{(inner_index if inner_index is not None else 0):08d}",
            "session_id": session_id,
            "source_session_id": session_id.removeprefix("agentx-"),
            "source_line": source_line,
            "source_request_index": outer_index,
            "source_inner_index": inner_index if inner_index is not None else -1,
            "actor_id": actor,
            "actor_role": role,
            "parent_actor_id": "lead" if role == "subagent" else None,
            "source_model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "block_size": block_size,
            "hash_id_scope": "local",
            "hash_ids": hashes,
            "t": started,
            "api_time": duration,
            "context_after": None,
            "prompt_ref": None,
            "timing_inference": "latest_completed_global",
        }


def analyze_agentx_trace_ir(trace_ir: UnifiedTraceIR) -> ReplayAnalysis:
    """Build the explicit timing graph while leaving full hash lists on disk."""

    if trace_ir.prompt_source_kind != "hash_snapshot":
        raise ValueError("AgentX analysis requires hash_snapshot Trace IR")
    rows_by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    compact_fields = (
        "request_id",
        "session_id",
        "source_session_id",
        "source_line",
        "source_request_index",
        "source_inner_index",
        "actor_id",
        "actor_role",
        "parent_actor_id",
        "source_model",
        "input_tokens",
        "output_tokens",
        "block_size",
        "hash_id_scope",
        "t",
        "api_time",
        "sequence_index",
        "send_after",
        "source_gap_seconds",
        "timing_inference",
    )
    with trace_ir.requests_path.open("rb") as handle:
        while line := handle.readline():
            offset = handle.tell() - len(line)
            raw = json.loads(line)
            row = {key: raw[key] for key in compact_fields}
            row["block_count"] = len(raw["hash_ids"])
            row["recipe_offset"] = offset
            rows_by_session[str(row["session_id"])].append(row)

    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    sessions: list[ReplaySession] = []
    zero_output = subagent_requests = overlap_pairs = 0
    for session_id, rows in sorted(rows_by_session.items()):
        by_key = {str(row["request_id"]): row for row in rows}
        overlaps: dict[str, list[str]] = {key: [] for key in by_key}
        active: list[dict[str, object]] = []
        for row in rows:
            started = float(row["t"])
            active = [old for old in active if float(old["t"]) + float(old["api_time"]) > started]
            key = str(row["request_id"])
            for old in active:
                old_key = str(old["request_id"])
                overlaps[key].append(old_key)
                overlaps[old_key].append(key)
                overlap_pairs += 1
            active.append(row)
        requests: list[ReplayRequest] = []
        seen_actors: set[str] = set()
        for row in rows:
            key = str(row["request_id"])
            predecessor = str(row["send_after"]) if row["send_after"] is not None else None
            predecessor_actor = str(by_key[predecessor]["actor_id"]) if predecessor else None
            actor = str(row["actor_id"])
            role = str(row["actor_role"])
            if predecessor is None:
                kind = "session_root"
            elif predecessor_actor == actor:
                kind = "same_agent"
            else:
                kind = "cross_agent_completion"
            gap = float(row["source_gap_seconds"])
            started = epoch + timedelta(seconds=float(row["t"]))
            finished = started + timedelta(seconds=float(row["api_time"]))
            out = int(row["output_tokens"])
            zero_output += out == 0
            subagent_requests += role == "subagent"
            requests.append(
                ReplayRequest(
                    key=key,
                    source_line=int(row["source_line"]),
                    source_request_id=key,
                    actor_id=actor,
                    actor_role="subagent" if role == "subagent" else "lead",
                    parent_actor_id=str(row["parent_actor_id"]) if row["parent_actor_id"] else None,
                    started_at=started,
                    finished_at=finished,
                    historical_status="unknown",
                    source_input_tokens=int(row["input_tokens"]),
                    source_output_tokens=out,
                    source_cached_tokens=None,
                    input_tokens=int(row["input_tokens"]),
                    output_tokens=out,
                    replay_kind="request",
                    prompt_kind=("subagent_first" if role == "subagent" else "lead_main")
                    if actor not in seen_actors
                    else "continuation",
                    prompt_kind_source="explicit",
                    prompt_kind_inference_rule=None,
                    retry_match_key=None,
                    prompt_recipe_key=str(row["recipe_offset"]),
                    prompt_ref=None,
                    send_after=predecessor,
                    context_after=None,
                    delay_seconds=gap,
                    dependency_kind=kind,  # type: ignore[arg-type]
                    same_agent_gap_seconds=gap if kind == "same_agent" else None,
                    parallel_with=tuple(overlaps[key]),
                    source_details={
                        "source_session_id": row["source_session_id"],
                        "source_model": row["source_model"],
                        "source_request_index": row["source_request_index"],
                        "source_inner_index": row["source_inner_index"],
                        "block_size": row["block_size"],
                        "block_count": row["block_count"],
                        "hash_id_scope": row["hash_id_scope"],
                        "timing_inference": row["timing_inference"],
                    },
                )
            )
            seen_actors.add(actor)
        sessions.append(ReplaySession(session_id, None, tuple(requests)))
    total = sum(len(session.requests) for session in sessions)
    return ReplayAnalysis(
        source_path=str(trace_ir.requests_path),
        source_sha256=sha256_file(trace_ir.requests_path),
        total_rows=total,
        valid_rows=total,
        row_analysis={
            "zero_output_requests": zero_output,
            "subagent_requests": subagent_requests,
            "overlapping_request_pairs": overlap_pairs,
        },
        sessions=tuple(sessions),
    )
