# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Immutable source-analysis models for Trace Replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

from .unified_trace_ir import PromptReference

PromptKind = Literal["lead_title", "lead_name", "lead_main", "subagent_first", "continuation", "none"]


@dataclass(frozen=True)
class ReplayRequest:
    """One historical request attempt with inferred Replay relationships."""

    key: str
    source_line: int
    source_request_id: str | None
    actor_id: str
    actor_role: Literal["lead", "subagent", "unknown"]
    parent_actor_id: str | None
    started_at: datetime
    finished_at: datetime
    historical_status: str
    source_input_tokens: int | None
    source_output_tokens: int | None
    source_cached_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    replay_kind: Literal["request", "timing_dependency"]
    prompt_kind: PromptKind
    prompt_kind_source: Literal["explicit", "inferred"]
    prompt_kind_inference_rule: str | None
    retry_match_key: str | None
    prompt_recipe_key: str
    prompt_ref: PromptReference | None
    send_after: str | None
    context_after: str | None
    delay_seconds: float
    dependency_kind: Literal[
        "session_root",
        "same_agent",
        "parent_to_subagent",
        "blocking_subagent_to_parent",
        "cross_agent_completion",
    ]
    same_agent_gap_seconds: float | None
    parallel_with: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize timestamps and tuples for JSON artifacts."""

        value = asdict(self)
        value["started_at"] = self.started_at.isoformat()
        value["finished_at"] = self.finished_at.isoformat()
        value["parallel_with"] = list(self.parallel_with)
        return value


@dataclass(frozen=True)
class ReplaySession:
    """A source session reconstructed from historical attempts."""

    source_session_id: str
    source_task_id: str | None
    requests: tuple[ReplayRequest, ...]

    @property
    def replayable(self) -> bool:
        """Return whether the session contains a real Backend request."""

        return any(request.replay_kind == "request" for request in self.requests)


@dataclass(frozen=True)
class ReplayAnalysis:
    """Analyzer output plus data-quality counters."""

    source_path: str
    source_sha256: str
    total_rows: int
    valid_rows: int
    row_analysis: dict[str, int]
    sessions: tuple[ReplaySession, ...]


@dataclass(frozen=True)
class SampledSession:
    """One independent runtime sample of a source session."""

    task_index: int
    sample_ordinal: int
    runtime_session_id: str
    source: ReplaySession


def required_request_keys(session: ReplaySession) -> frozenset[str]:
    """Return replayable requests and their transitive send dependencies."""

    by_key = {request.key: request for request in session.requests}
    required = {request.key for request in session.requests if request.replay_kind == "request"}
    frontier = list(required)
    while frontier:
        predecessor = by_key[frontier.pop()].send_after
        if predecessor is not None and predecessor not in required:
            required.add(predecessor)
            frontier.append(predecessor)
    return frozenset(required)
