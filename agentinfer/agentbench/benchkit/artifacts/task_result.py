# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Persist per-task runtime results and wall-clock task boundaries."""

from dataclasses import asdict, dataclass
from typing import Literal

from ...agents.contracts import AgentRunResult
from ..metrics.request import ObservedTopology


@dataclass(frozen=True)
class AgentIdentity:
    type: str
    profile: str


@dataclass(frozen=True)
class TaskResultArtifact:
    """Per-task artifact preserving runtime evidence and wall-clock boundaries."""

    schema_version: Literal["1"]
    instance_id: str
    session_id: str
    agent: AgentIdentity
    outcome: str
    termination_reason: str | None
    duration_seconds: float
    has_patch: bool
    patch_bytes: int
    topology: ObservedTopology
    error: dict[str, str] | None
    # Defaults let pre-change (schema-v1) ``result.json`` that predate the
    # task-position/timestamp fields load via ``TypeAdapter``; ``-1``/``""``
    # are sentinel values that cannot be produced by a real run (positions are
    # non-negative, timestamps are ISO-8601), so a missing field stays visible.
    task_position: int = -1
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_task_result(
    result: AgentRunResult,
    topology: ObservedTopology,
    error: dict[str, str] | None = None,
    *,
    task_position: int,
) -> TaskResultArtifact:
    return TaskResultArtifact(
        "1",
        result.instance_id,
        result.session_id,
        AgentIdentity(result.agent_type, result.profile_name),
        result.outcome.value,
        result.termination_reason.value if result.termination_reason else None,
        result.duration_seconds,
        result.has_patch,
        result.patch_bytes,
        topology,
        error,
        task_position,
        result.started_at,
        result.finished_at,
    )
