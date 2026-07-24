"""Task result artifact schema and builder."""

from dataclasses import asdict, dataclass
from typing import Literal

from ...agents.contracts import AgentRunResult
from ..metrics.request import ObservedTopology
from ..session_registration import SessionRegistrationResult


@dataclass(frozen=True)
class AgentIdentity:
    type: str
    profile: str


@dataclass(frozen=True)
class TaskResultArtifact:
    schema_version: Literal["1"]
    instance_id: str
    session_id: str
    agent: AgentIdentity
    outcome: str
    termination_reason: str | None
    duration_seconds: float
    has_patch: bool
    patch_bytes: int
    registration: SessionRegistrationResult | None
    cleanup: SessionRegistrationResult | None
    topology: ObservedTopology
    error: dict[str, str] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_task_result(
    result: AgentRunResult,
    registration: SessionRegistrationResult | None,
    cleanup: SessionRegistrationResult | None,
    topology: ObservedTopology,
    error: dict[str, str] | None = None,
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
        registration,
        cleanup,
        topology,
        error,
    )
