# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Define shared requests, results, and outcome enums for agent runtimes.

The runner and runtime boundary starts with ``AgentRunRequest`` and returns
``AgentRunResult``. Runtime dispatch and provider-specific execution are owned
by later modules rather than this contract layer.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..benchkit.dataset import Task


class AgentRunOutcome(str, Enum):
    """Classify execution for coarse task-level aggregation."""

    COMPLETED = "completed"
    FAILED = "failed"


class TerminationReason(str, Enum):
    """Diagnose why execution ended without normal completion."""

    AGENT_STARTUP_FAILED = "agent_startup_failed"
    AGENT_STARTUP_TIMEOUT = "agent_startup_timeout"
    TIMEOUT = "timeout"
    PLAN_EXIT_LOOP = "plan_exit_loop"
    VALIDATION_ERROR_LOOP = "validation_error_loop"
    CONFIRMATION_HANG = "confirmation_hang"
    IDLE_AFTER_PATCH = "idle_after_patch"
    PROMPT_SUBMISSION_FAILED = "prompt_submission_failed"
    INTERRUPTED = "interrupted"
    HARNESS_ERROR = "harness_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AgentRunRequest:
    """Describe one task execution request passed to an agent runtime."""

    agent_type: str
    task: Task
    profile_name: str
    executable: Path
    model: str
    api_base_url: str
    workspace: Path
    artifact_dir: Path
    session_id: str
    timeout_seconds: int
    patch_flush_seconds: int
    prompt_delivery_timeout_seconds: int
    tmux_startup_seconds: float
    terminal_capture_interval_seconds: int


@dataclass
class AgentRunResult:
    """Record one runtime execution result and its generated evidence.

    Normal completion is represented by ``outcome=COMPLETED`` and no
    termination reason. Failed execution pairs ``outcome=FAILED`` with the
    reason it stopped. Correctness is evaluated separately from this result.
    """

    agent_type: str
    profile_name: str
    outcome: AgentRunOutcome = AgentRunOutcome.FAILED
    termination_reason: TerminationReason | None = None
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    session_id: str = ""
    tmux_session: str | None = None
    transcript: str | None = None
    auto_plan_approvals: int = 0
    auto_yes_confirmations: int = 0
    auto_generic_selections: int = 0
    prompt_submission_retries: int = 0
    instance_id: str = ""
    patch_bytes: int = 0
    has_patch: bool = False

    def __post_init__(self) -> None:
        """Enforce consistent outcome and termination reason pairs."""

        if self.outcome is AgentRunOutcome.COMPLETED:
            if self.termination_reason is not None:
                raise ValueError("completed outcome cannot have a termination reason")
        elif self.termination_reason is None:
            self.termination_reason = TerminationReason.HARNESS_ERROR
