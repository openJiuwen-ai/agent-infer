"""Define execution outcomes separately from task correctness."""

from enum import Enum


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
