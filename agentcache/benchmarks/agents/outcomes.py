"""Define execution status shared by agent runtimes and benchkit.

``AgentRunOutcome`` provides the coarse completed/failed dimension used for
aggregation. ``TerminationReason`` diagnoses why execution failed or stopped
without normal completion. Neither enum represents task correctness, which is
evaluated from separate evidence.
"""

from enum import Enum


class AgentRunOutcome(str, Enum):
    """Classify execution for coarse task-level aggregation."""

    COMPLETED = "completed"
    FAILED = "failed"


class TerminationReason(str, Enum):
    """Diagnose why execution ended without normal completion."""

    TIMEOUT = "timeout"
    PLAN_EXIT_LOOP = "plan_exit_loop"
    VALIDATION_ERROR_LOOP = "validation_error_loop"
    CONFIRMATION_HANG = "confirmation_hang"
    IDLE_AFTER_PATCH = "idle_after_patch"
    PROMPT_SUBMISSION_FAILED = "prompt_submission_failed"
    INTERRUPTED = "interrupted"
    HARNESS_ERROR = "harness_error"
    CANCELLED = "cancelled"
