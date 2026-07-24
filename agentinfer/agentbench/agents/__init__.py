"""Expose shared contracts for benchmark agent runtimes.

Provider dispatch and execution remain in their runtime packages; callers use
these exports to exchange requests, results, and terminal outcome values.
"""

from .contracts import AgentRunRequest, AgentRunResult
from .dispatch import run_agent
from .outcomes import AgentRunOutcome, TerminationReason

__all__ = [
    "AgentRunOutcome",
    "AgentRunRequest",
    "AgentRunResult",
    "TerminationReason",
    "run_agent",
]
