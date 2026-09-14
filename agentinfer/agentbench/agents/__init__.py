# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Expose shared contracts for benchmark agent runtimes.

Provider dispatch and execution remain in their runtime packages; callers use
these exports to exchange requests, results, and terminal outcome values.
"""

from .contracts import AgentRunOutcome, AgentRunRequest, AgentRunResult, TerminationReason
from .dispatch import run_agent
from .preflight import check_agent_preflight
from .registry import RUNTIMES, get_runtime
from .runtime import AgentProfile, AgentRuntime

__all__ = [
    "AgentProfile",
    "AgentRunOutcome",
    "AgentRunRequest",
    "AgentRunResult",
    "AgentRuntime",
    "RUNTIMES",
    "TerminationReason",
    "check_agent_preflight",
    "get_runtime",
    "run_agent",
]
