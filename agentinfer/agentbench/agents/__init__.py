# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Expose shared contracts for benchmark agent runtimes.

Provider dispatch and execution remain in their runtime packages; callers use
these exports to exchange requests, results, and terminal outcome values.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .contracts import AgentRunOutcome, AgentRunRequest, AgentRunResult, TerminationReason
from .runtime import AgentProfile, AgentRuntime

if TYPE_CHECKING:
    from .dispatch import run_agent
    from .preflight import check_agent_preflight
    from .registry import RUNTIMES, get_runtime

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


def __getattr__(name: str):
    # Harbor adapters can import a single runtime's helpers in Harbor's Python environment.
    modules = {
        "run_agent": "dispatch",
        "check_agent_preflight": "preflight",
        "RUNTIMES": "registry",
        "get_runtime": "registry",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f".{modules[name]}", __name__), name)
