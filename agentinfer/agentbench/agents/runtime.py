# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Declare the agent runtime contract shared by every benchmark agent runtime.

The harness dispatches profile lookup, preflight, endpoint requirement, and
execution through this interface. Concrete runtimes keep their own policy
fields (e.g. Claude's ``tool_policy``) on the concrete profile type; only the
common denominator (``name`` + ``build_prompt``) is declared here.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..benchkit.dataset import Task

if TYPE_CHECKING:
    from .contracts import AgentRunRequest, AgentRunResult


@runtime_checkable
class AgentProfile(Protocol):
    """Common read-side contract every runtime profile exposes to the harness."""

    name: str
    build_prompt: Callable[[Task], str]


@runtime_checkable
class AgentRuntime(Protocol):
    """Contract every benchmark agent runtime implements.

    The harness resolves a runtime by ``agent_type`` and delegates profile
    lookup, preflight, the required model endpoint, and task execution to it.
    Runtime-specific knobs (``tmux_startup_seconds`` etc.) live on the
    concrete runtime or its profile, not on this interface.
    """

    agent_type: str
    required_endpoint: str
    usage_observer_class: type

    def get_profile(self, name: str) -> AgentProfile: ...

    async def preflight(self, executable: Path) -> None: ...

    async def run(self, request: "AgentRunRequest") -> "AgentRunResult": ...
