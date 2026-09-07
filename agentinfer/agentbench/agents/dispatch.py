# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Dispatch benchmark tasks to concrete agent runtimes."""

from .claude.runner import run_claude
from .contracts import AgentRunRequest, AgentRunResult


async def run_agent(request: AgentRunRequest) -> AgentRunResult:
    """Run one task using the concrete runtime named by ``request.agent_type``."""

    if request.agent_type == "claude":
        return await run_claude(request)
    raise ValueError(f"Unsupported agent.type: {request.agent_type!r}")
