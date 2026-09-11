# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Dispatch benchmark tasks to concrete agent runtimes."""

from .contracts import AgentRunRequest, AgentRunResult
from .registry import get_runtime


async def run_agent(request: AgentRunRequest) -> AgentRunResult:
    """Run one task using the concrete runtime named by ``request.agent_type``."""

    return await get_runtime(request.agent_type).run(request)
