# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Claude Code agent runtime adapter for the benchmark harness."""

import shutil
from pathlib import Path

from ...request_proxy.observers import _AnthropicSSEUsageObserver
from ..preflight import check_executable_output
from ..runtime import AgentProfile
from .profiles import REQUIRED_ENDPOINT, get_profile
from .runner import run_claude


class ClaudeRuntime:
    """Agent runtime for Claude Code, dispatched through the registry."""

    agent_type = "claude"
    required_endpoint = REQUIRED_ENDPOINT
    usage_observer_class = _AnthropicSSEUsageObserver

    def get_profile(self, name: str) -> AgentProfile:
        return get_profile(name)

    async def preflight(self, executable: Path) -> None:
        await check_executable_output("tmux", ["-V"], label="tmux")
        await check_executable_output(str(executable), ["--version"], label="Agent")
        # bubblewrap is the Linux write-restricting sandbox backing Claude's
        # Bash sandbox; socat is its network-proxy component. The Bash sandbox
        # hard-fails at launch without either (failIfUnavailable).
        for tool in ("bwrap", "socat"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"{tool} is required for Claude isolation but was not found on PATH")

    async def run(self, request):  # type: ignore[no-untyped-def]
        return await run_claude(request)
