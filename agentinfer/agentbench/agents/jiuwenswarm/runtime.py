# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""JiuwenSwarm agent runtime adapter for the benchmark harness."""

import shutil
from pathlib import Path

from ...request_proxy.observers import _OpenAISSEUsageObserver
from ..preflight import check_executable_output
from ..runtime import AgentProfile
from .instance import _companion_executable
from .profiles import REQUIRED_ENDPOINT, get_profile
from .runner import run_jiuwenswarm


class JiuwenSwarmRuntime:
    """Agent runtime for JiuwenSwarm, dispatched through the registry."""

    agent_type = "jiuwenswarm"
    required_endpoint = REQUIRED_ENDPOINT
    usage_observer_class = _OpenAISSEUsageObserver

    def get_profile(self, name: str) -> AgentProfile:
        return get_profile(name)

    async def preflight(self, executable: Path) -> None:
        command = str(executable)
        await check_executable_output(command, ["--help"], label="Agent", expected=("chat",))
        await check_executable_output(
            command,
            ["chat", "--help"],
            label="Agent",
            expected=("--jsonl", "--mode", "--session", "--cwd", "--project-dir", "--gateway-url", "--timeout"),
        )
        # jiuwenbox-server and bwrap back the mandatory SysOperation SANDBOX.
        # Resolve the package-installed server beside an explicit Jiuwen CLI
        # before consulting PATH, matching the runtime launcher.
        jiuwenbox_server = _companion_executable(executable, "jiuwenbox-server")
        if shutil.which(jiuwenbox_server) is None:
            raise RuntimeError(
                f"jiuwenbox-server is required for JiuwenSwarm isolation but is not executable: {jiuwenbox_server}"
            )
        if shutil.which("bwrap") is None:
            raise RuntimeError("bwrap is required for JiuwenSwarm isolation but was not found on PATH")

    async def run(self, request):  # type: ignore[no-untyped-def]
        return await run_jiuwenswarm(request)
