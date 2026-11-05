# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Register the pinned ZCode command-line runtime."""

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from ...request_proxy.observers import _OpenAISSEUsageObserver
from ..contracts import AgentPreflightReport, AgentRunRequest, AgentRunResult
from ..preflight import check_executable_output
from .profiles import ZCodeProfile, get_profile
from .runner import run_zcode

ZCODE_VERSION = "0.17.1"


class ZCodeRuntime:
    agent_type = "zcode"
    required_endpoint = "/v1/chat/completions"
    supports_api_key_env = False
    usage_observer_class = _OpenAISSEUsageObserver

    def get_profile(self, name: str) -> ZCodeProfile:
        return get_profile(name)

    async def preflight(self, executable: Path) -> AgentPreflightReport:
        path = await check_executable_output(str(executable), ["--help"], label="ZCode", expected=("--prompt", "--cwd"))
        version = await asyncio.to_thread(
            subprocess.run,
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        if ZCODE_VERSION not in version.stdout:
            raise RuntimeError(f"Expected ZCode CLI {ZCODE_VERSION}, got {version.stdout.strip()}")
        doctor = await asyncio.to_thread(
            subprocess.run,
            [str(path), "doctor", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return AgentPreflightReport(
            runtime="zcode",
            executable=str(path),
            versions={"zcode": version.stdout.strip()},
            capabilities={"headless": True, "hooks": True, "sandbox": "external"},
            installation={
                "launcher_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "doctor": json.loads(doctor.stdout),
            },
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        return await run_zcode(request)
