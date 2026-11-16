# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Register the pinned ZCode command-line runtime."""

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from ...request_proxy.observers import _OpenAISSEUsageObserver
from ..contracts import AgentPreflightReport, AgentRunRequest, AgentRunResult
from ..preflight import check_executable_output
from .instance import prompt_environment
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
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            home = cwd / "zcode-home"
            home.mkdir()
            environment = {
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"}
            }
            environment.update(
                {
                    "HOME": str(home),
                    "USERPROFILE": str(home),
                    "TMPDIR": directory,
                    "ZCODE_DATA_BASE_DIR": str(home),
                    "ZCODE_STORAGE_DIR": str(home / ".zcode/cli"),
                    "ZCODE_LOG_DIR": str(home / "logs"),
                    "ZCODE_SESSION_DB_PATH": str(home / ".zcode/cli/db.sqlite"),
                }
            )
            path = await check_executable_output(
                str(executable),
                ["--help"],
                label="ZCode",
                expected=("--prompt", "--cwd"),
                cwd=cwd,
                env=environment,
            )

            async def probe(args, env=environment):
                try:
                    return await asyncio.to_thread(
                        subprocess.run,
                        [str(path), *args],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=True,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, subprocess.CalledProcessError):
                        detail += f"\n{exc.stderr}"
                    raise RuntimeError(
                        f"ZCode preflight failed: expected={ZCODE_VERSION}, path={path}\n{detail}"
                    ) from exc

            version = await probe(["--version"])
            if version.stdout.strip() != ZCODE_VERSION:
                raise RuntimeError(f"Expected ZCode CLI {ZCODE_VERSION}, got {version.stdout.strip()}")
            prompt = cwd / "prompt.txt"
            prompt.write_text("AgentBench prompt transport probe")
            await probe(
                ["--prompt", str(prompt), "--help"],
                env={**environment, **prompt_environment(prompt)},
            )
            if not (home / "prompt.loaded").exists():
                raise RuntimeError("ZCode launcher must preserve NODE_OPTIONS for prompt file loading")
            doctor = await probe(["doctor", "--json"])
        return AgentPreflightReport(
            runtime="zcode",
            executable=str(path),
            versions={"zcode": version.stdout.strip()},
            capabilities={"headless": True, "hooks": True, "prompt_file": True, "sandbox": "external"},
            installation={
                "launcher_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "doctor": json.loads(doctor.stdout),
            },
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        return await run_zcode(request)
