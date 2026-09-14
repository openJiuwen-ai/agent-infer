# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""DSH agent runtime adapter for the benchmark harness."""

import asyncio
import importlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from ...request_proxy.observers import _OpenAISSEUsageObserver
from ..preflight import check_executable_output
from ..runtime import AgentProfile
from .profiles import REQUIRED_ENDPOINT, get_profile
from .runner import run_dsh


def _require_zstandard() -> None:
    try:
        importlib.import_module("zstandard")
    except ImportError as exc:
        raise RuntimeError(
            "DSH requires the Python package 'zstandard' to materialize session artifacts; "
            "install the AgentInfer dependencies in the benchmark environment"
        ) from exc


# Probe the verified DSH-internal surface (dsh 0.1.1-rc.2, Node >= 22):
# @deepseek-ai/cordis Context + @deepseek-ai/dsh-sandbox-local
# LocalSandboxProvider with ctx.sandbox.confine and ctx.fiber.dispose. An
# upstream refactor of these internals fails closed and blocks all runs with
# "Agent sandbox preflight check failed".
_SANDBOX_PROBE = r"""
import { spawnSync } from "node:child_process";
import { realpathSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(realpathSync(process.argv[2]));
const [{ Context }, { LocalSandboxProvider }] = await Promise.all([
  import(pathToFileURL(require.resolve("@deepseek-ai/cordis"))),
  import(pathToFileURL(require.resolve("@deepseek-ai/dsh-sandbox-local"))),
]);
const ctx = new Context();
await ctx.plugin(LocalSandboxProvider, {});
try {
  const confined = ctx.sandbox.confine(
    [process.execPath, "-e", ""],
    { mode: "workspace-write", workspaceRoot: process.cwd() },
  );
  const result = spawnSync(confined.argv[0], confined.argv.slice(1), {
    stdio: "inherit",
    timeout: 10_000,
  });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
} finally {
  await ctx.fiber.dispose();
}
"""


def _probe_sandbox(executable: str) -> None:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"Agent executable is not available: {executable}")
    with tempfile.TemporaryDirectory() as workspace:
        try:
            result = subprocess.run(
                ["node", "--input-type=module", "-", str(Path(resolved).resolve())],
                input=_SANDBOX_PROBE.encode(),
                cwd=workspace,
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Agent sandbox preflight check failed") from exc
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip() or result.stdout.decode(errors="replace").strip()
        message = "Agent sandbox preflight check failed"
        raise RuntimeError(f"{message}\n{detail}" if detail else message)


class DshRuntime:
    """Agent runtime for the DeepSeek Harness headless CLI, dispatched through the registry."""

    agent_type = "dsh"
    required_endpoint = REQUIRED_ENDPOINT
    usage_observer_class = _OpenAISSEUsageObserver

    def get_profile(self, name: str) -> AgentProfile:
        return get_profile(name)

    async def preflight(self, executable: Path) -> None:
        _require_zstandard()
        command = str(executable)
        await check_executable_output(command, ["--version"], label="Agent")
        await asyncio.to_thread(_probe_sandbox, command)

    async def run(self, request):  # type: ignore[no-untyped-def]
        return await run_dsh(request)
