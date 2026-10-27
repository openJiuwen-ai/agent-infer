# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""DSH agent runtime adapter for the benchmark harness."""

import asyncio
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ...request_proxy.observers import _OpenAISSEUsageObserver
from ..contracts import AgentPreflightReport
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


# Probe the fixed DSH 0.2.1-rc.2 installation, including async confine():
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
  const confined = await ctx.sandbox.confine(
    [process.execPath, "-e", ""],
    { mode: "workspace-write", workspaceRoot: process.cwd() },
  );
  const result = spawnSync(confined.argv[0], confined.argv.slice(1), {
    stdio: "inherit",
    timeout: 10_000,
  });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
  console.log(JSON.stringify({ sandbox: confined.enforcement, sandbox_command: confined.argv.slice(0, 2) }));
} finally {
  await ctx.fiber.dispose();
}
"""


def _probe_sandbox(executable: str) -> dict[str, object]:
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
    return json.loads(result.stdout)


DSH_VERSION = "0.2.1-rc.2"
_METADATA_PROBE = r"""
import { realpathSync, readFileSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';
const require = createRequire(realpathSync(process.argv[2]));
const names = ['@deepseek-ai/dsh', '@deepseek-ai/cordis', '@deepseek-ai/dsh-llm-pi-ai',
  '@earendil-works/pi-ai', '@deepseek-ai/dsh-sandbox-local', '@deepseek-ai/dsh-session-persistence-jsonl'];
const packages = {};
for (const name of names) {
  const scopedRequire = name === '@earendil-works/pi-ai'
    ? createRequire(require.resolve('@deepseek-ai/dsh-llm-pi-ai')) : require;
  const path = scopedRequire.resolve.paths(name).map(dir => join(dir, name, 'package.json')).find(existsSync);
  if (!path) throw new Error('Missing DSH runtime package: ' + name);
  const pkg = JSON.parse(readFileSync(path, 'utf8'));
  packages[name] = { version: pkg.version, path };
}
console.log(JSON.stringify({ node: process.versions.node, packages }));
"""


def _runtime_report(executable: str) -> AgentPreflightReport:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"DSH executable is not available: {executable}")
    path = Path(resolved).resolve()
    version = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=20, check=True)
    actual = re.search(r"\b\d+\.\d+\.\d+(?:-[\w.]+)?", version.stdout + version.stderr)
    if actual is None or actual.group() != DSH_VERSION:
        raise RuntimeError(
            f"Unsupported DSH version: actual={version.stdout.strip()!r}, expected={DSH_VERSION}, path={path}"
        )
    metadata = subprocess.run(
        ["node", "--input-type=module", "-", str(path)],
        input=_METADATA_PROBE,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    data = json.loads(metadata.stdout)
    node = tuple(int(part) for part in data["node"].split("."))
    if not ((node[0] == 22 and node >= (22, 19, 0)) or node[0] >= 24):
        raise RuntimeError(f"Unsupported Node version: actual={data['node']}, expected=^22.19.0 || >=24, path={path}")
    versions = {name: pkg["version"] for name, pkg in data["packages"].items()}
    for name, value in versions.items():
        if name.startswith("@deepseek-ai/dsh") and value != DSH_VERSION:
            raise RuntimeError(
                f"DSH package version mismatch: {name} actual={value}, expected={DSH_VERSION}, path={path}"
            )
    versions["node"] = data["node"]
    lock = next(
        (parent / "package-lock.json" for parent in path.parents if (parent / "package-lock.json").is_file()), None
    )
    capabilities = _probe_sandbox(str(path))
    with tempfile.TemporaryDirectory() as workspace:
        environment = {key: value for key, value in os.environ.items() if not key.startswith(("DSH_", "DEEPSEEK_"))}
        environment["DSH_HOME"] = str(Path(workspace) / "home")
        for flag in ("--help", "--dump-config"):
            probe = subprocess.run(
                [str(path), "--profile", "headless", flag],
                env=environment,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            if flag == "--help" and not all(token in probe.stdout for token in ("--json", "--session-id", "stdin")):
                raise RuntimeError(f"DSH headless lacks stdin/JSON/session capabilities: {path}")
    capabilities.update(headless=True, json_stdout=True, stdin=True, dump_config=True, patch=True)
    return AgentPreflightReport(
        runtime="dsh",
        executable=str(path),
        versions=versions,
        capabilities=capabilities,
        installation={
            "packages": data["packages"],
            "lockfile": str(lock) if lock else None,
            "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock else None,
        },
    )


class DshRuntime:
    """Agent runtime for the DeepSeek Harness headless CLI, dispatched through the registry."""

    agent_type = "dsh"
    required_endpoint = REQUIRED_ENDPOINT
    usage_observer_class = _OpenAISSEUsageObserver

    def get_profile(self, name: str) -> AgentProfile:
        return get_profile(name)

    async def preflight(self, executable: Path) -> AgentPreflightReport:
        _require_zstandard()
        command = str(executable)
        await check_executable_output(
            command, ["--help"], label="Agent", expected=("--profile", "--patch", "--dump-config")
        )
        try:
            return await asyncio.to_thread(_runtime_report, command)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"DSH capability preflight failed: expected={DSH_VERSION}, path={command}\n{exc.stderr}"
            ) from exc

    async def run(self, request):  # type: ignore[no-untyped-def]
        return await run_dsh(request)
