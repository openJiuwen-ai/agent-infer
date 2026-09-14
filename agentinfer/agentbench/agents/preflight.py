# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Shared executable preflight helpers used by concrete agent runtimes.

Concrete runtimes own their preflight body (which commands to run, which output
to expect); this module provides the subprocess/output-checking primitive they
build on, plus a dispatch entry point that resolves the runtime via the
registry.
"""

import asyncio
import shutil
import subprocess
from pathlib import Path


async def check_agent_preflight(agent_type: str, executable: Path) -> None:
    """Run the executable checks owned by one agent runtime.

    Raises ``RuntimeError`` when an executable is missing or its help output
    omits an expected token.
    """

    from .registry import get_runtime

    await get_runtime(agent_type).preflight(executable)


async def check_executable_output(
    executable: str,
    args: list[str],
    *,
    label: str,
    expected: tuple[str, ...] = (),
) -> None:
    """Run one command and assert it exits 0 and prints expected tokens.

    ``expected`` is matched against the combined stdout+stderr so a runtime can
    require help-output flags without constraining which stream carries them.
    """

    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"{label} executable is not available: {executable}")
    command = " ".join((executable, *args))
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [resolved, *args],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{label} executable preflight check failed: {command}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip() or result.stdout.decode(errors="replace").strip()
        message = f"{label} executable preflight check failed: {command}"
        raise RuntimeError(f"{message}\n{detail}" if detail else message)
    if expected:
        output = (result.stdout + result.stderr).decode(errors="replace")
        missing = [value for value in expected if value not in output]
        if missing:
            raise RuntimeError(f"{label} executable help check failed: {executable}; missing {', '.join(missing)}")
