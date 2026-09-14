# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Fail-closed preflight tests for mandatory AgentBench isolation.

Each runtime verifies the executables used by its isolation implementation and
hard-fails when one is missing.
"""

import asyncio
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.claude.runtime import ClaudeRuntime
from agentinfer.agentbench.agents.jiuwenswarm.runtime import JiuwenSwarmRuntime


async def _pass_through_check(_exec, _args, **_kwargs):
    """Stand-in for check_executable_output that approves the runtime executable."""


@pytest.mark.parametrize("missing", ["bwrap", "socat"])
def test_claude_preflight_fails_closed_without_isolation_deps(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.claude.runtime.check_executable_output",
        _pass_through_check,
    )
    which = {"bwrap": "/usr/bin/bwrap", "socat": "/usr/bin/socat"}
    which[missing] = None
    monkeypatch.setattr("agentinfer.agentbench.agents.claude.runtime.shutil.which", lambda name: which.get(name))

    with pytest.raises(RuntimeError, match=f"{missing} is required for Claude isolation"):
        asyncio.run(ClaudeRuntime().preflight(Path("claude")))


def test_claude_preflight_passes_with_isolation_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.claude.runtime.check_executable_output",
        _pass_through_check,
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.claude.runtime.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

    asyncio.run(ClaudeRuntime().preflight(Path("claude")))


@pytest.mark.parametrize("missing", ["jiuwenbox-server", "bwrap"])
def test_jiuwen_preflight_fails_closed_without_isolation_deps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.jiuwenswarm.runtime.check_executable_output",
        _pass_through_check,
    )
    executable = tmp_path / "bin" / "jiuwenswarm"
    executable.parent.mkdir()
    executable.touch()
    server = executable.with_name("jiuwenbox-server")
    if missing != "jiuwenbox-server":
        server.touch()
    which = {"bwrap": "/usr/bin/bwrap"}
    which[missing] = None
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.jiuwenswarm.runtime.shutil.which",
        lambda name: str(Path(name)) if Path(name).exists() else which.get(name),
    )

    with pytest.raises(RuntimeError, match=f"{missing} is required for JiuwenSwarm isolation"):
        asyncio.run(JiuwenSwarmRuntime().preflight(executable))


def test_jiuwen_preflight_resolves_server_beside_explicit_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.jiuwenswarm.runtime.check_executable_output",
        _pass_through_check,
    )
    executable = tmp_path / "bin" / "jiuwenswarm"
    executable.parent.mkdir()
    executable.touch()
    executable.with_name("jiuwenbox-server").touch()
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.jiuwenswarm.runtime.shutil.which",
        lambda name: str(Path(name)) if Path(name).exists() else "/usr/bin/bwrap" if name == "bwrap" else None,
    )

    asyncio.run(JiuwenSwarmRuntime().preflight(executable))
