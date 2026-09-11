# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify DSH runtime preflight requirements."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentinfer.agentbench.agents.dsh.runtime import DshRuntime, _probe_sandbox


def test_dsh_preflight_requires_zstandard(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runtime.importlib.import_module", missing)

    with pytest.raises(RuntimeError, match="DSH requires the Python package 'zstandard'"):
        asyncio.run(DshRuntime().preflight(Path("dsh")))


def test_dsh_preflight_functionally_probes_native_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    check = AsyncMock()
    probe = AsyncMock()
    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runtime._require_zstandard", lambda: None)
    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runtime.check_executable_output", check)
    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runtime.asyncio.to_thread", probe)

    asyncio.run(DshRuntime().preflight(Path("dsh")))

    check.assert_awaited_once_with("dsh", ["--version"], label="Agent")
    probe.assert_awaited_once_with(_probe_sandbox, "dsh")


def test_sandbox_probe_uses_installed_dsh_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "dsh"
    executable.write_text("", encoding="utf-8")
    calls: list[tuple[list[str], dict]] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runtime.shutil.which", lambda _value: str(executable))
    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runtime.subprocess.run", run)

    _probe_sandbox("dsh")

    command, kwargs = calls[0]
    assert command == ["node", "--input-type=module", "-", str(executable.resolve())]
    assert b"LocalSandboxProvider" in kwargs["input"]
    assert b"workspace-write" in kwargs["input"]
    assert kwargs["timeout"] == 20


def test_sandbox_probe_rejects_failed_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable = tmp_path / "dsh"
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr("agentinfer.agentbench.agents.dsh.runtime.shutil.which", lambda _value: str(executable))
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.dsh.runtime.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b"sandbox unavailable"),
    )

    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        _probe_sandbox("dsh")
