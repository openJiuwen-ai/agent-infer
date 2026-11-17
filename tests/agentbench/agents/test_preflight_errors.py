# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Keep actionable subprocess diagnostics at the preflight boundary."""

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentinfer.agentbench.agents.dsh import runtime as dsh
from agentinfer.agentbench.agents.preflight import check_executable_output
from agentinfer.agentbench.agents.zcode import runtime as zcode
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


@pytest.fixture(params=["exit", "timeout", "permission"])
def probe_failure(request, monkeypatch):
    error = {
        "exit": subprocess.CalledProcessError(2, ["agent", "--version"], stderr="native probe failure"),
        "timeout": subprocess.TimeoutExpired(["agent", "--version"], 20),
        "permission": PermissionError(13, "Permission denied", "agent"),
    }[request.param]

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    return error


@pytest.mark.parametrize("agent", ["dsh", "zcode"])
def test_capability_failure_preserves_context(tmp_path, monkeypatch, probe_failure, agent):
    module = dsh if agent == "dsh" else zcode
    executable = tmp_path / agent
    monkeypatch.setattr(module, "check_executable_output", AsyncMock(return_value=executable))
    if agent == "dsh":
        monkeypatch.setattr(dsh, "_require_zstandard", lambda: None)
    runtime = dsh.DshRuntime() if agent == "dsh" else zcode.ZCodeRuntime()

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(runtime.preflight(executable))

    message = str(caught.value)
    assert str(executable) in message
    assert (dsh.DSH_VERSION if agent == "dsh" else zcode.ZCODE_VERSION) in message
    assert type(probe_failure).__name__ in message
    assert str(probe_failure) in message
    assert caught.value.__cause__ is probe_failure
    if isinstance(probe_failure, subprocess.CalledProcessError):
        assert "native probe failure" in message


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("error", [PermissionError("probe denied"), subprocess.TimeoutExpired(["agent"], 10)])
def test_preflight_system_failure_keeps_reason(monkeypatch, sandbox, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError) as caught:
        if sandbox:
            dsh._probe_sandbox(Path(sys.executable))
        else:
            asyncio.run(check_executable_output(sys.executable, ["--help"], label="Agent"))

    assert str(error) in str(caught.value)
    assert caught.value.__cause__ is error
