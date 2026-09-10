# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify explicit benchmark agent dispatch."""

import asyncio
from pathlib import Path

import pytest

from agentinfer.agentbench.agents import AgentRunRequest, AgentRunResult
from agentinfer.agentbench.agents.dispatch import run_agent
from agentinfer.agentbench.agents.registry import RUNTIMES, get_runtime
from agentinfer.agentbench.benchkit.dataset import Task


def _request(agent_type: str = "claude") -> AgentRunRequest:
    return AgentRunRequest(
        agent_type=agent_type,
        task=Task("instance", "owner/repo", "abc", "fix"),
        profile_name="code.normal" if agent_type == "jiuwenswarm" else "single",
        executable=Path("claude"),
        model="model",
        api_base_url="http://proxy",
        workspace=Path("workspace"),
        artifact_dir=Path("artifacts"),
        session_id="session",
        timeout_seconds=10,
        patch_flush_seconds=1,
        prompt_delivery_timeout_seconds=2,
        tmux_startup_seconds=0.1,
        terminal_capture_interval_seconds=1,
    )


def test_dispatches_claude_request(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    expected = AgentRunResult(agent_type="claude", profile_name="single", session_id=request.session_id)

    async def fake_run(received: AgentRunRequest) -> AgentRunResult:
        assert received is request
        return expected

    monkeypatch.setattr(RUNTIMES["claude"], "run", fake_run)

    assert asyncio.run(run_agent(request)) is expected


def test_dispatches_jiuwenswarm_request(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request("jiuwenswarm")
    expected = AgentRunResult(agent_type="jiuwenswarm", profile_name="code.normal", session_id=request.session_id)

    async def fake_run(received: AgentRunRequest) -> AgentRunResult:
        assert received is request
        return expected

    monkeypatch.setattr(RUNTIMES["jiuwenswarm"], "run", fake_run)

    assert asyncio.run(run_agent(request)) is expected


def test_dispatches_dsh_request(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request("dsh")
    expected = AgentRunResult(agent_type="dsh", profile_name="single", session_id=request.session_id)

    async def fake_run(received: AgentRunRequest) -> AgentRunResult:
        assert received is request
        return expected

    monkeypatch.setattr(RUNTIMES["dsh"], "run", fake_run)

    assert asyncio.run(run_agent(request)) is expected


def test_rejects_unsupported_agent() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(run_agent(_request("unsupported")))


def test_get_runtime_returns_registered_keys() -> None:
    assert set(RUNTIMES) == {"claude", "jiuwenswarm", "dsh"}
    assert get_runtime("claude").agent_type == "claude"
    assert get_runtime("jiuwenswarm").agent_type == "jiuwenswarm"
    assert get_runtime("dsh").agent_type == "dsh"
