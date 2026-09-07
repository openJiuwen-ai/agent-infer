# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify explicit benchmark agent dispatch."""

import asyncio
from pathlib import Path

import pytest

from agentinfer.agentbench.agents import AgentRunRequest, AgentRunResult
from agentinfer.agentbench.agents.dispatch import run_agent
from agentinfer.agentbench.benchkit.dataset import Task


def _request(agent_type: str = "claude") -> AgentRunRequest:
    return AgentRunRequest(
        agent_type=agent_type,
        task=Task("instance", "owner/repo", "abc", "fix"),
        profile_name="single",
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
    expected = AgentRunResult(session_id=request.session_id)

    async def fake_run_claude(received: AgentRunRequest) -> AgentRunResult:
        assert received is request
        return expected

    monkeypatch.setattr("agentinfer.agentbench.agents.dispatch.run_claude", fake_run_claude)

    assert asyncio.run(run_agent(request)) is expected


def test_rejects_unsupported_agent() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(run_agent(_request("unsupported")))
