# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify agent preflight dispatch through the registry."""

import asyncio
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.preflight import check_agent_preflight
from agentinfer.agentbench.agents.registry import RUNTIMES


def test_check_agent_preflight_delegates_to_registered_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    async def fake_preflight(executable: Path) -> None:
        calls.append(executable)

    monkeypatch.setattr(RUNTIMES["claude"], "preflight", fake_preflight)

    executable = Path("claude")
    asyncio.run(check_agent_preflight("claude", executable))

    assert calls == [executable]


def test_check_agent_preflight_rejects_unsupported_agent() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(check_agent_preflight("unsupported", Path("claude")))
