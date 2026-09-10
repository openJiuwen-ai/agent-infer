# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify the agent-runtime registry contract across every registered runtime."""

import pytest

from agentinfer.agentbench.agents.registry import RUNTIMES, get_runtime


@pytest.mark.parametrize("key", sorted(RUNTIMES))
def test_registered_runtime_has_known_endpoint(key: str) -> None:
    assert get_runtime(key).required_endpoint in {"/v1/messages", "/v1/chat/completions"}


@pytest.mark.parametrize("key", sorted(RUNTIMES))
def test_registered_runtime_rejects_unknown_profile(key: str) -> None:
    with pytest.raises(ValueError, match="profile"):
        get_runtime(key).get_profile("__nonexistent_profile__")


@pytest.mark.parametrize("key", sorted(RUNTIMES))
def test_registered_runtime_reports_its_agent_type(key: str) -> None:
    assert get_runtime(key).agent_type == key


def test_get_runtime_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        get_runtime("unsupported")
