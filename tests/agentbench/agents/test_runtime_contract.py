# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify shared AgentBench runtime, registry, dispatch, and preflight contracts."""

import asyncio
import typing
from pathlib import Path

import pytest

from agentinfer.agentbench.agents import AgentRunOutcome, AgentRunRequest, AgentRunResult, TerminationReason
from agentinfer.agentbench.agents.dispatch import run_agent
from agentinfer.agentbench.agents.preflight import check_agent_preflight
from agentinfer.agentbench.agents.registry import RUNTIMES, get_runtime
from agentinfer.agentbench.benchkit.config import AgentConfig
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


def test_agent_run_result_requires_agent_type_and_profile() -> None:
    with pytest.raises(TypeError, match="agent_type"):
        AgentRunResult()


def test_agent_run_result_defaults_to_harness_failure_without_artifacts() -> None:
    result = AgentRunResult(agent_type="claude", profile_name="single")

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.HARNESS_ERROR
    assert result.transcript is None
    assert result.has_patch is False
    assert result.patch_bytes == 0


def test_completed_result_has_no_termination_reason() -> None:
    result = AgentRunResult(agent_type="claude", profile_name="single", outcome=AgentRunOutcome.COMPLETED)

    assert result.termination_reason is None


def test_completed_result_rejects_termination_reason() -> None:
    with pytest.raises(ValueError, match="completed outcome cannot have"):
        AgentRunResult(
            agent_type="claude",
            profile_name="single",
            outcome=AgentRunOutcome.COMPLETED,
            termination_reason=TerminationReason.TIMEOUT,
        )


def test_runtime_status_values_serialize_for_artifacts() -> None:
    assert AgentRunOutcome.COMPLETED.value == "completed"
    assert AgentRunOutcome.FAILED.value == "failed"
    assert TerminationReason.TIMEOUT.value == "timeout"
    assert TerminationReason.HARNESS_ERROR.value == "harness_error"


@pytest.mark.parametrize("key", sorted(RUNTIMES))
def test_registered_runtime_matches_registry_contract(key: str) -> None:
    runtime = get_runtime(key)

    assert runtime.agent_type == key
    assert runtime.required_endpoint in {"/v1/messages", "/v1/chat/completions"}
    assert runtime.usage_observer_class is not None


def test_agent_config_types_match_runtime_registry() -> None:
    annotation = AgentConfig.model_fields["type"].annotation

    assert typing.get_origin(annotation) is typing.Literal
    assert set(typing.get_args(annotation)) == set(RUNTIMES)


@pytest.mark.parametrize("key", sorted(RUNTIMES))
def test_registered_runtime_rejects_unknown_profile(key: str) -> None:
    with pytest.raises(ValueError, match="profile"):
        get_runtime(key).get_profile("__nonexistent_profile__")


def test_get_runtime_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        get_runtime("unsupported")


@pytest.mark.parametrize("agent_type", sorted(RUNTIMES))
def test_dispatches_request_to_registered_runtime(agent_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(agent_type)
    expected = AgentRunResult(agent_type=agent_type, profile_name=request.profile_name, session_id=request.session_id)

    async def fake_run(received: AgentRunRequest) -> AgentRunResult:
        assert received is request
        return expected

    monkeypatch.setattr(RUNTIMES[agent_type], "run", fake_run)

    assert asyncio.run(run_agent(request)) is expected


def test_dispatch_rejects_unsupported_agent() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(run_agent(_request("unsupported")))


def test_preflight_delegates_to_registered_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    async def fake_preflight(executable: Path) -> None:
        calls.append(executable)

    monkeypatch.setattr(RUNTIMES["claude"], "preflight", fake_preflight)

    executable = Path("claude")
    asyncio.run(check_agent_preflight("claude", executable))

    assert calls == [executable]


def test_preflight_rejects_unsupported_agent() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(check_agent_preflight("unsupported", Path("claude")))
