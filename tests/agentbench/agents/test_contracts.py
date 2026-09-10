# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify shared result defaults for agent runtimes."""

import pytest

from agentinfer.agentbench.agents import AgentRunOutcome, AgentRunResult, TerminationReason


def test_agent_run_result_requires_agent_type_and_profile() -> None:
    """``agent_type`` and ``profile_name`` have no defaults, so a bare result is rejected."""

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
