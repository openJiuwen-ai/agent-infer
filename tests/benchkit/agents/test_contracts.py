"""Verify shared result defaults for agent runtimes."""

import pytest

from agentcache.benchmarks.agents import AgentRunOutcome, AgentRunResult, TerminationReason


def test_agent_run_result_defaults_to_harness_failure_without_artifacts() -> None:
    result = AgentRunResult()

    assert result.outcome is AgentRunOutcome.FAILED
    assert result.termination_reason is TerminationReason.HARNESS_ERROR
    assert result.transcript is None
    assert result.has_patch is False
    assert result.patch_bytes == 0


def test_completed_result_has_no_termination_reason() -> None:
    result = AgentRunResult(outcome=AgentRunOutcome.COMPLETED)

    assert result.termination_reason is None


def test_completed_result_rejects_termination_reason() -> None:
    with pytest.raises(ValueError, match="completed outcome cannot have"):
        AgentRunResult(
            outcome=AgentRunOutcome.COMPLETED,
            termination_reason=TerminationReason.TIMEOUT,
        )
