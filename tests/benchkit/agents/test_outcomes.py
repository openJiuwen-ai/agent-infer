"""Verify external string values used by runtime artifacts."""

from agentcache.benchmarks.agents import AgentRunOutcome, TerminationReason


def test_runtime_status_values_serialize_for_artifacts() -> None:
    assert AgentRunOutcome.COMPLETED.value == "completed"
    assert AgentRunOutcome.FAILED.value == "failed"
    assert TerminationReason.TIMEOUT.value == "timeout"
    assert TerminationReason.HARNESS_ERROR.value == "harness_error"
