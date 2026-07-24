from agentinfer.agentbench.agents import AgentRunOutcome, AgentRunResult, TerminationReason
from agentinfer.agentbench.benchkit.metrics.task import aggregate_task_results


def test_aggregate_task_results_uses_execution_outcomes_and_patch_evidence() -> None:
    results = [
        AgentRunResult(
            outcome=AgentRunOutcome.COMPLETED,
            termination_reason=None,
            duration_seconds=10.0,
            has_patch=True,
        ),
        AgentRunResult(
            outcome=AgentRunOutcome.FAILED,
            termination_reason=TerminationReason.IDLE_AFTER_PATCH,
            duration_seconds=20.0,
            has_patch=True,
        ),
        AgentRunResult(
            outcome=AgentRunOutcome.FAILED,
            termination_reason=TerminationReason.TIMEOUT,
            duration_seconds=30.0,
        ),
    ]

    metrics = aggregate_task_results(results)

    assert metrics.completed == 1
    assert metrics.failed == 2
    assert metrics.with_patch == 2
    assert metrics.duration_seconds == {"mean": 20.0, "p50": 20.0, "p95": 29.0}


def test_aggregate_task_results_handles_empty_input() -> None:
    metrics = aggregate_task_results([])

    assert metrics.completed == 0
    assert metrics.failed == 0
    assert metrics.with_patch == 0
    assert metrics.duration_seconds == {"mean": None, "p50": None, "p95": None}
