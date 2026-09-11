# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify task aggregation and evidence-source health."""

from pathlib import Path

from agentinfer.agentbench.agents import AgentRunOutcome, AgentRunResult, TerminationReason
from agentinfer.agentbench.benchkit.metrics.schema import EvidenceCapture
from agentinfer.agentbench.benchkit.metrics.source_health import evaluate_captures
from agentinfer.agentbench.benchkit.metrics.task import aggregate_task_results


def test_aggregate_task_results_uses_execution_outcomes_and_patch_evidence() -> None:
    results = [
        AgentRunResult(
            agent_type="claude",
            profile_name="single",
            outcome=AgentRunOutcome.COMPLETED,
            termination_reason=None,
            duration_seconds=10.0,
            has_patch=True,
        ),
        AgentRunResult(
            agent_type="claude",
            profile_name="single",
            outcome=AgentRunOutcome.FAILED,
            termination_reason=TerminationReason.IDLE_AFTER_PATCH,
            duration_seconds=20.0,
            has_patch=True,
        ),
        AgentRunResult(
            agent_type="claude",
            profile_name="single",
            outcome=AgentRunOutcome.FAILED,
            termination_reason=TerminationReason.TIMEOUT,
            duration_seconds=30.0,
        ),
    ]

    metrics = aggregate_task_results(results)

    assert metrics.completed == 1
    assert metrics.failed == 2
    assert metrics.with_patch == 2
    assert metrics.duration_seconds == {"mean": 20.0, "p50": 20.0, "p95": 29.0, "p99": 29.8}


def test_aggregate_task_results_handles_empty_input() -> None:
    metrics = aggregate_task_results([])

    assert metrics.completed == 0
    assert metrics.failed == 0
    assert metrics.with_patch == 0
    assert metrics.duration_seconds == {"mean": None, "p50": None, "p95": None, "p99": None}


def test_evaluate_captures_separates_unavailable_and_not_applicable() -> None:
    captures = [
        EvidenceCapture("environment", Path("environment.json"), True, None, {}),
        EvidenceCapture("vllm", None, False, "connection failed", {}),
        EvidenceCapture("correctness", None, False, "not requested", {}, applicable=False),
    ]

    health = evaluate_captures(captures)

    assert health.available == 1
    assert health.unavailable == 1
    assert health.not_applicable == 1
    assert health.reasons == ("connection failed",)
    assert health.sources["environment"][0]["path"] == "environment.json"


def test_evaluate_captures_strips_raw_metadata_from_duplicate_sources() -> None:
    captures = [
        EvidenceCapture("vllm", Path("start.prom"), True, None, {"text": "raw start"}),
        EvidenceCapture("vllm", Path("end.prom"), True, None, {"text": "raw end"}),
    ]

    health = evaluate_captures(captures)

    assert health.available == 2
    assert [row["path"] for row in health.sources["vllm"]] == ["start.prom", "end.prom"]
    assert [row["metadata"] for row in health.sources["vllm"]] == [{}, {}]
