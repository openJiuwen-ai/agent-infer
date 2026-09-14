# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from agentinfer.agentbench.benchkit.artifacts.summary import ExecutionSummary, _coerce_execution


def test_coerce_execution_provides_default_state() -> None:
    assert _coerce_execution(None) == ExecutionSummary(
        available=True,
        reason=None,
        metadata={},
    )


def test_coerce_execution_preserves_existing_summary() -> None:
    execution = ExecutionSummary(
        available=False,
        reason="planning only",
        metadata={"planned_tasks": 2},
    )

    assert _coerce_execution(execution) is execution


def test_coerce_execution_converts_mapping() -> None:
    assert _coerce_execution(
        {
            "available": False,
            "reason": "planning only",
            "metadata": {"planned_tasks": 2},
        }
    ) == ExecutionSummary(
        available=False,
        reason="planning only",
        metadata={"planned_tasks": 2},
    )
