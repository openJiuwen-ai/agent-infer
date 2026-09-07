# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from pathlib import Path

from agentinfer.agentbench.benchkit.metrics.schema import EvidenceCapture
from agentinfer.agentbench.benchkit.metrics.source_health import evaluate_captures


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
