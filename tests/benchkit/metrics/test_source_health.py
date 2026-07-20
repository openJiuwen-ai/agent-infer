from pathlib import Path

from agentcache.benchmarks.benchkit.metrics.schema import EvidenceCapture
from agentcache.benchmarks.benchkit.metrics.source_health import evaluate_captures


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


def test_evaluate_captures_preserves_duplicate_sources() -> None:
    captures = [
        EvidenceCapture("vllm", None, True, None, {"snapshot": "start"}),
        EvidenceCapture("vllm", None, True, None, {"snapshot": "end"}),
    ]

    health = evaluate_captures(captures)

    assert health.available == 2
    assert [row["metadata"] for row in health.sources["vllm"]] == [
        {"snapshot": "start"},
        {"snapshot": "end"},
    ]
