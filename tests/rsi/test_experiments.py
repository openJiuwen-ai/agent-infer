# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
from pathlib import Path

import pytest

from agentinfer.rsi.dashboard.experiments import render_experiments
from agentinfer.rsi.experiments import append_experiment, list_experiments, validate_experiment


def record(round_id: str = "tp4-baseline") -> dict[str, object]:
    return {
        "round_id": round_id,
        "timestamp": "2026-09-22T08:00:00+00:00",
        "hypothesis": "A fixed TP4 server provides a reproducible reference.",
        "change": "Start the official model with TP4 and prefix caching.",
        "model": "Qwen/Qwen3.8-27B",
        "backend": "cuda",
        "component_version": "vllm-0.29.0",
        "config": {"precision": "bf16", "tensor_parallel": 4},
        "commands": ["vllm serve ...", "vllm bench serve --agentinfer replay ..."],
        "metrics": {"throughput_output_tokens_per_s": 123.4, "gsm8k_accuracy": None},
        "status": "measured",
        "evidence": [{"path": "/tmp/replay/summary.json", "sha256": "a" * 64}],
        "failure_reason": None,
        "next_test": "Compare a two-replica TP2 layout.",
    }


def test_append_and_list_are_ordered_and_duplicate_safe(tmp_path: Path) -> None:
    first = append_experiment(tmp_path, record())
    append_experiment(tmp_path, {**record("tp2x2"), "baseline_round_id": "tp4-baseline"})

    assert first["metrics"]["throughput_input_tokens_per_s"] is None
    assert [item["round_id"] for item in list_experiments(tmp_path)] == ["tp4-baseline", "tp2x2"]
    with pytest.raises(ValueError, match="Duplicate round_id"):
        append_experiment(tmp_path, record())


def test_validation_rejects_nonfinite_metrics() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_experiment({**record(), "metrics": {"throughput_output_tokens_per_s": float("nan")}})


def test_renderer_escapes_records_and_marks_read_only(tmp_path: Path) -> None:
    item = {**record(), "hypothesis": "<script>alert(1)</script>"}
    append_experiment(tmp_path, item)
    html = render_experiments(list_experiments(tmp_path))

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "experiments.jsonl only" in html
    assert "Not measured (null)" in html
    assert "<script>alert(1)</script>" not in html


def test_ledger_lines_are_strict_json(tmp_path: Path) -> None:
    append_experiment(tmp_path, record())
    line = (tmp_path / "experiments.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["round_id"] == "tp4-baseline"
