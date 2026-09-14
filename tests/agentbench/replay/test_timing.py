# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.replay.analyzer import analyze_replay_trace
from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.timing import build_interval_model, effective_interval_seconds


def _same_agent_followup(tmp_path: Path):
    source = tmp_path / "requests.jsonl"
    rows = [
        {
            "request_id": "lead-1",
            "session_id": "s1",
            "actor_id": "lead",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "input_tokens": 10,
            "output_tokens": 2,
            "status": "success",
        },
        {
            "request_id": "lead-2",
            "session_id": "s1",
            "actor_id": "lead",
            "started_at": "2026-01-01T00:00:03+00:00",
            "finished_at": "2026-01-01T00:00:04+00:00",
            "input_tokens": 20,
            "output_tokens": 2,
            "status": "success",
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return analyze_replay_trace(source).sessions[0].requests[1]


def _cross_agent_followup(tmp_path: Path):
    source = tmp_path / "cross-agent-requests.jsonl"
    rows = [
        {
            "request_id": "lead-1",
            "session_id": "s1",
            "actor_id": "lead",
            "actor_role": "lead",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "input_tokens": 10,
            "output_tokens": 2,
            "status": "success",
        },
        {
            "request_id": "child-1",
            "session_id": "s1",
            "actor_id": "child",
            "actor_role": "subagent",
            "started_at": "2026-01-01T00:00:03+00:00",
            "finished_at": "2026-01-01T00:00:04+00:00",
            "input_tokens": 20,
            "output_tokens": 2,
            "status": "success",
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return analyze_replay_trace(source).sessions[0].requests[1]


def _config(tmp_path: Path, *, scale: float, offset_seconds: float) -> ReplayBenchConfig:
    return ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_path": tmp_path / "requests.jsonl",
                "trace_same_agent_gap_scale": scale,
                "trace_same_agent_gap_offset_seconds": offset_seconds,
            }
        }
    )


def test_trace_same_agent_gap_uses_scale_then_offset_stably(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, scale=0.5, offset_seconds=1)
    request = _same_agent_followup(tmp_path)

    first = effective_interval_seconds(
        config.replay,
        request,
    )
    assert first == 2.0
    assert (
        effective_interval_seconds(
            config.replay,
            request,
        )
        == first
    )


def test_trace_same_agent_gap_clamps_negative_result_to_zero(tmp_path: Path) -> None:
    config = _config(tmp_path, scale=0.5, offset_seconds=-2)

    assert effective_interval_seconds(config.replay, _same_agent_followup(tmp_path)) == 0.0


def test_trace_cross_agent_delay_is_not_adjusted(tmp_path: Path) -> None:
    config = _config(tmp_path, scale=10, offset_seconds=7)
    request = _cross_agent_followup(tmp_path)

    assert request.dependency_kind == "parent_to_subagent"
    assert effective_interval_seconds(config.replay, request) == 2.0


def test_three_quantile_lognormal_is_deterministic_and_matches_distribution() -> None:
    config = ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_path": "requests.jsonl",
                "interval_mode": "lognormal",
                "interval_lognormal": {
                    "p50_seconds": 2,
                    "p95_seconds": 30,
                    "p99_seconds": 90,
                },
            }
        }
    )
    model = build_interval_model(config.replay)
    requests = [
        SimpleNamespace(
            key=f"request-{index}",
            send_after="prior",
            delay_seconds=0.0,
            dependency_kind="same_agent",
            same_agent_gap_seconds=0.0,
        )
        for index in range(200_000)
    ]
    values = [effective_interval_seconds(config.replay, request, model, "runtime-session") for request in requests]
    repeated = [effective_interval_seconds(config.replay, request, model, "runtime-session") for request in requests]
    ordered = sorted(values)

    assert values == repeated
    assert all(value > 0 for value in values)
    assert model.anchor_predictions is not None
    assert abs(statistics.median(ordered) / model.anchor_predictions["p50_seconds"] - 1) < 0.02
    assert abs(ordered[round(0.95 * (len(ordered) - 1))] / model.anchor_predictions["p95_seconds"] - 1) < 0.02
    assert abs(ordered[round(0.99 * (len(ordered) - 1))] / model.anchor_predictions["p99_seconds"] - 1) < 0.02
    assert model.mean_predicted is not None
    assert abs(statistics.mean(values) / model.mean_predicted - 1) < 0.05
    assert model.anchor_residual_ratios is not None
    assert max(abs(value) for value in model.anchor_residual_ratios.values()) < 0.02


def test_three_quantile_lognormal_rejects_inconsistent_anchors() -> None:
    config = ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_path": "requests.jsonl",
                "interval_mode": "lognormal",
                "interval_lognormal": {"p50_seconds": 1, "p95_seconds": 2, "p99_seconds": 100},
            }
        }
    )

    with pytest.raises(ValueError, match="inconsistent with one Lognormal"):
        build_interval_model(config.replay)
