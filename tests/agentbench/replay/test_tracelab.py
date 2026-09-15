# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.converters import TraceLabConverter
from agentinfer.agentbench.replay.planner import TokenRecipeProfile, build_replay_plan
from agentinfer.agentbench.replay.prompt import PromptBuilder, PromptExchange, SyntheticPrompt
from agentinfer.agentbench.replay.unified_trace_ir import analyze_explicit_trace_ir, validate_trace_ir


def _round(index: int, *, input_tokens: int, prefix_tokens: int, start: int, end: int) -> dict[str, object]:
    return {
        "provider": "codex",
        "project": "fixture",
        "session_id": "session",
        "round_index": index,
        "round_id": f"round-{index}",
        "model": "source-model",
        "input_tokens_total": input_tokens,
        "prefix_tokens": prefix_tokens,
        "newly_append_tokens": input_tokens - prefix_tokens,
        "output_tokens": 7 + index,
        "reasoning_output_tokens": 1,
        "timing_events": [
            {"event_type": "user_message", "timestamp": f"2026-01-01T00:00:{start:02d}Z"},
            {"event_type": "text", "timestamp": f"2026-01-01T00:00:{end:02d}Z"},
        ],
        "tools": [{"is_error": True}],
    }


def _write_source(path: Path) -> None:
    rows = [
        _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1),
        _round(1, input_tokens=35, prefix_tokens=29, start=3, end=4),
    ]
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def test_tracelab_converter_emits_valid_explicit_token_recipe_ir(tmp_path: Path) -> None:
    source = tmp_path / "rounds.jsonl"
    _write_source(source)
    output = tmp_path / "converted"

    summary = TraceLabConverter().convert(source, output)
    trace_ir = validate_trace_ir(output / "requests.jsonl")
    analysis = analyze_explicit_trace_ir(trace_ir)

    assert summary.to_dict() == {"dataset": "tracelab", "sessions": 1, "requests": 2, "text_files": 0}
    assert trace_ir.prompt_source_kind == "token_recipe"
    assert not (output / "texts").exists()
    first, second = analysis.sessions[0].requests
    assert first.historical_status == second.historical_status == "unknown"
    assert first.input_after is None
    assert second.input_after == first.key
    assert second.send_after == first.key
    assert second.same_agent_gap_seconds == 2.0
    assert second.source_cached_tokens == 29
    assert second.source_evidence is not None
    assert second.source_evidence.newly_append_tokens == 6
    assert second.source_line == 2
    assert second.source_evidence.provider == "codex"
    assert second.source_evidence.session_id == "session"
    assert second.source_evidence.round_index == 1
    assert second.source_evidence.round_id == "round-1"
    assert second.source_evidence.model == "source-model"
    assert second.source_evidence.timing.basis == "event_proxy_v1"
    assert second.source_evidence.timing.input_event_type == "user_message"
    assert second.source_evidence.timing.output_event_type == "text"
    assert second.source_evidence.tool_count == 1
    assert second.source_evidence.tool_error_count == 1


def test_tracelab_plan_repeats_sessions_with_isolated_recipe_ids(tmp_path: Path) -> None:
    source = tmp_path / "rounds.jsonl"
    _write_source(source)
    output = tmp_path / "converted"
    TraceLabConverter().convert(source, output)
    trace_ir = validate_trace_ir(output / "requests.jsonl")
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 8, "max_concurrency": 4},
            "backend": {"endpoint": "/v1/chat/completions"},
            "replay": {
                "trace_type": "tracelab",
                "trace_path": source,
                "prompt_shape": "tracelab_synthetic",
                "prompt_calibration_tolerance_tokens": 0,
            },
        }
    )

    plan = build_replay_plan(
        config,
        analyze_explicit_trace_ir(trace_ir),
        trace_ir,
        TokenRecipeProfile(2, 2, "boundary"),
    )

    assert plan.schema_version == "2"
    assert plan.planner_version == "agentinfer-replay-structural/v10"
    assert len(plan.tasks) == 8
    assert len({task.runtime_session_id for task in plan.tasks}) == 8
    assert all(task.requests[0].planned_reuse_tokens == 0 for task in plan.tasks)
    assert all(task.requests[1].planned_reuse_tokens == 27 for task in plan.tasks)
    assert all(task.requests[1].input_after == task.requests[0].source_key for task in plan.tasks)


class _RecipeTokenizer:
    def __init__(self) -> None:
        self.token_recipe_prefix_ids = (1, 2)
        self.token_recipe_suffix_ids = (3, 4)

    async def token_text(self, namespace: str, count: int, *, cache: bool = True) -> str:
        start = 100 + int(hashlib.sha256(namespace.encode()).hexdigest()[:2], 16)
        return "".join(chr(start + index % 7) for index in range(count))

    async def text_token_ids(self, value: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in value)

    async def detokenize_tokens(self, tokens: tuple[int, ...] | list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    async def prompt_token_ids(self, prompt: SyntheticPrompt) -> tuple[int, ...]:
        content = str(prompt.messages[0]["content"])
        return self.token_recipe_prefix_ids + await self.text_token_ids(content) + self.token_recipe_suffix_ids


def test_token_recipe_builder_preserves_exact_length_and_planned_lcp() -> None:
    async def build() -> tuple[SyntheticPrompt, SyntheticPrompt]:
        config = ReplayBenchConfig.model_validate(
            {
                "backend": {"endpoint": "/v1/chat/completions"},
                "replay": {
                    "trace_type": "tracelab",
                    "trace_path": "source.jsonl",
                    "prompt_shape": "tracelab_synthetic",
                    "prompt_calibration_tolerance_tokens": 0,
                },
            }
        )
        builder = PromptBuilder(config, _RecipeTokenizer(), SimpleNamespace(prompt_source_kind="token_recipe"))
        task = SimpleNamespace(runtime_session_id="runtime")
        first_node = SimpleNamespace(
            source_key="first", input_after=None, planned_input_tokens=20, planned_reuse_tokens=0
        )
        first = await builder.build(task, first_node, None)
        second_node = SimpleNamespace(
            source_key="second", input_after="first", planned_input_tokens=24, planned_reuse_tokens=15
        )
        second = await builder.build(task, second_node, PromptExchange(first, "ignored live output"))
        return first, second

    first, second = asyncio.run(build())

    assert first.wire_token_ids is not None and len(first.wire_token_ids) == 20
    assert second.wire_token_ids is not None and len(second.wire_token_ids) == 24
    assert second.calibration is not None
    assert second.calibration.actual_shared_prefix_tokens == 15
    assert second.messages[0]["content"] != "ignored live output"


def test_tracelab_rejects_invalid_token_split_with_source_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    row = _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1)
    row["newly_append_tokens"] = 9
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1 token split"):
        TraceLabConverter().convert(source, tmp_path / "converted")
