# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.converters import TraceLabConverter
from agentinfer.agentbench.replay.planner import build_replay_plan
from agentinfer.agentbench.replay.prompt import PromptBuilder, PromptExchange, SyntheticPrompt
from agentinfer.agentbench.replay.runner import run_replay
from agentinfer.agentbench.replay.unified_trace_ir import (
    analyze_explicit_trace_ir,
    validate_trace_ir,
    write_trace_ir_manifest,
)


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
    assert first.context_after is None
    assert second.context_after == first.key
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
    assert second.source_evidence.timing.basis == "event_proxy"
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
    )

    assert plan.schema_version == "2"
    assert plan.planner_version == "agentinfer-replay-structural"
    assert len(plan.tasks) == 8
    assert len({task.runtime_session_id for task in plan.tasks}) == 8
    assert all(task.requests[0].context_after is None for task in plan.tasks)
    assert all(task.requests[1].context_mode == "append" for task in plan.tasks)
    assert all(task.requests[1].context_after == task.requests[0].source_key for task in plan.tasks)


class _RecipeTokenizer:
    async def token_text(self, namespace: str, count: int, *, cache: bool = True) -> str:
        start = 100 + int(hashlib.sha256(namespace.encode()).hexdigest()[:2], 16)
        return "".join(chr(start + index % 7) for index in range(count))

    async def text_token_ids(self, value: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in value)

    async def detokenize_tokens(self, tokens: tuple[int, ...] | list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    async def count(self, prompt: SyntheticPrompt) -> int:
        return 3 + sum(4 + len(str(message["content"])) for message in prompt.messages)


def test_tracelab_builder_appends_live_history_and_preserves_exact_length() -> None:
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
            source_key="first",
            prompt_recipe_key="first",
            context_after=None,
            context_mode="independent",
            planned_input_tokens=20,
        )
        first = await builder.build(task, first_node, None)
        second_node = SimpleNamespace(
            source_key="second",
            prompt_recipe_key="second",
            context_after="first",
            context_mode="append",
            planned_input_tokens=60,
        )
        second = await builder.build(task, second_node, PromptExchange(first, "live assistant output"))
        return first, second

    first, second = asyncio.run(build())

    assert first.calibration is not None and first.calibration.final_tokens == 20
    assert second.calibration is not None and second.calibration.final_tokens == 60
    assert second.messages[0] == first.messages[0]
    assert second.messages[1] == {"role": "assistant", "content": "live assistant output"}
    assert second.messages[2]["role"] == "user"


def test_tracelab_preserves_source_token_counts_without_reconciling_split(tmp_path: Path) -> None:
    source = tmp_path / "rounds.jsonl"
    row = _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1)
    row["newly_append_tokens"] = 9
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = tmp_path / "converted"
    TraceLabConverter().convert(source, output)
    trace_ir = validate_trace_ir(output / "requests.jsonl")
    analysis = analyze_explicit_trace_ir(trace_ir)
    request = analysis.sessions[0].requests[0]
    assert request.source_input_tokens == 30
    assert request.source_cached_tokens == 20
    assert request.source_evidence is not None
    assert request.source_evidence.newly_append_tokens == 9

    config = ReplayBenchConfig.model_validate(
        {
            "replay": {
                "trace_type": "tracelab",
                "trace_path": source,
                "prompt_shape": "tracelab_synthetic",
                "prompt_calibration_tolerance_tokens": 0,
            }
        }
    )
    node = build_replay_plan(config, analysis, trace_ir).tasks[0].requests[0]
    assert node.planned_input_tokens == 30
    assert node.planned_output_tokens == 7


@pytest.mark.parametrize("predecessor", [None, "missing", "self"])
def test_tracelab_ir_rejects_invalid_context_dependency(tmp_path: Path, predecessor: str | None) -> None:
    source = tmp_path / "rounds.jsonl"
    _write_source(source)
    output = tmp_path / "converted"
    summary = TraceLabConverter().convert(source, output)
    requests = output / "requests.jsonl"
    rows = [json.loads(line) for line in requests.read_text().splitlines()]
    rows[1]["context_after"] = rows[1]["request_id"] if predecessor == "self" else predecessor
    requests.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_trace_ir_manifest(
        output,
        converter_name="tracelab",
        source_path=source,
        summary=summary.to_dict(),
        prompt_source_kind="token_recipe",
    )

    with pytest.raises(ValueError, match="context_after"):
        validate_trace_ir(requests)


@pytest.mark.parametrize("mode", ["strict", "adaptive"])
def test_tracelab_does_not_drop_live_history_to_fit_a_small_target(mode: str) -> None:
    async def build() -> None:
        config = ReplayBenchConfig.model_validate(
            {
                "replay": {
                    "trace_type": "tracelab",
                    "trace_path": "source.jsonl",
                    "prompt_shape": "tracelab_synthetic",
                    "prompt_calibration_tolerance_tokens": 0,
                    "context_adjustment_mode": mode,
                }
            }
        )
        builder = PromptBuilder(config, _RecipeTokenizer(), SimpleNamespace(prompt_source_kind="token_recipe"))
        previous = SyntheticPrompt("", (), ({"role": "user", "content": "previous"},))
        exchange = PromptExchange(previous, "live assistant output that exceeds the input target")
        node = SimpleNamespace(
            source_key="next",
            prompt_recipe_key="next",
            context_after="first",
            context_mode="append",
            planned_input_tokens=10,
        )
        with pytest.raises(ValueError, match="Prompt minimum"):
            await builder.build(SimpleNamespace(runtime_session_id="runtime"), node, exchange)
        assert exchange.assistant_content == "live assistant output that exceeds the input target"
        assert previous.messages == ({"role": "user", "content": "previous"},)

    asyncio.run(build())


@pytest.mark.parametrize("fail_first", [False, True])
def test_tracelab_runner_uses_live_context_and_skips_failed_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_first: bool
) -> None:
    source = tmp_path / "rounds.jsonl"
    rows = [
        _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1),
        _round(1, input_tokens=70, prefix_tokens=29, start=3, end=4),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(503)
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            if "prompt" in body:
                return httpx.Response(200, json={"tokens": [ord(c) for c in body["prompt"]]})
            count = 3 + sum(4 + len(message["content"]) for message in body["messages"])
            return httpx.Response(200, json={"count": count})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"prompt": "".join(chr(t) for t in body["tokens"])})
        assert request.url.path == "/v1/chat/completions"
        sent.append(body)
        count = 3 + sum(4 + len(message["content"]) for message in body["messages"])
        output = "A" * body["max_tokens"]
        chunk = {
            "choices": [{"delta": {"content": output}}],
            "usage": {"prompt_tokens": count + int(fail_first), "completion_tokens": len(output)},
        }
        return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1, "max_concurrency": 1, "result_dir": tmp_path / "run"},
            "replay": {
                "trace_type": "tracelab",
                "trace_path": source,
                "prompt_shape": "tracelab_synthetic",
                "prompt_calibration_tolerance_tokens": 0,
                "trace_same_agent_gap_scale": 0,
            },
        }
    )
    result = run_replay(config)
    execution = json.loads((result / "replay-execution.json").read_text())
    plan = json.loads((result / "replay-plan.json").read_text())
    nodes = plan["tasks"][0]["requests"]
    assert nodes[1]["context_after"] == nodes[0]["source_key"]
    assert nodes[1]["context_mode"] == "append"
    statuses = [node["status"] for node in execution["tasks"][0]["nodes"]]
    if fail_first:
        assert len(sent) == 1
        assert statuses == ["failed", "skipped_dependency_failed"]
        assert execution["summary"]["dependency_skipped_requests"] == 1
    else:
        assert len(sent) == 2
        assert statuses == ["success", "success"]
        assert sent[1]["messages"][:2] == [
            sent[0]["messages"][0],
            {"role": "assistant", "content": "AAAAAAA"},
        ]
        assert [message["role"] for message in sent[1]["messages"]] == ["user", "assistant", "user"]
        assert execution["summary"]["prompt_calibration_exact_requests"] == 2
