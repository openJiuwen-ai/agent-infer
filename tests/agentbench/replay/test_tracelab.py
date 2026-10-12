# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import hashlib
import json
import os
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
    assert trace_ir.summary is not None and trace_ir.summary["warmup_input_tokens"] == 21
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


@pytest.mark.parametrize(
    ("indices", "expected", "found"),
    [([1], 0, 1), ([0, 2], 1, 2)],
)
def test_tracelab_converter_rejects_noncontiguous_rounds(
    tmp_path: Path, indices: list[int], expected: int, found: int
) -> None:
    source = tmp_path / "rounds.jsonl"
    rows = [_round(index, input_tokens=30, prefix_tokens=20, start=index * 3, end=index * 3 + 1) for index in indices]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "converted"

    with pytest.raises(ValueError, match=f"expected {expected}, found {found}"):
        TraceLabConverter().convert(source, output)
    assert not output.exists()


def test_tracelab_converter_accepts_out_of_order_contiguous_rounds(tmp_path: Path) -> None:
    source = tmp_path / "rounds.jsonl"
    rows = [
        _round(1, input_tokens=35, prefix_tokens=29, start=3, end=4),
        _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    TraceLabConverter().convert(source, tmp_path / "converted")
    trace_ir = validate_trace_ir(tmp_path / "converted" / "requests.jsonl")
    analysis = analyze_explicit_trace_ir(trace_ir)
    assert [request.source_evidence.round_index for request in analysis.sessions[0].requests] == [0, 1]


@pytest.mark.parametrize("warmup_input_tokens", [0, 22])
def test_tracelab_ir_rejects_incorrect_warmup_length(tmp_path: Path, warmup_input_tokens: int) -> None:
    source = tmp_path / "rounds.jsonl"
    _write_source(source)
    output = tmp_path / "converted"
    summary = TraceLabConverter().convert(source, output)
    write_trace_ir_manifest(
        output,
        converter_name="tracelab",
        source_path=source,
        summary={**summary.to_dict(), "warmup_input_tokens": warmup_input_tokens},
        prompt_source_kind="token_recipe",
    )

    with pytest.raises(ValueError, match="warmup_input_tokens does not match first-round prefixes"):
        validate_trace_ir(output / "requests.jsonl")


def test_tracelab_zero_output_survives_conversion_and_planning(tmp_path: Path) -> None:
    source = tmp_path / "zero-output.jsonl"
    first = _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1)
    first["output_tokens"] = 0
    second = _round(1, input_tokens=35, prefix_tokens=29, start=3, end=4)
    source.write_text("".join(json.dumps(row) + "\n" for row in (first, second)), encoding="utf-8")
    converted = tmp_path / "converted"
    TraceLabConverter().convert(source, converted)
    trace_ir = validate_trace_ir(converted / "requests.jsonl")
    analysis = analyze_explicit_trace_ir(trace_ir)
    assert [request.output_tokens for request in analysis.sessions[0].requests] == [0, 8]
    config = ReplayBenchConfig.model_validate(
        {"experiment": {"task_num": 1}, "replay": {"trace_type": "tracelab", "trace_path": source}}
    )
    plan = build_replay_plan(config, analysis, trace_ir)
    assert [node.planned_output_tokens for node in plan.tasks[0].requests] == [0, 8]
    assert plan.tasks[0].requests[0].to_dict()["effective_output_tokens"] == 1


def test_tracelab_negative_output_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "negative-output.jsonl"
    row = _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1)
    row["output_tokens"] = -1
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output_tokens must be a non-negative integer"):
        TraceLabConverter().convert(source, tmp_path / "converted")


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
    async def filler_token_ids(
        self, namespace: str, count: int, *, forbidden_first: int | None = None
    ) -> tuple[int, ...]:
        first = 100 + int(hashlib.sha256(namespace.encode()).hexdigest()[:2], 16)
        if first == forbidden_first:
            first += 1
        return tuple(first + index % 7 for index in range(count))

    async def token_text(self, namespace: str, count: int, *, cache: bool = True) -> str:
        start = 100 + int(hashlib.sha256(namespace.encode()).hexdigest()[:2], 16)
        return "".join(chr(start + index % 7) for index in range(count))

    async def text_token_ids(self, value: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in value)

    async def detokenize_tokens(self, tokens: tuple[int, ...] | list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    async def count(self, prompt: SyntheticPrompt) -> int:
        return 3 + sum(4 + len(str(message["content"])) for message in prompt.messages)


def test_tracelab_warmup_uses_conversion_prefix_limit() -> None:
    async def prepare() -> tuple[int, ...]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 2},
                "backend": {"endpoint": "/v1/chat/completions"},
                "replay": {"trace_type": "tracelab", "trace_path": "source.jsonl"},
            }
        )
        builder = PromptBuilder(
            config,
            _RecipeTokenizer(),
            SimpleNamespace(prompt_source_kind="token_recipe", summary={"warmup_input_tokens": 25}),
        )

        def request(name: str, prefix: int, predecessor: str | None) -> SimpleNamespace:
            return SimpleNamespace(
                node_type="request",
                runtime_request_id=name,
                context_after=predecessor,
                planned_input_tokens=prefix,
                source_prefix_tokens=prefix,
            )

        plan = SimpleNamespace(
            workload_fingerprint="fixture",
            tasks=(
                SimpleNamespace(requests=(request("first-a", 20, None), request("later-a", 60, "first-a"))),
                SimpleNamespace(requests=(request("first-b", 24, None), request("later-b", 70, "first-b"))),
            ),
        )
        return await builder.prepare_tracelab_warmup(plan)

    assert len(asyncio.run(prepare())) == 25


def test_tracelab_warmup_includes_unselected_session_first_round(tmp_path: Path) -> None:
    source = tmp_path / "rounds.jsonl"
    small = _round(0, input_tokens=30, prefix_tokens=20, start=0, end=1)
    large = _round(0, input_tokens=60, prefix_tokens=50, start=0, end=1)
    large["session_id"] = "other-session"
    source.write_text("".join(json.dumps(row) + "\n" for row in (small, large)), encoding="utf-8")
    converted = tmp_path / "converted"
    TraceLabConverter().convert(source, converted)
    trace_ir = validate_trace_ir(converted / "requests.jsonl")
    assert trace_ir.summary is not None and trace_ir.summary["warmup_input_tokens"] == 51

    async def prepare() -> tuple[int, ...]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"endpoint": "/v1/chat/completions"},
                "replay": {"trace_type": "tracelab", "trace_path": source},
            }
        )
        builder = PromptBuilder(config, _RecipeTokenizer(), trace_ir)
        first = SimpleNamespace(
            node_type="request",
            runtime_request_id="selected-first",
            context_after=None,
            planned_input_tokens=30,
            source_prefix_tokens=20,
        )
        plan = SimpleNamespace(workload_fingerprint="fixture", tasks=(SimpleNamespace(requests=(first,)),))
        return await builder.prepare_tracelab_warmup(plan)

    assert len(asyncio.run(prepare())) == 51


def test_tracelab_builder_reuses_exact_live_token_prefix() -> None:
    async def build() -> tuple[SyntheticPrompt, SyntheticPrompt]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"endpoint": "/v1/chat/completions"},
                "replay": {
                    "trace_type": "tracelab",
                    "trace_path": "source.jsonl",
                    "prompt_calibration_tolerance_tokens": 0,
                },
            }
        )
        builder = PromptBuilder(
            config,
            _RecipeTokenizer(),
            SimpleNamespace(prompt_source_kind="token_recipe", summary={"warmup_input_tokens": 1}),
        )
        await builder.prepare_tracelab_warmup(SimpleNamespace(workload_fingerprint="fixture", tasks=()))
        task = SimpleNamespace(runtime_session_id="runtime")
        first_node = SimpleNamespace(
            source_key="first",
            prompt_recipe_key="first",
            context_after=None,
            context_mode="independent",
            planned_input_tokens=20,
            source_prefix_tokens=0,
        )
        first = await builder.build(task, first_node, None)
        second_node = SimpleNamespace(
            source_key="second",
            prompt_recipe_key="second",
            context_after="first",
            context_mode="append",
            planned_input_tokens=60,
            source_prefix_tokens=25,
        )
        output_ids = tuple(range(200, 207))
        second = await builder.build(
            task, second_node, PromptExchange(first, "live assistant output", None, output_ids)
        )
        return first, second

    first, second = asyncio.run(build())

    assert first.calibration is not None and first.calibration.final_tokens == 20
    assert second.calibration is not None and second.calibration.final_tokens == 60
    assert second.token_ids is not None and first.token_ids is not None
    assert second.token_ids[:25] == first.token_ids + tuple(range(200, 205))
    assert second.calibration.effective_prefix_tokens == 25
    assert second.calibration.prefix_shortfall_tokens == 0


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
            "experiment": {"task_num": 1},
            "replay": {
                "trace_type": "tracelab",
                "trace_path": source,
                "prompt_calibration_tolerance_tokens": 0,
            },
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
def test_tracelab_caps_unavailable_prefix_and_fills_input(mode: str) -> None:
    async def build() -> None:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "replay": {
                    "trace_type": "tracelab",
                    "trace_path": "source.jsonl",
                    "prompt_calibration_tolerance_tokens": 0,
                    "context_adjustment_mode": mode,
                },
            }
        )
        builder = PromptBuilder(config, _RecipeTokenizer(), SimpleNamespace(prompt_source_kind="token_recipe"))
        previous = SyntheticPrompt("", (), (), token_ids=(10, 11, 12, 13))
        exchange = PromptExchange(previous, "live output", None, (14, 15))
        node = SimpleNamespace(
            source_key="next",
            prompt_recipe_key="next",
            context_after="first",
            context_mode="append",
            planned_input_tokens=10,
            source_prefix_tokens=8,
        )
        prompt = await builder.build(SimpleNamespace(runtime_session_id="runtime"), node, exchange)
        assert prompt.token_ids is not None
        assert prompt.token_ids[:6] == (10, 11, 12, 13, 14, 15)
        assert len(prompt.token_ids) == 10
        assert prompt.calibration is not None
        assert prompt.calibration.source_prefix_tokens == 8
        assert prompt.calibration.effective_prefix_tokens == 6
        assert prompt.calibration.prefix_shortfall_tokens == 2
        assert prompt.calibration.requested_filler_tokens == 4

    asyncio.run(build())


@pytest.mark.parametrize("fail_first", [False, True])
def test_tracelab_runner_uses_token_prefix_and_skips_failed_dependencies(
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
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "max_model_len": 1000}]}
            )
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            if "prompt" in body:
                return httpx.Response(200, json={"tokens": [ord(c) for c in body["prompt"]]})
            count = 3 + sum(4 + len(message["content"]) for message in body["messages"])
            return httpx.Response(200, json={"count": count})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"prompt": "".join(chr(t) for t in body["tokens"])})
        assert request.url.path == "/v1/completions"
        sent.append(body)
        count = len(body["prompt"])
        if not body.get("stream"):
            return httpx.Response(200, json={"usage": {"prompt_tokens": count, "completion_tokens": 1}})
        output_ids = list(range(400, 400 + body["max_tokens"]))
        chunk = {
            "choices": [{"text": "A" * len(output_ids), "token_ids": output_ids}],
            "usage": {
                "prompt_tokens": count + int(fail_first and len(sent) == 2),
                "completion_tokens": len(output_ids),
            },
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
        assert len(sent) == 2
        assert statuses == ["failed", "skipped_dependency_failed"]
        assert execution["summary"]["dependency_skipped_requests"] == 1
        assert json.loads((result / "manifest.json").read_text())["status"] == "failed"
    else:
        assert len(sent) == 3
        assert statuses == ["success", "success"]
        assert sent[1]["prompt"][:20] == sent[0]["prompt"][:20]
        assert sent[2]["prompt"][:29] == sent[1]["prompt"][:29]
        assert execution["summary"]["prompt_calibration_exact_requests"] == 2
        assert json.loads((result / "manifest.json").read_text())["status"] == "completed"


def test_tracelab_eight_tasks_four_concurrent_replays_have_same_token_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "eight-sessions.jsonl"
    rows = []
    for session_index in range(8):
        for round_index, (input_count, prefix_count) in enumerate(((30, 20), (40, 34), (55, 52))):
            row = _round(
                round_index,
                input_tokens=input_count,
                prefix_tokens=prefix_count,
                start=round_index * 3,
                end=round_index * 3 + 1,
            )
            row["session_id"] = f"session-{session_index}"
            if session_index == 0 and round_index == 0:
                row["output_tokens"] = 0
            rows.append(row)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    run_index = 0
    requests_by_run: list[dict[str, list[tuple[list[int], list[int]]]]] = []
    warmups: list[list[int]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(503)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "max_model_len": 1000}]}
            )
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(character) for character in body["prompt"]]})
        assert request.url.path == "/v1/completions"
        prompt_ids = body["prompt"]
        if not body.get("stream"):
            warmups.append(prompt_ids)
            return httpx.Response(200, json={"usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": 1}})
        session = request.headers["x-claude-code-session-id"]
        per_session = requests_by_run[run_index].setdefault(session, [])
        round_index = len(per_session)
        output_ids = list(
            range(
                400 + run_index * 1000 + round_index * 10,
                400 + run_index * 1000 + round_index * 10 + body["max_tokens"],
            )
        )
        per_session.append((prompt_ids, output_ids))
        chunk = {
            "choices": [{"text": "A" * len(output_ids), "token_ids": output_ids}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(output_ids)},
        }
        return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    summaries = []
    for run_index in range(2):
        requests_by_run.append({})
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {
                    "task_num": 8,
                    "max_concurrency": 4,
                    "task_timeout_seconds": None,
                    "result_dir": tmp_path / f"run-{run_index}",
                },
                "replay": {"trace_type": "tracelab", "trace_path": source, "trace_same_agent_gap_scale": 0},
            }
        )
        result = run_replay(config)
        execution = json.loads((result / "replay-execution.json").read_text())
        summaries.append(execution["summary"])
        assert json.loads((result / "manifest.json").read_text())["status"] == "completed"
        assert execution["summary"]["planned_tasks"] == 8
        assert execution["summary"]["completed_tasks"] == 8
        assert execution["summary"]["planned_requests"] == 24
        assert execution["summary"]["successful_requests"] == 24
        assert execution["summary"]["requests_with_prefix_shortfall"] == 9
        assert execution["summary"]["prefix_shortfall_tokens"] == 35
        assert execution["summary"]["source_zero_output_requests"] == 1
        assert execution["summary"]["zero_output_requests_observed_one_token"] == 1
        assert len(requests_by_run[run_index]) == 8
        facts = [json.loads(line) for line in (result / "requests.jsonl").read_text().splitlines()]
        assert len(facts) == 24
        assert sum(fact["input_tokens"] for fact in facts) == 8 * (30 + 40 + 55)
        assert sum(fact["output_tokens"] for fact in facts) == 8 * (7 + 8 + 9) - 6
        first_branch_tokens = [requests[0][0][20] for requests in requests_by_run[run_index].values()]
        assert len(set(first_branch_tokens)) == 8
        for requests in requests_by_run[run_index].values():
            assert len(requests) == 3
            assert requests[0][0][:20] == warmups[run_index][:20]
            assert len(requests[0][0]) == 30
            reused = min(34, len(requests[0][0]) + len(requests[0][1]))
            assert requests[1][0][:reused] == (requests[0][0] + requests[0][1])[:reused]
            assert len(requests[1][0]) == 40
            assert requests[2][0][:48] == requests[1][0] + requests[1][1]
            assert len(requests[2][0]) == 55
    assert len(warmups) == 2
    assert len(warmups[0]) == len(warmups[1]) == 21
    for session, first_run in requests_by_run[0].items():
        second_run = requests_by_run[1][session]
        assert first_run[0][0] == second_run[0][0]
        assert first_run[1][0] != second_run[1][0]
    assert all(
        summaries[0][key] == summaries[1][key]
        for key in ("planned_requests", "successful_requests", "source_prefix_tokens", "effective_prefix_tokens")
    )


def test_tracelab_live_eight_tasks_four_concurrent(tmp_path: Path) -> None:
    base_url = os.environ.get("REPLAY_LIVE_BASE_URL")
    if not base_url:
        pytest.skip("set REPLAY_LIVE_BASE_URL to run against a live vLLM server")
    source = tmp_path / "live-eight-sessions.jsonl"
    rows = []
    for session_index in range(8):
        for round_index, (input_count, prefix_count) in enumerate(((64, 32), (80, 75))):
            row = _round(
                round_index,
                input_tokens=input_count,
                prefix_tokens=prefix_count,
                start=round_index * 3,
                end=round_index * 3 + 1,
            )
            row["session_id"] = f"live-session-{session_index}"
            if session_index == 0 and round_index == 0:
                row["output_tokens"] = 0
            rows.append(row)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {
                "task_num": 8,
                "max_concurrency": 4,
                "task_timeout_seconds": None,
                "result_dir": tmp_path / "live-run",
            },
            "backend": {"base_url": base_url},
            "replay": {"trace_type": "tracelab", "trace_path": source, "trace_same_agent_gap_scale": 0},
        }
    )
    result = run_replay(config)
    execution = json.loads((result / "replay-execution.json").read_text())
    summary = execution["summary"]
    assert summary["planned_tasks"] == summary["completed_tasks"] == 8
    assert summary["planned_requests"] == summary["successful_requests"] == 16
    assert summary["requests_with_prefix_shortfall"] == 8
    assert summary["prefix_shortfall_tokens"] == 38
    assert summary["source_zero_output_requests"] == 1
    assert summary["zero_output_requests_observed_one_token"] == 1
    facts = [json.loads(line) for line in (result / "requests.jsonl").read_text().splitlines()]
    assert len(facts) == 16
    assert sum(fact["input_tokens"] for fact in facts) == 8 * (64 + 80)
    assert sum(fact["output_tokens"] for fact in facts) == 8 * (7 + 8) - 6


def test_tracelab_rejects_context_limit_before_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "rounds.jsonl"
    _write_source(source)
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            return httpx.Response(200, json={"tokens": [ord(character) for character in body["prompt"]]})
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200, json={"data": [{"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "max_model_len": 31}]}
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1, "max_concurrency": 4, "result_dir": tmp_path / "too-long"},
            "replay": {"trace_type": "tracelab", "trace_path": source},
        }
    )
    with pytest.raises(ValueError, match="TraceLab request .* context tokens"):
        run_replay(config)
    assert requests[-1] == "/v1/models"
    assert "/v1/completions" not in requests
    assert json.loads((config.experiment.result_dir / "manifest.json").read_text())["status"] == "failed"
