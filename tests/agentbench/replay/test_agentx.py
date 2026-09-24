# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""AgentX full-snapshot conversion and token-ID Replay contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.converters.agentx import AgentXConverter, analyze_agentx_trace_ir
from agentinfer.agentbench.replay.hash_snapshot import HashSnapshotStore, TokenBlockRenderer
from agentinfer.agentbench.replay.planner import build_replay_plan
from agentinfer.agentbench.replay.prompt import PromptBuilder, PromptExchange
from agentinfer.agentbench.replay.runner import run_replay
from agentinfer.agentbench.replay.transport import ReplayTransport
from agentinfer.agentbench.replay.unified_trace_ir import validate_trace_ir, write_trace_ir_manifest
from agentinfer.agentbench.request_proxy.request_trace import RequestTraceWriter, load_request_facts


def _request(t: float, duration: float, hashes: list[int], *, output: int = 3, model: str = "model-a") -> dict:
    return {
        "t": t,
        "api_time": duration,
        "in": 64 * len(hashes),
        "out": output,
        "hash_ids": hashes,
        "model": model,
        "type": "n",
    }


def _source(path: Path, *, zero_output: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "source-session",
                "models": ["model-a", "model-b"],
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [
                    _request(0, 4, [0]),
                    _request(1, 1, [0, 1]),
                    {
                        "t": 3,
                        "type": "subagent",
                        "agent_id": "child",
                        "duration_ms": 3000,
                        "requests": [
                            _request(3, 2, [0, 2], model="model-b"),
                            _request(3.2, 1, [0, 2, 3], model="model-b"),
                        ],
                    },
                    _request(6, 1, [0], output=0 if zero_output else 2),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _config(source: Path, *, result_dir: Path | None = None) -> ReplayBenchConfig:
    return ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1, "result_dir": result_dir or source.parent / "result"},
            "backend": {"base_url": "http://backend", "model": "served-model", "endpoint": "/v1/completions"},
            "replay": {"trace_type": "agentX", "trace_path": source},
        }
    )


def _converted(source: Path, output: Path):
    summary = AgentXConverter().convert(source, output)
    ir = validate_trace_ir(output / "requests.jsonl")
    return summary, ir, analyze_agentx_trace_ir(ir)


def test_agentx_converter_preserves_nested_requests_and_timing(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    summary, ir, analysis = _converted(source, tmp_path / "convert")
    requests = analysis.sessions[0].requests

    assert (summary.sessions, summary.requests) == (1, 5)
    assert [request.actor_id for request in requests] == ["lead", "lead", "child", "child", "lead"]
    assert requests[1].send_after is None  # Starts while first lead request is still running.
    assert requests[2].send_after == requests[1].key
    assert requests[3].send_after == requests[1].key
    assert requests[4].send_after == requests[2].key
    assert requests[2].context_after is None
    assert requests[2].parent_actor_id == "lead"
    assert requests[2].dependency_kind == "cross_agent_completion"
    assert requests[3].key in requests[2].parallel_with
    assert analysis.row_analysis["subagent_requests"] == 2
    assert analysis.row_analysis["zero_output_requests"] == 0
    recipe = HashSnapshotStore(ir.requests_path).read(requests[2].prompt_recipe_key, requests[2].key)
    assert recipe["hash_ids"] == [0, 2]
    assert recipe["source_model"] == "model-b"
    assert build_replay_plan(_config(source), analysis, ir).tasks[0].requests[2].context_after is None


def test_agentx_zero_output_is_retained_and_selected_plan_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _source(source, zero_output=True)
    summary, ir, analysis = _converted(source, tmp_path / "convert")

    assert summary.requests == 5
    assert analysis.row_analysis["zero_output_requests"] == 1
    with pytest.raises(ValueError, match="zero-output request"):
        build_replay_plan(_config(source), analysis, ir)


def test_agentx_hash_block_count_uses_exact_integer_division(tmp_path: Path) -> None:
    block_size = 2**53 + 1
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "large-token-target",
                "block_size": block_size,
                "hash_id_scope": "local",
                "requests": [
                    {
                        "t": 0,
                        "api_time": 1,
                        "in": block_size + 1,
                        "out": 1,
                        "hash_ids": [0, 1],
                        "model": "source-model",
                        "type": "n",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary, ir, _ = _converted(source, tmp_path / "convert")
    assert summary.requests == 1
    assert validate_trace_ir(ir.requests_path).prompt_source_kind == "hash_snapshot"


@pytest.mark.parametrize("mutation", ["hash_count", "scope", "foreign_edge", "missing_hash"])
def test_agentx_ir_rejects_broken_recipes(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    summary, ir, _ = _converted(source, tmp_path / "convert")
    rows = [json.loads(line) for line in ir.requests_path.read_text().splitlines()]
    if mutation == "hash_count":
        rows[0]["hash_ids"].append(7)
    elif mutation == "scope":
        rows[0]["hash_id_scope"] = "global"
    elif mutation == "foreign_edge":
        rows[0]["send_after"] = "unknown"
    else:
        rows[0].pop("hash_ids")
    ir.requests_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original_summary = json.loads(ir.manifest_path.read_text())["summary"]
    write_trace_ir_manifest(
        ir.root,
        converter_name=AgentXConverter.name,
        source_path=source,
        summary={**summary.to_dict(), **original_summary},
        prompt_source_kind="hash_snapshot",
    )

    with pytest.raises(ValueError, match="hash_ids|hash_id_scope|send_after"):
        validate_trace_ir(ir.requests_path)


def test_agentx_token_blocks_preserve_prefix_and_runtime_isolation() -> None:
    renderer = TokenBlockRenderer(tuple(range(100, 120)), cache_blocks=2)
    recipe = {
        "hash_id_scope": "local",
        "block_size": 64,
        "input_tokens": 131,
        "hash_ids": [0, 1, 2],
        "source_model": "source-model",
    }
    first = renderer.render(recipe, "runtime-one")
    second = renderer.render({**recipe, "hash_ids": [0, 1, 3]}, "runtime-one")
    other_session = renderer.render(recipe, "runtime-two")
    other_model = renderer.render({**recipe, "source_model": "other-model"}, "runtime-one")

    assert len(first) == 131
    assert first[:128] == second[:128]
    assert first[128:] != second[128:]
    assert first[:64] != other_session[:64]
    assert first[:64] != other_model[:64]
    assert len(renderer.cache) <= 2


def test_agentx_prompt_is_complete_and_independent_of_live_answer(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    _, ir, analysis = _converted(source, tmp_path / "convert")
    config = _config(source)
    task = build_replay_plan(config, analysis, ir).tasks[0]

    class FakeTokenizer:
        async def text_token_ids(self, text: str) -> tuple[int, ...]:
            return tuple(range(50, 100))

    async def build():
        prompts = PromptBuilder(config, FakeTokenizer(), ir)  # type: ignore[arg-type]
        first = await prompts.build(task, task.requests[0], None)
        next_prompt = await prompts.build(task, task.requests[1], None)
        repeat = await prompts.build(task, task.requests[1], None)
        with pytest.raises(ValueError, match="cannot consume"):
            await prompts.build(task, task.requests[1], PromptExchange(first, "different live answer"))
        return first, next_prompt, repeat

    first, next_prompt, repeat = asyncio.run(build())
    assert next_prompt.token_ids == repeat.token_ids
    assert next_prompt.token_ids is not None and first.token_ids is not None
    assert next_prompt.token_ids[:64] == first.token_ids
    assert len(next_prompt.token_ids) == 128
    assert next_prompt.messages == ()


@pytest.mark.parametrize(
    "observed_input, observed_output, done, success",
    [
        (128, 3, True, True),
        (127, 3, True, False),
        (128, 2, True, False),
        (128, 3, False, False),
        (None, 3, True, False),
    ],
)
def test_agentx_completions_wire_and_usage(
    tmp_path: Path, observed_input: int | None, observed_output: int, done: bool, success: bool
) -> None:
    source = tmp_path / "source.jsonl"
    _source(source)
    _, ir, analysis = _converted(source, tmp_path / "convert")
    config = _config(source)
    task = build_replay_plan(config, analysis, ir).tasks[0]
    node = task.requests[2]
    body: dict[str, object] = {}

    class FakeTokenizer:
        async def text_token_ids(self, text: str) -> tuple[int, ...]:
            return tuple(range(50, 100))

    async def send():
        prompt = await PromptBuilder(config, FakeTokenizer(), ir).build(task, node, None)  # type: ignore[arg-type]
        writer = RequestTraceWriter(tmp_path / "observed.jsonl")
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.client.aclose()

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/completions"
            body.update(json.loads(request.content))
            usage = {"completion_tokens": observed_output}
            if observed_input is not None:
                usage["prompt_tokens"] = observed_input
            chunks = [
                'data: {"choices":[{"text":"reply"}]}\n\n',
                f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n",
            ]
            if done:
                chunks.append("data: [DONE]\n\n")
            return httpx.Response(200, content="".join(chunks), headers={"content-type": "text/event-stream"})

        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        result = await transport.send(task, node, prompt)
        await transport.close()
        await writer.close()
        return result

    result = asyncio.run(send())
    facts = load_request_facts(tmp_path / "observed.jsonl")
    assert result.success is success
    assert body["add_special_tokens"] is False
    assert len(body["prompt"]) == 128
    assert body["max_tokens"] == body["min_tokens"] == 3
    assert "messages" not in body
    assert facts[0].actor_id == "child"
    identity = body["vllm_xargs"]["agentic_context"]  # type: ignore[index]
    if isinstance(identity, str):
        identity = json.loads(identity)
    assert identity["blocks_parent"] is False
    assert result.assistant_content == "reply"


def test_agentx_runner_records_same_snapshots_with_different_live_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "session",
                "block_size": 64,
                "hash_id_scope": "local",
                "models": ["source-model"],
                "requests": [
                    _request(0, 0.01, [0]),
                    {
                        "t": 0.02,
                        "type": "subagent",
                        "agent_id": "child",
                        "requests": [
                            _request(0.02, 0.01, [0, 1]),
                        ],
                    },
                    _request(0.04, 0.01, [0, 2]),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    real_client = httpx.AsyncClient
    sent: list[list[dict[str, object]]] = [[], []]
    selected_run = 0

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(503)
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(character) for character in body["prompt"]]})
        assert request.url.path == "/v1/completions"
        sent[selected_run].append(body)
        content = "A" if selected_run == 0 else "Z"
        stream = (
            f"data: {json.dumps({'choices': [{'text': content}]})}\n\n"
            f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': len(body['prompt']), 'completion_tokens': body['max_tokens']}})}\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs)
    )
    fingerprints = []
    for run_index in range(2):
        selected_run = run_index
        result = run_replay(_config(source, result_dir=tmp_path / f"result-{run_index}"))
        execution = json.loads((result / "replay-execution.json").read_text())
        plan = json.loads((result / "replay-plan.json").read_text())
        summary = json.loads((result / "summary.json").read_text())
        assert execution["summary"]["successful_requests"] == 3
        assert summary["execution"]["metadata"]["planned_requests"] == 3
        assert summary["execution"]["metadata"]["snapshot_cache_usage_coverage"] == 0
        assert len(load_request_facts(result / "requests.jsonl")) == 3
        fingerprints.append(plan["workload_fingerprint"])

    assert fingerprints[0] == fingerprints[1]
    assert [body["prompt"] for body in sent[0]] == [body["prompt"] for body in sent[1]]
    assert [len(body["prompt"]) for body in sent[0]] == [64, 128, 128]
    assert sent[0][0]["prompt"] == sent[0][1]["prompt"][:64]


def test_agentx_runner_bounds_session_inflight_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "session",
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [_request(0, 0.01, [index]) for index in range(4)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    real_client = httpx.AsyncClient
    active = peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        if request.url.path == "/metrics":
            return httpx.Response(503)
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(character) for character in body["prompt"]]})
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        stream = (
            f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 64, 'completion_tokens': 3}})}\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs)
    )
    config = _config(source, result_dir=tmp_path / "result")
    config.replay.max_inflight_requests = 1
    result = run_replay(config)
    execution = json.loads((result / "replay-execution.json").read_text())
    assert execution["summary"]["successful_requests"] == 4
    assert peak == 1
