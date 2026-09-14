# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.prompt import SyntheticPrompt
from agentinfer.agentbench.replay.transport import ReplayTransport
from agentinfer.agentbench.request_proxy.request_trace import RequestTraceWriter, load_request_facts


def test_replay_transport_disables_keepalive_reuse(tmp_path: Path) -> None:
    config = ReplayBenchConfig.model_validate(
        {
            "backend": {"base_url": "http://backend", "endpoint": "/v1/messages"},
            "replay": {"trace_path": "source.jsonl"},
        }
    )

    with patch("agentinfer.agentbench.replay.transport.httpx.AsyncClient", wraps=httpx.AsyncClient) as client:
        transport = ReplayTransport(config, "run", RequestTraceWriter(tmp_path / "requests.jsonl"))

    limits = client.call_args.kwargs["limits"]
    assert limits.max_connections is None
    assert limits.max_keepalive_connections == 0
    asyncio.run(transport.close())


def test_chat_transport_streams_content_usage_and_plan_identity(tmp_path: Path) -> None:
    async def send() -> tuple[object, dict[str, object]]:
        config = ReplayBenchConfig.model_validate(
            {
                "backend": {"base_url": "http://backend", "endpoint": "/v1/chat/completions"},
                "replay": {"trace_path": "source.jsonl"},
            }
        )
        trace = tmp_path / "requests.jsonl"
        writer = RequestTraceWriter(trace)
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.client.aclose()
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            stream = (
                'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        task = SimpleNamespace(runtime_session_id="session")
        node = SimpleNamespace(
            runtime_request_id="request",
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_main",
            parent_actor_id=None,
            planned_output_tokens=3,
            backend_sampling_seed=7,
        )
        result = await transport.send(
            task,
            node,
            SyntheticPrompt("system", (), ({"role": "user", "content": "message"},)),
        )
        await transport.close()
        await writer.close()
        return result, captured

    result, body = asyncio.run(send())
    facts = load_request_facts(tmp_path / "requests.jsonl")

    assert result.assistant_content == "hello"
    assert body["min_tokens"] == body["max_tokens"] == 3
    assert body["seed"] == 7
    assert "tools" not in body
    assert facts[0].request_id == "request"
    assert facts[0].request_purpose == "lead_main"
    assert facts[0].input_tokens == 12
    assert facts[0].output_tokens == 3


def test_messages_transport_keeps_title_prompt_tool_free_and_bridges_sampling(tmp_path: Path) -> None:
    async def build() -> dict[str, object]:
        config = ReplayBenchConfig.model_validate(
            {
                "backend": {"base_url": "http://backend", "endpoint": "/v1/messages"},
                "replay": {"trace_path": "source.jsonl"},
            }
        )
        transport = ReplayTransport(config, "run", RequestTraceWriter(tmp_path / "requests.jsonl"))
        task = SimpleNamespace(runtime_session_id="session")
        node = SimpleNamespace(
            runtime_request_id="title",
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_title",
            parent_actor_id=None,
            planned_output_tokens=4,
            backend_sampling_seed=9,
        )
        body = transport._body(
            task,
            node,
            SyntheticPrompt("title system", (), ({"role": "user", "content": "title input"},)),
        )
        await transport.close()
        return body

    body = asyncio.run(build())

    assert body["tools"] == []
    sampling = body["metadata"]["_agentinfer_replay_sampling"]
    assert sampling == {"seed": 9, "min_tokens": 4, "ignore_eos": True}
