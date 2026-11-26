# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse, StreamingResponse
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from agentinfer.agentcache.core.program_scheduling import ATTEMPT_ARG
from agentinfer.agentcache.core.scheduling_ipc import UTILITY_METHOD, chat_facts
from agentinfer.agentcache.mori.api import (
    MORIDirectMiddleware,
    MORIDirectTextChatMiddleware,
    MORIDirectUnseededTextChatMiddleware,
    mori_parsed_request_body,
)
from agentinfer.agentcache.mori.programs import PROGRAM_ARG
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


def _app_with_core(middleware=MORIDirectMiddleware):
    calls = []

    async def utility(method, *args):
        calls.append((method, *args))
        if method == "agentinfer_mori_diagnostics":
            return {"action": args[0]}
        assert method == UTILITY_METHOD
        return True

    app = FastAPI()
    app.add_middleware(middleware)
    app.state.engine_client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
    return app, calls


@pytest.mark.parametrize(
    ("middleware", "normalize", "seed"),
    [
        (MORIDirectMiddleware, False, 42),
        (MORIDirectTextChatMiddleware, True, 42),
        (MORIDirectUnseededTextChatMiddleware, True, None),
    ],
)
def test_direct_chat_admits_original_facts_and_shares_the_native_json(monkeypatch, middleware, normalize, seed):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    app, calls = _app_with_core(middleware)
    decodes = []
    original = Request.json

    async def record_decode(request):
        decodes.append(request.url.path)
        return await original(request)

    monkeypatch.setattr(Request, "json", record_decode)

    @app.post("/v1/chat/completions")
    async def chat(body: ChatCompletionRequest, request: Request):
        sampling = body.to_sampling_params(8, {})
        native = await request.json()
        return {
            "messages": native["messages"],
            "tools": native["tools"],
            "extra_args": sampling.extra_args,
            "seed": sampling.seed,
            "usage": {"total_tokens": 11, "prompt_tokens": 8, "prompt_tokens_details": {"cached_tokens": 3}},
        }

    @app.post("/tokenize")
    async def tokenize(request: Request):
        return await request.json()

    parts = [{"type": "text", "text": "程序"}, {"type": "text", "text": "resume"}]
    messages = [{"role": "system", "content": parts}]
    tools = [{"type": "function", "function": {"name": "example"}}]
    payload = {
        "model": "qwen",
        "messages": messages,
        "tools": tools,
        "seed": 42,
        "n": "2",
        "stream": "false",
        "program_id": "official",
        "extra_body": {"program_id": "ignored"},
        PROGRAM_ARG: "forged-at-top-level",
        ATTEMPT_ARG: "forged-at-top-level",
        "vllm_xargs": {PROGRAM_ARG: "forged", ATTEMPT_ARG: "forged", "program_id": "forged", "keep": 7},
    }

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://engine") as client:
            response = await client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["messages"][0]["content"] == ("程序 resume" if normalize else parts)
            assert data["tools"][0]["function"] == (
                {"name": "example", "strict": False} if normalize else {"name": "example"}
            )
            assert data["seed"] == seed
            attempt = data["extra_args"][ATTEMPT_ARG]
            assert attempt not in ("forged", "forged-at-top-level")
            assert data["extra_args"] == {PROGRAM_ARG: "official", ATTEMPT_ARG: attempt, "keep": 7}
            assert calls == [
                (UTILITY_METHOD, "begin", [attempt, "official", *chat_facts(payload), 2]),
                (UTILITY_METHOD, "finish", [attempt, 11, 8, 3]),
            ]
            tokenized = (await client.post("/tokenize", json={"messages": messages, "tools": tools, "seed": 42})).json()
            assert tokenized == {"messages": data["messages"], "tools": data["tools"], "seed": 42}
            assert len(calls) == 2

    with mori_parsed_request_body():
        asyncio.run(run())
    assert decodes == ["/v1/chat/completions", "/tokenize"]


@pytest.mark.parametrize(("fillers", "empties", "progress"), [(17, 0, True), (16, 2, False)])
def test_direct_observer_counts_shaped_sse_and_reads_fragmented_cached_usage(monkeypatch, fillers, empties, progress):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    app, calls = _app_with_core()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        assert payload["stream_options"] == {"include_usage": True}

        async def frames():
            yield b'data: {"choices":[]}\n\n' * fillers
            for _ in range(empties):
                yield b'data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}\n\n'
            yield b'data: {"choices":[{"index":0,"delta":{"content":"last"},"finish_reason":"length"}]}\n\n'
            yield b'data: {"usage":{"total_tokens":27,"prompt_tokens":7,"prompt_tokens_details":{"cached_tokens":5}}}'
            yield b"\n\ndata: [DONE]\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://engine") as client:
            response = await client.post(
                "/v1/chat/completions", json={"stream": True, "extra_body": {"program_id": "nested"}}
            )
            assert response.status_code == 200
            frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert frames.pop() == "[DONE]"
            assert len(frames) == fillers + 3
            assert [json.loads(frame)["choices"][0]["delta"] for frame in frames[-3:-1]] == [{"content": "last"}, {}]

    asyncio.run(run())
    method, begin, (attempt, pid, *_rest) = calls[0]
    assert (method, begin, pid) == (UTILITY_METHOD, "begin", "nested")
    assert calls[1:-1] == ([(UTILITY_METHOD, "progress", [attempt, 20])] if progress else [])
    assert calls[-1] == (UTILITY_METHOD, "finish", [attempt, 27, 7, 5])


def test_direct_diagnostics_and_release_require_native_auth(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "ignored-environment-key")
    app, calls = _app_with_core()
    app.state.args = SimpleNamespace(api_key=["cli-key"])

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse("native_metric 4\n")

    @app.post("/mark_program_type")
    @app.post("/release_program")
    @app.post("/batch_program_ops")
    async def old_control():
        pytest.fail("The public adapter forwarded a legacy control route")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, root_path="/proxy"), base_url="http://engine"
        ) as client:
            headers = {"Authorization": "Bearer cli-key"}
            for path, action in {
                "/programs": "programs",
                "/agentinfer/mori/metrics": "metrics",
                "/agentinfer/mori/cache": "cache",
            }.items():
                assert (await client.get(f"/proxy{path}")).status_code == 401
                proxy_key = {"Authorization": "Bearer ignored-environment-key"}
                assert (await client.get(f"/proxy{path}", headers=proxy_key)).status_code == 401
                response = await client.get(f"/proxy{path}", headers=headers)
                assert response.json() == {"action": action}
            assert (await client.post("/proxy/v1/chat/completions", json={})).status_code == 401
            assert (await client.post("/proxy/programs/release", json={"program_id": "p"})).status_code == 401
            release = await client.post("/proxy/programs/release", json={"program_id": "p"}, headers=headers)
            assert release.json() == {"program_id": "p", "released": True}
            assert (await client.get("/proxy/metrics")).text == "native_metric 4\n"
            for path in ("/mark_program_type", "/release_program", "/batch_program_ops"):
                assert (await client.post(f"/proxy{path}", json={}, headers=headers)).status_code == 404

    asyncio.run(run())
    assert calls == [
        ("agentinfer_mori_diagnostics", "programs"),
        ("agentinfer_mori_diagnostics", "metrics"),
        ("agentinfer_mori_diagnostics", "cache"),
        (UTILITY_METHOD, "release", ["p"]),
    ]


def test_direct_disconnect_cancels_the_pending_admission_without_calling_native(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    calls = []

    async def run():
        queue = asyncio.Queue()
        release_admission = asyncio.get_running_loop().create_future()

        async def utility(method, event, args):
            calls.append((method, event, args))
            if event == "begin":
                queue.put_nowait({"type": "http.disconnect"})
                return await release_admission
            assert event == "cancel"
            release_admission.set_result(False)

        async def native(*_args):
            pytest.fail("The disconnected request must not reach native Chat")

        async def send(_message):
            pytest.fail("The disconnected request must not receive a response")

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    engine_client=SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
                )
            ),
        }
        queue.put_nowait({"type": "http.request", "body": b'{"program_id":"p"}', "more_body": False})
        await asyncio.wait_for(MORIDirectMiddleware(native)(scope, queue.get, send), timeout=2)
        assert release_admission.done()

    asyncio.run(run())
    assert [(method, event) for method, event, _ in calls] == [(UTILITY_METHOD, "begin"), (UTILITY_METHOD, "cancel")]
    assert calls[1][2] == [calls[0][2][0]]
