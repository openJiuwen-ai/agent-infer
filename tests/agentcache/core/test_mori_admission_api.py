# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agentinfer.agentcache.mori.admission_api import MoriSchedulingMiddleware
from agentinfer.agentcache.mori.ipc import ATTEMPT_ARG, CHILD_COUNT_ARG, PROGRAM_ARG, RESERVED_ARGS
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


def test_concurrent_http_requests_isolate_native_children_and_strip_reserved_args(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    arrivals = asyncio.Queue()
    unblock = asyncio.Event()
    operations = []

    async def add(request):
        operations.append(("add", request.sampling_params.extra_args))

    async def utility(_method, event, args):
        operations.append((event, args))

    core = SimpleNamespace(add_request_async=add, call_utility_async=utility)

    async def endpoint(request):
        payload = await request.json()
        arrivals.put_nowait(None)
        await unblock.wait()

        async def child():
            params = SimpleNamespace(extra_args=payload["vllm_xargs"])
            await core.add_request_async(SimpleNamespace(sampling_params=params))

        await asyncio.gather(*(child() for _ in range(payload.get("n", 1))))
        return JSONResponse({})

    app = Starlette(
        routes=[Route(path, endpoint, methods=["POST"]) for path in ("/v1/chat/completions", "/v1/completions")]
    )
    app.add_middleware(MoriSchedulingMiddleware)
    app.state.engine_client = SimpleNamespace(engine_core=core)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            injected = {"keep": 7, **dict.fromkeys(RESERVED_ARGS, "user")}
            calls = []
            for path, pid, n in [
                ("/v1/completions", "unmanaged", 1),
                ("/v1/chat/completions", "A", 2),
                ("/v1/chat/completions", "B", 1),
            ]:
                calls.append(
                    asyncio.create_task(client.post(path, json={"program_id": pid, "n": n, "vllm_xargs": injected}))
                )
            for _ in calls:
                await arrivals.get()
            unblock.set()
            assert all(response.status_code == 200 for response in await asyncio.gather(*calls))

    asyncio.run(asyncio.wait_for(run(), 3))
    added = [facts for event, facts in operations if event == "add"]
    assert [facts for facts in added if PROGRAM_ARG not in facts] == [{"keep": 7}]
    for pid, n in (("A", 2), ("B", 1)):
        facts = [args for args in added if args.get(PROGRAM_ARG) == pid]
        assert len(facts) == n and all(child == facts[0] for child in facts)
        assert facts[0][CHILD_COUNT_ARG] == n and facts[0]["keep"] == 7
        assert operations.count(("finish", [facts[0][ATTEMPT_ARG], None, 0, 0])) == 1
    assert len({facts[ATTEMPT_ARG] for facts in added if ATTEMPT_ARG in facts}) == 2


@pytest.mark.parametrize("submit_first", [False, True], ids=["before-add", "partial-fanout"])
def test_disconnected_chat_does_not_submit_late_children(monkeypatch, submit_first):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    operations = []

    async def run():
        send_late = asyncio.Event()
        delayed = []

        async def add(child):
            operations.append(("add", child.sampling_params.extra_args[ATTEMPT_ARG]))

        async def utility(_method, event, args):
            operations.append((event, args[0]))

        core = SimpleNamespace(add_request_async=add, call_utility_async=utility)
        queue = asyncio.Queue()
        queue.put_nowait({"type": "http.request", "body": json.dumps({"n": 2}).encode(), "more_body": False})
        queue.put_nowait({"type": "http.disconnect"})

        async def native(_scope, receive, _send):
            assert (await receive())["type"] == "http.request"

            async def child():
                await core.add_request_async(SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None)))

            async def sibling():
                await send_late.wait()
                await child()

            if submit_first:
                await child()
            delayed.append(asyncio.create_task(sibling()))
            assert (await receive())["type"] == "http.disconnect"

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(engine_client=SimpleNamespace(engine_core=core))),
        }
        await MoriSchedulingMiddleware(native)(scope, queue.get, None)
        send_late.set()
        with pytest.raises(asyncio.CancelledError):
            await delayed[0]

    asyncio.run(asyncio.wait_for(run(), 3))
    if submit_first:
        add, cancel = operations
        assert (add[0], cancel[0]) == ("add", "cancel") and add[1] == cancel[1]
    else:
        assert operations == []
