# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Test the Program scheduling boundary with an independent host."""

from types import SimpleNamespace

import pytest

from agentinfer.agentcache.core.program_scheduling import BackendObservation, ProgramSchedulingBridge
from agentinfer.agentcache.core.scheduling_ipc import chat_facts
from agentinfer.scheduling import runtime
from agentinfer.scheduling.mori import MoriConfig


class RecordingHost:
    def __init__(self, hbm: int, dram: int) -> None:
        self.observation = BackendObservation(True, hbm, dram)
        self.operations = []

    def observe_backend(self) -> BackendObservation:
        return self.observation

    def apply_program_ops(self, operations) -> None:
        self.operations.extend((op.program.program_id, op.placement and op.placement.value) for op in operations)


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=1.0)
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: state.now))
    return state


def test_admission_uses_pretemplate_chars_and_exact_effective_capacity():
    admitted = ProgramSchedulingBridge(RecordingHost(123, 0), MoriConfig())
    waiting = ProgramSchedulingBridge(RecordingHost(123, 0), MoriConfig())

    assert admitted.begin("a", "program", 115, "_empty_").result()
    assert not waiting.begin("b", "program", 158, "_empty_").done()


def test_mori_chat_facts_use_system_prefix_and_canonical_tools():
    payload = {"messages": [{"role": "system", "content": [{"text": "系统"}]}], "tools": [{"z": 1, "a": 2}]}
    chars, prefix = chat_facts(payload)
    new_chars, new_prefix = chat_facts({**payload, "stream_options": {"include_usage": True}})
    assert new_chars > chars
    assert new_prefix == prefix
    assert len(prefix) == 16
    assert chat_facts({"messages": [{"role": "user", "content": "hello"}]})[1] == "_empty_"


@pytest.mark.parametrize("dram,can_displace", [(160, True), (159, False)])
def test_idle_displacement_requires_dram_and_then_resumes_waiter(clock, dram, can_displace):
    host = RecordingHost(300, dram)
    bridge = ProgramSchedulingBridge(host, MoriConfig())
    assert bridge.begin("a", "A", 500, "_empty_").result()
    assert bridge.attach("a")
    clock.now = 2.0
    bridge.finish("a", 160, 100, 0)
    request = bridge.begin("b", "B", 500, "_empty_")
    assert not request.done()

    clock.now = 4.0
    assert bridge.tick() is can_displace
    assert request.done() is can_displace
    assert host.operations == ([("A", "async"), ("B", "interactive")] if can_displace else [])


@pytest.mark.parametrize(
    "dram,final_tokens,expected",
    [(0, 110, []), (0, 250, [("A", "inactive")]), (300, 250, [("A", "async")])],
)
def test_reasoning_overflow_defers_placement_until_response(clock, dram, final_tokens, expected):
    host = RecordingHost(300, dram)
    bridge = ProgramSchedulingBridge(host, MoriConfig())
    assert bridge.begin("a", "A", 500, "_empty_").result()
    bridge.progress("a", 150)
    assert not bridge.tick()
    assert not host.operations
    ref = bridge.runtime.registry.current_ref("A")
    assert bridge.runtime.registry.require(ref).marked_for_pause

    clock.now = 2.0
    bridge.finish("a", final_tokens, 100, 0)
    assert host.operations == expected
    assert not bridge.runtime.registry.require(ref).marked_for_pause


def test_cpu_overflow_moves_warm_program_to_waiting(clock):
    host = RecordingHost(300, 160)
    bridge = ProgramSchedulingBridge(host, MoriConfig())
    assert bridge.begin("a", "A", 500, "_empty_").result()
    bridge.finish("a", 160, 100, 0)
    pending = bridge.begin("b", "B", 500, "_empty_")
    clock.now = 3
    assert bridge.tick()
    assert pending.result()

    host.observation = BackendObservation(True, 300, 159)
    bridge.observe_backend()
    assert bridge.tick()
    assert host.operations == [("A", "async"), ("B", "interactive"), ("A", "inactive")]


def test_release_resolves_gate_and_ignores_old_attempt_completion(clock):
    host = RecordingHost(300, 10)
    bridge = ProgramSchedulingBridge(host, MoriConfig())
    waiting = bridge.begin("first", "A", 2000, "_empty_")
    assert not waiting.done()
    assert bridge.release("A")
    assert not waiting.result()
    assert not host.operations

    assert bridge.begin("second", "A", 100, "_empty_", 2).result()
    assert bridge.attach("second")
    assert bridge.attach("second")
    assert not bridge.attach("second")
    bridge.finish("first", 800, 100, 0)
    ref = bridge.runtime.registry.current_ref("A")
    assert bridge.runtime.strategy_factors.for_program(ref).tokens == 20


def test_cancel_and_missing_observation_do_not_force_an_admission():
    host = RecordingHost(300, 20)
    host.observation = BackendObservation(False, None, None)
    bridge = ProgramSchedulingBridge(host, MoriConfig())
    pending = bridge.begin("a", "A", 100, "_empty_")
    assert not pending.done()
    assert not bridge.tick()

    bridge.cancel_gates()
    assert not pending.result()
    assert not bridge.runtime.registry.views()


def test_host_failure_fails_pending_admission_and_prevents_future_decisions(clock):
    class FailedHost(RecordingHost):
        def apply_program_ops(self, operations):
            raise OSError("placement acknowledgement failed")

    host = FailedHost(300, 160)
    bridge = ProgramSchedulingBridge(host, MoriConfig())
    assert bridge.begin("a", "A", 500, "_empty_").result()
    bridge.finish("a", 160, 100, 0)
    pending = bridge.begin("b", "B", 500, "_empty_")
    clock.now = 3.0
    with pytest.raises(OSError, match="acknowledgement"):
        bridge.tick()
    with pytest.raises(OSError, match="acknowledgement"):
        pending.result()
    with pytest.raises(RuntimeError, match="must be restarted"):
        bridge.begin("c", "C", 100, "_empty_")


def test_config_selects_mori_on_the_existing_bridge():
    from argparse import Namespace

    from agentinfer.agentcache.entrypoints.cli.serve_profile import ASYNC_SCHEDULER_CLS, MORI_MIDDLEWARE, inject_profile

    args = Namespace(agentinfer_strategy="mori", additional_config=None, async_scheduling=True, middleware=[])
    inject_profile(args, set())
    assert args.scheduler_cls == ASYNC_SCHEDULER_CLS
    assert args.middleware == [MORI_MIDDLEWARE]
    assert args.additional_config["agentcache"]["strategy"] == "mori"


def test_shared_vllm_hook_consumes_the_pre_admission_permit():
    from vllm.v1.engine.core import EngineCore

    from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks
    from agentinfer.agentcache.core.scheduling_ipc import UTILITY_METHOD

    host = RecordingHost(600, 200)
    owner = SimpleNamespace(program_scheduling_host=host)
    config = SimpleNamespace(additional_config={"agentcache": {"strategy": "mori"}})
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=999), 16)
    assert hooks.program_scheduling.observation.hbm_capacity_tokens == 600
    core = SimpleNamespace(scheduler=SimpleNamespace(_agentcache=hooks))
    handle = getattr(EngineCore, UTILITY_METHOD)
    assert handle(core, "begin", ["attempt", "A", 100, "_empty_"]).result()
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"agentinfer_mori_attempt": "attempt"}))
    native = []
    hooks.add_request(owner, request, native.append)
    with pytest.raises(ValueError, match="no outstanding MORI admission"):
        hooks.add_request(owner, request, native.append)
    assert native == [request]


def test_api_preserves_begin_progress_end_and_release_order():
    import asyncio
    import json

    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    from agentinfer.agentcache.core.mori_api import MoriSchedulingMiddleware
    from agentinfer.agentcache.core.program_scheduling import ATTEMPT_ARG
    from agentinfer.agentcache.core.scheduling_ipc import UTILITY_METHOD, handle_event

    host = RecordingHost(300, 10)
    bridge = ProgramSchedulingBridge(host, MoriConfig())
    events = []

    class FakeCore:
        async def call_utility_async(self, method, event, args):
            assert method == UTILITY_METHOD
            events.append(event)
            result = handle_event(bridge, event, args)
            return await asyncio.wrap_future(result) if event == "begin" else result

    async def native_chat(request: Request):
        payload = await request.json()
        attempt = payload["vllm_xargs"][ATTEMPT_ARG]
        assert bridge.attach(attempt)
        assert payload["stream_options"]["include_usage"]

        async def frames():
            yield b'data: {"choices": []}\n\n' * 19
            usage = {"usage": {"total_tokens": 27, "prompt_tokens": 7, "prompt_tokens_details": {"cached_tokens": 5}}}
            yield ("data: " + json.dumps(usage)).encode()
            yield b"\n\ndata: [DONE]\n\n"

        return StreamingResponse(frames(), media_type="text/event-stream")

    app = Starlette(routes=[Route("/v1/chat/completions", native_chat, methods=["POST"])])
    app.state.engine_client = SimpleNamespace(engine_core=FakeCore())
    middleware = MoriSchedulingMiddleware(app)

    # Starlette normally sets this scope field before invoking the middleware.
    async def with_app(scope, receive, send):
        scope["app"] = app
        await middleware(scope, receive, send)

    async def start():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=with_app), base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "program_id": "A",
                    "messages": [{"role": "system", "content": "hi"}],
                    "stream": True,
                },
            )
            assert response.status_code == 200
            response = await client.post("/programs/release", json={"program_id": "A"})
            assert response.json()["released"]

    asyncio.run(start())
    assert events == ["config", "observe", "begin", "progress", "finish", "release"]
    assert host.operations == [("A", None)]
