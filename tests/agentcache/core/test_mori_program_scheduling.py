# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Test the Program scheduling boundary with an independent host."""

from types import SimpleNamespace

import pytest

from agentinfer.agentcache.core.program_scheduling import BackendObservation, MoriProgramAdmissionController
from agentinfer.agentcache.core.scheduling_ipc import chat_facts
from agentinfer.scheduling import runtime
from agentinfer.scheduling.mori import MoriConfig
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


class RecordingHost:
    def __init__(self, hbm: int, dram: int) -> None:
        self.observation = BackendObservation(True, hbm, dram)
        self.operations: list[tuple[str, str | None]] = []

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
    admitted = MoriProgramAdmissionController(RecordingHost(123, 0), MoriConfig())
    waiting = MoriProgramAdmissionController(RecordingHost(123, 0), MoriConfig())

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
    bridge = MoriProgramAdmissionController(host, MoriConfig())
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
    bridge = MoriProgramAdmissionController(host, MoriConfig())
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
    bridge = MoriProgramAdmissionController(host, MoriConfig())
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
    bridge = MoriProgramAdmissionController(host, MoriConfig())
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
    bridge = MoriProgramAdmissionController(host, MoriConfig())
    pending = bridge.begin("a", "A", 100, "_empty_")
    assert not pending.done()
    assert not bridge.tick()

    bridge.cancel_gates()
    assert not pending.result()
    assert not bridge.runtime.registry.views()


class FlakyCache(RecordingHost):
    fail_next = False

    def apply_program_ops(self, operations) -> None:
        if self.fail_next:
            self.fail_next = False
            raise OSError("placement acknowledgement failed")
        super().apply_program_ops(operations)


def test_failed_placement_keeps_pending_admission_and_retries(clock):
    cache = FlakyCache(300, 160)
    bridge = MoriProgramAdmissionController(cache, MoriConfig())
    assert bridge.begin("a", "A", 500, "_empty_").result()
    bridge.finish("a", 160, 100, 0)
    pending = bridge.begin("b", "B", 500, "_empty_")
    clock.now = 3.0

    cache.fail_next = True
    assert not bridge.tick()
    assert not pending.done()
    assert bridge.tick()
    assert pending.result()
    assert cache.operations == [("A", "async"), ("B", "interactive")]


def test_failed_batch_retries_in_order_after_a_partial_apply(clock):
    class PartialCache(RecordingHost):
        def __init__(self):
            super().__init__(300, 160)
            self.placement = None
            self.batches = []

        def apply_program_ops(self, operations):
            self.batches.append(operations)
            if self.placement is None:
                self.placement = {operations[0].program: operations[0].placement}
                raise OSError("failed after applying the first operation")
            for op in operations:
                self.placement[op.program] = op.placement

    cache = PartialCache()
    bridge = MoriProgramAdmissionController(cache, MoriConfig())
    assert bridge.begin("a", "A", 500, "_empty_").result()
    bridge.finish("a", 160, 100, 0)
    pending = bridge.begin("b", "B", 500, "_empty_")
    clock.now = 3.0

    assert not bridge.tick()
    assert len(cache.placement) == 1
    assert not pending.done()
    assert bridge.tick()
    assert pending.result()
    assert cache.batches[0] == cache.batches[1]
    assert {ref.program_id: placement.value for ref, placement in cache.placement.items()} == {
        "A": "async",
        "B": "interactive",
    }


def test_failed_release_is_retryable_and_does_not_stop_other_programs():
    cache = FlakyCache(600, 160)
    bridge = MoriProgramAdmissionController(cache, MoriConfig())
    assert bridge.begin("a", "A", 100, "_empty_").result()

    cache.fail_next = True
    assert not bridge.release("A")
    ref = bridge.runtime.registry.current_ref("A")
    assert ref is not None
    bridge.progress("a", 7)
    assert bridge.runtime.strategy_factors.for_program(ref).tokens == 27
    assert bridge.begin("b", "B", 100, "_empty_").result()
    assert bridge.release("A")
    assert bridge.attach("a")  # A request already granted admission can reach Core.
    bridge.finish("a", 50, 30, 0)
    assert bridge.runtime.registry.current_ref("A") is None
    assert bridge.attach("b")
    assert cache.operations == [("A", None)]
    assert not bridge.release("A")
    assert cache.operations == [("A", None)]


def test_inactive_generation_keeps_cache_ownership_until_release_ack(clock):
    cache = FlakyCache(300, 0)
    controller = MoriProgramAdmissionController(cache, MoriConfig())
    assert controller.begin("a", "A", 500, "_empty_").result()
    controller.progress("a", 150)
    controller.tick()
    clock.now = 2
    controller.finish("a", 250, 100, 0)
    ref = controller.runtime.registry.current_ref("A")
    assert cache.operations == [("A", "inactive")]

    cache.fail_next = True
    assert not controller.release("A")
    assert controller.runtime.registry.current_ref("A") == ref
    assert controller.release("A")
    assert cache.operations == [("A", "inactive"), ("A", None)]
    assert controller.begin("new", "A", 100, "_empty_").result()
    assert controller.runtime.registry.current_ref("A") != ref


def test_shared_vllm_hook_consumes_the_pre_admission_permit():
    from vllm.v1.engine.core import EngineCore

    from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks
    from agentinfer.agentcache.core.scheduling_ipc import UTILITY_METHOD

    host = RecordingHost(600, 200)
    owner = SimpleNamespace(program_cache_adapter=host)
    config = SimpleNamespace(additional_config={"agentcache": {"strategy": "mori"}})
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=999), 16)
    assert hooks.program_admission.observation.hbm_capacity_tokens == 600
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
    bridge = MoriProgramAdmissionController(host, MoriConfig())
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
        assert payload["messages"] == [{"role": "system", "content": "你好"}]

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
                    "messages": [{"role": "system", "content": "你好"}],
                    "stream": True,
                },
            )
            assert response.status_code == 200
            response = await client.post("/programs/release", json={"program_id": "A"})
            assert response.json()["released"]

    asyncio.run(start())
    assert events == ["begin", "progress", "finish", "release"]
    assert host.operations == [("A", None)]


def test_core_timer_wakes_idle_input_queue_and_executes_on_owner(monkeypatch):
    import pickle
    from queue import Queue
    from threading import get_ident

    from vllm.v1.engine.core import EngineCoreProc

    from agentinfer.agentcache.core.mori_timer import install_mori_timer

    called = []
    owner = get_ident()
    bridge = SimpleNamespace(
        config=MoriConfig(interval_seconds=0.01),
        observe_backend=lambda: called.append(("observe", get_ident())),
        tick=lambda: called.append(("tick", get_ident())),
    )

    def run_busy_loop(self):
        request = self.input_queue.get(timeout=2)
        assert called == []
        self._handle_client_request(*pickle.loads(pickle.dumps(request)))

    monkeypatch.setattr(EngineCoreProc, "run_busy_loop", run_busy_loop)
    monkeypatch.setattr(EngineCoreProc, "_handle_client_request", lambda *_: pytest.fail("unexpected vLLM event"))
    install_mori_timer()
    core = object.__new__(EngineCoreProc)
    core.scheduler = SimpleNamespace(_agentcache=SimpleNamespace(program_admission=bridge))
    core.input_queue = Queue()
    core.is_running = lambda: True

    core.run_busy_loop()
    assert called == [("observe", owner), ("tick", owner)]
    assert core._agentinfer_mori_timer is None


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_controller_rejects_invalid_interval_before_installing_timer(interval):
    from agentinfer.agentcache.core.controller import build_mori_controller

    with pytest.raises(ValueError, match="finite|positive"):
        build_mori_controller(
            SimpleNamespace(program_cache_adapter=RecordingHost(300, 10)),
            {"mori": {"interval_seconds": interval}},
            "local",
        )


def test_runtime_arrival_without_chat_facts_uses_known_prompt_tokens():
    from agentinfer.scheduling.identity import AgentIdentity

    controller = MoriProgramAdmissionController(RecordingHost(300, 160), MoriConfig())
    scheduler = controller.runtime
    assert scheduler.on_request_arrival("short", AgentIdentity("A"), 100, controller._backend, object())
    ref = scheduler.registry.current_ref("A")
    factors = scheduler.strategy_factors.for_program(ref)
    assert (factors.tokens, factors.steps) == (100, 1)
    scheduler.on_reported_completion("short", 120, 100, 50)
    assert not scheduler.strategy_factors.global_factors.prefix_lengths
    assert not scheduler.on_request_arrival("long", AgentIdentity("B"), 400, controller._backend, object())
    controller.tick()
    assert controller.release("B")


@pytest.mark.parametrize("finish_first", ["first", "second"])
def test_overlapping_finishes_keep_program_reasoning_until_last_request(clock, finish_first):
    from agentinfer.scheduling.domain import ProgramStatus

    controller = MoriProgramAdmissionController(RecordingHost(300, 300), MoriConfig())
    for request in ("first", "second"):
        assert controller.begin(request, "A", 500, "_empty_").result()
        assert controller.attach(request)
    controller.progress("first", 150)
    controller.tick()
    program = controller.runtime.registry.require(controller.runtime.registry.current_ref("A"))
    assert program.marked_for_pause
    clock.now = 2.0
    controller.finish(finish_first, 250, 100, 0)
    assert program.status is ProgramStatus.REASONING and program.marked_for_pause
    assert not controller.cache.operations
    clock.now = 3.0
    controller.finish("second" if finish_first == "first" else "first", 250, 100, 0)
    assert program.status is ProgramStatus.ACTING and not program.marked_for_pause
    assert controller.cache.operations == [("A", "async")]


@pytest.mark.parametrize("configured", [True, False], ids=["cli", "env"])
def test_authenticated_program_routes_share_id_resolution_before_core(configured, monkeypatch):
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from agentinfer.agentcache.core.mori_api import MoriSchedulingMiddleware

    events = []

    class Core:
        async def call_utility_async(self, method, event, args):
            events.append((event, args))
            return True

    async def chat(request: Request):
        return JSONResponse({})

    app = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])
    app.add_middleware(MoriSchedulingMiddleware)
    app.state.engine_client = SimpleNamespace(engine_core=Core())
    app.state.args = SimpleNamespace(api_key=["cli-key"] if configured else None)
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    headers = {"Authorization": "Bearer " + ("cli-key" if configured else "env-key")}

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, root_path="/proxy"), base_url="http://test/proxy/"
        ) as client:
            for path in ("v1/chat/completions", "programs/release"):
                assert (await client.post(path, json={})).status_code == 401
            assert not events
            for top, extra, expected in [
                (None, "nested", "nested"),
                ("", 0, "0"),
                ("top", "nested", "top"),
                ("  ", None, "default"),
            ]:
                payload = {"program_id": top, "extra_body": {"program_id": extra}}
                assert (await client.post("v1/chat/completions", json=payload, headers=headers)).status_code == 200
                assert [args[1] for kind, args in events if kind == "begin"][-1] == expected
                result = await client.post("programs/release", json=payload, headers=headers)
                if expected == "default":
                    assert result.status_code == 400
                else:
                    assert result.json() == {"program_id": expected, "released": True}
            assert (
                await client.post("programs/release", json={"program_id": "default"}, headers=headers)
            ).status_code == 200
            assert [args[0] for kind, args in events if kind == "release"] == ["nested", "0", "top", "default"]

    asyncio.run(run())
