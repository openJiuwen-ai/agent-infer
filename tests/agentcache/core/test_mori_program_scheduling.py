# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Test the Program scheduling boundary with an independent host."""

from types import SimpleNamespace

import pytest

from agentinfer.agentcache.mori.admission import (
    BackendObservation,
    InvalidMoriAdmission,
    MoriProgramAdmissionController,
)
from agentinfer.agentcache.mori.ipc import (
    ATTEMPT_ARG,
    CHILD_COUNT_ARG,
    CONTEXT_CHARS_ARG,
    PREFIX_ARG,
    PREPARATION_NS_ARG,
    PROGRAM_ARG,
    chat_facts,
)
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


def native(rid, attempt, program_id, context_chars, n=1, preparation_ns=0):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    fields = {
        ATTEMPT_ARG: attempt,
        PROGRAM_ARG: program_id,
        CONTEXT_CHARS_ARG: context_chars,
        PREFIX_ARG: "_empty_",
        CHILD_COUNT_ARG: n,
    }
    if preparation_ns:
        fields[PREPARATION_NS_ARG] = preparation_ns
    return Request(rid, [1, 2, 3], SamplingParams(extra_args=fields), None)


def submit(controller, attempt, program_id, chars, n=1, preparation_ns=0):
    dispatched = []
    controller.add_request(native(attempt, attempt, program_id, chars, n, preparation_ns), dispatched.append)
    return dispatched


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=1.0)
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: state.now))
    return state


def test_admission_uses_pretemplate_chars_and_exact_effective_capacity():
    admitted = MoriProgramAdmissionController(RecordingHost(123, 0), MoriConfig())
    waiting = MoriProgramAdmissionController(RecordingHost(123, 0), MoriConfig())

    assert submit(admitted, "a", "program", 115)
    assert not submit(waiting, "b", "program", 158)


@pytest.mark.parametrize("preparation_ns, demotes", [(0, True), (5_000_000, False)])
def test_ingress_credit_preserves_swe_current_cycle_displacement(clock, preparation_ns, demotes):
    # At SWE tick 3197, ACTING lasted 1.25764 s after a 1.2616 s REASONING period.
    host = RecordingHost(300, 1000)
    controller = MoriProgramAdmissionController(host, MoriConfig())
    clock.now = 20.005
    assert submit(controller, "first", "A", 500, preparation_ns=preparation_ns)
    clock.now = 21.2616
    controller.finish("first", 150, 100, 0)
    clock.now = 22
    assert not submit(controller, "waiting", "B", 500)
    clock.now = 22.51924
    controller.tick()
    assert (("A", "async") in host.operations) is demotes


def test_ingress_credit_stops_acting_but_parked_reasoning_starts_at_wake(clock):
    host = RecordingHost(1200, 1000)
    controller = MoriProgramAdmissionController(host, MoriConfig())
    clock.now = 10
    assert submit(controller, "first", "A", 500, preparation_ns=2_000_000_000)
    factors = controller.runtime.strategy_factors.for_program(controller.runtime.registry.current_ref("A"))
    assert factors.reasoning_since == 8
    clock.now = 11
    controller.finish("first", 100, 100, 0)
    host.observation = BackendObservation(True, 0, 1000)
    controller.observe_backend()
    controller.tick()
    clock.now = 15
    parked = submit(controller, "next", "A", 500, preparation_ns=1_000_000_000)
    assert not parked and factors.acting_history[-1] == 3 and factors.reasoning_since is None
    assert controller.runtime.request_pool.get("next").arrived_at_monotonic_s == 15
    clock.now = 19
    host.observation = BackendObservation(True, 1200, 1000)
    controller.observe_backend()
    controller.tick()
    assert parked and factors.reasoning_since == 19
    clock.now = 20
    controller.finish("next", 150, 100, 0)
    assert list(factors.reasoning_history) == [3, 1]


def test_overlapping_and_stale_preparation_preserve_the_program_clock(clock):
    controller = MoriProgramAdmissionController(RecordingHost(1200, 1000), MoriConfig())
    clock.now = 10
    submit(controller, "old", "A", 100, preparation_ns=500_000_000)
    old = controller.runtime.strategy_factors.for_program(controller.runtime.registry.current_ref("A"))
    clock.now = 11
    submit(controller, "parallel", "A", 100, preparation_ns=3_000_000_000)
    assert old.reasoning_since == 9.5
    clock.now = 12
    controller.finish("parallel", 100, 100, 0)
    controller.finish("old", 100, 100, 0)
    # This Chat was already being tokenized before the previous response finished.
    clock.now = 13
    submit(controller, "stale", "A", 100, preparation_ns=2_000_000_000)
    assert old.acting_history[-1] == 1 and old.reasoning_since == 13
    clock.now = 14
    assert controller.release("A")
    clock.now = 15
    submit(controller, "reused", "A", 100, preparation_ns=2_000_000_000)
    new = controller.runtime.strategy_factors.for_program(controller.runtime.registry.current_ref("A"))
    assert new.reasoning_since == 15 and old.reasoning_since == 13
    assert controller.release("A")
    clock.now = 17
    submit(controller, "fresh", "A", 100, preparation_ns=500_000_000)
    fresh = controller.runtime.strategy_factors.for_program(controller.runtime.registry.current_ref("A"))
    assert fresh.reasoning_since == 17
    clock.now = 18
    controller.finish("fresh", 100, 100, 0)
    clock.now = 20
    submit(controller, "same-generation", "A", 100, preparation_ns=500_000_000)
    assert fresh.reasoning_since == 19.5 and fresh.acting_history[-1] == 1.5


def test_legacy_overlap_without_preparation_keeps_latest_arrival_clock(clock):
    controller = MoriProgramAdmissionController(RecordingHost(1200, 1000), MoriConfig())
    clock.now = 10
    submit(controller, "first", "A", 100)
    clock.now = 12
    submit(controller, "second", "A", 100)
    factors = controller.runtime.strategy_factors.for_program(controller.runtime.registry.current_ref("A"))
    assert factors.reasoning_since == 12


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
    assert submit(bridge, "a", "A", 500)
    clock.now = 2.0
    bridge.finish("a", 160, 100, 0)
    request = submit(bridge, "b", "B", 500)
    assert not request

    clock.now = 4.0
    assert bridge.tick() is can_displace
    assert bool(request) is can_displace
    assert host.operations == ([("A", "async"), ("B", "interactive")] if can_displace else [])


@pytest.mark.parametrize(
    "dram,final_tokens,expected",
    [(0, 110, []), (0, 250, [("A", "inactive")]), (300, 250, [("A", "async")])],
)
def test_reasoning_overflow_defers_placement_until_response(clock, dram, final_tokens, expected):
    host = RecordingHost(300, dram)
    bridge = MoriProgramAdmissionController(host, MoriConfig())
    assert submit(bridge, "a", "A", 500)
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
    assert submit(bridge, "a", "A", 500)
    bridge.finish("a", 160, 100, 0)
    pending = submit(bridge, "b", "B", 500)
    clock.now = 3
    assert bridge.tick()
    assert len(pending) == 1

    host.observation = BackendObservation(True, 300, 159)
    bridge.observe_backend()
    assert bridge.tick()
    assert host.operations == [("A", "async"), ("B", "interactive"), ("A", "inactive")]


def test_release_discards_retained_children_and_ignores_old_attempt_completion(clock):
    host = RecordingHost(300, 10)
    bridge = MoriProgramAdmissionController(host, MoriConfig())
    waiting = submit(bridge, "first", "A", 2000)
    assert not waiting
    assert bridge.release("A")
    assert not waiting
    assert [req.request_id for req in bridge.take_aborted_requests()] == ["first"]
    assert not host.operations

    dispatched = submit(bridge, "second", "A", 100, 2)
    bridge.add_request(native("sibling", "second", "A", 100, 2), dispatched.append)
    assert [req.request_id for req in dispatched] == ["second", "sibling"]
    with pytest.raises(InvalidMoriAdmission):
        bridge.add_request(native("extra", "second", "A", 100, 2), dispatched.append)
    bridge.finish("first", 800, 100, 0)
    ref = bridge.runtime.registry.current_ref("A")
    assert bridge.runtime.strategy_factors.for_program(ref).tokens == 20


def test_online_beam_search_adds_each_depth_to_one_chat():
    admission = MoriProgramAdmissionController(RecordingHost(2000, 0), MoriConfig())
    sent = []
    for rid in ("depth-0-beam-0", "depth-1-beam-0", "depth-1-beam-1"):
        request = native(rid, "attempt", "P", 100)
        del request.sampling_params.extra_args[CHILD_COUNT_ARG]
        admission.add_request(request, sent.append)
    ref = admission.runtime.registry.current_ref("P")
    assert len(sent) == 3 and all(req._mori_program_ref == ref for req in sent)
    assert admission.runtime.strategy_factors.for_program(ref).steps == 1


def test_partial_child_abort_preserves_queued_sibling_and_prevents_duplicate_dispatch():
    from vllm.v1.request import RequestStatus

    cache = RecordingHost(100, 0)
    admission = MoriProgramAdmissionController(cache, MoriConfig())
    sent = []
    first = native("0_req", "attempt", "P", 2000, 2)
    sibling = native("1_req", "attempt", "P", 2000, 2)
    for request in (first, sibling):
        admission.add_request(request, sent.append)
    assert not sent
    assert admission.finish_requests([first.request_id], RequestStatus.FINISHED_ABORTED) == [first]
    assert not admission.take_aborted_requests()
    cache.observation = BackendObservation(True, 2000, 0)
    admission.observe_backend()
    admission.tick()
    admission.tick()
    assert sent == [sibling]


def test_aborting_only_known_child_before_fanout_completes_rejects_late_sibling():
    from vllm.v1.request import RequestStatus

    admission = MoriProgramAdmissionController(RecordingHost(100, 0), MoriConfig())
    sent = []
    child = native("0_req", "partial", "P", 2000, 2)
    admission.add_request(child, sent.append)
    assert admission.finish_requests(child.request_id, RequestStatus.FINISHED_ABORTED) == [child]
    with pytest.raises(InvalidMoriAdmission):
        admission.add_request(native("1_req", "partial", "P", 2000, 2), sent.append)
    admission.cancel("partial")
    assert not sent
    assert not admission.take_aborted_requests()


@pytest.mark.parametrize("event,argument", [("release", "P"), ("cancel", "waiting")])
def test_control_event_notifies_each_client_of_its_retained_children(event, argument):
    from queue import Queue

    from vllm.v1.engine import FinishReason
    from vllm.v1.engine.core import EngineCoreProc

    from agentinfer.agentcache.mori.ipc import UTILITY_METHOD, install_core_utility

    admission = MoriProgramAdmissionController(RecordingHost(100, 0), MoriConfig())
    core = object.__new__(EngineCoreProc)
    core.scheduler = SimpleNamespace(_agentcache=SimpleNamespace(program_admission=admission))
    core.output_queue = Queue()
    install_core_utility()
    for i in range(2):
        child = native(f"{i}_req", "waiting", "P", 4000, 2)
        child.client_index = i
        admission.add_request(child, lambda _: pytest.fail("parked request dispatched"))
    for _ in range(2):
        getattr(core, UTILITY_METHOD)(event, [argument])
    clients = set()
    for _ in range(2):
        client_index, outputs = core.output_queue.get_nowait()
        clients.add(client_index)
        assert [(item.request_id, item.finish_reason) for item in outputs.outputs] == [
            (f"{client_index}_req", FinishReason.ABORT)
        ]
    assert clients == {0, 1}
    assert core.output_queue.empty()


def test_cancel_and_missing_observation_do_not_force_an_admission():
    host = RecordingHost(300, 20)
    host.observation = BackendObservation(False, None, None)
    bridge = MoriProgramAdmissionController(host, MoriConfig())
    pending = submit(bridge, "a", "A", 100)
    assert not pending
    assert not bridge.tick()

    bridge.cancel("a")
    assert not pending
    assert [request.request_id for request in bridge.take_aborted_requests()] == ["a"]
    assert not bridge.take_aborted_requests()
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
    assert submit(bridge, "a", "A", 500)
    bridge.finish("a", 160, 100, 0)
    pending = submit(bridge, "b", "B", 500)
    clock.now = 3.0

    cache.fail_next = True
    assert not bridge.tick()
    assert not pending
    assert bridge.tick()
    assert len(pending) == 1
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
    assert submit(bridge, "a", "A", 500)
    bridge.finish("a", 160, 100, 0)
    pending = submit(bridge, "b", "B", 500)
    clock.now = 3.0

    assert not bridge.tick()
    assert len(cache.placement) == 1
    assert not pending
    assert bridge.tick()
    assert len(pending) == 1
    assert cache.batches[0] == cache.batches[1]
    assert {ref.program_id: placement.value for ref, placement in cache.placement.items()} == {
        "A": "async",
        "B": "interactive",
    }


def test_failed_release_is_retryable_and_does_not_stop_other_programs():
    cache = FlakyCache(600, 160)
    bridge = MoriProgramAdmissionController(cache, MoriConfig())
    assert submit(bridge, "a", "A", 100)

    cache.fail_next = True
    assert not bridge.release("A")
    ref = bridge.runtime.registry.current_ref("A")
    assert ref is not None
    bridge.progress("a", 7)
    assert bridge.runtime.strategy_factors.for_program(ref).tokens == 27
    assert submit(bridge, "b", "B", 100)
    assert bridge.release("A")
    bridge.finish("a", 50, 30, 0)
    assert bridge.runtime.registry.current_ref("A") is None
    assert cache.operations == [("A", None)]
    assert not bridge.release("A")
    assert cache.operations == [("A", None)]


def test_inactive_generation_keeps_cache_ownership_until_release_ack(clock):
    cache = FlakyCache(300, 0)
    controller = MoriProgramAdmissionController(cache, MoriConfig())
    assert submit(controller, "a", "A", 500)
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
    assert submit(controller, "new", "A", 100)
    assert controller.runtime.registry.current_ref("A") != ref


def test_shared_vllm_hook_counts_one_arrival_and_dispatches_each_native_child_once():
    from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks

    host = RecordingHost(600, 200)
    owner = SimpleNamespace(program_cache_adapter=host)
    config = SimpleNamespace(additional_config={"agentcache": {"strategy": "mori"}})
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=999), 16)
    controller = hooks.program_admission
    assert controller.observation.hbm_capacity_tokens == 600
    children = [native(f"{i}_request", "attempt", "A", 100, 2) for i in range(2)]
    dispatched = []
    for child in children:
        hooks.add_request(owner, child, dispatched.append)
    controller.tick()
    assert dispatched == children
    ref = controller.runtime.registry.current_ref("A")
    assert all(child._mori_program_ref == ref for child in children)
    assert controller.runtime.strategy_factors.for_program(ref).steps == 1
    with pytest.raises(InvalidMoriAdmission):
        hooks.add_request(owner, children[0], dispatched.append)
    assert dispatched == children


def test_api_preserves_add_progress_end_and_release_order():
    import asyncio
    import json

    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
    from starlette.routing import Route
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request as NativeRequest
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from agentinfer.agentcache.mori.admission_api import MoriSchedulingMiddleware
    from agentinfer.agentcache.mori.ipc import UTILITY_METHOD, handle_event

    host = RecordingHost(300, 10)
    bridge = MoriProgramAdmissionController(host, MoriConfig())
    events = []

    class FakeCore:
        async def add_request_async(self, req):
            wire = MsgpackEncoder().encode(req)
            received = MsgpackDecoder(EngineCoreRequest).decode(wire)
            native = NativeRequest.from_engine_core_request(received, None)
            dispatched = []
            bridge.add_request(native, dispatched.append)
            assert dispatched == [native]
            events.append("add")

        async def call_utility_async(self, method, event, args):
            assert method == UTILITY_METHOD
            events.append(event)
            return handle_event(bridge, event, args)

    async def native_chat(request: Request):
        payload = await request.json()
        native = EngineCoreRequest("request", [1, 2, 3], None, SamplingParams(), None, 1.0, None, None, None)
        await request.app.state.engine_client.engine_core.add_request_async(native)
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
    assert events == ["add", "progress", "finish", "release"]
    assert host.operations == [("A", None)]


def test_idle_owner_timer_dispatches_newly_admitted_work_and_closes_retained_requests(monkeypatch):
    from queue import Queue
    from threading import get_ident

    from vllm.v1.engine.core import EngineCoreProc

    from agentinfer.agentcache.mori.timer import install_mori_timer

    host = RecordingHost(200, 400)
    admission = MoriProgramAdmissionController(host, MoriConfig(interval_seconds=0.01))
    native_add = []
    req = native("ready", "ready", "A", 2000)
    admission.add_request(req, lambda child: native_add.append((get_ident(), child)))
    parked = native("parked", "parked", "B", 100000)
    parked.client_index = 3
    admission.add_request(parked, lambda _: pytest.fail("insufficient capacity"))
    assert not native_add
    host.observation = BackendObservation(True, 2000, 400)

    def run_busy_loop(self):
        self._handle_client_request(*self.input_queue.get(timeout=2))
        assert native_add == [(get_ident(), req)]

    monkeypatch.setattr(EngineCoreProc, "run_busy_loop", run_busy_loop)
    monkeypatch.setattr(EngineCoreProc, "_handle_client_request", lambda *_: pytest.fail("unexpected vLLM event"))
    install_mori_timer()
    core = object.__new__(EngineCoreProc)
    core.scheduler = SimpleNamespace(_agentcache=SimpleNamespace(program_admission=admission))
    core.input_queue = Queue()
    core.output_queue = Queue()
    core.is_running = lambda: True
    core.run_busy_loop()
    assert core._agentinfer_mori_timer is None
    index, outputs = core.output_queue.get_nowait()
    assert index == 3
    assert [output.request_id for output in outputs.outputs] == ["parked"]
    assert core.output_queue.empty()


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
        assert submit(controller, request, "A", 500)
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

    from agentinfer.agentcache.mori.admission_api import MoriSchedulingMiddleware

    events = []
    submitted = []

    class Core:
        async def add_request_async(self, child):
            submitted.append(child.sampling_params.extra_args[PROGRAM_ARG])

        async def call_utility_async(self, method, event, args):
            events.append((event, args))
            return True

    async def chat(request: Request):
        child = SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None))
        await request.app.state.engine_client.engine_core.add_request_async(child)
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
                assert submitted[-1] == expected
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
