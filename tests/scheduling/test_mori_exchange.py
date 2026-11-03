# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Cross-tier exchanges preserve feasible destinations and rejected CPU owners."""

import json
import logging
from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling import ProgramState, ProgramStatus, TransitionController, TransitionKind, TransitionResult
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, MORITier, build_mori_strategy
from tests.scheduling.test_mori_rebalance import pool, scheduler
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


def _controller(components, calls, reject=None):
    accepted = accepting_controller(components.strategy, components.initial_factors, calls)

    def apply(request):
        if reject is not None and reject(request):
            calls.append(request)
            return TransitionResult(request.kind, request.program, False, "test_rejected")
        return accepted.apply(request)

    return TransitionController(apply)


def _program(name, tokens, *, acting=False, gpu=False):
    return view(
        name,
        state=ProgramState.ACTIVE if gpu else ProgramState.PAUSED,
        status=ProgramStatus.ACTING if acting else ProgramStatus.REASONING,
        tokens=tokens,
        step_count=1,
    )


def test_exchange_keeps_victim_in_cpu_space_released_by_successful_promotion():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    victim = _program("victim", 700, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 400)
    set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(victim, running, waiting, hbm=800, dram=800)
    calls = []

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, waiting.ref),
    ]
    assert components.initial_factors.for_program(victim.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.GPU
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 500
    assert cpu.used == 700


def test_exchange_rejection_keeps_waiter_cpu_reservation():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    victim = _program("victim", 700, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 400)
    set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(victim, running, waiting, hbm=800, dram=800)
    calls = []
    controller = _controller(components, calls, reject=lambda request: request.kind is TransitionKind.RESUME)

    components.strategy.schedule_resume(fixed, components.initial_factors, controller)

    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(victim.ref) is None
    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, waiting.ref),
        (TransitionKind.RETIRE, victim.ref),
    ]
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 100
    assert cpu.used == 400


@pytest.mark.parametrize("dram", [650, 800])
def test_exchange_multiple_victims_share_the_released_cpu_capacity_once(dram):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 650)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(first, second, running, waiting, hbm=800, dram=dram)
    calls = []

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    expected = [
        (TransitionKind.PAUSE, first.ref),
        (TransitionKind.PAUSE, second.ref),
        (TransitionKind.RESUME, waiting.ref),
    ]
    if dram == 650:
        expected.append((TransitionKind.RETIRE, second.ref))
    assert [(call.kind, call.program) for call in calls] == expected
    assert components.initial_factors.for_program(first.ref).tier is MORITier.CPU
    if dram == 800:
        assert components.initial_factors.for_program(second.ref).tier is MORITier.CPU
    else:
        assert components.initial_factors.for_program(second.ref) is None
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 750
    assert cpu.used == (700 if dram == 800 else 350)


def test_rejected_second_pause_never_spends_unreleased_waiter_cpu_capacity():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 650)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(first, second, running, waiting, hbm=800, dram=1000)
    calls = []
    controller = _controller(
        components, calls, reject=lambda request: request.kind is TransitionKind.PAUSE and request.program == second.ref
    )

    components.strategy.schedule_resume(fixed, components.initial_factors, controller)

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, first.ref),
        (TransitionKind.PAUSE, second.ref),
    ]
    for program, expected in ((first, MORITier.CPU), (second, MORITier.GPU), (waiting, MORITier.CPU)):
        factors = components.initial_factors.for_program(program.ref)
        assert factors.tier is expected
        assert factors.pending_tier is None
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 450
    assert cpu.used == 1000


def test_exchange_cpu_relief_counts_only_unique_blocks_and_preserves_other_owner():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    victim = _program("victim", 600, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 400)
    resident = _program("resident", 400, acting=True)
    blocks = {
        resident.ref: frozenset(range(400)),
        waiting.ref: frozenset((*range(100, 400), *range(900, 1000))),
        victim.ref: frozenset(range(200, 800)),
    }

    class SharedCPU:
        def footprint(self, ref, tier):
            return blocks.get(ref, frozenset()) if tier == "cpu" else frozenset()

    components.strategy.bind_footprints(SharedCPU())
    set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    set_tier(components, resident, MORITier.CPU)
    fixed = snapshot(victim, running, waiting, resident, hbm=700, dram=800)
    calls = []
    assert components.strategy._ledgers(fixed, components.initial_factors)[1].used == 500

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, waiting.ref),
    ]
    assert components.initial_factors.for_program(resident.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(victim.ref).tier is MORITier.CPU
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 500
    assert cpu.used == 800


def test_exchange_does_not_spend_cpu_space_of_a_later_unadmitted_candidate():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("a-waiting", 500)
    later = _program("b-waiting", 300)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.4)
    set_tier(components, later, MORITier.CPU, idleness=0.5)
    fixed = snapshot(first, second, running, waiting, later, hbm=800, dram=800)
    calls = []

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, first.ref),
        (TransitionKind.PAUSE, second.ref),
        (TransitionKind.RESUME, waiting.ref),
        (TransitionKind.RETIRE, second.ref),
    ]
    assert components.initial_factors.for_program(first.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(later.ref).tier is MORITier.CPU
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 600
    assert cpu.used == 650


@pytest.mark.parametrize("failure", ["first_pause", "second_pause", "resume"])
def test_exchange_exception_finishes_accepted_pauses_without_spending_candidate_capacity(failure, caplog):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    components.strategy.bind_observability(True)
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 650)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(first, second, running, waiting, hbm=800, dram=1000)
    calls = []
    accepted = _controller(components, calls)
    caplog.set_level(logging.INFO, logger="agentinfer.scheduling.mori.strategy")

    def apply(request):
        if (
            (failure == "first_pause" and request.kind is TransitionKind.PAUSE and request.program == first.ref)
            or (failure == "second_pause" and request.kind is TransitionKind.PAUSE and request.program == second.ref)
            or (failure == "resume" and request.kind is TransitionKind.RESUME)
        ):
            raise RuntimeError("transition failed before mutation")
        return accepted.apply(request)

    with pytest.raises(RuntimeError, match="transition failed before mutation"):
        components.strategy.schedule_resume(fixed, components.initial_factors, TransitionController(apply))

    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.CPU
    expected = {
        "first_pause": (MORITier.GPU, MORITier.GPU, 800, 650),
        "second_pause": (MORITier.CPU, MORITier.GPU, 450, 1000),
        "resume": (MORITier.CPU, MORITier.WAITING, 100, 1000),
    }[failure]
    for program, tier in zip((first, second), expected[:2], strict=True):
        factor = components.initial_factors.for_program(program.ref)
        assert factor.tier is tier
        assert factor.pending_tier is None
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert (gpu.used, cpu.used) == expected[2:]
    assert components.strategy.counters["replacement_transition_exceptions"] == 1
    records = [record.message for record in caplog.records if record.message.startswith("MORI replacement plan ")]
    assert len(records) == 1
    payload = json.loads(records[0].removeprefix("MORI replacement plan "))
    assert payload["transition_rejected"] is True
    assert payload["transition_applied"] is False
    assert payload["transition_reason"] == "transition_exception"
    assert payload["transition_exception"] == "RuntimeError: transition failed before mutation"
    expected_placements = {
        "first_pause": [],
        "second_pause": [{"ref": "a-victim:0", "tier": "cpu"}],
        "resume": [{"ref": "a-victim:0", "tier": "cpu"}, {"ref": "b-victim:0", "tier": "waiting"}],
    }[failure]
    assert payload["final_placements"] == expected_placements


@pytest.mark.parametrize("rejected", [False, True])
def test_exchange_runtime_preserves_ownership_through_admission_boundary(rejected):
    runtime = scheduler()
    capacity = pool(hbm=800, dram=800)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("first", AgentIdentity("waiting"), 400, capacity, "first")
        waiting = runtime.program_ref_for_request("first")
        runtime.on_native_execution("first", running=True)
        clock.return_value = 1.0
        runtime.on_request_completion("first", 400)
        runtime.schedule_cycle(pool(hbm=0, dram=800))
        assert runtime.strategy_factors.for_program(waiting).tier is MORITier.CPU

        clock.return_value = 2.0
        assert runtime.on_request_arrival("victim", AgentIdentity("victim"), 700, capacity, "victim")
        victim = runtime.program_ref_for_request("victim")
        runtime.on_native_execution("victim", running=True)
        assert runtime.on_request_arrival("running", AgentIdentity("running"), 100, capacity, "running")
        runtime.on_native_execution("running", running=True)
        clock.return_value = 3.0
        runtime.on_request_completion("victim", 700)
        assert not runtime.on_request_arrival("second", AgentIdentity("waiting"), 400, capacity, "second")
        for ref, idle in ((victim, 0.6), (waiting, 0.45)):
            runtime.strategy_factors.set_program_factors(
                ref,
                replace(
                    runtime.strategy_factors.for_program(ref),
                    cycles=((1.0 - idle, idle),),
                    pending_reasoning_seconds=0.0,
                    acting_started_at_monotonic_s=None,
                ),
            )
        apply = runtime._apply_transition

        def observed(request):
            if request.kind is TransitionKind.RESUME:
                # Neither owner is dropped while the CPU->GPU outcome is unknown.
                assert runtime.strategy_factors.for_program(waiting).tier is MORITier.CPU
                assert runtime.strategy_factors.for_program(victim).tier is MORITier.GPU
                assert runtime.registry.get(victim) is not None
                if rejected:
                    return TransitionResult(request.kind, request.program, False, "test_rejected")
            return apply(request)

        with patch.object(runtime, "_apply_transition", side_effect=observed):
            runtime.schedule_cycle(capacity)

        if rejected:
            assert runtime.registry.get(victim) is None
            assert runtime.strategy_factors.for_program(waiting).tier is MORITier.CPU
            assert runtime.request_pool.waiting_request_ids == ("second",)
            assert runtime.consume_admitted_requests() == ()
        else:
            assert runtime.registry.get(victim) is not None
            assert runtime.strategy_factors.for_program(victim).tier is MORITier.CPU
            assert runtime.strategy_factors.for_program(waiting).tier is MORITier.GPU
            assert runtime.registry.require(victim).state is ProgramState.PAUSED
            assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["second"]
        gpu, cpu = runtime.strategy._ledgers(runtime._snapshot(), runtime.strategy_factors)
        assert gpu.used <= 800
        assert cpu.used <= 800
