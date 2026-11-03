# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Unit tests for MORI placement, idleness, and capacity invariants."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agentinfer.scheduling import (
    AdmissionDisposition,
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
    SchedulingEvent,
    SchedulingEventKind,
    SchedulingSnapshot,
    TokenObservationSource,
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)
from agentinfer.scheduling.mori import (
    MORIConfig,
    MORIMode,
    MORIProgramFactors,
    MORITier,
    build_mori_strategy,
)

pytestmark = pytest.mark.cpu_test

EVENT_KIND = {
    TransitionKind.ADMIT: SchedulingEventKind.REQUEST_ADMITTED,
    TransitionKind.QUEUE: SchedulingEventKind.REQUEST_PENDING,
    TransitionKind.PAUSE: SchedulingEventKind.PROGRAM_PAUSED,
    TransitionKind.MARK_FOR_PAUSE: SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE,
    TransitionKind.CANCEL_PAUSE: SchedulingEventKind.PROGRAM_PAUSE_CANCELLED,
    TransitionKind.RESUME: SchedulingEventKind.PROGRAM_RESUMED,
    TransitionKind.RELEASE: SchedulingEventKind.PROGRAM_RELEASED,
    TransitionKind.RETIRE: SchedulingEventKind.PROGRAM_RELEASED,
}


def view(
    name: str,
    *,
    state: ProgramState,
    status: ProgramStatus,
    tokens: int,
    step_count: int = 0,
    marked_for_pause: bool = False,
) -> ProgramView:
    return ProgramView(
        ref=ProgramRef(name, 0),
        state=state,
        status=status,
        tokens=ProgramTokenObservation(estimated_context_tokens=tokens),
        backend_id="backend" if state is ProgramState.ACTIVE else None,
        step_count=step_count,
        marked_for_pause=marked_for_pause,
    )


def snapshot(*programs: ProgramView, hbm: int = 1000, dram: int = 1000) -> SchedulingSnapshot:
    return SchedulingSnapshot(
        observed_at_monotonic_s=100.0,
        backend_id="backend",
        total_kv_tokens=hbm,
        total_dram_kv_tokens=dram,
        hbm_block_size_tokens=1,
        dram_chunk_size_tokens=1,
        programs=programs,
        waiting_programs=tuple(program.ref for program in programs if program.state is ProgramState.PAUSED),
    )


def accepting_controller(strategy, factors, calls: list[TransitionRequest]) -> TransitionController:
    def apply(request: TransitionRequest) -> TransitionResult:
        calls.append(request)
        event = SchedulingEvent(
            event_id=f"event-{len(calls)}",
            sequence=len(calls),
            kind=EVENT_KIND[request.kind],
            occurred_at_monotonic_s=100.0,
            reason=request.reason,
            program=request.program,
            current_state=(
                ProgramState.PAUSED
                if request.kind is TransitionKind.PAUSE
                else ProgramState.ACTIVE
                if request.kind in (TransitionKind.ADMIT, TransitionKind.RESUME)
                else None
            ),
            current_status=ProgramStatus.ACTING if request.kind is TransitionKind.PAUSE else ProgramStatus.REASONING,
        )
        strategy.handle_scheduling_event(factors, event)
        return TransitionResult(request.kind, request.program, True, request.reason, event)

    return TransitionController(apply)


def set_tier(components, program: ProgramView, tier: MORITier, *, idleness: float = 0.0) -> None:
    sidecar = MORIProgramFactors(tier=tier, cycles=((1.0 - idleness, idleness),))
    components.strategy._set_factor(components.initial_factors, program.ref, sidecar)


def with_token_source(program: ProgramView, source: TokenObservationSource) -> ProgramView:
    return replace(program, tokens=replace(program.tokens, source=source))


def test_no_fit_admission_preserves_acting_reservation_until_planning() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400)
    set_tier(components, resident, MORITier.GPU)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(resident, candidate, hbm=1000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert [call.kind for call in calls] == [TransitionKind.QUEUE]
    assert components.initial_factors.for_program(resident.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(candidate.ref).tier is MORITier.WAITING


def test_no_fit_admission_keeps_reasoning_victim_lazy_until_completion() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400)
    set_tier(components, resident, MORITier.GPU)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(resident, candidate, hbm=1000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert [call.kind for call in calls] == [TransitionKind.QUEUE]
    resident_factors = components.initial_factors.for_program(resident.ref)
    assert resident_factors.tier is MORITier.GPU
    assert resident_factors.pending_tier is None


def test_oversized_admission_does_not_displace_a_legal_gpu_program() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=1100)
    set_tier(components, resident, MORITier.GPU)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(resident, candidate, hbm=1000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert [call.kind for call in calls] == [TransitionKind.QUEUE]
    assert components.initial_factors.for_program(resident.ref).tier is MORITier.GPU


def test_capacity_repair_demotes_acting_before_reasoning_and_more_idle_first() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    idle_acting = view("idle-acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=350)
    busy_acting = view("busy-acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=350)
    reasoning = view("reasoning", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=350)
    set_tier(components, idle_acting, MORITier.GPU, idleness=0.9)
    set_tier(components, busy_acting, MORITier.GPU, idleness=0.1)
    set_tier(components, reasoning, MORITier.GPU, idleness=1.0)
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(idle_acting, busy_acting, reasoning, hbm=800),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert calls[0].program == idle_acting.ref
    assert calls[0].kind is TransitionKind.PAUSE
    assert calls[1].program == busy_acting.ref
    assert calls[1].kind is TransitionKind.PAUSE
    assert all(call.program != reasoning.ref for call in calls)


def test_lazy_demotion_remains_gpu_capacity_debt_until_completion() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    victim = view("victim", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=200, step_count=1)
    set_tier(components, victim, MORITier.GPU, idleness=0.8)
    set_tier(components, waiting, MORITier.CPU, idleness=0.1)
    calls: list[TransitionRequest] = []
    controller = accepting_controller(components.strategy, components.initial_factors, calls)
    fixed = snapshot(victim, waiting, hbm=600)

    components.strategy.repair_capacity(fixed, components.initial_factors, controller)
    components.strategy.schedule_resume(fixed, components.initial_factors, controller)

    assert [call.kind for call in calls] == [TransitionKind.MARK_FOR_PAUSE]
    victim_factors = components.initial_factors.for_program(victim.ref)
    assert victim_factors.tier is MORITier.GPU
    assert victim_factors.pending_tier is MORITier.CPU


def test_existing_waiter_reclaims_more_idle_acting_without_capacity_debt() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    set_tier(components, resident, MORITier.GPU, idleness=0.9)
    set_tier(components, waiting, MORITier.CPU)
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(resident, waiting, hbm=1000, dram=2000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, resident.ref),
        (TransitionKind.RESUME, waiting.ref),
    ]
    assert components.initial_factors.for_program(resident.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.GPU


def test_wait_age_does_not_preempt_reasoning_without_capacity_debt() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    waiting = replace(waiting, wait_started_at_monotonic_s=0.0)
    set_tier(components, resident, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU)
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(resident, waiting, hbm=1000, dram=2000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert calls == []
    resident_factors = components.initial_factors.for_program(resident.ref)
    assert resident_factors.tier is MORITier.GPU
    assert resident_factors.pending_tier is None


def test_pending_cpu_destinations_prevent_lazy_demotion_overcommit() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    first = view("first", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    second = view("second", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    set_tier(components, first, MORITier.GPU, idleness=0.9)
    set_tier(components, second, MORITier.GPU, idleness=0.8)
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(first, second, hbm=100, dram=450),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert components.initial_factors.for_program(first.ref).pending_tier is MORITier.CPU
    assert components.initial_factors.for_program(second.ref).pending_tier is MORITier.WAITING


def test_more_valuable_demotion_replaces_a_more_idle_cpu_owner() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    valuable = view("valuable", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    idle_cpu = view("idle-cpu", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=400, step_count=1)
    set_tier(components, valuable, MORITier.GPU, idleness=0.1)
    set_tier(components, idle_cpu, MORITier.CPU, idleness=0.9)
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(valuable, idle_cpu, hbm=100, dram=400),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert [call.program for call in calls] == [valuable.ref]
    assert components.initial_factors.for_program(valuable.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(valuable.ref).pending_tier is None
    assert components.initial_factors.for_program(idle_cpu.ref).tier is MORITier.WAITING
    assert components.strategy.counters["cpu_relative_replacements"] == 1


def test_less_valuable_demotion_cannot_displace_a_busy_cpu_owner() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    idle_gpu = view("idle-gpu", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    valuable_cpu = view(
        "valuable-cpu", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=400, step_count=1
    )
    set_tier(components, idle_gpu, MORITier.GPU, idleness=0.9)
    set_tier(components, valuable_cpu, MORITier.CPU, idleness=0.1)
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(idle_gpu, valuable_cpu, hbm=100, dram=400),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert [call.program for call in calls] == [idle_gpu.ref]
    assert components.initial_factors.for_program(idle_gpu.ref).tier is MORITier.WAITING
    assert components.initial_factors.for_program(idle_gpu.ref).pending_tier is None
    assert components.initial_factors.for_program(valuable_cpu.ref).tier is MORITier.CPU
    assert components.strategy.counters["cpu_relative_replacements"] == 0


def test_rejected_demotion_does_not_commit_planned_cpu_replacements() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    valuable = view("valuable", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    idle_cpu = view("idle-cpu", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=400, step_count=1)
    set_tier(components, valuable, MORITier.GPU, idleness=0.1)
    set_tier(components, idle_cpu, MORITier.CPU, idleness=0.9)

    def reject(request: TransitionRequest) -> TransitionResult:
        return TransitionResult(request.kind, request.program, False, "still_authorized")

    components.strategy.repair_capacity(
        snapshot(valuable, idle_cpu, hbm=100, dram=400),
        components.initial_factors,
        TransitionController(reject),
    )

    assert components.initial_factors.for_program(valuable.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(valuable.ref).pending_tier is None
    assert components.initial_factors.for_program(idle_cpu.ref).tier is MORITier.CPU
    assert components.strategy.counters["cpu_relative_replacements"] == 0


@pytest.mark.parametrize("status", [ProgramStatus.ACTING, ProgramStatus.REASONING])
def test_completed_demotion_is_not_a_persistent_plan(status: ProgramStatus) -> None:
    """The original drained-plan state cannot lock later planning cycles."""
    components = build_mori_strategy()
    survivor = view("survivor", state=ProgramState.ACTIVE, status=status, tokens=700)
    drained = view("drained", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=400)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    set_tier(components, survivor, MORITier.GPU, idleness=0.9)
    set_tier(components, drained, MORITier.CPU)
    calls = []
    components.strategy.schedule_resume(
        snapshot(survivor, drained, waiting, hbm=1500, dram=2000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [call.kind for call in calls] == [TransitionKind.RESUME]
    assert calls[0].program == waiting.ref
    assert components.initial_factors.for_program(survivor.ref).pending_tier is None


def test_unfittable_high_priority_waiter_does_not_block_smaller_request() -> None:
    components = build_mori_strategy()
    victim = view("victim", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    backfill = view("backfill", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=50)
    set_tier(components, victim, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU)
    calls = []
    components.strategy.schedule_resume(
        snapshot(victim, waiting, backfill),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.ADMIT, backfill.ref)]


@pytest.mark.parametrize(("resident_idle", "waiting_idle"), [(1.0, 1.0), (0.01, 0.99), (0.0, 0.0), (1.0, 0.0)])
def test_idle_gpu_can_reclaim_acting_regardless_of_relative_score(resident_idle, waiting_idle) -> None:
    components = build_mori_strategy()
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400)
    set_tier(components, resident, MORITier.GPU, idleness=resident_idle)
    set_tier(components, waiting, MORITier.CPU, idleness=waiting_idle)
    calls = []
    components.strategy.schedule_resume(
        snapshot(resident, waiting, dram=2000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, resident.ref),
        (TransitionKind.ADMIT, waiting.ref),
    ]


@pytest.mark.parametrize("reverse_snapshot", [False, True])
@pytest.mark.parametrize("wait_seconds", [59.0, 60.0, 61.0])
def test_context_replacement_k1_is_deterministic_and_ignores_wait_age(reverse_snapshot, wait_seconds) -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    cheaper = with_token_source(
        view("cheaper", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=450),
        TokenObservationSource.BACKEND_USAGE,
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    candidate = replace(
        view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1),
        wait_started_at_monotonic_s=100.0 - wait_seconds,
    )
    set_tier(components, legacy, MORITier.GPU, idleness=0.9)
    set_tier(components, cheaper, MORITier.GPU, idleness=0.8)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, candidate, MORITier.CPU, idleness=0.1)
    residents = [legacy, cheaper, running]
    if reverse_snapshot:
        residents.reverse()
    calls = []

    components.strategy.schedule_resume(
        snapshot(*residents, candidate, hbm=1350, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, cheaper.ref),
        (TransitionKind.RESUME, candidate.ref),
    ]
    assert components.strategy.counters["replacement_plans_changed"] == 1
    assert components.strategy.counters["replacement_reasoning_victims"] == 0


def test_context_replacement_k2_keeps_count_and_improves_both_slots() -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy_a = with_token_source(
        view("legacy-a", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    legacy_b = with_token_source(
        view("legacy-b", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=450),
        TokenObservationSource.BACKEND_USAGE,
    )
    cheaper_c = with_token_source(
        view("cheaper-c", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400),
        TokenObservationSource.BACKEND_USAGE,
    )
    cheaper_d = with_token_source(
        view("cheaper-d", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=350),
        TokenObservationSource.BACKEND_USAGE,
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=800, step_count=1)
    for program, idle in ((legacy_a, 0.9), (legacy_b, 0.8), (cheaper_c, 0.7), (cheaper_d, 0.6)):
        set_tier(components, program, MORITier.GPU, idleness=idle)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, candidate, MORITier.CPU, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(legacy_a, legacy_b, cheaper_c, cheaper_d, running, candidate, hbm=2300, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, cheaper_c.ref),
        (TransitionKind.PAUSE, cheaper_d.ref),
        (TransitionKind.RESUME, candidate.ref),
    ]
    assert components.strategy.counters["replacement_swaps"] == 2
    assert components.strategy.counters["replacement_plans_changed"] == 1


@pytest.mark.parametrize(
    ("legacy_source", "incoming_source"),
    [
        (TokenObservationSource.BACKEND_USAGE, TokenObservationSource.ROUTER_ESTIMATE),
        (TokenObservationSource.ROUTER_ESTIMATE, TokenObservationSource.BACKEND_USAGE),
        (TokenObservationSource.BACKEND_USAGE, TokenObservationSource.MIXED),
    ],
)
def test_unknown_context_cannot_enter_or_leave_the_legacy_plan(legacy_source, incoming_source) -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500), legacy_source
    )
    incoming = with_token_source(
        view("incoming", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=450), incoming_source
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    set_tier(components, legacy, MORITier.GPU, idleness=0.9)
    set_tier(components, incoming, MORITier.GPU, idleness=0.8)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, candidate, MORITier.CPU, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(legacy, incoming, running, candidate, hbm=1350, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert [(call.kind, call.program) for call in calls[:2]] == [
        (TransitionKind.PAUSE, legacy.ref),
        (TransitionKind.RESUME, candidate.ref),
    ]
    assert components.strategy.counters["replacement_plans_legacy_fallback"] == 1


def test_equal_context_cost_does_not_change_the_legacy_plan() -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    equal = with_token_source(
        view("equal", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    set_tier(components, legacy, MORITier.GPU, idleness=0.9)
    set_tier(components, equal, MORITier.GPU, idleness=0.8)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, candidate, MORITier.CPU, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(legacy, equal, running, candidate, hbm=1400, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert calls[0].program == legacy.ref
    assert components.strategy.counters["replacement_equal_cost_guard"] == 1
    assert components.strategy.counters["replacement_plans_changed"] == 0


def test_cheaper_context_with_more_overshoot_falls_back_to_legacy() -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    cheaper = with_token_source(
        view("cheaper", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400),
        TokenObservationSource.BACKEND_USAGE,
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=500)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100, step_count=1)

    class Shared:
        def footprint(self, ref, tier):
            if tier != "gpu":
                return frozenset()
            if ref in (legacy.ref, running.ref):
                return frozenset(range(1, 501))
            if ref == cheaper.ref:
                return frozenset(range(501, 901))
            return frozenset()

    components.strategy.bind_footprints(Shared())
    set_tier(components, legacy, MORITier.GPU, idleness=0.9)
    set_tier(components, cheaper, MORITier.GPU, idleness=0.8)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, candidate, MORITier.CPU, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(legacy, cheaper, running, candidate, hbm=1300, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert calls[0].program == legacy.ref
    assert components.strategy.counters["replacement_overshoot_guard"] == 1
    assert components.strategy.counters["replacement_plans_changed"] == 0


def test_cheaper_context_that_cannot_fit_falls_back_to_legacy() -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    cheaper = with_token_source(
        view("cheaper", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400),
        TokenObservationSource.BACKEND_USAGE,
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500, step_count=1)
    set_tier(components, legacy, MORITier.GPU, idleness=0.9)
    set_tier(components, cheaper, MORITier.GPU, idleness=0.8)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, candidate, MORITier.CPU, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(legacy, cheaper, running, candidate, hbm=1300, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert calls[0].program == legacy.ref
    assert components.strategy.counters["replacement_capacity_guard"] == 1
    assert components.strategy.counters["replacement_plans_changed"] == 0


def test_replacement_scope_excludes_ineligible_and_reasoning_programs() -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    marked = with_token_source(
        view(
            "marked",
            state=ProgramState.ACTIVE,
            status=ProgramStatus.ACTING,
            tokens=300,
            marked_for_pause=True,
        ),
        TokenObservationSource.BACKEND_USAGE,
    )
    reasoning = with_token_source(
        view("reasoning", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=300),
        TokenObservationSource.BACKEND_USAGE,
    )
    pending = with_token_source(
        view("pending", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=300),
        TokenObservationSource.BACKEND_USAGE,
    )
    non_gpu = with_token_source(
        view("non-gpu", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=300),
        TokenObservationSource.BACKEND_USAGE,
    )
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    for program, idle in ((legacy, 0.9), (marked, 0.8), (reasoning, 0.7), (pending, 0.6)):
        set_tier(components, program, MORITier.GPU, idleness=idle)
    pending_factor = components.initial_factors.for_program(pending.ref)
    assert pending_factor is not None
    components.strategy._set_factor(
        components.initial_factors,
        pending.ref,
        replace(pending_factor, pending_tier=MORITier.CPU),
    )
    set_tier(components, non_gpu, MORITier.CPU, idleness=0.5)
    set_tier(components, candidate, MORITier.CPU, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(legacy, marked, reasoning, pending, non_gpu, candidate, hbm=2000, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert calls[0].program == legacy.ref
    assert components.strategy.counters["replacement_reasoning_victims"] == 0


def test_multiple_candidates_replan_from_committed_arbitrary_subset() -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    cheaper = with_token_source(
        view("cheaper", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=450),
        TokenObservationSource.BACKEND_USAGE,
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    first = view("first", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    second = view("second", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100, step_count=1)
    set_tier(components, legacy, MORITier.GPU, idleness=0.9)
    set_tier(components, cheaper, MORITier.GPU, idleness=0.8)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, first, MORITier.CPU, idleness=0.1)
    set_tier(components, second, MORITier.WAITING, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(legacy, cheaper, running, first, second, hbm=1350, dram=5000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, cheaper.ref),
        (TransitionKind.RESUME, first.ref),
        (TransitionKind.PAUSE, legacy.ref),
        (TransitionKind.RESUME, second.ref),
    ]
    assert components.strategy.counters["replacement_plans_considered"] == 2


def test_rejected_arbitrary_subset_pause_preserves_real_slack_for_later_fit() -> None:
    components = build_mori_strategy(MORIConfig(enable_wait_credit=False))
    legacy = with_token_source(
        view("legacy", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500),
        TokenObservationSource.BACKEND_USAGE,
    )
    cheaper = with_token_source(
        view("cheaper", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=450),
        TokenObservationSource.BACKEND_USAGE,
    )
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    blocked = view("blocked", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    backfill = view("backfill", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=0)
    set_tier(components, legacy, MORITier.GPU, idleness=0.9)
    set_tier(components, cheaper, MORITier.GPU, idleness=0.8)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, blocked, MORITier.CPU, idleness=0.1)
    attempts = []
    accepted_calls = []
    accepted = accepting_controller(components.strategy, components.initial_factors, accepted_calls)

    def reject_selected(request):
        attempts.append(request)
        if request.kind is TransitionKind.PAUSE:
            return TransitionResult(request.kind, request.program, False, "still_authorized")
        return accepted.apply(request)

    components.strategy.schedule_resume(
        snapshot(legacy, cheaper, running, blocked, backfill, hbm=1450, dram=5000),
        components.initial_factors,
        TransitionController(reject_selected),
    )

    assert [(call.kind, call.program) for call in attempts] == [
        (TransitionKind.PAUSE, cheaper.ref),
        (TransitionKind.ADMIT, backfill.ref),
    ]
    assert components.initial_factors.for_program(cheaper.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(blocked.ref).tier is MORITier.CPU
    assert components.strategy.counters["replacement_transition_rejections"] == 1


@pytest.mark.parametrize(("resident_idle", "waiting_idle"), [(1.0, 1.0), (0.01, 0.99), (0.0, 0.0)])
def test_active_gpu_reasoning_preserves_equal_or_less_idle_acting(resident_idle, waiting_idle) -> None:
    components = build_mori_strategy()
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400)
    set_tier(components, resident, MORITier.GPU, idleness=resident_idle)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=waiting_idle)
    calls = []
    components.strategy.schedule_resume(
        snapshot(resident, running, waiting),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert calls == []


def test_insufficient_eligible_acting_relief_does_not_evict_before_backfill():
    components = build_mori_strategy()
    idle = view("idle", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=100)
    busy = view("busy", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=600)
    large = view("large", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    small = view("small", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=1)
    for program, tier, score in [(idle, MORITier.GPU, 0.9), (busy, MORITier.GPU, 0.1), (large, MORITier.CPU, 0.5)]:
        set_tier(components, program, tier, idleness=score)
    calls = []
    components.strategy.schedule_resume(
        snapshot(idle, busy, large, small, hbm=1020),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.ADMIT, small.ref)]
    assert components.initial_factors.for_program(idle.ref).tier is MORITier.GPU


def test_idle_gpu_progress_preserves_cpu_before_waiting_priority():
    components = build_mori_strategy()
    victim = view("victim", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    cpu = view("cpu", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500, step_count=1)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500, step_count=1)
    set_tier(components, victim, MORITier.GPU, idleness=0.5)
    set_tier(components, cpu, MORITier.CPU, idleness=0.9)
    set_tier(components, waiting, MORITier.WAITING, idleness=0.1)
    calls = []
    components.strategy.schedule_resume(
        snapshot(victim, cpu, waiting, hbm=1000, dram=2000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, cpu.ref),
    ]
    assert components.initial_factors.for_program(victim.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(cpu.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.WAITING


def test_rejected_capacity_demotion_keeps_debt_and_blocks_admission():
    components = build_mori_strategy()
    victim = view("victim", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    backfill = view("backfill", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=50)
    set_tier(components, victim, MORITier.GPU, idleness=0.9)
    set_tier(components, waiting, MORITier.CPU, idleness=0.1)
    calls = []
    accepted = accepting_controller(components.strategy, components.initial_factors, calls)

    def reject_demotion(request):
        if request.kind is TransitionKind.PAUSE:
            return TransitionResult(request.kind, request.program, False, "still_authorized")
        return accepted.apply(request)

    fixed = snapshot(victim, waiting, backfill, hbm=600)
    controller = TransitionController(reject_demotion)
    components.strategy.repair_capacity(fixed, components.initial_factors, controller)
    components.strategy.schedule_resume(fixed, components.initial_factors, controller)
    assert calls == []
    assert components.initial_factors.for_program(victim.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(victim.ref).pending_tier is None
    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.CPU
    assert components.strategy.counters["demotion_rejected"] == 1


def test_rejected_rebalance_demotion_keeps_reservation_and_allows_only_actual_slack():
    components = build_mori_strategy()
    victim = view("victim", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    backfill = view("backfill", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=50)
    set_tier(components, victim, MORITier.GPU, idleness=0.9)
    set_tier(components, waiting, MORITier.CPU, idleness=0.1)
    calls = []
    accepted = accepting_controller(components.strategy, components.initial_factors, calls)

    def reject_demotion(request):
        if request.kind is TransitionKind.PAUSE:
            return TransitionResult(request.kind, request.program, False, "still_authorized")
        return accepted.apply(request)

    components.strategy.schedule_resume(
        snapshot(victim, waiting, backfill, hbm=1000),
        components.initial_factors,
        TransitionController(reject_demotion),
    )

    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.ADMIT, backfill.ref)]
    assert components.initial_factors.for_program(victim.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(victim.ref).pending_tier is None
    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.CPU
    assert components.strategy.counters["demotion_rejected"] == 1


def test_idleness_uses_only_committed_reasoning_and_elapsed_acting() -> None:
    components = build_mori_strategy(MORIConfig(window_size=5, mode=MORIMode.FULL))
    program = ProgramRef("program", 0)
    factors = MORIProgramFactors(
        tier=MORITier.GPU,
        cycles=((2.0, 2.0),),
        pending_reasoning_seconds=2.0,
        acting_started_at_monotonic_s=98.0,
    )
    components.strategy._set_factor(components.initial_factors, program, factors)

    assert components.strategy.idleness_for_program(program, 100.0) == pytest.approx(4.0 / 8.0)


def test_decode_buffer_is_fixed_and_rounded_at_hbm_block_granularity() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=101)
    fixed = replace(snapshot(candidate, hbm=1000), hbm_block_size_tokens=16)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        fixed,
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
        candidate.ref,
    )

    assert outcome.required_tokens == 208
    assert outcome.disposition is AdmissionDisposition.ADMITTED


def test_capacity_targets_do_not_deduct_unproven_shared_context() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    candidate = replace(
        view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=201),
        tokens=ProgramTokenObservation(estimated_context_tokens=201, shared_prefix_tokens=80),
    )
    fixed = replace(snapshot(candidate, hbm=1000, dram=1000), hbm_block_size_tokens=16, dram_chunk_size_tokens=64)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        fixed,
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
        candidate.ref,
    )

    assert outcome.required_tokens == 304
    assert components.strategy._dram_target(candidate, fixed) == 256
    assert outcome.disposition is AdmissionDisposition.ADMITTED


def test_completion_accumulates_cache_hit_report_from_server_observations() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    program = view("program", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    set_tier(components, program, MORITier.GPU)
    event = SchedulingEvent(
        event_id="finished",
        sequence=1,
        kind=SchedulingEventKind.REQUEST_FINISHED,
        occurred_at_monotonic_s=100.0,
        reason="request_completed",
        program=program.ref,
        current_state=ProgramState.ACTIVE,
        current_status=ProgramStatus.ACTING,
        fields=(
            ("prompt_tokens", 400),
            ("cached_prefix_tokens", 300),
            ("hbm_cached_prefix_tokens", 250),
            ("hbm_hit_ref_zero_tokens", 50),
            ("hbm_cached_prefix_exact_present", True),
            ("hbm_hit_ref_zero_exact_present", True),
            ("request_latency_seconds", 2.0),
            ("scheduler_queue_seconds", 0.25),
        ),
    )

    components.strategy.handle_scheduling_event(components.initial_factors, event)
    report = dict(components.strategy.diagnostics(snapshot(program), components.initial_factors)[0].fields)

    assert report["cache_observed_requests"] == 1
    assert report["cache_prompt_tokens"] == 400
    assert report["cache_prefix_hit_tokens"] == 300
    assert report["cache_hbm_hit_tokens"] == 250
    assert report["cache_hbm_ref_zero_hit_tokens"] == 50
    assert report["cache_prefix_hit_ratio"] == pytest.approx(0.75)
    assert report["cache_hbm_hit_ratio"] == pytest.approx(0.625)
    shadow = components.initial_factors.for_program(program.ref)
    assert shadow.last_hbm_cached_prefix_exact_present is True
    assert shadow.last_hbm_cached_prefix_tokens == 250
    assert shadow.last_hbm_hit_ref_zero_exact_present is True
    assert shadow.last_hbm_hit_ref_zero_tokens == 50


def test_shadow_cache_observation_distinguishes_exact_zero_from_combined_fallback() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    program = view("program", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=100)
    set_tier(components, program, MORITier.GPU)

    def finished(sequence, *, exact):
        return SchedulingEvent(
            event_id=f"finished-{sequence}",
            sequence=sequence,
            kind=SchedulingEventKind.REQUEST_FINISHED,
            occurred_at_monotonic_s=100.0 + sequence,
            reason="request_completed",
            program=program.ref,
            current_state=ProgramState.ACTIVE,
            current_status=ProgramStatus.ACTING,
            fields=(
                ("prompt_tokens", 100),
                ("cached_prefix_tokens", 80),
                ("hbm_cached_prefix_tokens", 0 if exact else 80),
                ("hbm_hit_ref_zero_tokens", 0),
                ("hbm_cached_prefix_exact_present", exact),
                ("hbm_hit_ref_zero_exact_present", exact),
            ),
        )

    components.strategy.handle_scheduling_event(components.initial_factors, finished(1, exact=True))
    exact_zero = components.initial_factors.for_program(program.ref)
    assert exact_zero.last_hbm_cached_prefix_exact_present is True
    assert exact_zero.last_hbm_cached_prefix_tokens == 0
    assert exact_zero.last_hbm_hit_ref_zero_exact_present is True
    assert exact_zero.last_hbm_hit_ref_zero_tokens == 0

    components.strategy.handle_scheduling_event(components.initial_factors, finished(2, exact=False))
    fallback = components.initial_factors.for_program(program.ref)
    assert fallback.last_hbm_cached_prefix_exact_present is False
    assert fallback.last_hbm_cached_prefix_tokens is None
    assert fallback.last_hbm_hit_ref_zero_exact_present is False
    assert fallback.last_hbm_hit_ref_zero_tokens is None


def test_overlapping_completions_end_reasoning_only_after_the_last_request() -> None:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    program = view("program", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    set_tier(components, program, MORITier.GPU)
    started = SchedulingEvent(
        event_id="started",
        sequence=1,
        kind=SchedulingEventKind.PROGRAM_STATUS_CHANGED,
        occurred_at_monotonic_s=10.0,
        reason="request_started",
        program=program.ref,
        current_state=ProgramState.ACTIVE,
        current_status=ProgramStatus.REASONING,
    )
    first_finished = SchedulingEvent(
        event_id="first-finished",
        sequence=2,
        kind=SchedulingEventKind.REQUEST_FINISHED,
        occurred_at_monotonic_s=12.0,
        reason="request_completed",
        program=program.ref,
        current_state=ProgramState.ACTIVE,
        current_status=ProgramStatus.REASONING,
        fields=(("request_latency_seconds", 2.0), ("scheduler_queue_seconds", 0.0)),
    )
    last_finished = SchedulingEvent(
        event_id="last-finished",
        sequence=3,
        kind=SchedulingEventKind.REQUEST_FINISHED,
        occurred_at_monotonic_s=15.0,
        reason="request_completed",
        program=program.ref,
        current_state=ProgramState.ACTIVE,
        current_status=ProgramStatus.ACTING,
        fields=(("request_latency_seconds", 4.0), ("scheduler_queue_seconds", 0.0)),
    )

    components.strategy.handle_scheduling_event(components.initial_factors, started)
    components.strategy.handle_scheduling_event(components.initial_factors, first_finished)
    during_overlap = components.initial_factors.for_program(program.ref)
    assert during_overlap is not None
    assert during_overlap.pending_reasoning_seconds == 0.0
    assert during_overlap.reasoning_started_at_monotonic_s == 10.0
    assert during_overlap.acting_started_at_monotonic_s is None

    components.strategy.handle_scheduling_event(components.initial_factors, last_finished)
    completed = components.initial_factors.for_program(program.ref)
    assert completed is not None
    assert completed.pending_reasoning_seconds == 5.0
    assert completed.reasoning_started_at_monotonic_s is None
    assert completed.acting_started_at_monotonic_s == 15.0


@pytest.mark.parametrize("tier,steps", [(MORITier.CPU, 1), (MORITier.WAITING, 1), (MORITier.WAITING, 0)])
def test_wait_age_does_not_override_paper_promotion_priority(tier, steps) -> None:
    components = build_mori_strategy()
    old = replace(
        view("old", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=700),
        wait_started_at_monotonic_s=0.0,
    )
    young = replace(
        view("young", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=600, step_count=steps),
        wait_started_at_monotonic_s=99.0,
    )
    set_tier(components, old, MORITier.WAITING, idleness=1.0)
    set_tier(components, young, tier)
    calls = []
    fixed = snapshot(old, young)
    components.strategy.schedule_resume(
        fixed, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
    )
    assert [call.program for call in calls] == [young.ref]
    assert dict(components.strategy.diagnostics(fixed, components.initial_factors)[0].fields)["max_wait_seconds"] == 100


def test_normal_promotion_preserves_mori_decision_table() -> None:
    components = build_mori_strategy()
    programs = [
        view("new-large", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=200),
        view("new-small", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100),
        view("return-idle", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100, step_count=1),
        view("return-busy", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100, step_count=1),
        view("cpu", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=300, step_count=1),
    ]
    for program in programs:
        set_tier(
            components,
            program,
            MORITier.CPU if program.ref.program_id == "cpu" else MORITier.WAITING,
            idleness=0.9 if program.ref.program_id == "return-idle" else 0.1,
        )
    calls = []
    components.strategy.schedule_resume(
        snapshot(*programs, hbm=10000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [call.program.program_id for call in calls] == [
        "cpu",
        "return-busy",
        "return-idle",
        "new-small",
        "new-large",
    ]


def test_new_arrival_joins_the_same_ranked_selection_between_ticks() -> None:
    components = build_mori_strategy()
    old = view("old", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500)
    new = view("new", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    calls = []
    outcome = components.strategy.handle_admission(
        snapshot(old, new),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
        new.ref,
    )
    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert [call.kind for call in calls] == [TransitionKind.QUEUE]
    calls.clear()
    components.strategy.schedule_resume(
        snapshot(old, new),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [call.program for call in calls] == [new.ref, old.ref]


def test_later_fitting_candidate_runs_without_speculative_demotion() -> None:
    components = build_mori_strategy()
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
    large = view("large", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    small = view("small", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=50)
    set_tier(components, resident, MORITier.GPU)
    set_tier(components, large, MORITier.CPU)
    calls = []
    components.strategy.schedule_resume(
        snapshot(resident, large, small),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.ADMIT, small.ref),
    ]
    assert components.initial_factors.for_program(large.ref).tier is MORITier.CPU


def test_over_total_quota_candidate_does_not_block_a_later_fit() -> None:
    components = build_mori_strategy()
    large = view("large", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=1100, step_count=1)
    small = view("small", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=50)
    set_tier(components, large, MORITier.CPU)
    calls = []
    components.strategy.schedule_resume(
        snapshot(large, small),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [call.program for call in calls] == [small.ref]


def test_decode_buffer_does_not_block_a_prompt_that_fits_the_total_quota() -> None:
    components = build_mori_strategy()
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=950)
    calls = []
    outcome = components.strategy.handle_admission(
        snapshot(candidate, hbm=1000),
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
        candidate.ref,
    )
    assert outcome.disposition is AdmissionDisposition.ADMITTED
    assert outcome.required_tokens == 1000
