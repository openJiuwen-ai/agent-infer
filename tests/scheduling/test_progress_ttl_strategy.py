# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Unit tests for Progress-TTL decision factors, deadlines, and capacity decisions."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
    StrategyFactors,
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)
from agentinfer.scheduling.progress_ttl import (
    ProgressTTLConfig,
    ProgressTTLGlobalFactors,
    ProgressTTLProgramFactors,
    ProgressTTLStrategy,
)

pytestmark = pytest.mark.cpu_test

EVENT_KIND = {
    TransitionKind.ADMIT: SchedulingEventKind.REQUEST_ADMITTED,
    TransitionKind.QUEUE: SchedulingEventKind.REQUEST_PENDING,
    TransitionKind.PAUSE: SchedulingEventKind.PROGRAM_PAUSED,
    TransitionKind.MARK_FOR_PAUSE: SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE,
    TransitionKind.RESUME: SchedulingEventKind.PROGRAM_RESUMED,
    TransitionKind.RELEASE: SchedulingEventKind.PROGRAM_RELEASED,
}


def build_progress_ttl_strategy(config: ProgressTTLConfig) -> SimpleNamespace:
    """Build policy components locally so the policy-core PR does not include runtime wiring."""
    return SimpleNamespace(
        strategy=ProgressTTLStrategy(config),
        initial_factors=StrategyFactors(global_factors=ProgressTTLGlobalFactors()),
    )


def view(
    program_id: str,
    *,
    state: ProgramState,
    status: ProgramStatus,
    tokens: int = 100,
    backend_id: str | None = None,
    task_id: str | None = None,
    parent_program_id: str | None = None,
    marked_for_pause: bool = False,
) -> ProgramView:
    """Build one deterministic policy input."""
    return ProgramView(
        ref=ProgramRef(program_id, 0),
        state=state,
        status=status,
        tokens=ProgramTokenObservation(estimated_context_tokens=tokens),
        backend_id=backend_id,
        task_id=task_id,
        parent_program_id=parent_program_id,
        marked_for_pause=marked_for_pause,
    )


def snapshot(*programs: ProgramView, capacity: int = 1000, waiting: tuple[ProgramRef, ...] = ()) -> SchedulingSnapshot:
    """Build one fixed backend snapshot."""
    return SchedulingSnapshot(
        observed_at_monotonic_s=100.0,
        backend_id="backend",
        total_kv_tokens=capacity,
        programs=programs,
        waiting_programs=waiting,
    )


def controller(calls: list[TransitionRequest]) -> TransitionController:
    """Return an accepting transition controller that records exact plans."""

    def apply(request: TransitionRequest) -> TransitionResult:
        calls.append(request)
        return TransitionResult(
            request.kind,
            request.program,
            True,
            request.reason,
            SchedulingEvent(
                event_id=f"event-{len(calls)}",
                sequence=len(calls),
                kind=EVENT_KIND[request.kind],
                occurred_at_monotonic_s=100.0,
                reason=request.reason,
                program=request.program,
            ),
        )

    return TransitionController(apply)


def test_progress_ttl_config_rejects_invalid_round_and_ttl_bounds() -> None:
    with pytest.raises(ValueError, match="target_max"):
        ProgressTTLConfig(target_min_segment_rounds=3, target_max_segment_rounds=2)
    with pytest.raises(ValueError, match="ttl_max"):
        ProgressTTLConfig(ttl_min_seconds=2, ttl_max_seconds=1)
    with pytest.raises(ValueError, match="resume_capacity_ratio"):
        ProgressTTLConfig(resume_capacity_ratio=0)
    with pytest.raises(ValueError, match="pause_capacity_ratio"):
        ProgressTTLConfig(pause_capacity_ratio=1.1)
    with pytest.raises(ValueError, match="privileged_max_context_tokens"):
        ProgressTTLConfig(privileged_max_context_tokens=0)


def test_admission_and_resume_reserve_the_same_minimum_segment_growth() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            target_min_segment_rounds=2,
            resume_capacity_ratio=1,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 150
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    candidate = view(
        "candidate",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
        tokens=200,
    )
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=1, is_evictable_after_min_rounds=False),
    )
    admission_calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(active, candidate, capacity=1000),
        components.initial_factors,
        controller(admission_calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert outcome.required_tokens == 200
    assert outcome.reserve_tokens == 450
    assert outcome.deficit_tokens == 50
    assert [call.kind for call in admission_calls] == [TransitionKind.QUEUE]

    resume_calls: list[TransitionRequest] = []
    components.strategy.schedule_resume(
        snapshot(active, candidate, capacity=1000, waiting=(candidate.ref,)),
        components.initial_factors,
        controller(resume_calls),
    )

    assert resume_calls == []

    fitting_admission_calls: list[TransitionRequest] = []
    fitting_outcome = components.strategy.handle_admission(
        snapshot(active, candidate, capacity=1050),
        components.initial_factors,
        controller(fitting_admission_calls),
        candidate.ref,
    )
    fitting_resume_calls: list[TransitionRequest] = []
    components.strategy.schedule_resume(
        snapshot(active, candidate, capacity=1050, waiting=(candidate.ref,)),
        components.initial_factors,
        controller(fitting_resume_calls),
    )

    assert fitting_outcome.disposition is AdmissionDisposition.ADMITTED
    assert [call.kind for call in fitting_admission_calls] == [TransitionKind.ADMIT]
    assert [call.kind for call in fitting_resume_calls] == [TransitionKind.RESUME]


def test_fixed_growth_overrides_rolling_growth_for_capacity_projection() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            target_min_segment_rounds=2,
            use_fixed_input_token_growth=True,
            fixed_input_token_growth_per_round=25,
        )
    )
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        backend_id="backend",
    )
    candidate = view(
        "candidate",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
    )
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=1),
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 999

    first_reserve = components.strategy._continuous_growth_reserve_tokens(
        (active,),
        components.initial_factors,
        candidate=candidate,
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 10_000
    second_reserve = components.strategy._continuous_growth_reserve_tokens(
        (active,),
        components.initial_factors,
        candidate=candidate,
    )

    assert first_reserve == 75
    assert second_reserve == 75


def test_resume_ignores_paused_acting_program_without_a_pending_request() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    acting = view("acting", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=100)
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(acting, capacity=1000, waiting=(acting.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert calls == []


def test_force_resume_timeout_bypasses_capacity_projection() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            force_resume_timeout_seconds=30,
        )
    )
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=900)
    components.initial_factors.set_program_factors(
        waiting.ref,
        ProgressTTLProgramFactors(
            wait_started_at_monotonic_s=10,
            request_wait_started_at_monotonic_s=60,
        ),
    )
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(waiting, capacity=100, waiting=(waiting.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert [(call.kind, call.reason) for call in calls] == [
        (TransitionKind.RESUME, "progress_ttl_force_resume_timeout")
    ]


def test_force_resume_timeout_does_not_reactivate_idle_acting_program() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            force_resume_timeout_seconds=30,
        )
    )
    acting = view("acting", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=900)
    components.initial_factors.set_program_factors(
        acting.ref,
        ProgressTTLProgramFactors(
            wait_started_at_monotonic_s=10,
            request_wait_started_at_monotonic_s=60,
        ),
    )
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(acting, capacity=100, waiting=(acting.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert calls == []


def test_force_resume_timeout_does_not_prioritize_idle_acting_program() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0, force_resume_timeout_seconds=30))
    reasoning = view("reasoning", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    acting = view("acting", state=ProgramState.PAUSED, status=ProgramStatus.ACTING)
    components.initial_factors.set_program_factors(
        acting.ref,
        ProgressTTLProgramFactors(
            wait_started_at_monotonic_s=10,
            request_wait_started_at_monotonic_s=10,
        ),
    )

    ordered = sorted(
        (reasoning, acting),
        key=lambda program: components.strategy._resume_key(
            program,
            components.initial_factors,
            60,
        ),
    )

    assert ordered == [reasoning, acting]


@pytest.mark.parametrize("wait_started", [0.0, 10.0])
def test_pending_request_keeps_pause_age_but_starts_its_own_force_resume_timer(wait_started: float) -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    program = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(wait_started_at_monotonic_s=wait_started),
    )
    pending = SchedulingEvent(
        event_id="pending",
        sequence=1,
        kind=SchedulingEventKind.REQUEST_PENDING,
        occurred_at_monotonic_s=90,
        reason="already_waiting",
        program=program.ref,
    )

    components.strategy.handle_scheduling_event(components.initial_factors, pending)

    sidecar = components.initial_factors.for_program(program.ref)
    assert sidecar is not None
    assert sidecar.wait_started_at_monotonic_s == wait_started
    assert sidecar.request_wait_started_at_monotonic_s == 90


def test_admission_does_not_use_parent_relationship_as_a_capacity_handoff() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0, target_min_segment_rounds=2))
    parent = view(
        "task:lead",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=700,
        backend_id="backend",
        task_id="task",
    )
    child = view(
        "task:child",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
        tokens=400,
        task_id="task",
        parent_program_id="task:lead",
    )
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(parent, child, capacity=1000),
        components.initial_factors,
        controller(calls),
        child.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    assert [call.kind for call in calls] == [TransitionKind.QUEUE]


def test_privileged_relationship_handoff_transfers_one_task_slot() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            privileged_lookahead_rounds=14,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    parent_waiting = view(
        "temp:session:lead",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
        tokens=700,
        task_id="task",
    )
    parent_calls: list[TransitionRequest] = []

    parent_outcome = components.strategy.handle_admission(
        snapshot(parent_waiting, capacity=1000),
        components.initial_factors,
        controller(parent_calls),
        parent_waiting.ref,
    )

    assert parent_outcome.disposition is AdmissionDisposition.ADMITTED
    assert components.initial_factors.for_program(parent_waiting.ref).is_privileged is True

    parent_active = replace(
        parent_waiting, state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, backend_id="backend"
    )
    child = view(
        "temp:session:child",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
        tokens=400,
        task_id="task",
        parent_program_id=parent_active.ref.program_id,
    )
    child_calls: list[TransitionRequest] = []
    child_outcome = components.strategy.handle_admission(
        snapshot(parent_active, child, capacity=1000, waiting=(child.ref,)),
        components.initial_factors,
        controller(child_calls),
        child.ref,
    )

    assert child_outcome.disposition is AdmissionDisposition.ADMITTED
    assert [call.kind for call in child_calls] == [TransitionKind.PAUSE, TransitionKind.ADMIT]
    assert components.initial_factors.for_program(parent_active.ref).is_privileged is False
    assert components.initial_factors.for_program(child.ref).is_privileged is True


def test_capacity_fit_prefers_fewer_programs_for_the_same_partial_relief() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    programs = [
        view("a", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=3),
        view("b", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=3),
        view("c", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=6),
        view("d", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=1),
    ]

    selected = components.strategy._select_capacity_fit(programs, 7)

    assert selected is not None
    assert tuple(program.ref.program_id for program in selected) == ("c", "d")


def test_ttl_expiry_transfers_privilege_to_paused_same_task_reasoning() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            privileged_lookahead_rounds=14,
            ttl_min_seconds=5,
            ttl_max_seconds=5,
        )
    )
    source = view(
        "source",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        backend_id="backend",
        task_id="task",
    )
    target = view(
        "target",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
        task_id="task",
    )
    components.initial_factors.set_program_factors(
        source.ref,
        ProgressTTLProgramFactors(is_privileged=True, ttl_deadline_monotonic_s=99),
    )
    calls: list[TransitionRequest] = []

    components.strategy.handle_scheduled_check(
        snapshot(source, target, waiting=(target.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert [call.kind for call in calls] == [TransitionKind.PAUSE, TransitionKind.RESUME]
    assert components.initial_factors.for_program(source.ref).is_privileged is False
    assert components.initial_factors.for_program(target.ref).is_privileged is True


def test_capacity_repair_protects_privilege_while_an_ordinary_victim_fits() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            privileged_lookahead_rounds=14,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    privileged = view(
        "privileged",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=600,
        backend_id="backend",
        task_id="task-a",
    )
    ordinary = view(
        "ordinary",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=600,
        backend_id="backend",
        task_id="task-b",
    )
    components.initial_factors.set_program_factors(
        privileged.ref,
        ProgressTTLProgramFactors(is_privileged=True),
    )
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(privileged, ordinary, capacity=700),
        components.initial_factors,
        controller(calls),
    )

    assert [call.program for call in calls] == [ordinary.ref]
    assert components.initial_factors.for_program(privileged.ref).is_privileged is True


def test_privileged_program_uses_longer_growth_target_in_shared_capacity_projection() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            capacity_safety_margin_tokens=25,
            target_min_segment_rounds=2,
            privileged_lookahead_rounds=4,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 100
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        backend_id="backend",
    )
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=1, is_privileged=True),
    )

    reserve = components.strategy._continuous_growth_reserve_tokens(
        (active,),
        components.initial_factors,
        candidate=candidate,
    )

    assert reserve == 525


def test_request_completion_counts_round_and_arms_then_expires_ttl() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            target_min_segment_rounds=1,
            target_max_segment_rounds=2,
            ttl_min_seconds=5,
            ttl_max_seconds=5,
        )
    )
    program = view(
        "program",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        backend_id="backend",
    )
    finished = SchedulingEvent(
        event_id="finished",
        sequence=1,
        kind=SchedulingEventKind.REQUEST_FINISHED,
        occurred_at_monotonic_s=100.0,
        reason="completed",
        program=program.ref,
        current_status=ProgramStatus.ACTING,
        fields=(("total_tokens", 1000), ("prompt_tokens", 900), ("completion_tokens", 100)),
    )

    components.strategy.handle_scheduling_event(components.initial_factors, finished)
    sidecar = components.initial_factors.for_program(program.ref)

    assert sidecar is not None
    assert sidecar.segment_served_rounds == 1
    assert sidecar.is_evictable_after_min_rounds is True
    assert sidecar.ttl_deadline_monotonic_s == 105.0
    calls: list[TransitionRequest] = []
    due = replace(snapshot(program), observed_at_monotonic_s=105.0)
    components.strategy.handle_scheduled_check(due, components.initial_factors, controller(calls))
    assert [(call.kind, call.reason) for call in calls] == [(TransitionKind.PAUSE, "progress_ttl_expired")]


def test_periodic_check_rebuilds_missing_acting_deadline_from_acting_since() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(ttl_min_seconds=5, ttl_max_seconds=5))
    program = view(
        "program",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        backend_id="backend",
    )
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(acting_since_monotonic_s=95, ttl_deadline_monotonic_s=None),
    )
    calls: list[TransitionRequest] = []

    components.strategy.handle_scheduled_check(
        snapshot(program),
        components.initial_factors,
        controller(calls),
    )

    assert [(call.kind, call.reason) for call in calls] == [(TransitionKind.PAUSE, "progress_ttl_expired")]


def test_paused_acting_program_is_released_after_retention_ttl() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(paused_program_ttl_seconds=30))
    program = view("program", state=ProgramState.PAUSED, status=ProgramStatus.ACTING)
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(release_deadline_monotonic_s=100),
    )
    calls: list[TransitionRequest] = []

    components.strategy.handle_scheduled_check(
        snapshot(program, waiting=(program.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert [(call.kind, call.reason) for call in calls] == [
        (TransitionKind.RELEASE, "progress_ttl_paused_retention_expired")
    ]


def test_paused_reasoning_program_is_not_released_after_retention_deadline() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(paused_program_ttl_seconds=30))
    program = view("program", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(release_deadline_monotonic_s=100),
    )
    calls: list[TransitionRequest] = []

    components.strategy.handle_scheduled_check(
        snapshot(program, waiting=(program.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert calls == []


def test_resume_uses_evictable_acting_capacity_without_ranking_by_candidate_size() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0, target_min_segment_rounds=2))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=600,
        backend_id="backend",
    )
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=700)
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=2, is_evictable_after_min_rounds=True),
    )
    components.initial_factors.set_program_factors(
        waiting.ref,
        ProgressTTLProgramFactors(wait_started_at_monotonic_s=10.0),
    )
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(active, waiting, capacity=1000, waiting=(waiting.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert [call.kind for call in calls] == [TransitionKind.PAUSE, TransitionKind.RESUME]
    assert calls[0].program == active.ref
    assert calls[1].program == waiting.ref


def test_resume_reclaim_switch_preserves_acting_program_and_keeps_waiter_paused() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            target_min_segment_rounds=2,
            resume_reclaim_acting_programs=False,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=600,
        backend_id="backend",
    )
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=700)
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=2, is_evictable_after_min_rounds=True),
    )
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(active, waiting, capacity=1000, waiting=(waiting.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert calls == []


def test_resume_reclaims_program_at_max_rounds_before_minimum_eligible_program() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            target_min_segment_rounds=2,
            target_max_segment_rounds=4,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    minimum = view(
        "minimum",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    maximum = view(
        "maximum",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=300)
    components.initial_factors.set_program_factors(
        minimum.ref,
        ProgressTTLProgramFactors(segment_served_rounds=2, is_evictable_after_min_rounds=True),
    )
    components.initial_factors.set_program_factors(
        maximum.ref,
        ProgressTTLProgramFactors(segment_served_rounds=4, is_evictable_after_min_rounds=True),
    )
    components.initial_factors.set_program_factors(
        waiting.ref,
        ProgressTTLProgramFactors(wait_started_at_monotonic_s=10.0),
    )
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(minimum, maximum, waiting, capacity=1000, waiting=(waiting.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert [call.kind for call in calls] == [TransitionKind.PAUSE, TransitionKind.RESUME]
    assert calls[0].program == maximum.ref


def test_max_round_completion_pauses_for_uncovered_top_waiting_demand() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(decode_buffer_tokens=0, target_min_segment_rounds=2, target_max_segment_rounds=3)
    )
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=300)
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=3, is_evictable_after_min_rounds=True),
    )
    calls: list[TransitionRequest] = []

    components.strategy.handle_request_completion(
        snapshot(active, waiting, waiting=(waiting.ref,)),
        components.initial_factors,
        controller(calls),
        active.ref,
    )

    assert [call.kind for call in calls] == [TransitionKind.PAUSE]
    assert calls[0].program == active.ref
    assert calls[0].reason == "progress_ttl_max_round_queue_pressure"


def test_max_round_completion_counts_future_pause_against_only_top_waiting_demand() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(decode_buffer_tokens=0, target_min_segment_rounds=2, target_max_segment_rounds=3)
    )
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    marked = view(
        "marked",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=300,
        backend_id="backend",
        marked_for_pause=True,
    )
    top = view("a-top", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=300)
    later = view("z-later", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=700)
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=3, is_evictable_after_min_rounds=True),
    )
    calls: list[TransitionRequest] = []

    components.strategy.handle_request_completion(
        snapshot(active, marked, top, later, waiting=(top.ref, later.ref)),
        components.initial_factors,
        controller(calls),
        active.ref,
    )

    assert calls == []


def test_capacity_repair_uses_segment_duration_before_program_status() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            pause_capacity_ratio=1,
            pause_capacity_lookahead_rounds=0,
        )
    )
    acting = view(
        "acting",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=700,
        backend_id="backend",
    )
    reasoning = view(
        "reasoning",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=300,
        backend_id="backend",
    )
    components.initial_factors.set_program_factors(
        acting.ref,
        ProgressTTLProgramFactors(
            segment_served_rounds=8,
            segment_started_at_monotonic_s=90,
            last_request_prompt_tokens=700,
        ),
    )
    components.initial_factors.set_program_factors(
        reasoning.ref,
        ProgressTTLProgramFactors(
            segment_served_rounds=1,
            segment_started_at_monotonic_s=20,
        ),
    )
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(acting, reasoning, capacity=750),
        components.initial_factors,
        controller(calls),
    )

    assert [call.kind for call in calls] == [TransitionKind.MARK_FOR_PAUSE]
    assert calls[0].program == reasoning.ref


def test_capacity_repair_scales_segment_duration_by_current_input_tokens() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            pause_capacity_ratio=1,
            pause_capacity_lookahead_rounds=0,
        )
    )
    components.initial_factors.global_factors.avg_prompt_tokens = 500
    components.initial_factors.global_factors.avg_completion_tokens = 100
    smaller = view(
        "smaller",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=100,
        backend_id="backend",
    )
    larger = view(
        "larger",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=900,
        backend_id="backend",
    )
    for program, prompt_tokens in ((smaller, 100), (larger, 900)):
        components.initial_factors.set_program_factors(
            program.ref,
            ProgressTTLProgramFactors(
                segment_served_rounds=4,
                segment_started_at_monotonic_s=80,
                last_request_prompt_tokens=prompt_tokens,
            ),
        )
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(smaller, larger, capacity=950),
        components.initial_factors,
        controller(calls),
    )

    assert [call.program for call in calls] == [smaller.ref]


def test_capacity_pause_score_does_not_scale_by_output_tokens() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    components.initial_factors.global_factors.avg_prompt_tokens = 500
    short_output = view(
        "short-output",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=600,
        backend_id="backend",
    )
    long_output = replace(short_output, ref=ProgramRef("long-output", 0))
    for program, completion_tokens in ((short_output, 1), (long_output, 10_000)):
        components.initial_factors.set_program_factors(
            program.ref,
            ProgressTTLProgramFactors(
                segment_started_at_monotonic_s=80,
                last_request_prompt_tokens=500,
                segment_completion_tokens=completion_tokens,
            ),
        )

    short_score = components.strategy._pause_priority_score(
        short_output,
        components.initial_factors,
        100,
    )
    long_score = components.strategy._pause_priority_score(
        long_output,
        components.initial_factors,
        100,
    )

    assert short_score == long_score == 20


def test_program_segment_clock_spans_status_changes_and_resets_on_pause_resume() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    program = view(
        "program",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        backend_id="backend",
    )
    events = (
        SchedulingEvent(
            event_id="admitted",
            sequence=1,
            kind=SchedulingEventKind.REQUEST_ADMITTED,
            occurred_at_monotonic_s=10,
            reason="admitted",
            program=program.ref,
        ),
        SchedulingEvent(
            event_id="acting",
            sequence=2,
            kind=SchedulingEventKind.PROGRAM_STATUS_CHANGED,
            occurred_at_monotonic_s=20,
            reason="acting",
            program=program.ref,
            current_status=ProgramStatus.ACTING,
        ),
    )
    for event in events:
        components.strategy.handle_scheduling_event(components.initial_factors, event)

    assert components.initial_factors.for_program(program.ref).segment_started_at_monotonic_s == 10

    components.strategy.handle_scheduling_event(
        components.initial_factors,
        SchedulingEvent(
            event_id="paused",
            sequence=3,
            kind=SchedulingEventKind.PROGRAM_PAUSED,
            occurred_at_monotonic_s=30,
            reason="capacity",
            program=program.ref,
        ),
    )
    assert components.initial_factors.for_program(program.ref).segment_started_at_monotonic_s is None

    components.strategy.handle_scheduling_event(
        components.initial_factors,
        SchedulingEvent(
            event_id="resumed",
            sequence=4,
            kind=SchedulingEventKind.PROGRAM_RESUMED,
            occurred_at_monotonic_s=40,
            reason="resume",
            program=program.ref,
        ),
    )
    assert components.initial_factors.for_program(program.ref).segment_started_at_monotonic_s == 40


def test_pause_capacity_counts_marked_reasoning_as_future_relief() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            capacity_safety_margin_tokens=500,
            decode_buffer_tokens=0,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    marked = view(
        "marked",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=600,
        backend_id="backend",
        marked_for_pause=True,
    )
    ordinary = view(
        "ordinary",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    calls: list[TransitionRequest] = []

    components.strategy.repair_capacity(
        snapshot(marked, ordinary, capacity=1000),
        components.initial_factors,
        controller(calls),
    )

    assert calls == []


def test_resume_capacity_does_not_count_future_paused_tokens() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    marked = view(
        "marked",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=600,
        backend_id="backend",
        marked_for_pause=True,
    )
    waiting = view(
        "waiting",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
        tokens=500,
    )
    calls: list[TransitionRequest] = []

    components.strategy.schedule_resume(
        snapshot(marked, waiting, capacity=1000, waiting=(waiting.ref,)),
        components.initial_factors,
        controller(calls),
    )

    assert calls == []


def test_capacity_watermarks_reserve_resume_and_pause_headroom() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            resume_capacity_ratio=0.9,
            pause_capacity_ratio=0.95,
            pause_capacity_lookahead_rounds=2,
        )
    )
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=940,
        backend_id="backend",
    )
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=10)
    admission_calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(active, waiting, capacity=1000),
        components.initial_factors,
        controller(admission_calls),
        waiting.ref,
    )

    assert outcome.disposition is AdmissionDisposition.QUEUED
    repair_calls: list[TransitionRequest] = []
    components.strategy.repair_capacity(
        snapshot(active, capacity=1000),
        components.initial_factors,
        controller(repair_calls),
    )
    assert [call.kind for call in repair_calls] == [TransitionKind.PAUSE]


def test_pause_lookahead_reserve_scales_with_active_program_count() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            pause_capacity_lookahead_rounds=1,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 150
    first = view(
        "first",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    second = view(
        "second",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=400,
        backend_id="backend",
    )
    one_active_calls: list[TransitionRequest] = []
    components.strategy.repair_capacity(
        snapshot(first, capacity=1000),
        components.initial_factors,
        controller(one_active_calls),
    )
    two_active_calls: list[TransitionRequest] = []
    components.strategy.repair_capacity(
        snapshot(first, second, capacity=1000),
        components.initial_factors,
        controller(two_active_calls),
    )

    assert one_active_calls == []
    assert len(two_active_calls) == 1
    assert two_active_calls[0].kind is TransitionKind.PAUSE


def test_ttl_impact_multiplier_scales_avoided_prefill_cost() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            ttl_min_seconds=0,
            ttl_max_seconds=100,
            ttl_prefill_seconds_per_1k_uncached_tokens=1,
            ttl_decode_throughput_alpha=1,
            uncached_ratio_default=1,
            ttl_impact_multiplier=2,
        )
    )
    program = view(
        "program",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        backend_id="backend",
    )
    event = SchedulingEvent(
        event_id="finished",
        sequence=1,
        kind=SchedulingEventKind.REQUEST_FINISHED,
        occurred_at_monotonic_s=100,
        reason="completed",
        program=program.ref,
        current_status=ProgramStatus.ACTING,
        fields=(("total_tokens", 1000), ("previous_context_tokens", 0)),
    )

    components.strategy.handle_scheduling_event(components.initial_factors, event)

    sidecar = components.initial_factors.for_program(program.ref)
    assert sidecar is not None
    assert sidecar.ttl_deadline_monotonic_s == 102


def test_resume_score_uses_rolling_latency_load_and_last_request_gap() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            resume_fairness_weight=1,
            resume_resource_penalty_weight=1,
        )
    )
    program = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(
            last_segment_served_rounds=1,
            last_segment_prompt_tokens=100,
            last_segment_completion_tokens=100,
            wait_started_at_monotonic_s=90,
            last_inter_request_gap_seconds=10,
        ),
    )
    stats = components.initial_factors.global_factors
    stats.avg_prompt_tokens = 100
    stats.avg_completion_tokens = 100
    stats.avg_request_latency_seconds = 2
    stats.avg_active_programs = 4
    stats.avg_waiting_programs = 4

    fast_latency_key = components.strategy._resume_key(program, components.initial_factors, 100)
    stats.avg_request_latency_seconds = 4
    slow_latency_key = components.strategy._resume_key(program, components.initial_factors, 100)

    assert fast_latency_key[1] == -4
    assert slow_latency_key[1] == -1.5
    assert fast_latency_key < slow_latency_key


def test_resume_score_treats_zero_as_a_valid_wait_start() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    program = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(
            last_segment_served_rounds=1,
            last_segment_prompt_tokens=100,
            last_segment_completion_tokens=100,
            wait_started_at_monotonic_s=0,
        ),
    )
    stats = components.initial_factors.global_factors
    stats.avg_prompt_tokens = 100
    stats.avg_completion_tokens = 100
    stats.avg_request_latency_seconds = 1
    stats.avg_active_programs = 1
    stats.avg_waiting_programs = 0

    key = components.strategy._resume_key(program, components.initial_factors, 10)

    assert key[1] == -9


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_event_integer_parser_ignores_non_finite_values(invalid_value: float) -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())

    assert components.strategy._event_int(invalid_value) == 0
