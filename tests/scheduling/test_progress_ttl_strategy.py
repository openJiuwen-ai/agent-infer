# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Unit tests for Progress-TTL decision factors, deadlines, and capacity decisions."""

from __future__ import annotations

import math
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
    ProgressTTLMode,
    ProgressTTLProgramFactors,
    ProgressTTLResumeOrder,
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


def complete_growth_window(
    factors: StrategyFactors[ProgressTTLGlobalFactors, ProgressTTLProgramFactors], rounds: int
) -> None:
    """Fill the request window with deterministic samples before testing workload-derived capacity lookahead."""
    stats = factors.global_factors
    for _ in range(stats.request_window_size):
        stats.update_request(
            prompt_tokens=100,
            cached_prefix_tokens=0,
            completion_tokens=10,
            total_tokens=110,
            request_latency_seconds=1,
            active_programs=1,
            waiting_programs=0,
            input_token_growth=10,
            inter_request_gap_seconds=1,
            rounds_since_ttl_pause=rounds,
        )


def test_shared_prefix_freshness_uses_configured_global_kv_pool_turnovers() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(shared_prefix_freshness_warmup_seconds=100, shared_prefix_freshness_kv_turnovers=2)
    )
    factors = components.initial_factors
    factors.global_factors.request_window_size = 1
    factors.global_factors.update_request(
        prompt_tokens=1000,
        cached_prefix_tokens=600,
        completion_tokens=200,
        total_tokens=1200,
        request_latency_seconds=4,
        active_programs=2,
        waiting_programs=0,
        input_token_growth=200,
        inter_request_gap_seconds=1,
    )
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=8000)
    running_one = view("running-one", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, backend_id="backend")
    running_two = view("running-two", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, backend_id="backend")

    freshness = components.strategy.shared_prefix_freshness_seconds(
        snapshot(candidate, running_one, running_two), factors, candidate
    )

    assert factors.global_factors.avg_cache_churn_tokens_per_round == 600
    assert freshness == pytest.approx(2 * 1000 * 4 / 2 / 600)


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
        ProgressTTLConfig(target_max_segment_rounds=0)
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
            resume_capacity_ratio=1,
        )
    )
    complete_growth_window(components.initial_factors, rounds=1)
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


def test_batch_gain_can_consume_growth_reserve_but_not_current_capacity() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            resume_capacity_ratio=1,
            enable_batch_gain_admission=True,
        )
    )
    complete_growth_window(components.initial_factors, rounds=1)
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 10
    components.initial_factors.global_factors.avg_decode_seconds = 1
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=100,
        backend_id="backend",
    )
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(active, candidate, capacity=225),
        components.initial_factors,
        controller(calls),
        candidate.ref,
    )

    assert outcome.disposition is AdmissionDisposition.ADMITTED
    assert outcome.reason == "batch_gain_over_recovery_cost"
    assert [call.kind for call in calls] == [TransitionKind.ADMIT]

    no_capacity_calls: list[TransitionRequest] = []
    no_capacity = components.strategy.handle_admission(
        snapshot(active, candidate, capacity=175),
        components.initial_factors,
        controller(no_capacity_calls),
        candidate.ref,
    )

    assert no_capacity.disposition is AdmissionDisposition.QUEUED
    assert [call.kind for call in no_capacity_calls] == [TransitionKind.QUEUE]


def test_batch_gain_decode_surface_counts_complete_context_for_each_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    active = replace(
        view(
            "active-shared",
            state=ProgramState.ACTIVE,
            status=ProgramStatus.REASONING,
            tokens=1000,
            backend_id="backend",
        ),
        tokens=ProgramTokenObservation(estimated_context_tokens=1000, shared_prefix_tokens=800),
    )
    candidate = replace(
        view("candidate-shared", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=600),
        tokens=ProgramTokenObservation(estimated_context_tokens=600, shared_prefix_tokens=500),
    )
    throughput_inputs: list[tuple[int, int]] = []

    def decode_throughput(batch_size: int, total_context_tokens: int) -> float:
        throughput_inputs.append((batch_size, total_context_tokens))
        return float(batch_size)

    monkeypatch.setattr(components.strategy, "_decode_throughput", decode_throughput)
    monkeypatch.setattr(components.strategy, "_cache_recovery_impact_seconds", lambda *args: 0.0)
    monkeypatch.setattr(components.strategy, "_continuity_loss_seconds", lambda *args: 0.0)

    assert components.strategy._batch_gain_covers_recovery(
        (active,),
        components.initial_factors,
        candidate,
        remaining_tokens=1000,
    )
    assert throughput_inputs == [(1, 1000), (2, 1600)]


def test_batch_gain_accounts_for_active_continuity_rounds_lost_to_candidate_growth() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    complete_growth_window(components.initial_factors, rounds=2)
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 100
    active = view(
        "active",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=100,
        backend_id="backend",
    )
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)

    constrained = components.strategy._continuity_loss_seconds(
        (active,),
        components.initial_factors,
        candidate,
        remaining_tokens=600,
    )
    unconstrained = components.strategy._continuity_loss_seconds(
        (active,),
        components.initial_factors,
        candidate,
        remaining_tokens=1000,
    )

    assert constrained > 0
    assert unconstrained == 0


def test_continuity_loss_uses_post_admission_rounds_and_average_recovery_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    complete_growth_window(components.initial_factors, rounds=2)
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 100
    reasoning = view(
        "reasoning",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=100,
        backend_id="backend",
    )
    acting = view(
        "acting",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=200,
        backend_id="backend",
    )
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    monkeypatch.setattr(
        components.strategy,
        "_cache_recovery_impact_seconds",
        lambda state, context_tokens, shared_prefix_tokens: context_tokens / 50,
    )

    loss = components.strategy._continuity_loss_seconds(
        (reasoning, acting),
        components.initial_factors,
        candidate,
        remaining_tokens=700,
    )
    no_post_admission_rounds = components.strategy._continuity_loss_seconds(
        (reasoning, acting),
        components.initial_factors,
        candidate,
        remaining_tokens=500,
    )

    assert loss == pytest.approx((7 - 2) / 2 * ((2 + 4) / 2))
    assert no_post_admission_rounds == math.inf


def test_fixed_growth_overrides_rolling_growth_for_capacity_projection() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            use_fixed_input_token_growth=True,
            fixed_input_token_growth_per_round=25,
        )
    )
    complete_growth_window(components.initial_factors, rounds=1)
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


def test_capacity_diagnostic_reports_rolling_growth_but_projects_effective_fixed_growth() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
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
    components.initial_factors.set_program_factors(
        active.ref,
        ProgressTTLProgramFactors(segment_served_rounds=1),
    )
    complete_growth_window(components.initial_factors, rounds=1)
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 999

    diagnostic = components.strategy.diagnostics(
        snapshot(active),
        components.initial_factors,
    )[0]
    fields = dict(diagnostic.fields)

    assert fields["rolling_growth"] == 999
    assert fields["capacity_growth"] == 25
    assert fields["remaining_growth_rounds"] == 1
    assert fields["projected_active_reserve"] == 25


def test_cold_request_window_uses_zero_growth_lookahead_for_all_programs() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 100
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)

    non_privileged_reserve = components.strategy._continuous_growth_reserve_tokens(
        (),
        components.initial_factors,
        candidate=candidate,
    )
    privileged_reserve = components.strategy._continuous_growth_reserve_tokens(
        (),
        components.initial_factors,
        candidate=candidate,
    )

    assert non_privileged_reserve == 0
    assert privileged_reserve == 0


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
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    waiting = view("waiting", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=900)
    components.initial_factors.set_program_factors(
        waiting.ref,
        ProgressTTLProgramFactors(
            wait_started_at_monotonic_s=10,
            request_wait_started_at_monotonic_s=60,
            force_resume_timeout_seconds=30,
            force_resume_deadline_monotonic_s=90,
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
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    acting = view("acting", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=900)
    components.initial_factors.set_program_factors(
        acting.ref,
        ProgressTTLProgramFactors(
            wait_started_at_monotonic_s=10,
            request_wait_started_at_monotonic_s=60,
            force_resume_timeout_seconds=30,
            force_resume_deadline_monotonic_s=90,
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
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    reasoning = view("reasoning", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    acting = view("acting", state=ProgramState.PAUSED, status=ProgramStatus.ACTING)
    components.initial_factors.set_program_factors(
        acting.ref,
        ProgressTTLProgramFactors(
            wait_started_at_monotonic_s=10,
            request_wait_started_at_monotonic_s=10,
            force_resume_timeout_seconds=30,
            force_resume_deadline_monotonic_s=40,
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
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
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
    assert components.initial_factors.for_program(parent_waiting.ref).privilege_deadline_monotonic_s == pytest.approx(
        100.0 + components.strategy.config.privileged_ttl_seconds
    )

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


def test_all_programs_use_the_workload_growth_target_in_shared_capacity_projection() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    complete_growth_window(components.initial_factors, rounds=2)
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

    assert reserve == 700


def test_capacity_growth_target_is_bounded_by_max_segment_rounds() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            target_max_segment_rounds=3,
            decode_buffer_tokens=0,
        )
    )
    complete_growth_window(components.initial_factors, rounds=10)
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
        ProgressTTLProgramFactors(segment_served_rounds=1),
    )

    reserve = components.strategy._continuous_growth_reserve_tokens(
        (active,),
        components.initial_factors,
        candidate=candidate,
    )

    assert components.initial_factors.global_factors.estimated_continuity_rounds() == 20
    assert components.strategy._protected_min_segment_rounds(components.initial_factors) == 3
    assert reserve == 500


def test_request_completion_counts_round_and_warmup_ttl_expires_immediately() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            target_max_segment_rounds=2,
            ttl_min_seconds=5,
            ttl_max_seconds=5,
            ttl_max_cache_miss_impact_ratio=1,
            ttl_prefill_model_intercept_seconds=0,
            ttl_prefill_model_linear_seconds_per_1k_tokens=6,
            ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared=0,
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
    assert sidecar.ttl_deadline_monotonic_s == 100.0
    calls: list[TransitionRequest] = []
    due = replace(snapshot(program), observed_at_monotonic_s=100.0)
    components.strategy.handle_scheduled_check(due, components.initial_factors, controller(calls))
    assert [(call.kind, call.reason) for call in calls] == [(TransitionKind.PAUSE, "progress_ttl_expired")]


def test_request_completion_counts_ref_zero_churn_only_on_first_segment_round() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    program = view(
        "program",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        backend_id="backend",
    )

    for sequence in (1, 2):
        finished = SchedulingEvent(
            event_id=f"finished-{sequence}",
            sequence=sequence,
            kind=SchedulingEventKind.REQUEST_FINISHED,
            occurred_at_monotonic_s=100.0 + sequence,
            reason="completed",
            program=program.ref,
            current_status=ProgramStatus.ACTING,
            fields=(
                ("total_tokens", 1100),
                ("previous_context_tokens", 1000),
                ("prompt_tokens", 1000),
                ("completion_tokens", 100),
                ("cached_prefix_tokens", 800),
                ("hbm_cached_prefix_tokens", 800),
                ("hbm_hit_ref_zero_tokens", 600),
            ),
        )
        components.strategy.handle_scheduling_event(components.initial_factors, finished)

    samples = tuple(components.initial_factors.global_factors._request_samples)
    assert samples[0].cache_churn_tokens == 900
    assert samples[1].cache_churn_tokens == 300


def test_request_completion_exposes_armed_ttl_diagnostic() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            ttl_min_seconds=5,
            ttl_max_seconds=5,
            ttl_prefill_model_intercept_seconds=0,
            ttl_prefill_model_linear_seconds_per_1k_tokens=10,
            ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared=0,
            ttl_decode_throughput_alpha=1,
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
    diagnostics = components.strategy.event_diagnostics(finished, components.initial_factors)

    assert len(diagnostics) == 1
    assert diagnostics[0].name == "progress_ttl_armed"
    fields = dict(diagnostics[0].fields)
    assert fields["program"] == "program"
    assert fields["generation"] == 0
    assert fields["total_tokens"] == 1000
    assert fields["acting_since"] == 100.0
    assert fields["ttl_seconds"] == 0.0
    assert fields["ttl_deadline"] == 100.0
    assert fields["is_privileged"] is False


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
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
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
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0, target_max_segment_rounds=3))
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
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0, target_max_segment_rounds=3))
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


def test_adaptive_cache_miss_impact_uses_uncached_prompt_tokens() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            ttl_min_seconds=0,
            ttl_max_seconds=100,
            ttl_prefill_model_intercept_seconds=0,
            ttl_prefill_model_linear_seconds_per_1k_tokens=1,
            ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared=0,
            ttl_decode_throughput_alpha=1,
        )
    )
    assert components.strategy._cache_miss_impact_seconds(1000, 0) == 1
    assert components.strategy._cache_miss_impact_seconds(1000, 400) == 0.6


def test_cold_prefill_cost_uses_quadratic_calibration() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            ttl_prefill_model_intercept_seconds=1,
            ttl_prefill_model_linear_seconds_per_1k_tokens=2,
            ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared=3,
            ttl_decode_throughput_alpha=1,
        )
    )

    assert components.strategy._cache_miss_impact_seconds(2000, 0) == 17
    assert components.strategy._cache_miss_impact_seconds(2000, 2000) == 0


def test_warmup_ttl_zero_overrides_configured_minimum() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            ttl_min_seconds=10,
            ttl_max_seconds=100,
            ttl_max_cache_miss_impact_ratio=0.5,
            ttl_prefill_model_intercept_seconds=0,
            ttl_prefill_model_linear_seconds_per_1k_tokens=1,
            ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared=0,
            ttl_decode_throughput_alpha=1,
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
        fields=(("prompt_tokens", 1000), ("total_tokens", 1000), ("previous_context_tokens", 0)),
    )

    components.strategy.handle_scheduling_event(components.initial_factors, event)

    sidecar = components.initial_factors.for_program(program.ref)
    assert sidecar is not None
    assert sidecar.ttl_deadline_monotonic_s == 100


def test_off_mode_directly_admits_without_capacity_or_queueing() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(mode=ProgressTTLMode.OFF, decode_buffer_tokens=0))
    program = view("queued", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    calls: list[TransitionRequest] = []

    outcome = components.strategy.handle_admission(
        snapshot(program, capacity=1),
        components.initial_factors,
        controller(calls),
        program.ref,
    )

    assert outcome.disposition is AdmissionDisposition.ADMITTED
    assert outcome.reason == "progress_ttl_off_direct"
    assert [call.kind for call in calls] == [TransitionKind.ADMIT]


def test_off_mode_warns_when_paused_reasoning_resume_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(mode=ProgressTTLMode.OFF))
    program = view("paused", state=ProgramState.PAUSED, status=ProgramStatus.REASONING)
    calls: list[TransitionRequest] = []

    def reject_transition(request: TransitionRequest) -> TransitionResult:
        calls.append(request)
        return TransitionResult(
            kind=request.kind,
            program=request.program,
            applied=False,
            reason="resume_requires_paused",
        )

    components.strategy.schedule_resume(
        snapshot(program),
        components.initial_factors,
        TransitionController(reject_transition),
    )

    assert [call.kind for call in calls] == [TransitionKind.RESUME]
    assert "Progress-TTL transition rejected" in caplog.text
    assert "resume_requires_paused" in caplog.text


def test_off_mode_records_due_ttl_without_transitioning_program_state() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(mode=ProgressTTLMode.OFF, ttl_min_seconds=1))
    program = view("acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, backend_id="backend")
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(
            rounds_since_ttl_pause=5,
            acting_since_monotonic_s=1,
            ttl_deadline_monotonic_s=10,
        ),
    )
    calls: list[TransitionRequest] = []

    components.strategy.handle_scheduled_check(
        snapshot(program),
        components.initial_factors,
        controller(calls),
    )

    factors = components.initial_factors.for_program(program.ref)
    assert factors is not None
    assert calls == []
    assert factors.rounds_since_ttl_pause == 0
    assert factors.pause_reason == "progress_ttl_theoretical_expired"
    assert factors.ttl_expiry_observed is True


def test_native_running_usage_is_combined_with_share_adjusted_acting_reservations() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    reasoning = view(
        "reasoning",
        state=ProgramState.ACTIVE,
        status=ProgramStatus.REASONING,
        tokens=900,
        backend_id="backend",
    )
    acting = replace(
        view("acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=500, backend_id="backend"),
        tokens=ProgramTokenObservation(estimated_context_tokens=500, shared_prefix_tokens=200),
    )
    observed = SchedulingSnapshot(
        observed_at_monotonic_s=100,
        backend_id="backend",
        total_kv_tokens=1000,
        native_used_kv_tokens=600,
        native_waiting_kv_tokens=250,
        programs=(reasoning, acting),
        waiting_programs=(),
    )

    assert components.strategy._remaining_tokens(observed, 1.0) == -150


def test_resume_order_prefers_the_most_recent_request_completion() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    recent = view("recent", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    old = view("old", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    components.initial_factors.set_program_factors(
        recent.ref,
        ProgressTTLProgramFactors(
            last_request_finished_at_monotonic_s=90,
            last_inter_request_gap_seconds=1,
        ),
    )
    components.initial_factors.set_program_factors(
        old.ref,
        ProgressTTLProgramFactors(
            last_request_finished_at_monotonic_s=10,
            last_inter_request_gap_seconds=1000,
        ),
    )
    ordered = sorted(
        (old, recent),
        key=lambda program: components.strategy._resume_key(program, components.initial_factors, 100),
    )

    assert ordered == [recent, old]


def test_fcfs_resume_order_prefers_the_earliest_waiting_request() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(resume_order=ProgressTTLResumeOrder.FCFS))
    first = view("first", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    second = view("second", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)
    components.initial_factors.set_program_factors(
        first.ref,
        ProgressTTLProgramFactors(request_wait_started_at_monotonic_s=10, last_request_finished_at_monotonic_s=90),
    )
    components.initial_factors.set_program_factors(
        second.ref,
        ProgressTTLProgramFactors(request_wait_started_at_monotonic_s=20, last_request_finished_at_monotonic_s=10),
    )
    ordered = sorted(
        (second, first),
        key=lambda program: components.strategy._resume_key(program, components.initial_factors, 100),
    )

    assert ordered == [first, second]


def test_privileged_resume_bypasses_unknown_capacity() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())
    program = view(
        "privileged",
        state=ProgramState.PAUSED,
        status=ProgramStatus.REASONING,
        tokens=1000,
        task_id="task",
    )
    components.initial_factors.set_program_factors(
        program.ref,
        ProgressTTLProgramFactors(is_privileged=True),
    )
    calls: list[TransitionRequest] = []
    unknown_capacity = replace(snapshot(program, waiting=(program.ref,)), total_kv_tokens=None)

    components.strategy.schedule_resume(unknown_capacity, components.initial_factors, controller(calls))

    assert [(call.kind, call.reason) for call in calls] == [(TransitionKind.RESUME, "progress_ttl_privileged_resume")]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_event_integer_parser_ignores_non_finite_values(invalid_value: float) -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig())

    assert components.strategy._event_int(invalid_value) == 0
