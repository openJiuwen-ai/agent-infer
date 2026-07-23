# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Contract tests for PR 1 scheduling inputs and execution boundaries.

These tests cover generation-safe Program references, token and snapshot validation, lock-protected decision factors, the
single lifecycle callback path, admission outcomes, lock-scoped multi-step transitions, and the optional next-check
deadline. They intentionally use no concrete scheduling policy or runtime implementation, which keeps failures scoped
to the shared contracts introduced by this PR.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentinfer.scheduling import (
    AdmissionDisposition,
    AdmissionOutcome,
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
    SchedulingEvent,
    SchedulingEventKind,
    SchedulingSnapshot,
    SchedulingStrategy,
    StrategyDiagnostic,
    StrategyFactors,
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)

pytestmark = pytest.mark.cpu_test


@dataclass(frozen=True)
class ExampleGlobalFactors:
    """Minimal immutable global decision factors used by contract tests."""

    enabled: bool = True


@dataclass(frozen=True)
class ExampleProgramFactors:
    """Minimal immutable per-Program decision factors used by contract tests."""

    credit: float = 0.0


def make_program(ref: ProgramRef) -> ProgramView:
    """Build one active acting program with a runtime token estimate.

    Args:
        ref: Generation-safe identity assigned to the program view.
    """
    return ProgramView(
        ref=ref,
        state=ProgramState.ACTIVE,
        status=ProgramStatus.ACTING,
        tokens=ProgramTokenObservation(estimated_context_tokens=128),
    )


def make_event(
    kind: SchedulingEventKind,
    ref: ProgramRef,
) -> SchedulingEvent:
    """Build one valid lifecycle event for callback dispatch tests.

    Args:
        kind: Event boundary to dispatch.
        ref: Program generation associated with the event.
    """
    return SchedulingEvent(
        event_id=f"event-{kind.value}",
        sequence=1,
        kind=kind,
        occurred_at_monotonic_s=2.0,
        reason="test",
        program=ref,
    )


def make_snapshot(ref: ProgramRef) -> SchedulingSnapshot:
    """Build one valid single-program scheduling snapshot.

    Args:
        ref: Program generation included in the snapshot.
    """
    return SchedulingSnapshot(
        observed_at_monotonic_s=2.0,
        backend_id="backend-a",
        total_kv_tokens=1024,
        programs=(make_program(ref),),
        waiting_programs=(),
    )


def make_strategy_factors() -> StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors]:
    """Build empty policy-owned decision factors for a locked Hook invocation."""
    return StrategyFactors(global_factors=ExampleGlobalFactors())


class RecordingStrategy(SchedulingStrategy[ExampleGlobalFactors, ExampleProgramFactors]):
    """Strategy test double that records the callback selected by the base dispatcher."""

    def __init__(self) -> None:
        self.callbacks: list[str] = []

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        return AdmissionOutcome(AdmissionDisposition.QUEUED, reason="test")

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        transitions: TransitionController,
    ) -> None:
        return None

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        transitions: TransitionController,
    ) -> None:
        return None

    def on_request_pending(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("pending")

    def on_program_admitted(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("admitted")

    def on_request_finished(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("finished")

    def on_program_status_changed(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("status")

    def on_program_paused(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("paused")

    def on_program_resumed(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("resumed")

    def on_program_marked_for_pause(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("marked")

    def on_program_released(
        self,
        strategy_factors: StrategyFactors[ExampleGlobalFactors, ExampleProgramFactors],
        event: SchedulingEvent,
    ) -> None:
        self.callbacks.append("released")


def test_program_ref_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="program_id"):
        ProgramRef("", 0)
    with pytest.raises(ValueError, match="generation"):
        ProgramRef("program-a", -1)


def test_missing_physical_token_observation_is_not_zero() -> None:
    observation = ProgramTokenObservation(estimated_context_tokens=256)

    assert observation.actual_resident_tokens is None
    assert observation.actual_allocated_blocks is None


def test_token_observations_reject_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ProgramTokenObservation(estimated_context_tokens=-1)


def test_strategy_factors_is_generation_safe() -> None:
    current = ProgramRef("program-a", 2)
    stale = ProgramRef("program-a", 1)
    state = ExampleProgramFactors(credit=1.0)
    strategy_factors = StrategyFactors(
        global_factors=ExampleGlobalFactors(),
        program_factors=((current, state),),
    )

    assert strategy_factors.for_program(current) == state
    assert strategy_factors.for_program(stale) is None


def test_strategy_factors_rejects_duplicate_program_refs() -> None:
    ref = ProgramRef("program-a", 0)
    factors = ExampleProgramFactors()

    with pytest.raises(ValueError, match="duplicate"):
        StrategyFactors(
            global_factors=ExampleGlobalFactors(),
            program_factors=((ref, factors), (ref, factors)),
        )


def test_strategy_factors_applies_locked_updates() -> None:
    first = ProgramRef("program-a", 0)
    second = ProgramRef("program-b", 0)
    original = ExampleProgramFactors(credit=1.0)
    strategy_factors = StrategyFactors(
        global_factors=ExampleGlobalFactors(enabled=True),
        program_factors=((first, original),),
    )

    strategy_factors.set_global_factors(ExampleGlobalFactors(enabled=False))
    strategy_factors.remove_program_factors(first)
    strategy_factors.set_program_factors(second, ExampleProgramFactors(credit=2.0))

    assert strategy_factors.global_factors == ExampleGlobalFactors(enabled=False)
    assert strategy_factors.for_program(first) is None
    assert strategy_factors.for_program(second) == ExampleProgramFactors(credit=2.0)
    assert strategy_factors.program_factors == ((second, ExampleProgramFactors(credit=2.0)),)


def test_snapshot_rejects_unknown_or_duplicate_waiting_programs() -> None:
    ref = ProgramRef("program-a", 0)
    unknown = ProgramRef("program-b", 0)
    common = {
        "observed_at_monotonic_s": 2.0,
        "backend_id": "backend-a",
        "total_kv_tokens": 1024,
        "programs": (make_program(ref),),
    }

    with pytest.raises(ValueError, match="absent"):
        SchedulingSnapshot(waiting_programs=(unknown,), **common)
    with pytest.raises(ValueError, match="duplicate"):
        SchedulingSnapshot(waiting_programs=(ref, ref), **common)


def test_snapshot_indexes_programs_and_rejects_duplicate_references() -> None:
    first = ProgramRef("program-a", 0)
    second = ProgramRef("program-b", 0)
    first_view = make_program(first)
    second_view = make_program(second)
    snapshot = SchedulingSnapshot(
        observed_at_monotonic_s=2.0,
        backend_id="backend-a",
        total_kv_tokens=1024,
        programs=(first_view, second_view),
        waiting_programs=(second,),
    )

    assert snapshot.program(first) is first_view
    assert snapshot.program(second) is second_view
    assert snapshot.program(ProgramRef("unknown", 0)) is None
    with pytest.raises(ValueError, match="duplicate program references"):
        SchedulingSnapshot(
            observed_at_monotonic_s=2.0,
            backend_id="backend-a",
            total_kv_tokens=1024,
            programs=(first_view, first_view),
            waiting_programs=(),
        )


def test_snapshot_detaches_from_mutable_sequence_inputs() -> None:
    first = ProgramRef("program-a", 0)
    second = ProgramRef("program-b", 0)
    first_view = make_program(first)
    programs = [first_view]
    waiting_programs = [first]
    snapshot = SchedulingSnapshot(
        observed_at_monotonic_s=2.0,
        backend_id="backend-a",
        total_kv_tokens=1024,
        programs=programs,
        waiting_programs=waiting_programs,
    )

    programs.append(make_program(second))
    waiting_programs.clear()

    assert snapshot.programs == (first_view,)
    assert snapshot.waiting_programs == (first,)
    assert snapshot.program(first) is first_view
    assert snapshot.program(second) is None


@pytest.mark.parametrize("invalid_timestamp", [float("nan"), float("inf"), float("-inf")])
def test_scheduler_contracts_reject_non_finite_timestamps(invalid_timestamp: float) -> None:
    ref = ProgramRef("program-a", 0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ProgramTokenObservation(estimated_context_tokens=1, observed_at_monotonic_s=invalid_timestamp)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ProgramView(
            ref=ref,
            state=ProgramState.ACTIVE,
            status=ProgramStatus.ACTING,
            tokens=ProgramTokenObservation(estimated_context_tokens=1),
            wait_started_at_monotonic_s=invalid_timestamp,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        SchedulingSnapshot(
            observed_at_monotonic_s=invalid_timestamp,
            backend_id="backend-a",
            total_kv_tokens=1024,
            programs=(),
            waiting_programs=(),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        SchedulingEvent(
            event_id="event-invalid-time",
            sequence=1,
            kind=SchedulingEventKind.CAPACITY_OBSERVED,
            occurred_at_monotonic_s=invalid_timestamp,
            reason="test",
        )


def test_snapshot_rejects_invalid_backend_capacity() -> None:
    ref = ProgramRef("program-a", 0)

    with pytest.raises(ValueError, match="total_kv_tokens"):
        SchedulingSnapshot(
            observed_at_monotonic_s=2.0,
            backend_id="backend-a",
            total_kv_tokens=-1,
            programs=(make_program(ref),),
            waiting_programs=(),
        )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (SchedulingEventKind.REQUEST_PENDING, "pending"),
        (SchedulingEventKind.REQUEST_ADMITTED, "admitted"),
        (SchedulingEventKind.REQUEST_FINISHED, "finished"),
        (SchedulingEventKind.PROGRAM_STATUS_CHANGED, "status"),
        (SchedulingEventKind.PROGRAM_PAUSED, "paused"),
        (SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE, "marked"),
        (SchedulingEventKind.PROGRAM_RESUMED, "resumed"),
        (SchedulingEventKind.PROGRAM_RELEASED, "released"),
    ],
)
def test_lifecycle_event_uses_exactly_one_specific_callback(kind: SchedulingEventKind, expected: str) -> None:
    strategy = RecordingStrategy()
    strategy_factors = make_strategy_factors()

    strategy.handle_scheduling_event(strategy_factors, make_event(kind, ProgramRef("program-a", 0)))

    assert strategy.callbacks == [expected]


@pytest.mark.parametrize(
    "kind",
    [SchedulingEventKind.CAPACITY_OBSERVED, SchedulingEventKind.STRATEGY_FACTORS_CHANGED],
)
def test_observation_and_strategy_factors_events_do_not_invoke_lifecycle_callback(
    kind: SchedulingEventKind,
) -> None:
    strategy = RecordingStrategy()
    strategy_factors = make_strategy_factors()

    strategy.handle_scheduling_event(strategy_factors, make_event(kind, ProgramRef("program-a", 0)))

    assert strategy.callbacks == []


def test_transition_result_rejects_event_for_another_generation() -> None:
    current = ProgramRef("program-a", 2)
    stale_event = make_event(
        SchedulingEventKind.PROGRAM_PAUSED,
        ProgramRef("program-a", 1),
    )

    with pytest.raises(ValueError, match="transitioned program generation"):
        TransitionResult(
            kind=TransitionKind.PAUSE,
            program=current,
            applied=True,
            reason="test",
            event=stale_event,
        )


def test_transition_result_rejects_mismatched_event_kind() -> None:
    ref = ProgramRef("program-a", 0)

    with pytest.raises(ValueError, match="event kind"):
        TransitionResult(
            kind=TransitionKind.PAUSE,
            program=ref,
            applied=True,
            reason="test",
            event=make_event(SchedulingEventKind.PROGRAM_RESUMED, ref),
        )


def test_transition_result_requires_event_only_after_commit() -> None:
    ref = ProgramRef("program-a", 0)

    with pytest.raises(ValueError, match="applied transition"):
        TransitionResult(
            kind=TransitionKind.PAUSE,
            program=ref,
            applied=True,
            reason="test",
        )


def test_scheduling_event_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValueError):
        SchedulingEvent(
            event_id="event-invalid",
            sequence=1,
            kind=SchedulingEventKind.CAPACITY_OBSERVED,
            occurred_at_monotonic_s=2.0,
            reason="test",
            schema_version=0,
        )


def test_program_scoped_event_requires_exact_program_generation() -> None:
    with pytest.raises(ValueError, match="exact program generation"):
        SchedulingEvent(
            event_id="event-missing-program",
            sequence=1,
            kind=SchedulingEventKind.PROGRAM_PAUSED,
            occurred_at_monotonic_s=2.0,
            reason="test",
        )


def test_observation_event_may_omit_program() -> None:
    event = SchedulingEvent(
        event_id="event-capacity",
        sequence=1,
        kind=SchedulingEventKind.CAPACITY_OBSERVED,
        occurred_at_monotonic_s=2.0,
        reason="test",
    )

    assert event.program is None


def test_events_diagnostics_and_outcomes_detach_from_mutable_sequences() -> None:
    ref = ProgramRef("program-a", 0)
    event_fields = [("tokens", 128)]
    diagnostic_fields = [("score", 1.0)]
    transitioned = [ref]
    event = SchedulingEvent(
        event_id="event-fields",
        sequence=1,
        kind=SchedulingEventKind.CAPACITY_OBSERVED,
        occurred_at_monotonic_s=2.0,
        reason="test",
        fields=event_fields,
    )
    diagnostic = StrategyDiagnostic("capacity", "test", diagnostic_fields)
    outcome = AdmissionOutcome(AdmissionDisposition.ADMITTED, "test", transitioned_programs=transitioned)

    event_fields.clear()
    diagnostic_fields.clear()
    transitioned.clear()

    assert event.fields == (("tokens", 128),)
    assert event.field("tokens") == 128
    assert event.field("missing") is None
    assert diagnostic.fields == (("score", 1.0),)
    assert outcome.transitioned_programs == (ref,)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_event_and_diagnostic_fields_reject_non_finite_numbers(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="numeric fields must be finite"):
        SchedulingEvent(
            event_id="event-invalid-field",
            sequence=1,
            kind=SchedulingEventKind.CAPACITY_OBSERVED,
            occurred_at_monotonic_s=2.0,
            reason="test",
            fields=(("value", invalid_value),),
        )
    with pytest.raises(ValueError, match="numeric fields must be finite"):
        StrategyDiagnostic("capacity", "test", (("value", invalid_value),))


def test_admission_outcome_rejects_negative_capacity_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AdmissionOutcome(
            disposition=AdmissionDisposition.QUEUED,
            reason="capacity",
            deficit_tokens=-1,
        )


def test_transition_helpers_preserve_program_generation_reason_and_backend() -> None:
    calls: list[TransitionRequest] = []
    event_kinds = {
        TransitionKind.ADMIT: SchedulingEventKind.REQUEST_ADMITTED,
        TransitionKind.QUEUE: SchedulingEventKind.REQUEST_PENDING,
        TransitionKind.PAUSE: SchedulingEventKind.PROGRAM_PAUSED,
        TransitionKind.MARK_FOR_PAUSE: SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE,
        TransitionKind.RESUME: SchedulingEventKind.PROGRAM_RESUMED,
        TransitionKind.RELEASE: SchedulingEventKind.PROGRAM_RELEASED,
    }

    def apply_transition(request: TransitionRequest) -> TransitionResult:
        calls.append(request)
        return TransitionResult(
            kind=request.kind,
            program=request.program,
            applied=True,
            reason="recorded",
            event=make_event(event_kinds[request.kind], request.program),
        )

    controller = TransitionController(apply_transition=apply_transition)
    ref = ProgramRef("program-a", 0)

    controller.admit(ref, reason="capacity", backend_id="backend-a")
    controller.queue(ref, reason="waiting")
    controller.pause(ref, reason="pressure")
    controller.mark_for_pause(ref, reason="pressure")
    controller.resume(ref, reason="waiting", backend_id="backend-a")
    controller.release(ref, reason="finished")

    assert calls == [
        TransitionRequest(TransitionKind.ADMIT, ref, "capacity", "backend-a"),
        TransitionRequest(TransitionKind.QUEUE, ref, "waiting"),
        TransitionRequest(TransitionKind.PAUSE, ref, "pressure"),
        TransitionRequest(TransitionKind.MARK_FOR_PAUSE, ref, "pressure"),
        TransitionRequest(TransitionKind.RESUME, ref, "waiting", "backend-a"),
        TransitionRequest(TransitionKind.RELEASE, ref, "finished"),
    ]


def test_transition_request_validates_backend_usage() -> None:
    ref = ProgramRef("program-a", 0)

    with pytest.raises(ValueError, match="require backend_id"):
        TransitionRequest(TransitionKind.ADMIT, ref, "capacity")
    with pytest.raises(ValueError, match="valid only"):
        TransitionRequest(TransitionKind.PAUSE, ref, "capacity", "backend-a")


def test_transition_controller_rejects_mismatched_result() -> None:
    ref = ProgramRef("program-a", 0)

    def mismatched_result(request: TransitionRequest) -> TransitionResult:
        return TransitionResult(
            kind=TransitionKind.PAUSE,
            program=request.program,
            applied=False,
            reason="stale",
        )

    controller = TransitionController(apply_transition=mismatched_result)
    with pytest.raises(ValueError, match="does not match"):
        controller.queue(ref, reason="waiting")


def test_strategy_timer_contract_defaults_to_no_deadline_or_transition() -> None:
    strategy = RecordingStrategy()
    ref = ProgramRef("program-a", 0)
    transition_calls: list[TransitionRequest] = []

    def reject_transition(request: TransitionRequest) -> TransitionResult:
        transition_calls.append(request)
        return TransitionResult(
            kind=request.kind,
            program=request.program,
            applied=False,
            reason="test_rejection",
        )

    strategy_factors = make_strategy_factors()
    controller = TransitionController(apply_transition=reject_transition)

    assert strategy.next_check_at(strategy_factors) is None
    strategy.handle_scheduled_check(make_snapshot(ref), strategy_factors, controller)

    assert transition_calls == []
