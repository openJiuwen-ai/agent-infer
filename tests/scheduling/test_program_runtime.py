# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Tests for generation-safe Programs, retained requests, and runtime lifecycle."""

from dataclasses import dataclass

import pytest

from agentinfer.scheduling import (
    AdmissionDisposition,
    AdmissionOutcome,
    AgentIdentity,
    BackendInfo,
    BackendPoolInfo,
    DispatchTarget,
    DpRankInfo,
    ProgramLifecycle,
    ProgramRef,
    ProgramRegistry,
    ProgramScheduler,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
    RequestPool,
    RequestPoolEntry,
    RuntimeProgram,
    SchedulerObservabilityConfig,
    SchedulingSnapshot,
    SchedulingStrategy,
    StaleProgramReferenceError,
    StrategyDiagnostic,
    StrategyFactors,
    TransitionController,
)

pytestmark = pytest.mark.cpu_test


@dataclass
class _GlobalFactors:
    """Global decision factors containing committed lifecycle events for runtime tests."""

    event_kinds: list[str]


@dataclass(frozen=True)
class _ProgramFactors:
    """Unused per-Program decision factors required by the strategy generic."""


class _AdmissionStrategy(SchedulingStrategy[_GlobalFactors, _ProgramFactors]):
    """Deterministic strategy that either admits or queues every candidate."""

    def __init__(self, *, admit: bool) -> None:
        self.admit = admit
        self.completed_programs: list[ProgramRef] = []
        self.diagnostic_calls = 0

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        if self.admit:
            transitions.admit(candidate, reason="test_admit", backend_id=snapshot.backend_id)
            return AdmissionOutcome(AdmissionDisposition.ADMITTED, "test_admit")
        transitions.queue(candidate, reason="test_queue")
        return AdmissionOutcome(AdmissionDisposition.QUEUED, "test_queue")

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
    ) -> None:
        return None

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
    ) -> None:
        return None

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        self.completed_programs.append(completed)

    def diagnostics(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
    ) -> tuple[StrategyDiagnostic, ...]:
        self.diagnostic_calls += 1
        return (StrategyDiagnostic("test_state", "periodic_sample", (("programs", len(snapshot.programs)),)),)

    def handle_scheduling_event(self, strategy_factors, event) -> None:
        strategy_factors.global_factors.event_kinds.append(event.kind.value)
        super().handle_scheduling_event(strategy_factors, event)


class _InvalidDeadlineStrategy(_AdmissionStrategy):
    """Strategy fixture that violates the optional timer contract."""

    def next_check_at(self, strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors]) -> float:
        return float("nan")


class _ScheduledCheckStrategy(_AdmissionStrategy):
    """Strategy fixture that records timer callbacks and optionally requests release."""

    def __init__(self, deadline: float | None, *, release_on_check: bool = False) -> None:
        super().__init__(admit=True)
        self.deadline = deadline
        self.release_on_check = release_on_check
        self.check_count = 0
        self.release_result: tuple[bool, str] | None = None

    def next_check_at(self, strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors]) -> float | None:
        return self.deadline

    def handle_scheduled_check(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
    ) -> None:
        self.check_count += 1
        if self.release_on_check:
            result = transitions.release(snapshot.programs[0].ref, reason="test_scheduled_release")
            self.release_result = (result.applied, result.reason)


class _SnapshotLedgerStrategy(_AdmissionStrategy):
    """Exercise state propagation across all three synchronous cycle Hooks."""

    def __init__(self) -> None:
        super().__init__(admit=False)
        self.phase_views: list[ProgramView] = []

    def next_check_at(self, strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors]) -> float:
        return 0.0

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
    ) -> None:
        self.phase_views.append(snapshot.programs[0])
        transitions.resume(snapshot.programs[0].ref, reason="test_resume", backend_id=snapshot.backend_id)

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
    ) -> None:
        self.phase_views.append(snapshot.programs[0])
        transitions.mark_for_pause(snapshot.programs[0].ref, reason="test_mark")

    def handle_scheduled_check(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: StrategyFactors[_GlobalFactors, _ProgramFactors],
        transitions: TransitionController,
    ) -> None:
        self.phase_views.append(snapshot.programs[0])


def _backend() -> BackendPoolInfo:
    """Build one embedded backend observation."""
    return BackendPoolInfo(
        (
            BackendInfo(
                backend_id="vllm-local",
                backend_url="embedded://vllm",
                healthy=True,
                dp_ranks=(DpRankInfo(3, True, True, total_hbm_kv_tokens=4096),),
            ),
        )
    )


def _scheduler(*, admit: bool) -> ProgramScheduler[str, _GlobalFactors, _ProgramFactors]:
    """Build a Program runtime with deterministic admission."""
    return ProgramScheduler(
        _AdmissionStrategy(admit=admit),
        StrategyFactors(_GlobalFactors([])),
        _backend(),
    )


def test_request_pool_consumes_exact_request_to_target_decision() -> None:
    pool: RequestPool[str] = RequestPool()
    entry = RequestPoolEntry("request-a", ProgramRef("program-a", 0), 1.0, "native")
    target = DispatchTarget("backend-a")

    pool.add(entry)
    assert pool.admit(entry.request_id, target) is True
    assert pool.consume_admitted() == ((entry, target),)
    assert pool.unfinished_count == 0


@pytest.mark.parametrize("invalid_timestamp", [float("nan"), float("inf"), float("-inf")])
def test_runtime_boundaries_reject_non_finite_monotonic_times(invalid_timestamp: float) -> None:
    with pytest.raises(ValueError, match="request arrival time"):
        RequestPoolEntry("request-a", ProgramRef("program-a", 0), invalid_timestamp, "native")
    with pytest.raises(ValueError, match="program wait start"):
        RuntimeProgram(
            ref=ProgramRef("program-a", 0),
            state=ProgramState.PAUSED,
            status=ProgramStatus.ACTING,
            tokens=ProgramTokenObservation(estimated_context_tokens=0),
            wait_started_at_monotonic_s=invalid_timestamp,
        )

    scheduler = _scheduler(admit=True)
    with pytest.raises(ValueError, match="schedule-cycle time"):
        scheduler.needs_schedule_cycle(invalid_timestamp)


def test_runtime_rejects_non_finite_strategy_deadline() -> None:
    with pytest.raises(ValueError, match="strategy next-check deadline"):
        ProgramScheduler(
            _InvalidDeadlineStrategy(admit=True),
            StrategyFactors(_GlobalFactors([])),
            _backend(),
        )


def test_observability_is_disabled_by_default(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    scheduler = _scheduler(admit=True)

    scheduler.on_request_arrival("r1", AgentIdentity("p1"), 100, _backend(), "native")

    assert "AgentInfer admission" not in caplog.text
    assert "AgentInfer event" not in caplog.text


def test_observability_logs_events_and_throttles_periodic_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    strategy = _AdmissionStrategy(admit=True)
    scheduler = ProgramScheduler(
        strategy,
        StrategyFactors(_GlobalFactors([])),
        _backend(),
        observability=SchedulerObservabilityConfig(enabled=True, log_interval_seconds=30),
    )
    scheduler.on_request_arrival("r1", AgentIdentity("p1"), 100, _backend(), "native")
    snapshot = scheduler._snapshot()

    scheduler._log_periodic_diagnostics(snapshot, 10)
    scheduler._log_periodic_diagnostics(snapshot, 20)
    scheduler._log_periodic_diagnostics(snapshot, 40)

    assert "AgentInfer admission request=r1 program=p1" in caplog.text
    assert "retained_requests=0" in caplog.text
    assert "AgentInfer event kind=request_admitted" in caplog.text
    assert "fields=backend_id=vllm-local marked_for_pause=False" in caplog.text
    assert caplog.text.count("AgentInfer diagnostic name=test_state") == 2
    assert strategy.diagnostic_calls == 2


def test_observability_reports_queued_request_as_retained(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    scheduler = ProgramScheduler(
        _AdmissionStrategy(admit=False),
        StrategyFactors(_GlobalFactors([])),
        _backend(),
        observability=SchedulerObservabilityConfig(enabled=True),
    )

    admitted = scheduler.on_request_arrival("r1", AgentIdentity("p1"), 100, _backend(), "native")

    assert admitted is False
    assert scheduler.retained_request_count == 1
    assert "disposition=queued" in caplog.text
    assert "retained_requests=1" in caplog.text


def test_registry_retains_child_before_parent_edge_and_generation() -> None:
    registry = ProgramRegistry()
    child, _ = registry.materialize(
        AgentIdentity(
            "child",
            task_id="task-a",
            parent_program_id="parent",
            blocks_parent=True,
            expected_resume=False,
        )
    )
    parent, _ = registry.materialize(AgentIdentity("parent", task_id="task-a"))

    assert parent.blocked_on_child is True
    assert registry.children(parent.ref) == (child.ref,)
    registry.release(child.ref)
    assert parent.blocked_on_child is False

    registry.release(parent.ref)
    replacement, _ = registry.materialize(AgentIdentity("parent", task_id="task-a"))
    assert replacement.ref.generation == parent.ref.generation + 1
    with pytest.raises(StaleProgramReferenceError):
        registry.require(parent.ref)


def test_runtime_admits_and_completes_without_success_semantics() -> None:
    scheduler = _scheduler(admit=True)

    admitted = scheduler.on_request_arrival("request-a", AgentIdentity("program-a"), 100, _backend(), "native")
    scheduler.on_request_completion("request-a", 120)

    ref = scheduler.registry.current_ref("program-a")
    assert admitted is True
    assert ref is not None
    assert scheduler.registry.require(ref).state is ProgramState.ACTIVE
    assert scheduler.registry.require(ref).status is ProgramStatus.ACTING
    assert scheduler.retained_request_count == 0
    assert scheduler.strategy.completed_programs == [ref]


def test_runtime_uses_the_single_dp_rank_capacity_and_dispatch_target() -> None:
    scheduler = _scheduler(admit=False)

    assert scheduler._snapshot().total_kv_tokens == 4096
    assert scheduler._dispatch_target == DispatchTarget("vllm-local", 3)


def test_runtime_rejects_multiple_dp_rank_domains() -> None:
    backend = BackendPoolInfo(
        (
            BackendInfo(
                backend_id="vllm-local",
                backend_url="embedded://vllm",
                healthy=True,
                dp_ranks=(DpRankInfo(0, True, True), DpRankInfo(1, True, True)),
            ),
        )
    )

    with pytest.raises(ValueError, match="exactly one DP rank"):
        ProgramScheduler(_AdmissionStrategy(admit=True), StrategyFactors(_GlobalFactors([])), backend)


def test_cancel_queued_request_releases_never_admitted_program() -> None:
    scheduler = _scheduler(admit=False)

    assert scheduler.on_request_arrival("request-a", AgentIdentity("program-a"), 100, _backend(), "native") is False
    entry = scheduler.cancel_request("request-a")

    assert entry is not None and entry.retained_request == "native"
    assert scheduler.registry.current_ref("program-a") is None
    assert scheduler.retained_request_count == 0


def test_terminal_response_releases_only_idle_current_generation() -> None:
    scheduler = _scheduler(admit=True)
    identity = AgentIdentity("program-a")

    assert scheduler.on_request_arrival("request-a", identity, 100, _backend(), "native") is True
    assert scheduler.on_response_completion(identity.program_id, ProgramLifecycle.TERMINAL) is False
    scheduler.on_request_completion("request-a", 120)
    assert scheduler.on_response_completion(identity.program_id, ProgramLifecycle.CONTINUE) is False
    assert scheduler.on_response_completion(identity.program_id, ProgramLifecycle.TERMINAL) is True
    assert scheduler.registry.current_ref(identity.program_id) is None


@pytest.mark.parametrize("deadline", [None, 1.0e20])
def test_periodic_cycle_does_not_run_scheduled_check_before_deadline(deadline: float | None) -> None:
    strategy = _ScheduledCheckStrategy(deadline)
    scheduler = ProgramScheduler(strategy, StrategyFactors(_GlobalFactors([])), _backend())

    assert scheduler.on_request_arrival("request-a", AgentIdentity("program-a"), 100, _backend(), "native") is True
    scheduler.schedule_cycle(_backend())

    assert strategy.check_count == 0


def test_due_scheduled_check_cannot_release_program_with_inflight_request() -> None:
    factors = StrategyFactors(_GlobalFactors([]))
    strategy = _ScheduledCheckStrategy(0.0, release_on_check=True)
    scheduler = ProgramScheduler(strategy, factors, _backend())

    assert scheduler.on_request_arrival("request-a", AgentIdentity("program-a"), 100, _backend(), "native") is True
    ref = scheduler.registry.current_ref("program-a")
    scheduler.schedule_cycle(_backend())

    assert strategy.check_count == 1
    assert strategy.release_result == (False, "release_requires_idle")
    assert scheduler.registry.current_ref("program-a") == ref
    scheduler.on_request_completion("request-a", 120)
    assert "request_finished" in factors.global_factors.event_kinds
    assert ref is not None and scheduler.registry.require(ref).status is ProgramStatus.ACTING
    assert scheduler.on_response_completion("program-a", ProgramLifecycle.TERMINAL) is True


def test_cycle_derives_each_phase_snapshot_from_applied_transition_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _SnapshotLedgerStrategy()
    scheduler = ProgramScheduler(strategy, StrategyFactors(_GlobalFactors([])), _backend())
    assert scheduler.on_request_arrival("request-a", AgentIdentity("program-a"), 100, _backend(), "native") is False
    original_views = scheduler.registry.views
    view_calls = 0

    def counted_views() -> tuple[ProgramView, ...]:
        """Count full Registry snapshot materializations during one cycle."""
        nonlocal view_calls
        view_calls += 1
        return original_views()

    monkeypatch.setattr(scheduler.registry, "views", counted_views)
    scheduler.schedule_cycle(_backend())

    resume_view, repair_view, check_view = strategy.phase_views
    assert resume_view.state is ProgramState.PAUSED
    assert repair_view.state is ProgramState.ACTIVE
    assert repair_view.marked_for_pause is False
    assert check_view.state is ProgramState.ACTIVE
    assert check_view.marked_for_pause is True
    assert view_calls == 1
