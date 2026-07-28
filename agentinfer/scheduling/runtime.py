# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Environment-neutral AgentInfer Scheduler with an embedded RequestPool.

``ProgramScheduler`` owns request retention, request-to-Program bindings, mutable Program records, strategy
sidecars, transition execution, and periodic scheduling. Host adapters supply normalized request identity and backend
facts, then decide whether retained objects are coroutines, HTTP gates, or native engine requests. This synchronous
runtime assumes its Adapter serializes calls under one mutation boundary.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, replace
from typing import Generic

from agentinfer.scheduling.admission_outcome import AdmissionDisposition
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DispatchTarget, DpRankInfo
from agentinfer.scheduling.domain import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    SharedPrefixAttribution,
    TokenObservationSource,
)
from agentinfer.scheduling.events import SchedulingEvent, SchedulingEventKind, StrategyDiagnostic
from agentinfer.scheduling.factors import StrategyFactors, StrategyGlobalFactorsT, StrategyProgramFactorsT
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.observability import SchedulerObservabilityConfig
from agentinfer.scheduling.program_registry import ProgramRegistry
from agentinfer.scheduling.program_runtime import RuntimeProgram
from agentinfer.scheduling.request_pool import RequestPool, RequestPoolEntry, RequestPoolStatus, RetainedRequestT
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from agentinfer.scheduling.strategy import SchedulingStrategy
from agentinfer.scheduling.transitions import (
    TRANSITION_EVENT_KINDS,
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class _RequestBinding:
    """Exact Program and token counters associated with one request attempt."""

    program: ProgramRef
    prompt_tokens: int
    previous_context_tokens: int
    started_at_monotonic_s: float
    inter_request_gap_seconds: float
    track_segment_share: bool
    output_tokens: int = 0


class ProgramScheduler(Generic[RetainedRequestT, StrategyGlobalFactorsT, StrategyProgramFactorsT]):
    """Scheduler-owned RequestPool, Program runtime, and strategy execution boundary."""

    def __init__(
        self,
        strategy: SchedulingStrategy[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        strategy_factors: StrategyFactors[StrategyGlobalFactorsT, StrategyProgramFactorsT],
        backend_pool_info: BackendPoolInfo,
        schedule_interval_seconds: float = 1.0,
        observability: SchedulerObservabilityConfig | None = None,
    ) -> None:
        backend, dp_rank = self._embedded_backend_rank(backend_pool_info)
        if not math.isfinite(schedule_interval_seconds) or schedule_interval_seconds <= 0:
            raise ValueError("schedule_interval_seconds must be finite and positive")
        self.strategy = strategy
        self.strategy_factors = strategy_factors
        self.backend_pool_info = backend_pool_info
        self._dispatch_target = DispatchTarget(backend.backend_id, dp_rank.dp_rank)
        self.request_pool: RequestPool[RetainedRequestT] = RequestPool()
        self.registry = ProgramRegistry()
        self._request_bindings: dict[str, _RequestBinding] = {}
        self._program_requests: dict[ProgramRef, set[str]] = {}
        self._last_request_finished_at: dict[ProgramRef, float] = {}
        self._shared_prefix_freshness_anchor_at: dict[ProgramRef, float] = {}
        self._event_sequence = 0
        self._cycle_dirty = False
        self._schedule_interval_seconds = schedule_interval_seconds
        self._observability = observability or SchedulerObservabilityConfig()
        self._next_observability_log_at_monotonic_s = 0.0
        self._next_periodic_check_at_monotonic_s = 0.0
        self._next_check_at_monotonic_s = self._strategy_next_check_at()

    @property
    def retained_request_count(self) -> int:
        """Return attempts still held before native admission."""
        return self.request_pool.unfinished_count

    @property
    def retained_request_ids(self) -> tuple[str, ...]:
        """Return all retained attempts for host-wide cancellation."""
        return tuple(entry.request_id for entry in self.request_pool.entries)

    def prefix_cache_candidates(self) -> tuple[tuple[str, RetainedRequestT], ...]:
        """Return retained paused reasoning requests whose reusable prefix observation should be refreshed."""
        candidates: list[tuple[str, RetainedRequestT]] = []
        now = time.monotonic()
        for entry in self.request_pool.entries:
            binding = self._request_bindings.get(entry.request_id)
            program = self.registry.get(binding.program) if binding is not None else None
            if (
                binding is not None
                and binding.track_segment_share
                and program is not None
                and program.state is ProgramState.PAUSED
                and program.status is ProgramStatus.REASONING
                and self._shared_prefix_observation_due(program, now)
            ):
                candidates.append((entry.request_id, entry.retained_request))
        return tuple(candidates)

    def needs_schedule_cycle(self, now_monotonic_s: float) -> bool:
        """Return whether a host poll must enter the strategy instead of staying on its hot path."""
        if not math.isfinite(now_monotonic_s) or now_monotonic_s < 0:
            raise ValueError("schedule-cycle time must be finite and non-negative")
        deadline_due = (
            self._next_check_at_monotonic_s is not None and self._next_check_at_monotonic_s <= now_monotonic_s
        )
        if deadline_due:
            return True
        if self._next_periodic_check_at_monotonic_s > now_monotonic_s:
            return False
        return self.retained_request_count > 0 or self._cycle_dirty or self.registry.has_programs

    def on_request_arrival(
        self,
        request_id: str,
        metadata: AgentIdentity,
        prompt_tokens: int,
        backend_pool_info: BackendPoolInfo,
        retained_request: RetainedRequestT,
    ) -> bool:
        """Bind, retain, and run immediate Program-level admission."""
        if request_id in self._request_bindings:
            raise ValueError(f"duplicate request_id: {request_id}")
        self._cycle_dirty = True
        self._replace_backend_info(backend_pool_info)
        program, created = self.registry.materialize(metadata)
        previous_context_tokens = program.tokens.estimated_context_tokens
        track_segment_share = created or program.state is ProgramState.PAUSED
        program.tokens = replace(
            program.tokens,
            estimated_context_tokens=max(program.tokens.estimated_context_tokens, max(0, prompt_tokens)),
        )
        previous_status = self.registry.mark_request_started(program.ref)
        now = time.monotonic()
        if created:
            self._shared_prefix_freshness_anchor_at[program.ref] = now
        last_finished_at = self._last_request_finished_at.get(program.ref)
        self._request_bindings[request_id] = _RequestBinding(
            program.ref,
            max(0, prompt_tokens),
            previous_context_tokens,
            now,
            max(0.0, now - last_finished_at) if last_finished_at is not None else 0.0,
            track_segment_share,
        )
        self._program_requests.setdefault(program.ref, set()).add(request_id)
        self.request_pool.add(
            RequestPoolEntry(
                request_id=request_id,
                program=program.ref,
                arrived_at_monotonic_s=now,
                retained_request=retained_request,
            )
        )
        if previous_status is not ProgramStatus.REASONING:
            self._dispatch_event(
                self._event(
                    SchedulingEventKind.PROGRAM_STATUS_CHANGED,
                    program.ref,
                    "request_started",
                    program.state,
                    program.state,
                    previous_status,
                    ProgramStatus.REASONING,
                )
            )
        outcome = self.strategy.handle_admission(
            self._snapshot(exclude_waiting_program=program.ref if created else None),
            self.strategy_factors,
            TransitionController(self._apply_transition),
            program.ref,
        )
        admitted = outcome.disposition is not AdmissionDisposition.QUEUED
        if admitted:
            entry = self.request_pool.get(request_id)
            if entry is not None and entry.status is RequestPoolStatus.WAITING:
                self.request_pool.admit(request_id, self._dispatch_target)
            immediate = self.request_pool.consume_admitted_request(request_id)
            if immediate is None:
                raise RuntimeError("admitted request was not committed to RequestPool.recent_admit")
        if self._observability.enabled:
            logger.info(
                "AgentInfer admission request=%s program=%s generation=%d disposition=%s reason=%s "
                "required_tokens=%d reserve_tokens=%d deficit_tokens=%d retained_requests=%d",
                request_id,
                program.ref.program_id,
                program.ref.generation,
                outcome.disposition.value,
                outcome.reason,
                outcome.required_tokens,
                outcome.reserve_tokens,
                outcome.deficit_tokens,
                self.retained_request_count,
            )
        return admitted

    def schedule_cycle(self, backend_pool_info: BackendPoolInfo) -> None:
        """Run resume, capacity repair, and due TTL checks from fresh backend facts."""
        self._replace_backend_info(backend_pool_info)
        now = time.monotonic()
        if not self.needs_schedule_cycle(now):
            return
        controller = TransitionController(self._apply_transition)
        snapshot = self._snapshot()
        self.strategy.on_schedule_cycle_start(snapshot, self.strategy_factors)
        self._log_periodic_diagnostics(snapshot, now)
        self.strategy.schedule_resume(snapshot, self.strategy_factors, controller)
        snapshot = self._advance_snapshot(snapshot, controller.take_applied_requests())
        self.strategy.repair_capacity(snapshot, self.strategy_factors, controller)
        if self._next_check_at_monotonic_s is not None and self._next_check_at_monotonic_s <= now:
            snapshot = self._advance_snapshot(snapshot, controller.take_applied_requests())
            self.strategy.handle_scheduled_check(snapshot, self.strategy_factors, controller)
        self._cycle_dirty = False
        self._next_periodic_check_at_monotonic_s = now + self._schedule_interval_seconds
        self._next_check_at_monotonic_s = self._strategy_next_check_at()

    def consume_admitted_requests(
        self,
    ) -> tuple[tuple[RequestPoolEntry[RetainedRequestT], DispatchTarget], ...]:
        """Transfer newly admitted retained objects to the host Adapter."""
        return self.request_pool.consume_admitted()

    def on_stream_output(self, request_id: str, total_output_tokens: int) -> None:
        """Update request progress without changing Program reasoning status."""
        binding = self._request_bindings.get(request_id)
        if binding is not None:
            binding.output_tokens = max(binding.output_tokens, max(0, total_output_tokens))

    def on_prefix_cache_observation(self, request_id: str, cached_prefix_tokens: int) -> None:
        """Refresh one paused reasoning Program's reusable prefix hit after its pause-based freshness expires."""
        binding = self._request_bindings.get(request_id)
        if binding is None or not binding.track_segment_share:
            return
        program = self.registry.get(binding.program)
        now = time.monotonic()
        if program is None or not self._shared_prefix_observation_due(program, now):
            return
        observed = min(binding.prompt_tokens, max(0, cached_prefix_tokens))
        snapshot = self._snapshot()
        view = snapshot.program(binding.program)
        freshness_seconds = (
            self.strategy.shared_prefix_freshness_seconds(snapshot, self.strategy_factors, view)
            if view is not None
            else None
        )
        program.tokens = replace(
            program.tokens,
            shared_prefix_tokens=observed,
            shared_prefix_fresh_until_monotonic_s=self._shared_prefix_fresh_until(binding.program, freshness_seconds),
            shared_prefix_attribution=SharedPrefixAttribution.PARTIAL,
            source=TokenObservationSource.MIXED,
        )
        if self._observability.enabled:
            logger.info(
                "AgentInfer shared-prefix observed request=%s program=%s generation=%d cached_prefix_tokens=%d "
                "freshness_seconds=%s freshness_anchor=%s fresh_until=%s",
                request_id,
                binding.program.program_id,
                binding.program.generation,
                observed,
                freshness_seconds,
                self._shared_prefix_freshness_anchor_at.get(binding.program),
                program.tokens.shared_prefix_fresh_until_monotonic_s,
            )

    def on_request_completion(self, request_id: str, total_tokens: int) -> None:
        """Commit final token facts and transition to acting only after overlapping requests finish."""
        binding = self._pop_binding(request_id)
        if binding is None:
            return
        self._cycle_dirty = True
        program = self.registry.get(binding.program)
        if program is None:
            return
        now = time.monotonic()
        self._last_request_finished_at[binding.program] = now
        previous_status = self.registry.record_completion(binding.program, total_tokens=max(0, total_tokens))
        current_status = (
            ProgramStatus.REASONING if self._program_requests.get(binding.program) else ProgramStatus.ACTING
        )
        program.status = current_status
        completion_tokens = max(binding.output_tokens, max(0, total_tokens - binding.prompt_tokens))
        request_latency_seconds = max(0.0, now - binding.started_at_monotonic_s)
        views = self.registry.views()
        active_programs = sum(view.state is ProgramState.ACTIVE for view in views)
        waiting_programs = sum(view.state is ProgramState.PAUSED for view in views)
        self._dispatch_event(
            self._event(
                SchedulingEventKind.REQUEST_FINISHED,
                binding.program,
                "request_completed",
                program.state,
                program.state,
                previous_status,
                current_status,
                fields=(
                    ("total_tokens", max(0, total_tokens)),
                    ("previous_context_tokens", binding.previous_context_tokens),
                    ("prompt_tokens", binding.prompt_tokens),
                    ("completion_tokens", completion_tokens),
                    ("request_latency_seconds", request_latency_seconds),
                    ("active_programs", active_programs),
                    ("waiting_programs", waiting_programs),
                    ("inter_request_gap_seconds", binding.inter_request_gap_seconds),
                    ("shared_prefix_tokens", program.tokens.shared_prefix_tokens),
                ),
            )
        )
        if program.marked_for_pause and current_status is ProgramStatus.ACTING:
            self._apply_transition(TransitionRequest(TransitionKind.PAUSE, program.ref, "marked_pause_completion"))
        else:
            self.strategy.handle_request_completion(
                self._snapshot(),
                self.strategy_factors,
                TransitionController(self._apply_transition),
                program.ref,
            )

    def on_response_completion(self, program_id: str, lifecycle: ProgramLifecycle) -> bool:
        """Apply an optional API-layer lifecycle fact after response semantic parsing.

        Terminal facts release only an idle exact generation. A late signal never interrupts a newer reasoning
        request for the same stable Program id. Continue and unknown facts intentionally leave TTL ownership to the
        scheduling strategy.

        Args:
            program_id: Stable Program identity retained by the API Adapter.
            lifecycle: Protocol-neutral result produced after tool parsing.

        Returns:
            Whether this call released a live Program generation.
        """
        if lifecycle is not ProgramLifecycle.TERMINAL:
            return False
        ref = self.registry.current_ref(program_id)
        if ref is None or self._program_requests.get(ref):
            return False
        program = self.registry.get(ref)
        if program is None or program.status is not ProgramStatus.ACTING:
            return False
        result = self._apply_transition(TransitionRequest(TransitionKind.RELEASE, ref, "api_terminal_response"))
        if result.applied:
            self._cycle_dirty = True
        return result.applied

    def cancel_request(self, request_id: str) -> RequestPoolEntry[RetainedRequestT] | None:
        """Remove a retained attempt and clear its binding without fabricating completion state."""
        entry = self.request_pool.cancel(request_id)
        binding = self._pop_binding(request_id)
        if entry is not None or binding is not None:
            self._cycle_dirty = True
        if binding is not None:
            program = self.registry.get(binding.program)
            if program is not None:
                previous_status = program.status
                program.status = (
                    ProgramStatus.REASONING if self._program_requests.get(binding.program) else ProgramStatus.ACTING
                )
                if previous_status is not program.status:
                    self._dispatch_event(
                        self._event(
                            SchedulingEventKind.PROGRAM_STATUS_CHANGED,
                            program.ref,
                            "request_cancelled",
                            program.state,
                            program.state,
                            previous_status,
                            program.status,
                        )
                    )
                if (
                    not self._program_requests.get(binding.program)
                    and program.state is ProgramState.PAUSED
                    and program.step_count == 0
                ):
                    self._apply_transition(
                        TransitionRequest(TransitionKind.RELEASE, program.ref, "cancelled_before_admission")
                    )
        return entry

    def _apply_transition(self, request: TransitionRequest) -> TransitionResult:
        program = self.registry.get(request.program)
        if program is None:
            return TransitionResult(request.kind, request.program, False, "stale_program")
        previous_state = program.state
        previous_status = program.status
        if request.kind in (TransitionKind.ADMIT, TransitionKind.RESUME):
            if program.state is not ProgramState.PAUSED:
                return TransitionResult(request.kind, request.program, False, f"{request.kind.value}_requires_paused")
            program.state = ProgramState.ACTIVE
            program.backend_id = request.backend_id
            program.marked_for_pause = False
            self._admit_program_requests(program.ref)
        elif request.kind is TransitionKind.QUEUE:
            if program.state is not ProgramState.PAUSED:
                return TransitionResult(request.kind, request.program, False, "queue_requires_paused")
        elif request.kind is TransitionKind.PAUSE:
            if program.state is not ProgramState.ACTIVE or program.status is not ProgramStatus.ACTING:
                return TransitionResult(request.kind, request.program, False, "pause_requires_active_acting")
            program.state = ProgramState.PAUSED
            program.backend_id = None
            program.marked_for_pause = False
            self._restart_shared_prefix_freshness(program, time.monotonic())
        elif request.kind is TransitionKind.MARK_FOR_PAUSE:
            if program.state is not ProgramState.ACTIVE or program.status is not ProgramStatus.REASONING:
                return TransitionResult(request.kind, request.program, False, "mark_requires_active_reasoning")
            if program.marked_for_pause:
                return TransitionResult(request.kind, request.program, False, "already_marked")
            program.marked_for_pause = True
        elif request.kind is TransitionKind.RELEASE:
            if self._program_requests.get(program.ref):
                return TransitionResult(request.kind, request.program, False, "release_requires_idle")
            self.registry.release(program.ref)
            self._last_request_finished_at.pop(program.ref, None)
            self._shared_prefix_freshness_anchor_at.pop(program.ref, None)
        else:
            return TransitionResult(request.kind, request.program, False, "unsupported_transition")
        current_state = ProgramState.TERMINATED if request.kind is TransitionKind.RELEASE else program.state
        event = self._event(
            TRANSITION_EVENT_KINDS[request.kind],
            request.program,
            request.reason,
            previous_state,
            current_state,
            previous_status,
            program.status,
            fields=(("backend_id", request.backend_id), ("marked_for_pause", program.marked_for_pause)),
        )
        self._dispatch_event(event)
        return TransitionResult(request.kind, request.program, True, request.reason, event)

    def _admit_program_requests(self, program: ProgramRef) -> None:
        for entry in self.request_pool.entries:
            if entry.program == program and entry.status is RequestPoolStatus.WAITING:
                self.request_pool.admit(entry.request_id, self._dispatch_target)

    @staticmethod
    def _advance_snapshot(
        snapshot: SchedulingSnapshot,
        requests: tuple[TransitionRequest, ...],
    ) -> SchedulingSnapshot:
        """Derive post-Hook facts from the accepted transition ledger without rereading the Registry."""
        if not requests:
            return snapshot
        programs = {program.ref: program for program in snapshot.programs}
        for request in requests:
            program = programs[request.program]
            if request.kind is TransitionKind.RELEASE:
                programs.pop(request.program)
            elif request.kind in (TransitionKind.ADMIT, TransitionKind.RESUME):
                programs[request.program] = replace(
                    program, state=ProgramState.ACTIVE, backend_id=request.backend_id, marked_for_pause=False
                )
            elif request.kind is TransitionKind.PAUSE:
                programs[request.program] = replace(
                    program, state=ProgramState.PAUSED, backend_id=None, marked_for_pause=False
                )
            elif request.kind is TransitionKind.MARK_FOR_PAUSE:
                programs[request.program] = replace(program, marked_for_pause=True)
        values = tuple(programs.values())
        return replace(
            snapshot,
            programs=values,
            waiting_programs=tuple(program.ref for program in values if program.state is ProgramState.PAUSED),
        )

    def _snapshot(self, *, exclude_waiting_program: ProgramRef | None = None) -> SchedulingSnapshot:
        """Build a snapshot whose waiting view is exactly the live PAUSED Program set.

        A newly materialized admission candidate may be excluded before its first decision so existing paused work
        retains precedence without making every new Program appear already queued. Once a Program has actually been
        queued or paused, its lifecycle state alone keeps it in the waiting view, even when it has no pending request.
        """
        backend, dp_rank = self._embedded_backend_rank(self.backend_pool_info)
        programs = self.registry.views()
        waiting = tuple(
            program.ref
            for program in programs
            if program.state is ProgramState.PAUSED and program.ref != exclude_waiting_program
        )
        return SchedulingSnapshot(
            observed_at_monotonic_s=time.monotonic(),
            backend_id=backend.backend_id,
            total_kv_tokens=dp_rank.total_hbm_kv_tokens,
            programs=programs,
            waiting_programs=waiting,
            native_used_kv_tokens=dp_rank.used_hbm_kv_tokens,
            native_waiting_kv_tokens=dp_rank.waiting_hbm_kv_tokens,
        )

    def _shared_prefix_observation_due(self, program: RuntimeProgram, now_monotonic_s: float) -> bool:
        """Return whether the current pause interval is old enough to refresh shared-prefix evidence."""
        anchor = self._shared_prefix_freshness_anchor_at.get(program.ref)
        fresh_until = program.tokens.shared_prefix_fresh_until_monotonic_s
        return anchor is None or fresh_until is None or now_monotonic_s >= fresh_until

    def _restart_shared_prefix_freshness(self, program: RuntimeProgram, pause_at_monotonic_s: float) -> None:
        """Restart the existing freshness duration from a newly accepted pause.

        The stored deadline belongs to the previous pause anchor. Moving only the anchor would make that stale
        absolute deadline immediately due after a later pause, so preserve its duration while rebasing both values.
        """
        previous_anchor = self._shared_prefix_freshness_anchor_at.get(program.ref)
        previous_deadline = program.tokens.shared_prefix_fresh_until_monotonic_s
        self._shared_prefix_freshness_anchor_at[program.ref] = pause_at_monotonic_s
        if previous_anchor is None or previous_deadline is None:
            return
        freshness_seconds = max(0.0, previous_deadline - previous_anchor)
        program.tokens = replace(
            program.tokens,
            shared_prefix_fresh_until_monotonic_s=pause_at_monotonic_s + freshness_seconds,
        )

    def _shared_prefix_fresh_until(self, program: ProgramRef, freshness_seconds: float | None) -> float | None:
        """Derive a fixed freshness deadline from the latest pause, or initial Program entry."""
        anchor = self._shared_prefix_freshness_anchor_at.get(program)
        if anchor is None or freshness_seconds is None:
            return None
        return anchor + max(0.0, freshness_seconds)

    def _replace_backend_info(self, backend_pool_info: BackendPoolInfo) -> None:
        backend, dp_rank = self._embedded_backend_rank(backend_pool_info)
        if DispatchTarget(backend.backend_id, dp_rank.dp_rank) != self._dispatch_target:
            raise ValueError("embedded backend and DP-rank identity cannot change at runtime")
        self.backend_pool_info = backend_pool_info

    @staticmethod
    def _embedded_backend_rank(backend_pool_info: BackendPoolInfo) -> tuple[BackendInfo, DpRankInfo]:
        """Return the one rank-local capacity domain owned by an embedded Scheduler."""
        if len(backend_pool_info.backends) != 1:
            raise ValueError("embedded ProgramScheduler requires exactly one backend")
        backend = backend_pool_info.backends[0]
        if len(backend.dp_ranks) != 1:
            raise ValueError("embedded ProgramScheduler requires exactly one DP rank")
        return backend, backend.dp_ranks[0]

    def _pop_binding(self, request_id: str) -> _RequestBinding | None:
        binding = self._request_bindings.pop(request_id, None)
        if binding is None:
            return None
        requests = self._program_requests.get(binding.program)
        if requests is not None:
            requests.discard(request_id)
            if not requests:
                self._program_requests.pop(binding.program, None)
        return binding

    def _dispatch_event(self, event: SchedulingEvent) -> None:
        self.strategy.handle_scheduling_event(self.strategy_factors, event)
        if self._observability.enabled:
            fields = " ".join(f"{key}={value}" for key, value in event.fields)
            logger.info(
                "AgentInfer event kind=%s reason=%s program=%s generation=%s previous_state=%s current_state=%s "
                "previous_status=%s current_status=%s fields=%s",
                event.kind.value,
                event.reason,
                event.program.program_id if event.program is not None else None,
                event.program.generation if event.program is not None else None,
                event.previous_state.value if event.previous_state is not None else None,
                event.current_state.value if event.current_state is not None else None,
                event.previous_status.value if event.previous_status is not None else None,
                event.current_status.value if event.current_status is not None else None,
                fields,
            )
            for diagnostic in self.strategy.event_diagnostics(event, self.strategy_factors):
                self._log_diagnostic(diagnostic)
        self._next_check_at_monotonic_s = self._strategy_next_check_at()

    def _log_periodic_diagnostics(self, snapshot: SchedulingSnapshot, now_monotonic_s: float) -> None:
        """Emit strategy calculations no more often than the configured interval."""
        if not self._observability.enabled or now_monotonic_s < self._next_observability_log_at_monotonic_s:
            return
        for diagnostic in self.strategy.diagnostics(snapshot, self.strategy_factors):
            self._log_diagnostic(diagnostic)
        self._next_observability_log_at_monotonic_s = now_monotonic_s + self._observability.log_interval_seconds

    @staticmethod
    def _log_diagnostic(diagnostic: StrategyDiagnostic) -> None:
        """Serialize one enabled diagnostic without changing its typed source record."""
        fields = " ".join(f"{key}={value}" for key, value in diagnostic.fields)
        logger.info(
            "AgentInfer diagnostic name=%s reason=%s %s",
            diagnostic.name,
            diagnostic.reason,
            fields,
        )

    def _strategy_next_check_at(self) -> float | None:
        """Return one strategy deadline after validating the runtime timer contract."""
        deadline = self.strategy.next_check_at(self.strategy_factors)
        if deadline is not None and (not math.isfinite(deadline) or deadline < 0):
            raise ValueError("strategy next-check deadline must be finite and non-negative")
        return deadline

    def _event(
        self,
        kind: SchedulingEventKind,
        program: ProgramRef,
        reason: str,
        previous_state: ProgramState | None,
        current_state: ProgramState | None,
        previous_status: ProgramStatus | None,
        current_status: ProgramStatus | None,
        *,
        fields: tuple[tuple[str, str | int | float | bool | None], ...] = (),
    ) -> SchedulingEvent:
        event = SchedulingEvent(
            event_id=uuid.uuid4().hex,
            sequence=self._event_sequence,
            kind=kind,
            occurred_at_monotonic_s=time.monotonic(),
            reason=reason,
            program=program,
            previous_state=previous_state,
            current_state=current_state,
            previous_status=previous_status,
            current_status=current_status,
            fields=fields,
        )
        self._event_sequence += 1
        return event
