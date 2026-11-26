# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""ThunderAgent MORI Program rules for one embedded capacity domain."""

from agentinfer.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from agentinfer.scheduling.domain import ProgramRef, ProgramStatus
from agentinfer.scheduling.events import SchedulingEvent
from agentinfer.scheduling.mori.config import MoriConfig
from agentinfer.scheduling.mori.factors import MoriFactors, MoriGlobalFactors, MoriProgramFactors
from agentinfer.scheduling.mori.ledger import MoriLedger
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from agentinfer.scheduling.strategy import SchedulingStrategy
from agentinfer.scheduling.transitions import ProgramPlacement, TransitionController, TransitionKind, TransitionRequest


class MoriStrategy(SchedulingStrategy[MoriGlobalFactors, MoriProgramFactors]):
    """Place Programs using ThunderAgent arrival, usage, and periodic rules."""

    observes_request_arrivals = True
    host_driven_checks = True
    resolve_marked_pause = True

    def __init__(self, config: MoriConfig) -> None:
        self.config = config

    def on_request_arrived(self, factors: MoriFactors, event: SchedulingEvent) -> None:
        """Port ThunderAgent update_program_before_request bookkeeping to an API arrival event."""
        ref = event.program
        program = factors.for_program(ref)
        if program is None:
            program = MoriProgramFactors(prefix_group=str(event.field("prefix_group") or ""))
            factors.set_program_factors(ref, program)
        program.steps += 1
        context_chars = event.field("context_chars")
        program.context_chars = int(context_chars or 0)
        program.tokens = (
            int(program.context_chars / factors.global_factors.char_to_token_ratio)
            if context_chars is not None
            else max(0, int(event.field("prompt_tokens")))
        )
        program.start_reasoning(event.occurred_at_monotonic_s)

    def on_request_progress(self, factors: MoriFactors, event: SchedulingEvent) -> None:
        """Port ThunderAgent update_program_tokens_streaming to the shared event stream."""
        program = factors.for_program(event.program)
        if program is not None:
            program.tokens += int(event.field("delta"))

    def on_request_finished(self, factors: MoriFactors, event: SchedulingEvent) -> None:
        """Port ThunderAgent usage accounting, including its first cached sample per Program."""
        program = factors.for_program(event.program)
        if program is None:
            return
        if event.current_status is ProgramStatus.ACTING:
            program.finish_reasoning(event.occurred_at_monotonic_s)
        total = event.field("reported_total_tokens")
        if total is None:
            return
        program.tokens = int(total)
        prompt = int(event.field("prompt_tokens"))
        state = factors.global_factors
        if prompt > 0 and program.context_chars > 0:
            sample = program.context_chars / prompt
            state.char_to_token_ratio = (
                0.2 * sample + 0.8 * state.char_to_token_ratio if state.ratio_initialized else sample
            )
            state.ratio_initialized = True
        if program.prefix_group and not program.sampled_prefix:
            program.sampled_prefix = True
            cached = int(event.field("cached_prefix_tokens"))
            if cached > state.prefix_lengths.get(program.prefix_group, 0):
                state.prefix_lengths[program.prefix_group] = cached

    def on_program_admitted(self, factors: MoriFactors, event: SchedulingEvent) -> None:
        """Mirror ThunderAgent _promote_to_running after the shared Runtime commits admission."""
        ref = event.program
        program = factors.for_program(ref)
        state = factors.global_factors
        if ref not in state.active:
            state.active[ref] = None
            state.warm.pop(ref, None)
            state.waiting.pop(ref, None)
        program.reasoning_since = event.occurred_at_monotonic_s

    on_program_resumed = on_program_admitted

    def on_program_paused(self, factors: MoriFactors, event: SchedulingEvent) -> None:
        """Mirror ThunderAgent warm and waiting queue changes on a committed transition."""
        ref = event.program
        program = factors.for_program(ref)
        state = factors.global_factors
        state.active.pop(ref, None)
        state.warm.pop(ref, None)
        state.waiting.pop(ref, None)
        program.future_credit = None
        if event.field("placement") == ProgramPlacement.ASYNC.value:
            state.warm[ref] = program.tokens
        else:
            state.waiting[ref] = None

    def on_program_released(self, factors: MoriFactors, event: SchedulingEvent) -> None:
        """Drop the MORI indexes when the shared Runtime releases a Program."""
        state = factors.global_factors
        state.active.pop(event.program, None)
        state.warm.pop(event.program, None)
        state.waiting.pop(event.program, None)
        factors.remove_program_factors(event.program)

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        factors: MoriFactors,
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        """Port ThunderAgent arrival promotion and new-Program selection to one rank.

        The Runtime owns queue and admission state, so this hook requests typed
        transitions at the original decision points.
        """
        ledger = MoriLedger(snapshot, factors)
        state = factors.global_factors
        program = factors.for_program(candidate)
        required = ledger.required(candidate)
        if candidate in state.active:
            program.reasoning_since = snapshot.observed_at_monotonic_s
            return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_active", required)
        if candidate in state.warm and snapshot.total_kv_tokens is not None and ledger.hbm_room >= required:
            self._resume(ledger, transitions, candidate, "mori_warm_arrival")
            return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_warm_arrival", required)
        waiting_returner_fits = any(
            ref != candidate
            and snapshot.program(ref).status is ProgramStatus.REASONING
            and (ref in state.warm or factors.for_program(ref).steps > 1)
            and ledger.required(ref) <= ledger.hbm_room
            for ref in (*state.warm, *state.waiting)
        )
        if (
            candidate not in state.warm
            and candidate not in state.waiting
            and snapshot.backend_healthy
            and snapshot.total_kv_tokens is not None
            and ledger.hbm_room >= required
            and not waiting_returner_fits
        ):
            self._apply(transitions, TransitionKind.ADMIT, candidate, "mori_new", snapshot.backend_id)
            return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_new", required)
        if candidate not in state.warm:
            state.waiting.setdefault(candidate, None)
        self._apply(transitions, TransitionKind.QUEUE, candidate, "mori_queued")
        return AdmissionOutcome(AdmissionDisposition.QUEUED, "mori_queued", required)

    def schedule_resume(self, snapshot, strategy_factors, transitions) -> None:
        """Resume runs inside the ordered MORI tick."""

    def repair_capacity(self, snapshot, strategy_factors, transitions) -> None:
        """Capacity repair runs inside the ordered MORI tick."""

    def handle_request_completion(self, snapshot, strategy_factors, transitions, completed) -> None:
        """Port ThunderAgent _clear_mark_and_pause and _place_off_gpu to Runtime transitions."""
        program = strategy_factors.for_program(completed)
        if program.future_credit is None or snapshot.program(completed).status is not ProgramStatus.ACTING:
            return
        self._apply(transitions, TransitionKind.CLEAR_PAUSE_MARK, completed, "mori_complete_mark")
        program.future_credit = None
        ledger = MoriLedger(snapshot, strategy_factors)
        if ledger.dram_room >= program.tokens:
            self._pause(transitions, completed, ProgramPlacement.ASYNC, "mori_complete_warm")
        else:
            room = ledger.hbm_room
            freed = room + ledger.required(completed)
            if room < 0 or any(room < need <= freed for need in self._parked_demand(ledger)):
                self._pause(transitions, completed, ProgramPlacement.INACTIVE, "mori_complete_waiting")

    def handle_scheduled_check(self, snapshot, strategy_factors, transitions) -> None:
        """Port ThunderAgent _scheduled_check with one rank and local capacity observations."""
        if snapshot.total_kv_tokens is None:
            return
        ledger = MoriLedger(snapshot, strategy_factors)
        now = snapshot.observed_at_monotonic_s
        ledger.update_shared_prefix(now)
        state = strategy_factors.global_factors
        for ref in (*state.warm, *state.waiting):
            program = strategy_factors.for_program(ref)
            if self._reasoning(ledger, ref) and program.reasoning_since is None and program.context_chars > 0:
                program.tokens = int(program.context_chars / state.char_to_token_ratio)
        self._displace_idle(ledger, transitions, now)
        self._admit_parked(ledger, transitions, now)
        self._repair_hbm(ledger, transitions, now)
        self._repair_dram(ledger, transitions, now)

    def _displace_idle(self, ledger: MoriLedger, transitions: TransitionController, now: float) -> None:
        """Port ThunderAgent _step_idle_displacement, including its DRAM prerequisite and quota."""
        demand = self._parked_demand(ledger)
        if not demand:
            return
        need = max(demand)
        candidates = []
        for ref in ledger.factors.global_factors.active:
            program = ledger.factors.for_program(ref)
            if self._reasoning(ledger, ref) or program.acting_since is None or program.future_credit is not None:
                continue
            acting = max(0.0, now - program.acting_since)
            reasoning = program.reasoning_history[-1] if program.reasoning_history else 0.0
            score = acting / (acting + reasoning) if acting + reasoning > 0 else 0.0
            if score >= 0.5:
                candidates.append((ref, score))
        quota = max(1, len(candidates) // 2)
        for ref, _ in sorted(candidates, key=lambda candidate: -candidate[1]):
            if quota == 0 or ledger.hbm_room >= need:
                break
            if ledger.dram_room >= ledger.factors.for_program(ref).tokens:
                self._pause(transitions, ref, ProgramPlacement.ASYNC, "mori_idle_displacement")
                quota -= 1

    def _admit_parked(self, ledger: MoriLedger, transitions: TransitionController, now: float) -> None:
        """Port the two ordered ThunderAgent WARM and WAITING admission phases."""
        state = ledger.factors.global_factors
        warm = sorted(
            (ref for ref in state.warm if self._reasoning(ledger, ref)),
            key=lambda ref: ledger.factors.for_program(ref).idleness(now),
        )
        returning: list[ProgramRef] = []
        new: list[ProgramRef] = []
        for ref in state.waiting:
            if self._reasoning(ledger, ref):
                (new if ledger.factors.for_program(ref).steps <= 1 else returning).append(ref)
        returning.sort(key=lambda ref: ledger.factors.for_program(ref).idleness(now))
        new.sort(key=ledger.required)
        for candidates, reason in ((warm, "mori_admit_warm"), (returning + new, "mori_admit_waiting")):
            for ref in candidates:
                if ledger.hbm_room >= ledger.required(ref):
                    self._resume(ledger, transitions, ref, reason)

    def _repair_hbm(self, ledger: MoriLedger, transitions: TransitionController, now: float) -> None:
        """Port ThunderAgent GPU overflow and victim selection with deferred HBM credit."""
        state = ledger.factors.global_factors
        while ledger.hbm_room + ledger.future_release < 0:
            candidates = [
                ref
                for ref in state.active
                if not self._reasoning(ledger, ref) or ledger.factors.for_program(ref).future_credit is None
            ]
            if not candidates:
                break
            ref = min(
                candidates,
                key=lambda ref: (self._reasoning(ledger, ref), -ledger.factors.for_program(ref).idleness(now)),
            )
            program = ledger.factors.for_program(ref)
            if self._reasoning(ledger, ref):
                self._apply(transitions, TransitionKind.MARK_FOR_PAUSE, ref, "mori_hbm_deferred")
                program.future_credit = program.tokens
                continue
            placement = ProgramPlacement.ASYNC if ledger.dram_room >= program.tokens else ProgramPlacement.INACTIVE
            self._pause(transitions, ref, placement, "mori_hbm_overflow")

    def _repair_dram(self, ledger: MoriLedger, transitions: TransitionController, now: float) -> None:
        """Port ThunderAgent CPU overflow by updating the already parked Program's placement."""
        state = ledger.factors.global_factors
        while ledger.dram_room < 0 and state.warm:
            ref = min(
                state.warm,
                key=lambda ref: (self._reasoning(ledger, ref), -ledger.factors.for_program(ref).idleness(now)),
            )
            self._pause(transitions, ref, ProgramPlacement.INACTIVE, "mori_dram_overflow")

    @staticmethod
    def _reasoning(ledger: MoriLedger, ref: ProgramRef) -> bool:
        return ledger.snapshot.program(ref).status is ProgramStatus.REASONING

    def _parked_demand(self, ledger: MoriLedger) -> list[int]:
        state = ledger.factors.global_factors
        return [ledger.required(ref) for ref in (*state.warm, *state.waiting) if self._reasoning(ledger, ref)]

    def _resume(self, ledger, transitions, ref, reason) -> None:
        """Emit the local equivalent of a ThunderAgent INTERACTIVE mark."""
        self._apply(
            transitions, TransitionKind.RESUME, ref, reason, ledger.snapshot.backend_id, ProgramPlacement.INTERACTIVE
        )

    def _pause(self, transitions, ref, placement, reason) -> None:
        """Carry ThunderAgent ASYNC or INACTIVE placement on the shared pause transition."""
        self._apply(transitions, TransitionKind.PAUSE, ref, reason, placement=placement)

    @staticmethod
    def _apply(transitions, kind, ref, reason, backend_id=None, placement=None) -> None:
        result = transitions.apply(TransitionRequest(kind, ref, reason, backend_id, placement))
        if not result.applied:
            raise RuntimeError(f"MORI transition {kind.value} rejected for {ref}: {result.reason}")
