# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Bridge between API-owned request gates and a Program scheduling runtime."""

from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MoriConfig, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler
from agentinfer.scheduling.transitions import ProgramPlacement, TransitionRequest

ATTEMPT_ARG = "agentinfer_mori_attempt"


@dataclass(frozen=True)
class BackendObservation:
    """Effective capacities and health already computed by the host."""

    healthy: bool
    hbm_capacity_tokens: int | None
    dram_capacity_tokens: int | None


@dataclass(frozen=True)
class ProgramOperation:
    """Place a Program or release it when placement is absent."""

    program: ProgramRef
    placement: ProgramPlacement | None


class ProgramSchedulingHost(Protocol):
    """Scheduler-side capability supplied by the serving backend."""

    def observe_backend(self) -> BackendObservation: ...

    def apply_program_ops(self, operations: tuple[ProgramOperation, ...]) -> None: ...


@dataclass(frozen=True)
class _Admission:
    program: ProgramRef
    reply: Future[bool]
    native_count: int


class ProgramSchedulingBridge:
    """Run one ProgramScheduler on the EngineCore owner thread."""

    def __init__(self, host: ProgramSchedulingHost, config: MoriConfig, backend_id: str = "vllm-local") -> None:
        self.host = host
        self.config = config
        self.backend_id = backend_id
        self._pending: dict[str, _Admission] = {}
        self._ready: dict[str, tuple[ProgramRef, int]] = {}
        self._operations: list[ProgramOperation] = []
        self._failure: Exception | None = None
        self.observation = host.observe_backend()
        self._backend = self._backend_pool()
        components = build_mori_strategy(config)
        self.runtime = ProgramScheduler(
            components.strategy, components.initial_factors, self._backend, on_transition=self._on_transition
        )

    def _backend_pool(self) -> BackendPoolInfo:
        facts = self.observation
        now = time.monotonic()
        return BackendPoolInfo(
            (
                BackendInfo(
                    self.backend_id,
                    "embedded://vllm",
                    facts.healthy,
                    (
                        DpRankInfo(
                            0,
                            facts.healthy,
                            facts.hbm_capacity_tokens is not None,
                            total_hbm_kv_tokens=facts.hbm_capacity_tokens,
                            total_dram_kv_tokens=facts.dram_capacity_tokens,
                        ),
                    ),
                    now,
                ),
            ),
            observed_at_monotonic_s=now,
        )

    def observe_backend(self) -> None:
        self.observation = self.host.observe_backend()
        self._backend = self._backend_pool()

    def begin(
        self, attempt: str, program_id: str, context_chars: int, prefix_group: str, native_count: int = 1
    ) -> Future[bool]:
        self._check_operations()
        if not attempt or context_chars < 0 or not prefix_group or native_count < 1:
            raise ValueError("Invalid MORI admission facts")
        if attempt in self._pending or attempt in self._ready:
            raise ValueError("Duplicate MORI attempt")
        reply: Future[bool] = Future()
        admitted = self.runtime.on_request_arrival(
            attempt,
            AgentIdentity(program_id),
            0,
            self._backend,
            reply,
            context_chars=context_chars,
            prefix_group=prefix_group,
        )
        ref = self.runtime.registry.current_ref(program_id)
        assert ref is not None
        self._apply_operations()
        if admitted:
            self._ready[attempt] = (ref, native_count)
            reply.set_result(True)
        else:
            self._pending[attempt] = _Admission(ref, reply, native_count)
        return reply

    def attach(self, attempt: str) -> bool:
        """Consume the one-shot native dispatch permit."""
        self._check_operations()
        admission = self._ready.pop(attempt, None)
        if admission is None:
            return False
        ref, remaining = admission
        if remaining > 1:
            self._ready[attempt] = (ref, remaining - 1)
        return True

    def progress(self, attempt: str, delta: int) -> None:
        self._check_operations()
        if delta <= 0:
            raise ValueError("MORI progress must be positive")
        self.runtime.on_response_progress(attempt, delta)

    def finish(self, attempt: str, total: int | None, prompt: int, cached: int) -> None:
        self._check_operations()
        self._ready.pop(attempt, None)
        if attempt in self._pending:
            self.cancel(attempt)
            return
        self.runtime.on_reported_completion(attempt, total, prompt, cached)
        self._apply_operations()

    def cancel(self, attempt: str) -> None:
        self._ready.pop(attempt, None)
        pending = self._pending.pop(attempt, None)
        self.runtime.cancel_request(attempt)
        self._apply_operations()
        if pending is not None:
            pending.reply.set_result(False)

    def cancel_gates(self) -> None:
        """Resolve pending API waiters when the native scheduler is stopping."""
        for attempt in (*self._pending, *self._ready):
            self.cancel(attempt)

    def tick(self) -> bool:
        self._check_operations()
        self.runtime.run_host_check(self._backend)
        applied = self._apply_operations()
        for entry, _ in self.runtime.consume_admitted_requests():
            admission = self._pending.pop(entry.request_id, None)
            if admission is not None:
                self._ready[entry.request_id] = (admission.program, admission.native_count)
                admission.reply.set_result(True)
        return applied

    def release(self, program_id: str) -> bool:
        self._check_operations()
        ref = self.runtime.registry.current_ref(program_id)
        if ref is None:
            return False
        pending = [(attempt, entry) for attempt, entry in self._pending.items() if entry.program == ref]
        for attempt, (program, _) in tuple(self._ready.items()):
            if program == ref:
                del self._ready[attempt]
        active = self.runtime.strategy_factors.global_factors.active
        warm = self.runtime.strategy_factors.global_factors.warm
        if ref in active or ref in warm:
            self._operations.append(ProgramOperation(ref, None))
        released = self.runtime.release_program(ref)
        self._apply_operations()
        for attempt, entry in pending:
            del self._pending[attempt]
            entry.reply.set_result(False)
        return released

    def _on_transition(self, transition: TransitionRequest) -> None:
        if transition.placement is not None:
            self._operations.append(ProgramOperation(transition.program, transition.placement))

    def _apply_operations(self) -> bool:
        if not self._operations:
            return False
        operations = tuple(self._operations)
        self._operations.clear()
        try:
            self.host.apply_program_ops(operations)
        except Exception as error:
            # The host may have applied part of the batch. Stop at the first
            # failed acknowledgement so no later request sees a false commit.
            self._failure = error
            self._ready.clear()
            for attempt, pending in tuple(self._pending.items()):
                self.runtime.cancel_request(attempt)
                pending.reply.set_exception(error)
            self._pending.clear()
            raise
        return True

    def _check_operations(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "MORI Program operation failed; the scheduling host must be restarted"
            ) from self._failure
