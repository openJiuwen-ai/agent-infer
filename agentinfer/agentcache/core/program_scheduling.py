# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Bridge between API-owned request gates and a Program scheduling runtime."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MoriConfig, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler
from agentinfer.scheduling.transitions import ProgramPlacement, TransitionKind, TransitionRequest

logger = logging.getLogger(__name__)

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


class ProgramCacheAdapter(Protocol):
    """Read effective cache facts and apply cache operations in the local engine.

    Operations set placement or release an exact Program generation. The
    adapter must allow reapplying a batch after a partial failure.
    """

    def observe_backend(self) -> BackendObservation: ...

    def apply_program_ops(self, operations: tuple[ProgramOperation, ...]) -> None: ...


@dataclass(frozen=True)
class _Admission:
    program: ProgramRef
    reply: Future[bool]
    native_count: int


class MoriProgramAdmissionController:
    """Coordinate Program admission, dispatch permits and cache placement on the Core owner."""

    def __init__(self, cache: ProgramCacheAdapter, config: MoriConfig, backend_id: str = "vllm-local") -> None:
        self.cache = cache
        self.config = config
        self.backend_id = backend_id
        self._pending: dict[str, _Admission] = {}
        self._ready: dict[str, tuple[ProgramRef, int]] = {}
        self._operations: list[ProgramOperation] = []
        self._inactive: set[ProgramRef] = set()
        self._admitted: dict[str, None] = {}
        self.observation = cache.observe_backend()
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
        self.observation = self.cache.observe_backend()
        self._backend = self._backend_pool()

    def begin(
        self, attempt: str, program_id: str, context_chars: int, prefix_group: str, native_count: int = 1
    ) -> Future[bool]:
        if (
            not attempt
            or not isinstance(program_id, str)
            or not program_id.strip()
            or context_chars < 0
            or not prefix_group
            or native_count < 1
        ):
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
        self._pending[attempt] = _Admission(ref, reply, native_count)
        if admitted:
            self._admitted[attempt] = None
        self._flush_and_publish()
        return reply

    def attach(self, attempt: str) -> bool:
        """Consume the one-shot native dispatch permit."""
        admission = self._ready.pop(attempt, None)
        if admission is None:
            return False
        ref, remaining = admission
        if remaining > 1:
            self._ready[attempt] = (ref, remaining - 1)
        return True

    def progress(self, attempt: str, delta: int) -> None:
        if delta <= 0:
            raise ValueError("MORI progress must be positive")
        self.runtime.on_response_progress(attempt, delta)

    def finish(self, attempt: str, total: int | None, prompt: int, cached: int) -> None:
        self._ready.pop(attempt, None)
        if attempt in self._pending:
            self.cancel(attempt)
            return
        self.runtime.on_reported_completion(attempt, total, prompt, cached)
        self._flush_and_publish()

    def _cancel_attempt(self, attempt: str) -> None:
        self._ready.pop(attempt, None)
        self._admitted.pop(attempt, None)
        pending = self._pending.pop(attempt, None)
        self.runtime.cancel_request(attempt)
        if pending is not None:
            pending.reply.set_result(False)

    def cancel(self, attempt: str) -> None:
        self._cancel_attempt(attempt)
        self._flush_and_publish()

    def cancel_gates(self) -> None:
        """Resolve pending API waiters when the native scheduler is stopping."""
        for attempt in (*self._pending, *self._ready):
            self._cancel_attempt(attempt)

    def tick(self) -> bool:
        prior = self._flush_operations()
        if prior is None:
            return False
        self.runtime.run_host_check(self._backend)
        applied = self._flush_and_publish()
        return bool(prior or applied)

    def release(self, program_id: str) -> bool:
        ref = self.runtime.registry.current_ref(program_id)
        if ref is None:
            return False

        # ThunderAgent removes a Program from its queues and releases its cache.
        # Commit the local release after acknowledgement so this caller can retry
        # a failed release, and leave already-granted native dispatches valid.
        if any(op.program == ref for op in self._operations) and self._flush_operations() is None:
            return False
        factors = self.runtime.strategy_factors.global_factors
        if ref in factors.active or ref in factors.warm or ref in self._inactive:
            if not self._send_operations((ProgramOperation(ref, None),)):
                return False
            self._inactive.discard(ref)

        # All calls run on the Core owner. release_program cancels this generation's
        # runtime bindings before it commits RELEASE, including admitted dispatches.
        # Keep both those bindings and the generation intact until the cache ACK.
        if not self.runtime.release_program(ref):
            return False
        for attempt, admission in tuple(self._pending.items()):
            if admission.program == ref:
                self._pending.pop(attempt)
                self._admitted.pop(attempt, None)
                admission.reply.set_result(False)
        self._flush_and_publish()
        return True

    def _on_transition(self, transition: TransitionRequest) -> None:
        if transition.placement is not None:
            self._operations.append(ProgramOperation(transition.program, transition.placement))
            if transition.placement is ProgramPlacement.INACTIVE:
                self._inactive.add(transition.program)
            else:
                self._inactive.discard(transition.program)
        elif transition.kind is TransitionKind.RELEASE and transition.program in self._inactive:
            # Cancellation of the last pending attempt can retire an inactive Program.
            self._inactive.remove(transition.program)
            self._operations.append(ProgramOperation(transition.program, None))

    def _flush_and_publish(self) -> int | None:
        applied = self._flush_operations()
        if applied is None:
            return None
        for entry, _ in self.runtime.consume_admitted_requests():
            self._admitted[entry.request_id] = None
        for attempt in self._admitted:
            admission = self._pending.pop(attempt, None)
            if admission is not None:
                self._ready[attempt] = (admission.program, admission.native_count)
                admission.reply.set_result(True)
        self._admitted.clear()
        return applied

    def _flush_operations(self) -> int | None:
        if not self._operations:
            return 0
        operations = tuple(self._operations)
        if not self._send_operations(operations):
            return None
        self._operations.clear()
        return len(operations)

    def _send_operations(self, operations: tuple[ProgramOperation, ...]) -> bool:
        try:
            self.cache.apply_program_ops(operations)
        except Exception:
            logger.warning("MORI Program cache operation was not acknowledged", exc_info=True)
            return False
        return True
