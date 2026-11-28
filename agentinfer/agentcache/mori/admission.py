# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Retain native vLLM ADDs until their logical Chat is admitted by MORI."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from agentinfer.agentcache.mori.ipc import (
    ATTEMPT_ARG,
    CHILD_COUNT_ARG,
    CONTEXT_CHARS_ARG,
    PREFIX_ARG,
    PREPARATION_NS_ARG,
    PROGRAM_ARG,
)
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MoriConfig, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler
from agentinfer.scheduling.transitions import ProgramPlacement, TransitionKind, TransitionRequest

if TYPE_CHECKING:
    from vllm.v1.request import Request, RequestStatus

logger = logging.getLogger(__name__)


class InvalidMoriAdmission(ValueError):
    """The native child cannot join this logical Chat attempt."""


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


@dataclass
class _Admission:
    """One Chat arrival and its native sampling children, all on the Core owner."""

    program: ProgramRef
    context_chars: int
    prefix_group: str
    child_count: int | None
    preparation_ns: int | None
    waiting: dict[str, tuple[Request, Callable[[Request], None]]]
    seen: set[str] = field(default_factory=set)
    admitted: bool = False
    cancelled: bool = False


class MoriProgramAdmissionController:
    """Hand native children to vLLM after their Program placement is acknowledged."""

    def __init__(self, cache: ProgramCacheAdapter, config: MoriConfig, backend_id: str = "vllm-local") -> None:
        self.cache = cache
        self.config = config
        self.backend_id = backend_id
        self._attempts: dict[str, _Admission] = {}
        self._aborted: list[Request] = []
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

    def add_request(self, request: Request, native_add: Callable[[Request], None]) -> None:
        params = request.sampling_params
        extra = (params.extra_args or {}) if params is not None else {}
        request._mori_program_ref = None
        if ATTEMPT_ARG not in extra:
            native_add(request)
            return
        attempt = extra[ATTEMPT_ARG]
        program_id = extra.get(PROGRAM_ARG)
        context_chars = extra.get(CONTEXT_CHARS_ARG)
        prefix_group = extra.get(PREFIX_ARG)
        child_count = extra.get(CHILD_COUNT_ARG)
        preparation_ns = extra.get(PREPARATION_NS_ARG)
        if (
            not isinstance(attempt, str)
            or not attempt
            or not isinstance(program_id, str)
            or not program_id.strip()
            or type(context_chars) is not int
            or context_chars < 0
            or (preparation_ns is not None and (type(preparation_ns) is not int or not 0 <= preparation_ns < 1 << 63))
            or not isinstance(prefix_group, str)
            or not prefix_group
            or (child_count is not None and (type(child_count) is not int or child_count < 1))
        ):
            raise InvalidMoriAdmission("Invalid MORI Chat admission metadata")

        group = self._attempts.get(attempt)
        if group is None:
            waiting: dict[str, tuple[Request, Callable[[Request], None]]] = {}
            admitted = self.runtime.on_request_arrival(
                attempt,
                AgentIdentity(program_id),
                request.num_prompt_tokens,
                self._backend,
                waiting,
                context_chars=context_chars,
                prefix_group=prefix_group,
                frontend_preparation_ns=preparation_ns,
            )
            ref = self.runtime.registry.current_ref(program_id)
            assert ref is not None
            group = self._attempts[attempt] = _Admission(
                ref, context_chars, prefix_group, child_count, preparation_ns, waiting
            )
            if admitted:
                self._admitted[attempt] = None
        elif (
            group.program.program_id != program_id
            or group.context_chars != context_chars
            or group.prefix_group != prefix_group
            or group.child_count != child_count
            or group.preparation_ns != preparation_ns
            or request.request_id in group.seen
            or (child_count is not None and len(group.seen) >= child_count)
        ):
            raise InvalidMoriAdmission("Inconsistent or duplicate MORI sampling child")

        group.seen.add(request.request_id)
        if group.cancelled:
            raise InvalidMoriAdmission("MORI sampling group was cancelled before admission")
        if group.admitted:
            request._mori_program_ref = group.program
            native_add(request)
            return
        group.waiting[request.request_id] = (request, native_add)
        self._flush_and_publish()

    def progress(self, attempt: str, delta: int) -> None:
        if delta <= 0:
            raise ValueError("MORI progress must be positive")
        self.runtime.on_response_progress(attempt, delta)

    def finish(self, attempt: str, total: int | None, prompt: int, cached: int) -> None:
        group = self._attempts.get(attempt)
        if group is None:
            return
        if not group.admitted:
            self.cancel(attempt)
            return
        del self._attempts[attempt]
        self.runtime.on_reported_completion(attempt, total, prompt, cached)
        self._flush_and_publish()

    def _cancel_waiting(self, attempt: str, group: _Admission) -> None:
        self._admitted.pop(attempt, None)
        group.cancelled = True
        self._aborted.extend(request for request, _ in group.waiting.values())
        group.waiting.clear()

    def cancel(self, attempt: str) -> None:
        group = self._attempts.pop(attempt, None)
        if group is not None:
            self._cancel_waiting(attempt, group)
            self.runtime.cancel_request(attempt)
            self._flush_and_publish()

    def take_aborted_requests(self) -> list[Request]:
        """Transfer utility-triggered cancellations to the native Core output channel."""
        aborted, self._aborted = self._aborted, []
        return aborted

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus) -> list[Request]:
        """Return native children the engine cannot find in its own waiting/running queues."""
        ids = None if request_ids is None else {request_ids} if isinstance(request_ids, str) else set(request_ids)
        finished = []
        for attempt, group in self._attempts.items():
            children = tuple(group.waiting) if ids is None else group.waiting.keys() & ids
            for request_id in children:
                request, _ = group.waiting.pop(request_id)
                request.status = finished_status
                finished.append(request)
            if (ids is None or children) and not group.waiting and not group.cancelled and not group.admitted:
                group.cancelled = True
                self._admitted.pop(attempt, None)
                self.runtime.cancel_request(attempt)
        self._flush_and_publish()
        return finished

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
        for attempt, group in self._attempts.items():
            if group.program == ref and not group.admitted:
                self._cancel_waiting(attempt, group)
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
            group = self._attempts[attempt]
            group.admitted = True
            for request, native_add in group.waiting.values():
                request._mori_program_ref = group.program
                native_add(request)
            group.waiting.clear()
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
