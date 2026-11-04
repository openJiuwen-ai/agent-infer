# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Process-local bridge from native request ids to MORI Program generations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentinfer.agentcache.mori.footprints import FootprintBook
from agentinfer.scheduling.cache_placement import (
    CacheClass,
    ProgramCacheClassObserver,
    ProgramCacheClassProvider,
)
from agentinfer.scheduling.domain import ProgramRef


class _OwnerIndex(Protocol):
    def reclassify_program(self, program: ProgramRef) -> int: ...

    def release_program(self, program: ProgramRef) -> int: ...


@dataclass(frozen=True, eq=False)
class ProgramCacheContext:
    """Opaque engine-local identity for one Program's current prompt.

    Identity, rather than just Program generation, fences delayed cache work
    when successive requests replace the prompt of the same live Program.
    """

    program: ProgramRef


class CacheLifecycleObserver(ProgramCacheClassObserver, Protocol):
    def on_context_invalidated(self, program: ProgramRef) -> None:
        """Discard delayed work for a replaced prompt without retiring the Program."""


class MORIRuntimeState(ProgramCacheClassObserver):
    """Serialized engine-thread state shared with out-of-tree native policies."""

    def __init__(self) -> None:
        self._provider: ProgramCacheClassProvider | None = None
        self._requests: dict[str, ProgramRef] = {}
        self._program_requests: dict[ProgramRef, set[str]] = {}
        self._indexes: list[_OwnerIndex] = []
        self._contexts: dict[ProgramRef, ProgramCacheContext] = {}
        self._lifecycle_observers: list[CacheLifecycleObserver] = []
        self._hbm_pressure_pending = False
        self.footprints = FootprintBook()

    def bind_provider(self, provider: ProgramCacheClassProvider) -> None:
        if self._provider is not None and self._provider is not provider:
            raise RuntimeError("MORI runtime state is already bound to another scheduler")
        self._provider = provider
        self.footprints.configure_snapshot_cache(
            getattr(getattr(provider, "config", None), "fast_capacity_accounting", False) is True
        )
        provider.subscribe_cache_class_observer(self)
        bind = getattr(provider, "bind_footprints", None)
        if callable(bind):
            bind(self)

    def register_index(self, index: _OwnerIndex) -> None:
        if index not in self._indexes:
            self._indexes.append(index)

    def unregister_index(self, index: _OwnerIndex) -> None:
        if index in self._indexes:
            self._indexes.remove(index)

    def register_lifecycle_observer(self, observer: CacheLifecycleObserver) -> None:
        if observer not in self._lifecycle_observers:
            self._lifecycle_observers.append(observer)

    def unregister_lifecycle_observer(self, observer: CacheLifecycleObserver) -> None:
        if observer in self._lifecycle_observers:
            self._lifecycle_observers.remove(observer)

    def report_hbm_pressure(self) -> None:
        """Coalesce BUSY allocation advice; this neither evicts nor demotes."""
        self._hbm_pressure_pending = True

    def consume_hbm_pressure(self) -> bool:
        """Consume at most one wakeup at the next safe scheduling boundary."""
        pending = self._hbm_pressure_pending
        self._hbm_pressure_pending = False
        return pending

    def unbind_provider(self, provider: ProgramCacheClassProvider) -> None:
        if self._provider is not provider:
            return
        provider.unsubscribe_cache_class_observer(self)
        bind = getattr(provider, "bind_footprints", None)
        if callable(bind):
            bind(None)
        self.clear_requests()
        self.footprints.clear()
        self._hbm_pressure_pending = False
        self._provider = None

    def footprint(self, program: ProgramRef, tier: str) -> frozenset[int]:
        """Expose current-prompt block identity evidence to the placement planner."""
        return self.footprints.get(program, tier)

    @property
    def footprint_snapshot_hits(self) -> int:
        return self.footprints.snapshot_hits

    @property
    def footprint_snapshot_misses(self) -> int:
        return self.footprints.snapshot_misses

    def invalidate_footprint(self, program: ProgramRef) -> None:
        """Forget block proof without deleting native contents or LRU ownership."""
        self.footprints.invalidate(program)

    def _detach_context(self, program: ProgramRef) -> None:
        self._contexts.pop(program, None)
        self.footprints.invalidate(program)
        for index in tuple(self._indexes):
            index.release_program(program)
        for request_id in self._program_requests.pop(program, ()):
            self._requests.pop(request_id, None)

    def replace_context(self, program: ProgramRef) -> None:
        """Drop old content ownership, preserving shared blocks and native pins.

        Ingress calls this before binding the new request. Native cache hits
        subsequently reattach only content actually used by the new prompt.
        """
        self._detach_context(program)
        self._contexts[program] = ProgramCacheContext(program)
        for observer in tuple(self._lifecycle_observers):
            observer.on_context_invalidated(program)

    def capture_context(self, request_id: str) -> ProgramCacheContext | None:
        program = self.program_for_request(request_id)
        return self._contexts.get(program) if program is not None else None

    def context_for_program(self, program: ProgramRef) -> ProgramCacheContext | None:
        return self._contexts.get(program)

    def program_for_context(self, context: object) -> ProgramRef | None:
        """Resolve only the current opaque receipt, including after request finish."""
        if isinstance(context, ProgramCacheContext) and self._contexts.get(context.program) is context:
            return context.program
        return None

    def bind_request(self, request_id: str, program: ProgramRef) -> None:
        current = self._requests.get(request_id)
        if current is not None and current != program:
            raise RuntimeError("native request id was rebound to another Program generation")
        if program not in self._contexts:
            self._contexts[program] = ProgramCacheContext(program)
        self._requests[request_id] = program
        self._program_requests.setdefault(program, set()).add(request_id)

    def unbind_request(self, request_id: str) -> None:
        owner = self._requests.pop(request_id, None)
        if owner is not None:
            requests = self._program_requests[owner]
            requests.discard(request_id)
            if not requests:
                del self._program_requests[owner]

    def program_for_request(self, request_id: str) -> ProgramRef | None:
        return self._requests.get(request_id)

    def clear_requests(self) -> None:
        """Invalidate native bindings and delayed cache work on scheduler shutdown."""
        for program in tuple(self._contexts):
            self._detach_context(program)
            for observer in tuple(self._lifecycle_observers):
                observer.on_context_invalidated(program)
        self._requests.clear()
        self._program_requests.clear()

    def cache_class_for_program(self, program: ProgramRef) -> CacheClass:
        provider = self._provider
        return provider.cache_class_for_program(program) if provider is not None else CacheClass.INACTIVE

    def on_cache_class_changed(
        self,
        program: ProgramRef,
        previous: CacheClass,
        current: CacheClass,
    ) -> None:
        for index in tuple(self._indexes):
            index.reclassify_program(program)
        for observer in tuple(self._lifecycle_observers):
            observer.on_cache_class_changed(program, previous, current)

    def on_program_released(self, program: ProgramRef) -> None:
        self._detach_context(program)
        for observer in tuple(self._lifecycle_observers):
            observer.on_program_released(program)

    def reset(self) -> None:
        """Reset process-local registrations after shutdown or in unit tests."""
        if self._provider is not None:
            self.unbind_provider(self._provider)
        self.clear_requests()
        self._indexes.clear()
        self._lifecycle_observers.clear()
        self._hbm_pressure_pending = False
        self.footprints.clear()
        self.footprints.enabled = False
        self.footprints.configure_snapshot_cache(False)


MORI_RUNTIME_STATE = MORIRuntimeState()
