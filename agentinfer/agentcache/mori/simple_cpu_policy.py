# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Typed victim provider for vLLM's SimpleCPUOffloadConnector."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.typed_lru import TypedLRUIndex
from agentinfer.scheduling.cache_placement import CacheClass
from agentinfer.scheduling.domain import ProgramRef


class MORISimpleCPUOffloadPolicy:
    """Propose SimpleCPU blocks in INACTIVE, BUSY, IDLE order.

    vLLM remains authoritative for free-queue membership, refcounts, cached
    hashes, allocation, and transfer completion. This provider only mirrors
    owner/class/readiness deltas and proposes an exact ordered victim set.
    """

    HOOK_API_VERSION = 3

    def __init__(self, runtime_state: MORIRuntimeState) -> None:
        self.runtime_state = runtime_state
        self.index = TypedLRUIndex[int](
            eviction_order=(CacheClass.INACTIVE, CacheClass.BUSY, CacheClass.IDLE),
            protection_rank={
                CacheClass.INACTIVE: 0,
                CacheClass.BUSY: 1,
                CacheClass.IDLE: 2,
            },
            class_for_program=runtime_state.cache_class_for_program,
        )
        runtime_state.register_index(self.index)
        self._retry_contexts: dict[ProgramRef, object] = {}
        self._retry_listener: Callable[[object, bool], None] | None = None
        runtime_state.register_lifecycle_observer(self)

    def bind_store_retry_listener(self, listener: Callable[[object, bool], None] | None) -> None:
        """Connect native best-effort retry without retaining completed requests."""
        self._retry_listener = listener
        if listener is None:
            self._retry_contexts.clear()

    def capture_store_retry_context(self, request_id: str) -> object | None:
        context = self.runtime_state.capture_context(request_id)
        owner = self.runtime_state.program_for_context(context)
        if owner is not None:
            self._retry_contexts[owner] = context
        return context

    def store_retry_context_is_current(self, context: object) -> bool:
        owner = self.runtime_state.program_for_context(context)
        return owner is not None and self.runtime_state.cache_class_for_program(owner) is CacheClass.IDLE

    def on_cache_class_changed(self, program: ProgramRef, previous: CacheClass, current: CacheClass) -> None:
        if previous is CacheClass.BUSY and current is CacheClass.IDLE:
            context = self._retry_contexts.get(program)
            if context is not None and self._retry_listener is not None:
                self._retry_listener(context, True)

    def on_context_invalidated(self, program: ProgramRef) -> None:
        context = self._retry_contexts.pop(program, None)
        if context is not None and self._retry_listener is not None:
            self._retry_listener(context, False)

    def on_program_released(self, program: ProgramRef) -> None:
        self.on_context_invalidated(program)

    def _store_owner(self, owner_key: object) -> ProgramRef | None:
        if isinstance(owner_key, str):
            return self.runtime_state.program_for_request(owner_key)
        return self.runtime_state.program_for_context(owner_key)

    def on_block_ready(self, block_id: int, native_lru_seq: int) -> None:
        """Observe a ref-zero CPU block after native queue order is committed."""
        self.index.upsert(
            block_id,
            owners=self.index.owners(block_id),
            native_lru_seq=native_lru_seq,
            ready=True,
        )

    def on_block_owner_added(self, block_id: int, owner_request_id: object) -> None:
        """Resolve a native request ID through server-maintained Program state."""
        owner = self._store_owner(owner_request_id)
        if owner is None:
            return
        self.runtime_state.footprints.add("cpu", block_id, owner)
        owners = set(self.index.owners(block_id))
        if owner in owners:
            return
        owners.add(owner)
        if self.index.cache_class(block_id) is None:
            # Store destinations are observed while pinned. Their exact native
            # sequence arrives only after transfer completion frees the block.
            self.index.upsert(block_id, owners=owners, native_lru_seq=0, ready=False)
        else:
            self.index.update_owners(block_id, owners)

    def on_block_evictability_changed(self, block_id: int, ready: bool) -> None:
        self.index.mark_ready(block_id, ready)

    def on_block_removed(self, block_id: int) -> None:
        self.runtime_state.footprints.remove_block("cpu", block_id)
        self.index.remove(block_id)

    def on_cache_reset(self) -> None:
        self.runtime_state.footprints.clear("cpu")
        self.index.clear()

    def propose(
        self,
        count: int,
        protected_block_ids: Iterable[int] = (),
    ) -> tuple[int, ...] | None:
        """Return advice only; SimpleCPU validates and performs allocation."""
        return self.index.propose_exact(count, protected_block_ids)

    def _store_classes(self, request_id: object) -> tuple[CacheClass, ...]:
        owner = self._store_owner(request_id)
        if owner is None and not isinstance(request_id, str):
            return ()
        incoming = self.runtime_state.cache_class_for_program(owner) if owner is not None else CacheClass.INACTIVE
        order = self.index.eviction_order
        return order[: order.index(incoming) + 1]

    def store_capacity(self, request_id: object) -> int:
        """Count free/lower-or-equal protection destinations for this incoming store."""
        return self.index.ready_count(self._store_classes(request_id))

    def propose_for_store(self, count: int, request_id: object) -> tuple[int, ...] | None:
        """Reject a backup that would evict a more protected CPU working set."""
        return self.index.propose_exact(count, allowed_classes=self._store_classes(request_id))

    def close(self) -> None:
        """Detach the SimpleCPU index from Program class notifications."""
        self.index.clear()
        self.runtime_state.unregister_index(self.index)
        self.runtime_state.unregister_lifecycle_observer(self)
        self.bind_store_retry_listener(None)


__all__ = ["MORISimpleCPUOffloadPolicy"]
