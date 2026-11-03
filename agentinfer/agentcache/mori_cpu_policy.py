# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Out-of-tree vLLM 0.31 CPU cache policy implementing MORI typed LRU."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from typing_extensions import override
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus, order_request_keys

from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori.typed_lru import TypedLRUIndex
from agentinfer.scheduling.cache_placement import CacheClass
from agentinfer.scheduling.domain import ProgramRef


class MoriCPUCachePolicy(CachePolicy):
    """Evict INACTIVE, then BUSY, then IDLE chunks in native LRU order.

    Configure vLLM with ``eviction_policy=MoriCPUCachePolicy`` and
    ``cache_policy_module_path=agentinfer.agentcache.mori_cpu_policy``. The
    manager keeps ownership of ref counts, protected keys, job/fence safety,
    chunk allocation, and transfer completion; this class owns replacement
    order only.
    """

    MORI_CPU_POLICY = True

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._owners: dict[OffloadKey, set[ProgramRef]] = {}
        self._pending_owners: dict[OffloadKey, set[ProgramRef]] = {}
        self._pending_program_keys: dict[ProgramRef, set[OffloadKey]] = {}
        self._next_rank = 0
        self.index = TypedLRUIndex[OffloadKey](
            eviction_order=(CacheClass.INACTIVE, CacheClass.BUSY, CacheClass.IDLE),
            protection_rank={
                CacheClass.INACTIVE: 0,
                CacheClass.BUSY: 1,
                CacheClass.IDLE: 2,
            },
            class_for_program=MORI_RUNTIME_STATE.cache_class_for_program,
        )
        MORI_RUNTIME_STATE.register_index(self)

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.chunks[key] = chunk
        owners = self._pending_owners.pop(key, set())
        for owner in owners:
            pending = self._pending_program_keys.get(owner)
            if pending is not None:
                pending.discard(key)
                if not pending:
                    self._pending_program_keys.pop(owner, None)
        self._owners[key] = set(owners)
        self._next_rank += 1
        self.index.upsert(
            key,
            owners=owners,
            native_lru_seq=self._next_rank,
            ready=chunk.ref_cnt == 0,
        )

    @override
    def remove(self, key: OffloadKey) -> None:
        self.chunks.pop(key, None)
        self._owners.pop(key, None)
        pending_owners = self._pending_owners.pop(key, set())
        for owner in pending_owners:
            pending = self._pending_program_keys.get(owner)
            if pending is not None:
                pending.discard(key)
                if not pending:
                    self._pending_program_keys.pop(owner, None)
        self.index.remove(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        ordered = list(keys)
        self._remember_owner(ordered, req_context)
        self._update_recency(reversed(ordered))

    @override
    def on_store_miss(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        self._remember_owner(keys, req_context)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        del insertion_only_keys, reused_keys
        ordered = order_request_keys(key_groups, req_context)
        self._remember_owner(ordered, req_context)
        self._update_recency(reversed(ordered))

    @override
    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        with self.index.reserve_exact(n, protected) as proposed:
            if proposed is None:
                return None
            result: list[tuple[OffloadKey, ChunkStatus]] = []
            for key in proposed:
                chunk = self.chunks.get(key)
                if chunk is None or chunk.ref_cnt != 0 or key in protected:
                    # Native state changed without the expected callback. Fail
                    # closed; the reservation restores every live selection.
                    if chunk is None:
                        self.remove(key)
                        self.index.stats["missing_native_entries"] += 1
                    else:
                        self.index.mark_ready(key, chunk.ref_cnt == 0)
                    return None
                result.append((key, chunk))
            for key, _ in result:
                self.remove(key)
            return result

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self._owners.clear()
        self._pending_owners.clear()
        self._pending_program_keys.clear()
        self._next_rank = 0
        self.index.clear()

    def close(self) -> None:
        """Detach the policy index after the native offload manager shuts down."""
        self.clear()
        MORI_RUNTIME_STATE.unregister_index(self)

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self.index.mark_ready(key, True)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.index.mark_ready(key, False)

    def _remember_owner(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        owner = MORI_RUNTIME_STATE.program_for_request(req_context.req_id)
        if owner is None:
            return
        for key in keys:
            if key in self.chunks:
                owners = self._owners.setdefault(key, set())
                if owner not in owners:
                    owners.add(owner)
                    self.index.update_owners(key, owners)
            else:
                self._pending_owners.setdefault(key, set()).add(owner)
                self._pending_program_keys.setdefault(owner, set()).add(key)

    def reclassify_program(self, program: ProgramRef) -> int:
        """Reclassify live chunks owned by one Program generation."""
        return self.index.reclassify_program(program)

    def release_program(self, program: ProgramRef) -> int:
        """Remove a released generation from live and pending CPU ownership."""
        live_keys = self.index.entries_for_program(program)
        for key in live_keys:
            owners = self._owners.get(key)
            if owners is not None:
                owners.discard(program)
        changed = self.index.release_program(program)
        for key in self._pending_program_keys.pop(program, set()):
            owners = self._pending_owners.get(key)
            if owners is None:
                continue
            owners.discard(program)
            if not owners:
                self._pending_owners.pop(key, None)
        return changed

    def _update_recency(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if key not in self.chunks:
                continue
            self._next_rank += 1
            self.index.touch(key, self._next_rank)
