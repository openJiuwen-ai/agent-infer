# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""HBM typed victim provider for the vLLM MORI hook contract."""

from __future__ import annotations

from collections.abc import Iterable

from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.typed_lru import TypedLRUIndex
from agentinfer.scheduling.cache_placement import CacheClass
from agentinfer.scheduling.domain import ProgramRef


class MORIHBMPolicy:
    """Propose INACTIVE, then IDLE, then BUSY blocks in native LRU order."""

    HOOK_API_VERSION = 1

    def __init__(self, runtime_state: MORIRuntimeState) -> None:
        self.runtime_state = runtime_state
        self.index = TypedLRUIndex[int](
            eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
            protection_rank={
                CacheClass.INACTIVE: 0,
                CacheClass.IDLE: 1,
                CacheClass.BUSY: 2,
            },
            class_for_program=runtime_state.cache_class_for_program,
        )
        runtime_state.register_index(self.index)

    def on_block_ready(self, block_id: int, native_lru_seq: int) -> None:
        """Observe a ref-zero free-queue entry after native order is committed."""
        owners = self.index.owners(block_id)
        self.index.upsert(block_id, owners=owners, native_lru_seq=native_lru_seq, ready=True)

    def on_block_owner_added(self, block_id: int, owner_request_id: str) -> None:
        """Attach one live request owner without asking native code for Program identity."""
        owner = self.runtime_state.program_for_request(owner_request_id)
        if owner is None:
            return
        self.runtime_state.footprints.add("gpu", block_id, owner)
        owners = set(self.index.owners(block_id))
        if owner in owners:
            return
        owners.add(owner)
        if self.index.cache_class(block_id) is None:
            # Ownership is normally attached while a block is allocated. The
            # free-queue callback later supplies its exact native LRU sequence.
            self.index.upsert(block_id, owners=owners, native_lru_seq=0, ready=False)
        else:
            self.index.update_owners(block_id, owners)

    def on_block_touched(self, block_id: int, native_lru_seq: int) -> None:
        self.index.touch(block_id, native_lru_seq)

    def on_block_evictability_changed(self, block_id: int, ready: bool) -> None:
        self.index.mark_ready(block_id, ready)

    def on_block_owners_changed(self, block_id: int, owner_request_ids: Iterable[str]) -> None:
        self.index.update_owners(block_id, self._owners(owner_request_ids))

    def on_block_removed(self, block_id: int) -> None:
        self.runtime_state.footprints.remove_block("gpu", block_id)
        self.index.remove(block_id)

    def on_cache_reset(self) -> None:
        self.runtime_state.consume_hbm_pressure()
        self.runtime_state.footprints.clear("gpu")
        self.index.clear()

    def close(self) -> None:
        """Detach this index from the process-local Program class bridge."""
        self.index.clear()
        self.runtime_state.unregister_index(self.index)

    def propose(self, count: int, protected_block_ids: Iterable[int] = ()) -> tuple[int, ...] | None:
        """Plan one native allocation, recording BUSY pressure without performing eviction."""
        proposed = self.index.propose_exact(count, protected_block_ids)
        if proposed and any(self.index.cache_class(block) is CacheClass.BUSY for block in proposed):
            self.runtime_state.report_hbm_pressure()
        return proposed

    def _owners(self, request_ids: Iterable[str]) -> frozenset[ProgramRef]:
        return frozenset(
            program
            for request_id in request_ids
            if (program := self.runtime_state.program_for_request(request_id)) is not None
        )
