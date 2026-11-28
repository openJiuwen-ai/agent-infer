# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Scheduler-side vLLM overrides for MORI cache placement."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)

if TYPE_CHECKING:
    from agentinfer.agentcache.mori.hiradix import _TierIndex


class MoriBlockPool(BlockPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tier_index: _TierIndex | None = None
        self._hiradix_owns_prefix_index = False

    @property
    def hiradix_index(self) -> _TierIndex | None:
        return self._tier_index if self._hiradix_owns_prefix_index else None

    def get_cached_block(self, block_hash: BlockHash, kv_cache_group_ids: list[int]) -> list[KVCacheBlock] | None:
        """Use HiRadix even when a dense native path asks for an exact page."""
        index = self.hiradix_index
        if index is None:
            return super().get_cached_block(block_hash, kv_cache_group_ids)
        page = index.tree.pages.get(block_hash)
        if page is None:
            return None
        replicas = index.replicas(page)
        blocks = []
        for group_id in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, group_id)
            block = next((replica for replica in replicas.values() if replica.block_hash == key), None)
            if block is None:
                return None
            blocks.append(block)
        return blocks

    def _insert_block_hash(
        self, block_hash_with_group_id: BlockHashWithGroupId, block: KVCacheBlock, num_tokens: int | None
    ) -> None:
        if self.hiradix_index is None:
            super()._insert_block_hash(block_hash_with_group_id, block, num_tokens)
            return
        # Native allocation and DMA use the identity attached to each page.
        # Only HiRadix exposes the identity to shared prefix lookups.
        if block.block_hash is None:
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
        elif block.block_hash != block_hash_with_group_id:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(block_hash_with_group_id)

    def _remove_cached_block_hashes(self, block: KVCacheBlock) -> list[BlockHashWithGroupId]:
        if self.hiradix_index is None:
            return super()._remove_cached_block_hashes(block)
        hashes = [block.block_hash] if block.block_hash is not None else []
        hashes.extend(self.cached_block_hashes_by_block.pop(block.block_id, ()))
        block.reset_hash()
        return hashes

    def get_indexed_copy(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        index = self.hiradix_index
        if index is None:
            return self.cached_block_hash_to_block.get_one_block(key)
        page = index.tree.pages.get(get_block_hash(key))
        if page is None:
            return None
        return next((block for block in index.replicas(page).values() if block.block_hash == key), None)

    def get_usage(self) -> float:
        index = self.hiradix_index
        capacity = self.num_gpu_blocks - 1
        if index is None or not capacity:
            return super().get_usage()
        # HiCache reports capacity less both unused and evictable KV.
        return (capacity - len(index.ready)) / capacity

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        index = self._tier_index
        if index is None:
            return super().get_new_blocks(num_blocks)
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        selected = index.allocation_ids(num_blocks)
        if len(selected) != num_blocks or len(set(selected)) != num_blocks:
            raise RuntimeError("MORI HiRadix allocation bypassed eviction planning")
        if any(not 0 < block_id < len(self.blocks) for block_id in selected):
            raise RuntimeError("MORI HiRadix selected an invalid physical block ID")
        blocks = [self.blocks[block_id] for block_id in selected]
        if any(
            block.is_null or block.ref_cnt or block.prev_free_block is None or block.next_free_block is None
            for block in blocks
        ):
            raise RuntimeError("MORI HiRadix selected a block outside the physical free queue")

        for block in blocks:
            self.free_block_queue.remove(block)
            index.on_block_allocated(block.block_id)
        if self._reuse_watchers:
            self._notify_reuse(blocks)
        for block in blocks:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return blocks

    def unpin_blocks(
        self,
        blocks: Iterable[KVCacheBlock],
        on_reuse: Callable[[KVCacheBlock], None],
    ) -> None:
        """Release references to blocks that stay readable until reused.

        The blocks become last-resort eviction candidates regardless of prefix
        caching: they join the tail of the free queue and count as free.
        ``on_reuse`` fires when ``get_new_blocks`` hands a block out (or the
        cache is reset), so the caller can stop reading it.
        """
        index = self._tier_index
        if index is None:
            super().unpin_blocks(blocks, on_reuse)
            return
        released: list[KVCacheBlock] = []
        for block in blocks:
            assert block.ref_cnt > 0 and not block.is_null
            block.ref_cnt -= 1
            self._reuse_watchers[block.block_id] = on_reuse
            if block.ref_cnt == 0:
                released.append(block)
        self.free_block_queue.append_n(released)
        for block in released:
            index.on_block_ready(block.block_id)

    def is_idle_cached_block(self, block_id: int) -> bool:
        block = self.blocks[block_id]
        return block.ref_cnt == 0 and block.block_hash is not None and block_id not in self._reuse_watchers

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.

        """
        index = self._tier_index
        if index is None:
            super().touch(blocks)
            return
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
                index.on_block_pinned(block.block_id)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.

        """
        index = self._tier_index
        if index is None:
            super().free_blocks(ordered_blocks)
            return
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_to_evict_last = []
        blocks_to_evict_first = []
        other_pools: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in ordered_blocks:
            if block.pool is not None and block.pool is not self:
                other_pools.setdefault(block.pool, []).append(block)
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    # FIFO reuse of cached blocks for LRU eviction behavior.
                    blocks_to_evict_last.append(block)

        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
        # Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(blocks_to_evict_last)
        for block in (*blocks_to_evict_first, *blocks_to_evict_last):
            index.on_block_ready(block.block_id)
        if not index.host and not self.enable_kv_cache_events:
            index.discard_released_replicas(blocks_to_evict_last)
        for pool, blocks in other_pools.items():
            pool.free_blocks(blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        super().evict_blocks(block_ids)
        if (index := self._tier_index) is not None:
            for block_id in block_ids:
                if self.blocks[block_id].block_hash is None:
                    index.on_block_removed(block_id)

    def reset_prefix_cache(self) -> bool:
        if not super().reset_prefix_cache():
            return False
        if (index := self._tier_index) is not None:
            index.on_cache_reset()
        return True

    def bind_hiradix_index(self, index: _TierIndex, *, authoritative: bool = False) -> None:
        if self._tier_index is index:
            if authoritative:
                self._hiradix_owns_prefix_index = True
            return
        if self._tier_index is not None or index.pool is not self:
            raise RuntimeError("invalid HiRadix tier binding for the physical block pool")
        self._tier_index = index
        self._hiradix_owns_prefix_index = authoritative
        index.on_cache_reset()
        for block in self.free_block_queue.get_all_free_blocks():
            index.on_block_ready(block.block_id)

    def unbind_hiradix_index(self, index: _TierIndex) -> None:
        if self._tier_index is index:
            self._tier_index = None
            self._hiradix_owns_prefix_index = False
