# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Capture complete positional prefixes using vLLM's absolute hash endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, make_block_hash_with_group_id, resolve_block_hashes

from agentinfer.scheduling.mori import MORIStrategy


@dataclass(frozen=True)
class _WritebackSource:
    block_id: int
    block_hash: BlockHashWithGroupId
    # Native metadata names this num_tokens, but it is the absolute prefix end.
    num_tokens: int


class FullPrefixWritebackCapture:
    """Replace only the faulty capture hook; native submit/fences/copies remain authoritative."""

    def __init__(self, manager: Any, strategy: MORIStrategy) -> None:
        if (
            not callable(getattr(manager, "_mori_capture_finished_store", None))
            or not isinstance(getattr(manager, "_mori_store_retries", None), dict)
            or len(getattr(manager, "group_block_sizes", ())) != 1
            or set(getattr(manager, "prefix_cacheable_group_ids", ())) != {0}
            or not manager.cpu_coordinator.single_type_managers[0].has_positionally_stable_blocks
        ):
            raise RuntimeError("MORI writeback endpoint repair requires one positional SimpleCPU KV group")
        self.manager = manager
        self.strategy = strategy
        self._original = manager._mori_capture_finished_store
        self._wrapper = self._capture
        self.closed = False
        manager._mori_capture_finished_store = self._wrapper

    def _capture(self, request: Any, block_ids: tuple[list[int], ...]) -> None:
        started = time.thread_time()
        try:
            manager = self.manager
            provider, pool = manager._mori_offload_tier_provider, manager._gpu_block_pool
            if provider is None or pool is None:
                return
            context = provider.capture_store_retry_context(request.request_id)
            if context is None:
                return
            manager._mori_store_retries.pop(context, None)
            confirmed = request.num_computed_tokens - request.num_output_placeholders
            size = manager.group_block_sizes[0]
            hashes = resolve_block_hashes(request.block_hashes, manager.hash_block_size, size)
            ids = block_ids[0] if block_ids else ()
            ready = min(len(ids), confirmed // size, len(hashes))
            seen: set[int] = set()
            candidates: list[_WritebackSource] = []
            for position in range(ready):
                block = pool.blocks[ids[position]]
                expected = make_block_hash_with_group_id(hashes[position], 0)
                endpoint = (position + 1) * size
                if (
                    block.is_null
                    or block.block_id in seen
                    or block.block_hash != expected
                    or block.block_hash_num_tokens != endpoint
                ):
                    continue
                seen.add(block.block_id)
                candidates.append(_WritebackSource(block.block_id, expected, endpoint))
            if candidates:
                manager._mori_store_retries[context] = tuple(candidates)
            self.strategy.counters["writeback_capture_requests"] += 1
            self.strategy.counters["writeback_capture_blocks"] += len(candidates)
            self.strategy.counters["writeback_capture_noninitial_blocks"] += sum(
                item.num_tokens > size for item in candidates
            )
        finally:
            self.strategy.writeback_capture_cpu_seconds += time.thread_time() - started

    def close(self) -> None:
        """Restore only this instance's original capture hook."""
        if self.closed:
            return
        self.closed = True
        if self.manager._mori_capture_finished_store is self._wrapper:
            self.manager._mori_capture_finished_store = self._original
