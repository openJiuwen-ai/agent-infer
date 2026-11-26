# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Scheduler-side vLLM overrides for MORI cache placement."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.mori.block_pool import MoriBlockPool
from agentinfer.agentcache.mori.hiradix import HiRadix
from agentinfer.agentcache.mori.programs import request_program

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector

logger = init_logger("vllm.agentinfer.mori.kv_cache_manager")


class MoriKVCacheManager(KVCacheManager):
    hiradix: HiRadix | None = None
    mori_batch_loads = False
    skip_native_cascade_prefix = False
    release_replaced_hiradix_blocks: Callable[[list[KVCacheBlock], int], None] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._radix_depth: dict[str, int] = {}
        self._replayed_prefill_tokens: dict[str, int] = {}
        self._hiradix_block_table_updates: set[str] = set()
        self._unpublished_batch_loads: set[str] = set()
        self._eviction_stores: dict[str, tuple[int, ...]] = {}
        self._observed_allocation: tuple[str, int] | None = None
        self._native_capacity_check = None

    def bind_hiradix(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        connector: MoriCPUOffloadConnector,
        *,
        page_size: int,
        dense: bool,
        use_v2_model_runner: bool,
        share_issued_loads: bool,
        release_replaced: Callable[[list[KVCacheBlock], int], None],
    ) -> None:
        """Build the sole HiRadix and connect it to native pools and transfers."""
        if self.hiradix is not None:
            raise RuntimeError("MORI KV manager can bind only one HiRadix")
        gpu = self.block_pool
        if not isinstance(gpu, MoriBlockPool):
            raise RuntimeError("MORI HiRadix requires its own GPU block pool")
        tree = HiRadix.with_controller(gpu, page_size, vllm_config, kv_cache_config, connector.cpu_capacity_per_rank)
        self.hiradix = tree
        controller = tree.cache_controller
        assert controller is not None
        try:
            if controller._lazy_mode or not controller._mori_cpu_backend:
                raise ValueError("MORI HiRadix requires non-lazy CPU copies")
            gpu.bind_hiradix_index(tree.device, authoritative=dense)
            tree.host.pool.bind_hiradix_index(tree.host, authoritative=dense)
            # MRV2 consumes native block-table updates before attention.
            self.skip_native_cascade_prefix = use_v2_model_runner
            if dense and use_v2_model_runner and not gpu.enable_kv_cache_events:
                self.release_replaced_hiradix_blocks = release_replaced
            self._install_hiradix_capacity_check()
            if connector.mori_batch_loads:
                controller.mori_batch_loads = self.mori_batch_loads = True
                if not gpu.enable_kv_cache_events and not controller.enable_kv_cache_events:
                    tree.share_batch_loads = True
                    tree.share_issued_loads = share_issued_loads
                    controller.publish_batch_load = self.publish_batch_load
            connector.bind_controller(controller)
        except BaseException:
            self.close_hiradix(connector)
            raise

    def close_hiradix(self, connector: MoriCPUOffloadConnector) -> None:
        tree = self.hiradix
        if tree is None:
            return
        controller = tree.cache_controller
        try:
            tree.close_transfers()
        finally:
            tree.device.pool.unbind_hiradix_index(tree.device)
            tree.host.pool.unbind_hiradix_index(tree.host)
            if self._native_capacity_check is not None:
                self.coordinator.get_num_blocks_to_allocate = self._native_capacity_check
                self._native_capacity_check = None
            self.mori_batch_loads = False
            self.skip_native_cascade_prefix = False
            self.release_replaced_hiradix_blocks = None
            self._observed_allocation = None
            self._eviction_stores.clear()
            self._radix_depth.clear()
            self._replayed_prefill_tokens.clear()
            self._hiradix_block_table_updates.clear()
            self._unpublished_batch_loads.clear()
            self.hiradix = None
            if controller is not None:
                connector.unbind_controller(controller)

    def _install_hiradix_capacity_check(self) -> None:
        """Observe the native capacity check before it touches prefix hits or allocates."""
        original = self.coordinator.get_num_blocks_to_allocate

        def check_capacity(*args, **kwargs):
            count = original(*args, **kwargs)
            allocation = self._observed_allocation
            if allocation is not None and not kwargs.get("apply_admission_cap"):
                request_id, reserved = allocation
                if self._preserve_evictions(request_id, count, kwargs["new_computed_blocks"], reserved):
                    return self.block_pool.get_num_free_blocks() + 1
            return count

        self._native_capacity_check = original
        self.coordinator.get_num_blocks_to_allocate = check_capacity

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        tree = self.hiradix
        if tree is None:
            return super().get_computed_blocks(request)
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0
        device, _ = tree.match_for_native_admission(request)
        matched = len(device) * tree.page_size
        return KVCacheBlocks((device,)), matched, 0

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        if self.skip_native_cascade_prefix:
            return [0] * len(self.coordinator.single_type_managers)
        return super().get_num_common_prefix_blocks(running_request_id)

    def allocate_slots(self, request: Request, num_new_tokens: int, *args, **kwargs) -> KVCacheBlocks | None:
        if (
            self.mori_batch_loads
            and request.status == RequestStatus.RUNNING
            and num_new_tokens == 1
            and not args
            and len(kwargs) == 1
            and kwargs.get("num_lookahead_tokens") == 0
            and self._has_private_decode_slot(request)
        ):
            return self.empty_kv_cache_blocks
        starts_batch_load = False
        borrowed_block = None
        tree = self.hiradix
        if self.mori_batch_loads:
            starts_batch_load = kwargs.get("num_external_computed_tokens", 0) > 0
            hits = kwargs.get("new_computed_blocks")
            assert tree is not None
            if hits is not None and (tree.batch_load_destinations or tree.issued_load_destinations):
                borrowed_block = next(
                    (
                        block.block_id
                        for group in hits.blocks
                        for block in group
                        if tree.shares_load_destination(block.block_id)
                    ),
                    None,
                )
        pending_batch_load = starts_batch_load or borrowed_block is not None
        if pending_batch_load or self.mori_batch_loads and request.request_id in self._unpublished_batch_loads:
            kwargs["delay_cache_blocks"] = True
        waiting = request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
        # vLLM would hash all full pages during allocate_slots, before the
        # decode result. Keep decode-only pages private until retirement.
        replaying = waiting and request.num_output_tokens > 0 and tree is not None
        if (
            not waiting
            and request.num_output_tokens > 0
            and tree is not None
            and request.num_computed_tokens >= self._prefill_token_limit(request)
        ):
            kwargs["delay_cache_blocks"] = True
        if (
            self._native_capacity_check is not None
            and (waiting or request.status == RequestStatus.RUNNING)
            and not args
        ):
            request_id = request.request_id
            reserved = kwargs.get("reserved_blocks", 0)
            if waiting and kwargs.get("has_scheduled_reqs", True):
                reserved += self.watermark_blocks
            self._observed_allocation = (request_id, reserved)
        try:
            blocks = super().allocate_slots(request, num_new_tokens, *args, **kwargs)
        finally:
            self._observed_allocation = None
        if blocks is not None and self.enable_caching:
            if replaying:
                # SGLang considers already generated tokens part of fill_ids
                # when a retracted request is prefilling again.
                self._replayed_prefill_tokens[request.request_id] = request.num_tokens
            if pending_batch_load:
                self._unpublished_batch_loads.add(request.request_id)
            if borrowed_block is not None:
                assert tree is not None
                if borrowed_block in tree.issued_load_destinations:
                    logger.info_once("MORI borrower reuses H2D pages from the previous forward")
                else:
                    logger.info_once("MORI borrowers share a single H2D transfer within the prefill batch")
            self._eviction_stores.pop(request.request_id, None)
        return blocks

    def _has_private_decode_slot(self, request: Request) -> bool:
        """A single dense decode token can append to its existing private page."""
        request_id = request.request_id
        position = request.num_computed_tokens
        if (
            not request.num_output_tokens
            or position < self._prefill_token_limit(request)
            or position >= self.max_model_len
            or request_id in self._unpublished_batch_loads
            or request_id in self._eviction_stores
        ):
            return False
        (manager,) = self.coordinator.single_type_managers
        if type(manager) is not FullAttentionManager or request_id not in manager.num_cached_block:
            return False
        # Older compatible managers may not expose pending copy-on-write state.
        partial_hits = getattr(manager, "_partial_hit_reqs", None)
        if partial_hits is None or request_id in partial_hits:
            return False
        index, offset = divmod(position, manager.block_size)
        blocks = manager.req_to_blocks.get(request_id, ())
        if not offset or index >= len(blocks):
            return False
        block = blocks[index]
        return not block.is_null and block.ref_cnt == 1 and block.block_hash is None

    def _preserve_evictions(
        self, request_id: str, count: int, hits: tuple[list[KVCacheBlock], ...], reserved: int
    ) -> bool:
        tree = self.hiradix
        assert tree is not None
        if request_id in self._eviction_stores:
            if tree.eviction_blocks_pending(self._eviction_stores[request_id]):
                return True
            del self._eviction_stores[request_id]
        if count == 0 or count + reserved > self.block_pool.get_num_free_blocks():
            return False
        protected = frozenset(
            block.block_id for group in hits for block in group if not block.is_null and not block.ref_cnt
        )
        staged = tree.stage_evictions(count - len(protected), protected)
        if staged is None:
            return True
        if staged:
            self._eviction_stores[request_id] = staged
        return bool(staged)

    def _prefill_token_limit(self, request: Request) -> int:
        if request.num_output_tokens:
            return self._replayed_prefill_tokens.get(request.request_id, request.num_prompt_tokens)
        return request.num_prompt_tokens

    def _cache_and_publish(self, request: Request, num_computed_tokens: int, *, refresh_existing: bool = False) -> None:
        super().cache_blocks(request, num_computed_tokens)
        self.publish_hiradix(request, num_computed_tokens, refresh_existing=refresh_existing)
        if (tree := self.hiradix) is not None:
            tree.release_load(request.request_id)

    def cache_blocks(self, request: Request, num_computed_tokens: int, *, refresh_existing: bool = False) -> None:
        # AsyncScheduler invokes this after each decode result, including the
        # terminal result before the scheduler retires the request.
        tree = self.hiradix
        if tree is not None:
            num_computed_tokens = min(num_computed_tokens, self._prefill_token_limit(request))
            # Native AsyncScheduler calls this on every decode result. Once both
            # the dense block table and the tree cover the eligible prompt, only
            # the transfer reservation still needs its normal release.
            if self.mori_batch_loads and request.num_output_tokens > 1 and not refresh_existing:
                request_id = request.request_id
                pages = num_computed_tokens // tree.page_size
                if pages <= self._radix_depth.get(request_id, 0):
                    (manager,) = self.coordinator.single_type_managers
                    if pages <= manager.num_cached_block.get(request_id, 0):
                        tree.release_load(request_id)
                        return
        self._cache_and_publish(request, num_computed_tokens, refresh_existing=refresh_existing)

    def confirm_hiradix(self, request: Request, confirmed_tokens: int, *, refresh_existing: bool = False) -> None:
        confirmed_tokens = min(confirmed_tokens, self._prefill_token_limit(request))
        if self.mori_batch_loads and request.request_id in self._unpublished_batch_loads:
            self._cache_and_publish(request, confirmed_tokens, refresh_existing=refresh_existing)
            self._unpublished_batch_loads.remove(request.request_id)
        else:
            self.publish_hiradix(request, confirmed_tokens, refresh_existing=refresh_existing)

    def stash_hiradix_chunk(self, request: Request, *, latest_forward: bool = False) -> None:
        """Expose the previous local forward's full prefill pages to LPM."""
        # The local V2 runner has dispatched the writer on its compute stream.
        # Native block references retain the pages and subsequent forwards
        # run on the same stream. Mixed chunks require a completed DMA event.
        request_id = request.request_id
        tokens = min(self._prefill_token_limit(request), request.num_computed_tokens)
        if request_id in self._unpublished_batch_loads:
            if not latest_forward or not self._certify_stashed_load(request, tokens):
                return
            # The DMA is complete. Normal ACK still owns transfer cleanup.
            super().cache_blocks(request, tokens)
        previous_depth = self._radix_depth.get(request_id, 0)
        self.publish_hiradix(request, tokens, refresh_existing=False)
        if self._radix_depth.get(request_id, 0) > previous_depth:
            logger.debug_once("MORI stashes local chunked prefill KV before the previous forward's output arrives")

    def _certify_stashed_load(self, request: Request, tokens: int) -> bool:
        tree = self.hiradix
        assert tree is not None
        controller = tree.cache_controller
        if controller is None or (transfer := controller.completed_local_load(request)) is None:
            return False
        (manager,) = self.coordinator.single_type_managers
        pages = tokens // tree.page_size
        prefix = manager.req_to_blocks.get(request.request_id, ())[:pages]
        cached = manager.num_cached_block.get(request.request_id, 0)
        loaded = cached + len(transfer.gpu_block_ids)
        if (
            not cached < loaded <= pages <= len(request.block_hashes)
            or len(prefix) != pages
            or len(transfer.cpu_block_ids) != loaded - cached
            or [block.block_id for block in prefix[cached:loaded]] != transfer.gpu_block_ids
        ):
            return False
        for index, block in enumerate(prefix):
            if block.is_null or not block.ref_cnt:
                return False
            in_transfer = cached <= index < loaded
            if (block.block_id in tree.issued_load_destinations) != in_transfer:
                return False
            if index >= cached and block.block_hash is not None:
                return False
            if index < loaded:
                source = tree.host.pool.blocks[transfer.cpu_block_ids[index - cached]] if in_transfer else block
                expected_hash = make_block_hash_with_group_id(request.block_hashes[index], manager.kv_cache_group_id)
                if (
                    not source.ref_cnt
                    or source.block_hash != expected_hash
                    or source.block_hash_num_tokens != (index + 1) * tree.page_size
                ):
                    return False
        return True

    def publish_batch_load(
        self,
        request: Request,
        gpu_ids: list[int],
        cpu_ids: list[int],
        invalid_ids: set[int],
    ) -> None:
        """Publish acknowledged shared pages even when their DMA owner was canceled."""
        pool = self.block_pool
        tree = self.hiradix
        assert tree is not None
        (manager,) = self.coordinator.single_type_managers
        owned = manager.req_to_blocks.get(request.request_id, ())
        for gpu_id, cpu_id in zip(gpu_ids, cpu_ids, strict=True):
            if gpu_id in invalid_ids:
                break
            source = tree.host.pool.blocks[cpu_id]
            target = pool.blocks[gpu_id]
            page = tree.host.by_id[cpu_id]
            assert source.block_hash is not None and source.block_hash_num_tokens is not None
            assert target.block_hash is None or target.block_hash == source.block_hash
            pool._insert_block_hash(source.block_hash, target, num_tokens=source.block_hash_num_tokens)
            assert page.parent is not None
            parent_key = page.parent.key if page.parent is not tree.root else None
            tree.publish(target, page.key, parent_key, page.program)
            depth = source.block_hash_num_tokens // tree.page_size
            if depth <= len(owned) and owned[depth - 1] is target:
                manager.num_cached_block[request.request_id] = max(
                    manager.num_cached_block.get(request.request_id, 0), depth
                )

    def publish_hiradix(self, request: Request, confirmed_tokens: int, *, refresh_existing: bool) -> None:
        tree = self.hiradix
        if tree is None:
            return
        start = depth = self._radix_depth.get(request.request_id, 0)
        end = min(confirmed_tokens // tree.page_size, len(request.block_hashes))
        if end <= start and not refresh_existing:
            return
        keys = request.block_hashes
        if end > start:
            (blocks,) = self.coordinator.get_blocks(request.request_id)
            program_id = request_program(request)
            release_replaced = self.release_replaced_hiradix_blocks
            rebind = release_replaced is not None and not request.is_finished()
            replaced = []
            for index in range(start, min(end, len(blocks))):
                block = blocks[index]
                if block.is_null or block.block_hash_num_tokens != (index + 1) * tree.page_size:
                    break
                tree.publish(block, keys[index], keys[index - 1] if index else None, program_id)
                if rebind:
                    canonical = next(iter(tree.pages[keys[index]].device.values()))
                    if canonical is not block and canonical.block_hash == block.block_hash:
                        self.block_pool.touch([canonical])
                        blocks[index] = canonical
                        replaced.append(block)
                depth = index + 1
            if replaced:
                assert release_replaced is not None
                self._hiradix_block_table_updates.add(request.request_id)
                release_replaced(replaced, request.last_sched_seq)
                logger.info_once("MORI rebinds confirmed HiRadix pages through native V2 block-table updates")
            if depth > start:
                self._radix_depth[request.request_id] = depth
        if (depth > start or refresh_existing) and (published := min(end, depth)) > 0:
            tree.touch_prefix(keys, published)

    def take_hiradix_block_table_updates(self, scheduled_requests: dict[str, int]) -> dict[str, tuple[list[int], ...]]:
        # The model runner may skip applying metadata on an empty forward.
        ready = self._hiradix_block_table_updates.intersection(scheduled_requests)
        self._hiradix_block_table_updates.difference_update(ready)
        return {request_id: self.get_block_ids(request_id) for request_id in ready}

    def _retire_hiradix(self, request: Request) -> None:
        if self.hiradix is not None and request.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ):
            # Only a normal finish inserts private generated pages. Physical
            # retirement still follows the native request's reuse fence.
            self._cache_and_publish(
                request,
                min(request.num_tokens, request.num_computed_tokens - request.num_in_flight_tokens),
                refresh_existing=True,
            )
        self._replayed_prefill_tokens.pop(request.request_id, None)
        if self.mori_batch_loads:
            self._unpublished_batch_loads.discard(request.request_id)
        if self.release_replaced_hiradix_blocks is not None:
            self._hiradix_block_table_updates.discard(request.request_id)
        if (tree := self.hiradix) is not None:
            tree.release_load(request.request_id)
        self._radix_depth.pop(request.request_id, None)

    def free(self, request: Request) -> None:
        self._retire_hiradix(request)
        self._eviction_stores.pop(request.request_id, None)
        super().free(request)

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        self._retire_hiradix(request)
        self._eviction_stores.pop(request.request_id, None)
        return super().pop_blocks_for_free(request)
