# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Execute host KV transfers selected by MORI HiRadix."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus
from vllm.v1.simple_kv_offload.manager import (
    BlockStoreMeta,
    LoadRequestState,
    SimpleCPUOffloadScheduler,
    TransferMeta,
)

logger = init_logger("vllm.agentinfer.mori.offload")


class _TransferTree(Protocol):
    """Scheduler-side logical actions needed at native transfer boundaries."""

    def lookup_restore(self, request: Request, num_computed_tokens: int) -> list[KVCacheBlock] | None: ...

    def reserve_load(self, request_id: str, gpu_ids: list[int], cpu_ids: list[int]) -> None: ...

    def release_load(self, request_id: str) -> None: ...

    def discard_load_intent(self, request_id: str) -> None: ...

    def submit_batch_loads(self, gpu_ids: list[int]) -> None: ...

    def collect_restored_host(
        self,
        request_id: str,
        gpu_ids: list[int],
        cpu_ids: list[int],
        invalid: set[int],
        copies: dict[int, BlockHashWithGroupId],
    ) -> None: ...

    def release_restored_host(self, loaded: dict[int, BlockHashWithGroupId], *, published: bool) -> None: ...

    def drain_exclusive_loads(self) -> None: ...

    def register_writeback_replicas(self, gpu_ids: list[int], cpu_ids: list[int]) -> None: ...

    def retire_writeback_sources(self, gpu_ids: set[int]) -> None: ...

    def reset_loads(self) -> None: ...


class MoriHiCacheController(SimpleCPUOffloadScheduler):
    """Manage CPU slots and native transfers for one HiRadix cache."""

    inline_eviction_writeback: Callable[[list[int], list[int]], object] | None = None
    inline_retired_writeback: Callable[[list[int], list[int]], object] | None = None
    local_load_completed: Callable[[int], bool] | None = None
    _tree: _TransferTree | None = None
    mori_batch_loads = False
    publish_batch_load: Callable[[Request, list[int], list[int], set[int]], None] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mori_cpu_backend = self.kv_event_medium == MEDIUM_CPU
        self._eviction_store_gpu_ids: set[int] = set()
        self._prefetch_pins: dict[str, list[KVCacheBlock]] = {}

    def bind_tree(self, tree: _TransferTree) -> None:
        if self._tree is not None and self._tree is not tree:
            raise RuntimeError("MORI transfer controller is already bound")
        self._tree = tree

    def unbind_tree(self, tree: _TransferTree) -> None:
        if self._tree is tree:
            self._tree = None
            self.local_load_completed = None

    def completed_local_load(self, request: Request) -> TransferMeta | None:
        completed = self.local_load_completed
        if completed is None:
            return None
        state = self._reqs_to_load.get(request.request_id)
        if (
            state is None
            or state.request is not request
            or state.finished
            or state.load_event is None
            or state.load_event < 0
            or not completed(state.load_event)
        ):
            return None
        return state.transfer_meta

    def has_pending_evictions(self) -> bool:
        return bool(self._eviction_store_gpu_ids)

    def eviction_blocks_pending(self, blocks: tuple[int, ...]) -> bool:
        return not self._eviction_store_gpu_ids.isdisjoint(blocks)

    def reserve_host_pages(self, count: int) -> list[KVCacheBlock]:
        return self.cpu_block_pool.get_new_blocks(count)

    def writeback_queue_position(self) -> int:
        return len(self._pending_finished_stores)

    def writeback_inflight(self, gpu_id: int) -> bool:
        return gpu_id in self._in_flight_store_gpu_blocks

    def queue_writeback(self, gpu_ids: list[int], cpu_blocks: list[KVCacheBlock]) -> None:
        assert len(gpu_ids) == len(cpu_blocks)
        pool = self._gpu_block_pool
        assert pool is not None
        pool.touch([pool.blocks[block_id] for block_id in gpu_ids])
        self._in_flight_store_gpu_blocks.update(gpu_ids)
        self._eviction_store_gpu_ids.update(gpu_ids)
        self._pending_finished_stores.append(
            TransferMeta(
                gpu_ids,
                [block.block_id for block in cpu_blocks],
                [{} for _ in gpu_ids] if self.enable_kv_cache_events else None,
            )
        )

    def flush_inline_writeback(self, first_store: int, gpu_ids: list[int], *, retired: bool) -> bool:
        writeback = self.inline_eviction_writeback
        if writeback is None:
            return False
        if retired and self.inline_retired_writeback is not None:
            writeback = self.inline_retired_writeback
        stores = self._pending_finished_stores[first_store:]
        cpu_ids = [block_id for store in stores for block_id in store.cpu_block_ids]
        writeback(gpu_ids, cpu_ids)
        del self._pending_finished_stores[first_store:]
        for store in stores:
            self._in_flight_store_gpu_blocks.difference_update(store.gpu_block_ids)
            self._process_store_completion(store.gpu_block_ids, store.cpu_block_ids, store.block_meta)
        logger.info_once("MORI eviction writeback completes before the local scheduler selects its forward")
        return True

    def prefetch(self, request: Request, device: list[KVCacheBlock], host: list[KVCacheBlock]) -> None:
        """Use the native DMA queue for a restore even if the extend cannot run."""
        key = f"mori-prefetch-{uuid4().hex}"
        pool = self._gpu_block_pool
        destinations = pool.get_new_blocks(len(host))
        pool.touch(device)
        self.cpu_block_pool.touch(host)
        transfer = TransferMeta([block.block_id for block in destinations], [block.block_id for block in host])
        self._prefetch_pins[key] = device
        self._reqs_to_load[key] = LoadRequestState(request=request, transfer_meta=transfer)
        self._project_load(key, transfer)
        logger.debug_once("MORI restores a DRAM prefix before its waiting request can extend")

    def _project_load(self, request_id: str, transfer: TransferMeta) -> None:
        if self._tree is not None:
            self._tree.reserve_load(request_id, transfer.gpu_block_ids, transfer.cpu_block_ids)

    def _cleanup_load_request(self, req_id: str) -> int:
        if self._tree is not None:
            self._tree.discard_load_intent(req_id)
        if (ancestors := self._prefetch_pins.pop(req_id, None)) is not None:
            if self._tree is not None:
                self._tree.release_load(req_id)
            self._gpu_block_pool.free_blocks(reversed(ancestors))
        return super()._cleanup_load_request(req_id)

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        tree = self._tree
        if tree is None:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        stale = self._pending_cpu_hits.pop(request.request_id, None)
        if stale:
            self._free_pending_cpu_hit(stale)
        host = tree.lookup_restore(request, num_computed_tokens)
        if host is None:
            return None, False
        if not host:
            return 0, False
        self.cpu_block_pool.touch(host)
        matched = len(host) * self.block_size
        self._pending_cpu_hits[request.request_id] = ([host], matched, num_computed_tokens)
        return matched, not self.mori_batch_loads

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if self._tree is not None and num_external_tokens:
            state = self._reqs_to_load.get(request.request_id)
            if state is not None:
                self._project_load(request.request_id, state.transfer_meta)

    def release_unused_hits(self) -> None:
        # A deferred allocation did not claim these lookup-only pins.
        for hit in self._pending_cpu_hits.values():
            self._free_pending_cpu_hit(hit)
        self._pending_cpu_hits.clear()

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> SimpleCPUOffloadMetadata:
        tree = self._tree
        if tree is not None:
            tree.drain_exclusive_loads()
        metadata = super().build_connector_meta(scheduler_output)
        if tree is not None:
            tree.submit_batch_loads(metadata.load_gpu_blocks)
            if metadata.load_gpu_blocks and self._prefetch_pins:
                # Worker handles this exactly like other same-step prefill
                # loads, including the layer dependencies and native ACK.
                scheduler_output.has_sync_kv_loads = True
        return metadata

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None,
    ]:
        # MORI copies the exact pages selected by HiRadix eviction.
        return [], [], [], [] if self.enable_kv_cache_events else None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        tree = self._tree
        if tree is None:
            super().update_connector_output(connector_output)
            return
        if not connector_output.finished_recving:
            super().update_connector_output(connector_output)
            tree.drain_exclusive_loads()
            return

        publish = self.publish_batch_load
        invalid = connector_output.invalid_block_ids
        loaded: dict[int, BlockHashWithGroupId] = {}
        for request_id in connector_output.finished_recving:
            state = self._reqs_to_load.get(request_id)
            if state is None or state.load_event is None:
                continue
            transfer = state.transfer_meta
            if publish is not None:
                publish(state.request, transfer.gpu_block_ids, transfer.cpu_block_ids, invalid)
                tree.release_load(request_id)
            if state.finished and (
                not self.mori_batch_loads
                or state.request.status not in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)
            ):
                continue
            tree.collect_restored_host(request_id, transfer.gpu_block_ids, transfer.cpu_block_ids, invalid, loaded)
        super().update_connector_output(connector_output)
        if self.mori_batch_loads:
            # Native promotion applies only to requests parked for a transfer.
            connector_output.finished_recving = None
        tree.release_restored_host(loaded, published=publish is not None)

    def _process_store_completion(
        self,
        gpu_block_ids: list[int],
        cpu_block_ids: list[int],
        block_meta: list[dict[BlockHashWithGroupId, BlockStoreMeta]] | None = None,
    ) -> None:
        """Publish copied Host replicas before releasing native transfer refs."""
        tree = self._tree
        eviction_sources = self._eviction_store_gpu_ids.intersection(gpu_block_ids)
        if tree is not None:
            tree.register_writeback_replicas(gpu_block_ids, cpu_block_ids)
        # Native completion publishes hashes and events, then drops transfer refs.
        super()._process_store_completion(gpu_block_ids, cpu_block_ids, block_meta)
        self._eviction_store_gpu_ids.difference_update(eviction_sources)
        if tree is not None:
            tree.retire_writeback_sources(eviction_sources)

    def _release_transfer_refs(self, transfer: TransferMeta) -> None:
        self._eviction_store_gpu_ids.difference_update(transfer.gpu_block_ids)
        super()._release_transfer_refs(transfer)

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        if self._tree is not None:
            self._tree.release_load(request.request_id)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        result = self.request_finished(request, block_ids=[])
        if self._tree is not None:
            self._tree.drain_exclusive_loads()
        return result

    def reset(self) -> bool:
        if self._tree is not None:
            self._tree.reset_loads()
        return super().reset()
