# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Two-tier HiRadix over native vLLM allocations, owned by the request KV manager."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from itertools import chain, count, islice
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from agentinfer.agentcache.mori.programs import CacheClass, MoriProgramRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

    from agentinfer.agentcache.mori.block_pool import MoriBlockPool
    from agentinfer.agentcache.mori.offload import MoriHiCacheController
    from agentinfer.scheduling.domain import ProgramRef


_DEVICE_ORDER = {CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2}
_HOST_ORDER = {CacheClass.INACTIVE: 0, CacheClass.BUSY: 2.5, CacheClass.IDLE: 2}
LOAD_BACK_THRESHOLD = 10
logger = init_logger("vllm.agentinfer.mori.hiradix")


@dataclass(slots=True, eq=False)
class RadixNode:
    parent: RadixNode | None
    children: dict[bytes, RadixNode] = field(default_factory=dict)
    key: bytes = b""
    device: dict[int, KVCacheBlock] = field(default_factory=dict)
    host: dict[int, KVCacheBlock] = field(default_factory=dict)
    # A radix edge ends at an insertion boundary, a partial match, or a fork.
    boundary: bool = False
    program: str | ProgramRef | None = None
    shared: bool = False
    stamped_for: ProgramRef | None = None
    accessed: int = 0
    load_epoch: int = -1
    write_epoch: int = -1
    device_children: int = 0


class HiRadix:
    """Maintain cached prefix topology independently of the two block pools.

    Paths use vLLM's cumulative page hashes, so cache salt and other native
    namespace inputs remain part of the radix key. Native blocks retain their
    original ownership, reference counting and DMA lifetime.
    """

    def __init__(self, gpu: MoriBlockPool, cpu: MoriBlockPool, page_size: int):
        self.placement = MoriProgramRegistry()
        self.cache_controller: MoriHiCacheController | None = None
        self.page_size = page_size
        self.epoch = 0
        self.clock = count()
        self.root = RadixNode(None)
        self.pages: dict[bytes, RadixNode] = {}
        self.device = _TierIndex(self, gpu, host=False)
        self.host = _TierIndex(self, cpu, host=True)
        self._loading: dict[RadixNode, KVCacheBlock] = {}
        self._request_loads: dict[str, list[RadixNode]] = {}
        self._planned_load_edges: dict[str, tuple[tuple[RadixNode, ...], tuple[int, ...]]] = {}
        self._exclusive_load_edges: dict[str, tuple[int, ...]] = {}
        self._pending_exclusive_loads: dict[int, BlockHashWithGroupId] = {}
        self.share_batch_loads = False
        self.share_issued_loads = False
        self.batch_load_destinations: set[int] = set()
        self.issued_load_destinations: set[int] = set()
        self.prefill_host_sources: frozenset[int] = frozenset()
        self.native_schedule_active = False
        self._prepared_hbm_matches: dict[str, list[KVCacheBlock]] = {}
        # Canonical HBM hits already pinned in the native request block table.
        self.attached_hbm_prefixes: dict[str, int] = {}
        self._native_match: tuple[Request, list[KVCacheBlock], list[KVCacheBlock]] | None = None

    @classmethod
    def with_controller(
        cls,
        gpu: MoriBlockPool,
        page_size: int,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        cpu_capacity_per_rank: int,
    ) -> HiRadix:
        """Create the CPU controller and share its single physical pool."""
        from agentinfer.agentcache.mori.offload import MoriHiCacheController
        from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

        with mori_block_pool_constructor():
            controller = MoriHiCacheController(
                vllm_config,
                kv_cache_config,
                cpu_capacity_per_rank,
                scheduler_block_size=page_size,
                hash_block_size=page_size,
            )
        controller.bind_gpu_block_pool(gpu)
        tree = cls(gpu, controller.cpu_block_pool, page_size)
        tree.cache_controller = controller
        controller.bind_tree(tree)
        return tree

    def close_transfers(self) -> None:
        controller = self.cache_controller
        if controller is not None:
            controller.inline_eviction_writeback = None
            controller.inline_retired_writeback = None
            controller.publish_batch_load = None
            controller.unbind_tree(self)
        self.reset_loads()

    def has_inline_writeback(self) -> bool:
        return self.cache_controller is not None and self.cache_controller.inline_eviction_writeback is not None

    def evictions_pending(self) -> bool:
        return self.cache_controller is not None and self.cache_controller.has_pending_evictions()

    def eviction_blocks_pending(self, blocks: tuple[int, ...]) -> bool:
        return self.cache_controller is not None and self.cache_controller.eviction_blocks_pending(blocks)

    def release_unused_hits(self) -> None:
        if self.cache_controller is not None:
            self.cache_controller.release_unused_hits()

    def stage_evictions(
        self,
        count: int,
        protected: frozenset[int],
        *,
        reserved_hbm: int = 0,
        protected_host: frozenset[int] = frozenset(),
    ) -> tuple[int, ...] | None:
        """Demote whole device leaves, recycling host-only leaves as needed."""
        free_before = self.device.free_uncached()
        if count + reserved_hbm <= free_before:
            return ()
        controller = self.cache_controller
        assert controller is not None
        protected_host |= self.prefill_host_sources
        pending = []
        first_store = controller.writeback_queue_position()
        retired = controller.inline_retired_writeback is not None
        loading = {block.block_id for block in self._loading.values()} if retired else set()
        blocked = set(protected)
        reclaimed = 0
        # As in HiRadix, reclaim the entire demand when it does not fit.
        while reclaimed < count:
            last_reclaimed = reclaimed
            for group in self.device.plan(count - reclaimed, blocked, include_uncached=False):
                # Check native readiness before staging takes its own pin.
                if retired and any(
                    block_id in loading
                    or block_id not in self.device.ready
                    or block_id not in self.device.by_id
                    or not self.device.pool.is_idle_cached_block(block_id)
                    for block_id in group
                ):
                    retired = False
                staged, skipped = self._demote_device_edge(group, protected_host)
                pending.extend(staged)
                blocked.update(skipped)
            reclaimed = self.device.free_uncached() - free_before + len(pending)
            if reclaimed == last_reclaimed:
                break
        if pending and controller.flush_inline_writeback(first_store, pending, retired=retired):
            pending.clear()
        return tuple(pending) if pending or self.device.free_uncached() >= count + reserved_hbm else None

    def _demote_device_edge(self, group: list[int], protected_host: frozenset[int]) -> tuple[list[int], set[int]]:
        controller = self.cache_controller
        assert controller is not None
        pool = self.device.pool
        host_pool = self.host.pool
        victims = set(group)
        by_page: dict[RadixNode, int] = {}
        for block_id in group:
            page = self.device.by_id[block_id]
            key = pool.blocks[block_id].block_hash
            if (
                not page.host
                and page.device.keys() <= victims
                and key is not None
                and not controller.writeback_inflight(block_id)
                and (host_pool.hiradix_index is not None or host_pool.get_indexed_copy(key) is None)
            ):
                # Keep one Host copy per logical page after its last device replica.
                by_page.setdefault(page, block_id)
        destinations = self._reserve_host_pages(len(by_page), protected_host)
        staged = []
        if destinations:
            for page, block_id in by_page.items():
                page.write_epoch = self.epoch
                staged.append(block_id)
            # The victim plan walks leaf to root. Assign the allocated Host
            # slots root to leaf within this edge, as SGLang's node value does.
            destinations.reverse()
            controller.queue_writeback(staged, destinations)

        # Host exhaustion permits dropping an unbacked leaf, but the logical
        # path to a backed descendant must remain addressable.
        skip = set()
        staged_ids = set(staged)
        for block_id in group:
            if block_id in staged_ids:
                continue
            page = self.device.by_id[block_id]
            if page.host or len(page.device) > 1 or not page.children:
                pool.evict_blocks({block_id})
            else:
                skip.add(block_id)
        return staged, skip

    def _reserve_host_pages(self, count: int, protected: frozenset[int]) -> list[KVCacheBlock]:
        if count == 0:
            return []
        if count > self.host.free_uncached():
            groups = self.host.plan(count, protected, include_uncached=False)
            victims = {block_id for group in groups for block_id in group}
            for block_id in victims:
                self._pending_exclusive_loads.pop(block_id, None)
            self.host.pool.evict_blocks(victims)
        if count > self.host.free_uncached():
            return []
        controller = self.cache_controller
        assert controller is not None
        return controller.reserve_host_pages(count)

    def can_load(self, device: list[KVCacheBlock], host: list[KVCacheBlock]) -> bool:
        pages = len(host)
        if pages * self.page_size < LOAD_BACK_THRESHOLD:
            return False
        protected = {block.block_id for block in device if block.ref_cnt == 0}
        if pages > self.device.pool.get_num_free_blocks() - len(protected):
            return False
        return pages <= self.device.free_uncached() or sum(map(len, self.device.plan(pages, protected))) >= pages

    def lookup_restore(self, request: Request, num_computed_tokens: int) -> list[KVCacheBlock] | None:
        """Return Host sources for native allocation, or defer an active copy."""
        device, host = self.match_for_native_admission(request, connector=True)
        if len(device) * self.page_size != num_computed_tokens:
            return []
        if self.has_loading_blocks(host):
            return None
        return host if self.can_load(device, host) else []

    def prefetch(self, request: Request, device: list[KVCacheBlock], host: list[KVCacheBlock]) -> None:
        controller = self.cache_controller
        assert controller is not None
        controller.prefetch(request, device, host)

    def register_writeback_replicas(self, gpu_ids: list[int], cpu_ids: list[int]) -> None:
        for gpu_id, cpu_id in zip(gpu_ids, cpu_ids, strict=True):
            self.backup(gpu_id, self.host.pool.blocks[cpu_id])

    def retire_writeback_sources(self, gpu_ids: set[int]) -> None:
        pool = self.device.pool
        retired = {
            block_id
            for block_id in gpu_ids
            if pool.blocks[block_id].ref_cnt == 0 and pool.blocks[block_id].block_hash is not None
        }
        if retired:
            pool.evict_blocks(retired)

    def publish(
        self, block: KVCacheBlock, key: bytes, parent_key: bytes | None, program_id: str | ProgramRef | None
    ) -> None:
        parent = self.root if parent_key is None else self.pages.get(parent_key)
        if parent is None:
            raise RuntimeError("vLLM published a MORI page before its parent")
        page = self.pages.get(key)
        if page is None:
            if parent.children and parent is not self.root and not parent.boundary:
                # HiRadix touches the old compressed edge before splitting it
                # to insert a divergent suffix produced by a running request.
                self._touch_edge_suffix(parent, next(self.clock))
                parent.boundary = True
                parent.accessed = next(self.clock)
            page = RadixNode(parent, key=key, program=program_id, accessed=next(self.clock))
            self.pages[key] = page
            parent.children[key] = page
            self.host.refresh(parent)
        elif page.parent is not parent:
            raise RuntimeError("MORI page hash has incompatible radix parents")
        self.device.bind(page, block)

    def backup(self, gpu_id: int, cpu_block: KVCacheBlock) -> None:
        page = self.device.by_id.get(gpu_id)
        if page is not None:
            self.host.bind(page, cpu_block)

    @staticmethod
    def _load_edge_ends(pages: tuple[RadixNode, ...]) -> tuple[int, ...]:
        return tuple(i for i, page in enumerate(pages, 1) if page.boundary or i == len(pages))

    def reserve_load(self, request_id: str, gpu_ids: list[int], cpu_ids: list[int]) -> None:
        """Project the actual native H2D destinations into scheduling decisions."""
        assert request_id not in self._request_loads
        self._prepared_hbm_matches.clear()
        pages = []
        for gpu_id, cpu_id in zip(gpu_ids, cpu_ids, strict=True):
            page = self.host.by_id[cpu_id]
            assert page not in self._loading
            destination = self.device.pool.blocks[gpu_id]
            assert destination.ref_cnt > 0 and self.host.pool.blocks[cpu_id].ref_cnt > 0
            self._loading[page] = destination
            if self.share_batch_loads:
                self.batch_load_destinations.add(gpu_id)
            self.host.refresh(page)
            pages.append(page)
        self._request_loads[request_id] = pages
        planned = self._planned_load_edges.pop(request_id, None)
        if not self.placement.has_programs:
            return
        actual = tuple(pages)
        if planned is not None:
            # The prefix of the planned host path may be borrowed from an
            # earlier restore in the same pass.
            previous, ends = planned
            skipped = len(previous) - len(actual)
            if skipped >= 0 and previous[skipped:] == actual:
                self._exclusive_load_edges[request_id] = tuple(end - skipped for end in ends if end > skipped)
                return
        self._exclusive_load_edges[request_id] = self._load_edge_ends(actual)

    def collect_restored_host(
        self,
        request_id: str,
        gpu_ids: list[int],
        cpu_ids: list[int],
        invalid: set[int],
        copies: dict[int, BlockHashWithGroupId],
    ) -> None:
        edges = self._exclusive_load_edges.get(request_id)
        if not edges:
            return
        # A gap invalidates the rest of the radix prefix.
        valid = len(gpu_ids)
        if invalid:
            valid = next((i for i, block in enumerate(gpu_ids) if block in invalid), valid)
        pool = self.host.pool
        for block_id in self._host_blocks_to_release(cpu_ids, edges, valid):
            if (block_hash := pool.blocks[block_id].block_hash) is not None:
                copies[block_id] = block_hash

    def discard_load_intent(self, request_id: str) -> None:
        self._exclusive_load_edges.pop(request_id, None)

    def release_restored_host(self, loaded: dict[int, BlockHashWithGroupId], *, published: bool) -> None:
        if published:
            # Match SGLang: try once at ACK, leaving any still-pinned copy.
            pool = self.host.pool
            pool.evict_blocks(
                {
                    block_id
                    for block_id, key in loaded.items()
                    if pool.blocks[block_id].block_hash == key
                    and not pool.blocks[block_id].ref_cnt
                    and (page := self.host.by_id.get(block_id)) is not None
                    and page.device
                }
            )
        else:
            self._pending_exclusive_loads.update(loaded)
        self.drain_exclusive_loads()

    def drain_exclusive_loads(self) -> None:
        if not self._pending_exclusive_loads:
            return
        pool = self.host.pool
        disposable = set()
        retired = []
        for block_id, expected in self._pending_exclusive_loads.items():
            block = pool.blocks[block_id]
            if block.block_hash != expected:
                retired.append(block_id)
                continue
            if block.ref_cnt:
                continue
            if (page := self.host.by_id.get(block_id)) is not None and not page.device:
                continue
            disposable.add(block_id)
            retired.append(block_id)
        for block_id in retired:
            del self._pending_exclusive_loads[block_id]
        pool.evict_blocks(disposable)

    def _host_blocks_to_release(self, cpu_ids: list[int], edge_ends: tuple[int, ...], valid: int) -> Iterable[int]:
        start = 0
        for end in edge_ends:
            if start >= valid:
                break
            # Splitting an original SGLang node gives the prefix a new
            # identity. ACK still refers to the old node, now its suffix.
            for position in range(end - 2, start - 1, -1):
                if self.host.by_id[cpu_ids[position]].boundary:
                    start = position + 1
                    break
            yield from cpu_ids[start : min(end, valid)]
            start = end

    def release_load(self, request_id: str) -> None:
        reservation = self._request_loads.pop(request_id, None)
        if reservation is not None:
            self._prepared_hbm_matches.clear()
            for page in reservation:
                destination = self._loading.pop(page)
                self.issued_load_destinations.discard(destination.block_id)
                self.host.refresh(page)

    def reset_loads(self) -> None:
        for request_id in tuple(self._request_loads):
            self.release_load(request_id)
        self.batch_load_destinations.clear()
        self._planned_load_edges.clear()
        self._exclusive_load_edges.clear()
        self._pending_exclusive_loads.clear()

    def has_loading_blocks(self, blocks: Iterable[KVCacheBlock]) -> bool:
        return bool(self._loading) and any(self.host.by_id.get(block.block_id) in self._loading for block in blocks)

    def submit_batch_loads(self, gpu_ids: list[int]) -> None:
        if self.share_issued_loads and gpu_ids:
            self.issued_load_destinations.update(self.batch_load_destinations.intersection(gpu_ids))

    def shares_load_destination(self, block_id: int) -> bool:
        return block_id in self.batch_load_destinations or block_id in self.issued_load_destinations

    def waits_for_load(self, device: list[KVCacheBlock], host: list[KVCacheBlock]) -> bool:
        return bool(self._loading) and (
            any(
                block.block_id not in self.device.by_id and block.block_id not in self.issued_load_destinations
                for block in device
            )
            or self.has_loading_blocks(host)
        )

    def request_match(
        self,
        request,
        *,
        mark_load: bool = False,
        include_loading: bool = False,
        include_batch_loading: bool = False,
        include_last_token: bool = False,
        touch: bool = True,
        plan_load: bool = False,
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        if request.skip_reading_prefix_cache:
            return [], []
        max_pages = max(0, (request.num_tokens - int(not include_last_token)) // self.page_size)
        device, host = self.match(
            request.block_hashes[:max_pages],
            mark_load=mark_load,
            include_loading=include_loading,
            include_batch_loading=include_batch_loading,
            touch=touch,
        )
        if plan_load and self.share_batch_loads and host:
            pages = tuple(self.host.by_id[block.block_id] for block in host)
            self._planned_load_edges[request.request_id] = (pages, self._load_edge_ends(pages))
        return device, host

    def remember_hbm_match(self, request: Request, device: list[KVCacheBlock]) -> None:
        """Reuse the scheduler's selected HBM path until residency changes."""
        if not self._loading:
            self._prepared_hbm_matches[request.request_id] = device

    def match_for_native_admission(
        self, request: Request, *, connector: bool = False
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        # The scheduler asks its cache manager and connector consecutively
        # before either allocates. Only share that read within this pass.
        previous = self._native_match
        self._native_match = None
        if connector and previous is not None and previous[0] is request:
            return previous[1], previous[2]
        prepared = self._prepared_hbm_matches.get(request.request_id)
        if prepared is not None:
            device = prepared
            host: list[KVCacheBlock] = []
            logger.debug_once("MORI native admission reuses the selected HBM prefix")
        else:
            device, host = self.request_match(request, include_batch_loading=self.share_batch_loads, touch=False)
        if self.native_schedule_active and not connector:
            self._native_match = (request, device, host)
        return device, host

    def record_attached_hbm_prefix(self, request_id: str, blocks: Sequence[KVCacheBlock]) -> None:
        # Only accept the exact list produced by the scheduler's resident match.
        if self._prepared_hbm_matches.pop(request_id, None) is blocks:
            self.attached_hbm_prefixes[request_id] = len(blocks)

    def finish_native_schedule(self) -> None:
        self.native_schedule_active = False
        self._prepared_hbm_matches.clear()
        self._planned_load_edges.clear()
        self._native_match = None
        self.batch_load_destinations.clear()

    def match(
        self,
        keys: Iterable[bytes],
        *,
        mark_load: bool = False,
        include_loading: bool = False,
        include_batch_loading: bool = False,
        touch: bool = True,
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        device: list[KVCacheBlock] = []
        host: list[KVCacheBlock] = []
        page = self.root
        accessed = 0
        for key in keys:
            child = page.children.get(key)
            if child is None:
                break
            if not host and child.device:
                device.append(next(iter(child.device.values())))
            elif (
                not host
                and (include_loading or include_batch_loading)
                and (loading := self._loading.get(child)) is not None
                and (include_loading or self.shares_load_destination(loading.block_id))
            ):
                device.append(loading)
            elif child.host:
                host.append(next(iter(child.host.values())))
                if mark_load:
                    child.load_epoch = self.epoch
            else:
                break
            if touch:
                if page is self.root or page.boundary:
                    accessed = next(self.clock)
                child.accessed = accessed
            page = child
        if touch:
            self._finish_touch(page, accessed)
        return device, host

    def touch_prefix(self, keys: Iterable[bytes], depth: int, *, for_lpm: bool = False) -> int:
        """Touch match recency and count device hits without collecting physical blocks.

        LPM also counts in-flight device destinations and marks host hits for
        load-back. Publication sees only resident replicas and updates recency.
        """
        root = page = self.root
        accessed = 0
        on_host = False
        device_pages = 0
        for key in islice(keys, depth):
            child = page.children.get(key)
            if child is None:
                break
            if not on_host and (child.device or for_lpm and child in self._loading):
                device_pages += 1
            elif child.host:
                on_host = True
                if for_lpm:
                    child.load_epoch = self.epoch
            else:
                break
            if page is root or page.boundary:
                accessed = next(self.clock)
            child.accessed = accessed
            page = child
        self._finish_touch(page, accessed)
        return device_pages

    def _finish_touch(self, page: RadixNode, accessed: int) -> None:
        if page is not self.root and not page.boundary:
            # Matching into the middle of a compressed edge touches that
            # entire edge before splitting it, including its unmatched suffix.
            self._touch_edge_suffix(page, accessed)
            page.boundary = True
            page.accessed = next(self.clock)

    @staticmethod
    def _touch_edge_suffix(page: RadixNode, accessed: int) -> None:
        while not page.boundary and len(page.children) == 1:
            page = next(iter(page.children.values()))
            page.accessed = accessed

    def stamp(self, endpoint: KVCacheBlock, program_id: str | ProgramRef | None) -> None:
        page = self.device.by_id.get(endpoint.block_id)
        if page is not None:
            self._stamp(page, program_id)

    def stamp_prefill(self, key: bytes | None, program_id: str | ProgramRef) -> None:
        ref = self.placement.register(program_id)
        if key is None or ref is None:
            return
        endpoint = self.pages[key]
        self._stamp(endpoint, program_id, ref)
        endpoint.stamped_for = ref

    def _stamp(self, page: RadixNode, program_id: str | ProgramRef | None, ref: ProgramRef | None = None) -> None:
        if program_id is None:
            return
        while page.parent is not None:
            # As long as the same program is registered, an already stamped
            # ancestor can only keep its owner or become permanently shared.
            if ref is not None and page.stamped_for is ref:
                break
            if not page.shared and page.program is not program_id and page.program != program_id:
                if self.placement.has_program(page.program):
                    page.shared = True
                else:
                    page.program = program_id
            page = page.parent

    def forget(self, page: RadixNode) -> None:
        # A vLLM cache invalidation can make an orphaned suffix temporarily
        # unreachable. Keep its routes until its final backed page is gone.
        while page.parent is not None and not (page.children or page.device or page.host):
            parent = page.parent
            del parent.children[page.key]
            del self.pages[page.key]
            self.host.refresh(parent)
            self.device.refresh(parent)
            page = parent


class _TierIndex:
    """Pool callbacks and leaf-only victim selection for one memory tier."""

    def __init__(self, tree: HiRadix, pool: MoriBlockPool, *, host: bool):
        self.tree = tree
        self.pool = pool
        self.host = host
        self.by_id: dict[int, RadixNode] = {}
        self.unused: OrderedDict[int, None] = OrderedDict()
        self.ready: set[int] = set()
        self.leaves: set[RadixNode] = set()

    def replicas(self, page: RadixNode) -> dict[int, KVCacheBlock]:
        return page.host if self.host else page.device

    def bind(self, page: RadixNode, block: KVCacheBlock) -> None:
        block_id = block.block_id
        if self.by_id.get(block_id) is page:
            return
        self.tree._prepared_hbm_matches.clear()
        if block_id in self.by_id:
            self.on_block_removed(block_id)
        if not self.host and not page.device:
            assert page.parent is not None
            page.parent.device_children += 1
            self.leaves.discard(page.parent)
        self.by_id[block_id] = page
        self.unused.pop(block_id, None)
        self.replicas(page)[block_id] = block
        self.refresh(page)
        if not self.host:
            self.tree.host.refresh(page)

    def refresh(self, page: RadixNode) -> None:
        replicas = self.replicas(page)
        if (
            replicas
            and (
                not page.children and not page.device and page not in self.tree._loading
                if self.host
                else not page.device_children
            )
            and any(block_id in self.ready for block_id in replicas)
        ):
            self.leaves.add(page)
        else:
            self.leaves.discard(page)

    def discard_released_replicas(self, blocks: list[KVCacheBlock]) -> None:
        """A released second allocation no longer needs its own cache record."""
        redundant = set()
        for block in blocks:
            page = self.by_id.get(block.block_id)
            if page is None or len(page.device) < 2:
                continue
            canonical = next(iter(page.device.values()))
            if canonical is not block and canonical.block_hash == block.block_hash:
                redundant.add(block.block_id)
        if redundant:
            self.pool.evict_blocks(redundant)

    def on_block_ready(self, block_id: int) -> None:
        self.ready.add(block_id)
        page = self.by_id.get(block_id)
        if page is None:
            self.unused[block_id] = None
        else:
            self.refresh(page)

    def on_block_pinned(self, block_id: int) -> None:
        self.ready.discard(block_id)
        self.unused.pop(block_id, None)
        if (page := self.by_id.get(block_id)) is not None:
            self.refresh(page)

    def on_block_allocated(self, block_id: int) -> None:
        self.ready.discard(block_id)
        self.unused.pop(block_id, None)
        self.on_block_removed(block_id)

    def on_block_removed(self, block_id: int) -> None:
        page = self.by_id.pop(block_id, None)
        if page is None:
            return
        self.tree._prepared_hbm_matches.clear()
        replicas = self.replicas(page)
        del replicas[block_id]
        if not self.host:
            # Explicit invalidation can remove even a pinned canonical replica.
            if self.tree.attached_hbm_prefixes and self.pool.blocks[block_id].ref_cnt:
                self.tree.attached_hbm_prefixes.clear()
            if not replicas:
                assert page.parent is not None
                page.parent.device_children -= 1
                self.refresh(page.parent)
            self.tree.host.refresh(page)
        if block_id in self.ready:
            self.unused[block_id] = None
        self.refresh(page)
        self.tree.forget(page)

    def on_cache_reset(self) -> None:
        for block_id in tuple(self.by_id):
            self.on_block_removed(block_id)

    def _priority(self, page: RadixNode) -> tuple[int, float, int, int]:
        program = page.program
        placement = self.tree.placement
        if page.shared:
            priority: float = 3
        elif self.host:
            priority = _HOST_ORDER[placement.type_for_id(program)]
        else:
            priority = _DEVICE_ORDER[placement.type_for_id(program)]
        typed_host_pass = self.host and placement.has_programs
        return (
            int(typed_host_pass and page.load_epoch == self.tree.epoch),
            priority,
            int(typed_host_pass and page.write_epoch == self.tree.epoch),
            page.accessed,
        )

    def plan(self, count: int, protected: Iterable[int] = (), *, include_uncached: bool = True) -> list[list[int]]:
        """Plan whole compressed leaves, promoting their parents as they empty."""
        if count <= 0:
            return []
        protected = frozenset(protected)
        fresh: list[int] = []
        if include_uncached:
            for block_id in self.unused:
                if len(fresh) == count:
                    break
                if block_id not in protected:
                    fresh.append(block_id)
        groups = [fresh] if fresh else []
        remaining = count - len(fresh)
        if remaining <= 0:
            return groups

        heap = [(self._priority(page), page.key, page) for page in self.leaves]
        heapify(heap)
        removed: dict[RadixNode, int] = {}
        while heap and remaining > 0:
            _, _, leaf = heappop(heap)
            page = leaf
            edge = []
            while page.parent is not None:
                if self.host and page in self.tree._loading:
                    break
                replicas = self.replicas(page)
                ids = [block_id for block_id in replicas if block_id in self.ready and block_id not in protected]
                if not ids:
                    break
                edge.extend(ids)
                if len(ids) != len(replicas):
                    # A duplicate physical allocation may be reclaimed while
                    # a different replica keeps the content node resident.
                    break
                parent = page.parent
                child_count = len(parent.children) if self.host else parent.device_children
                removed[parent] = removed.get(parent, 0) + 1
                page = parent
                if removed[parent] < child_count or not self.replicas(parent) or (self.host and parent.device):
                    break
                if parent.boundary or child_count > 1:
                    heappush(heap, (self._priority(parent), parent.key, parent))
                    break
            if edge:
                groups.append(edge)
                remaining -= len(edge)
        return groups

    def free_uncached(self) -> int:
        return len(self.unused)

    def allocation_ids(self, count: int) -> list[int]:
        if self.pool.hiradix_index is self:
            # Writeback and host eviction retire shared entries before the
            # physical allocator takes only uncached pages in the dense path.
            return list(islice(self.unused, count))
        return list(islice(chain.from_iterable(self.plan(count)), count))
