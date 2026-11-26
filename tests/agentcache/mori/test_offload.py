# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

try:
    from vllm.v1.core.sched.output import KVConnectorBlockState as _KVConnectorBlockState  # noqa: F401
except ImportError:
    pytest.skip("MORI engine tests require the native KVConnectorBlockState ABI", allow_module_level=True)

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager
from tests.agentcache.mori.helpers import _request, _test_controller, insert, key, make_scheduler
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


def test_writeback_keeps_gpu_victim_order_and_assigns_host_slots_in_prefix_order(cache):
    controller = _test_controller(cache)
    controller.enable_kv_cache_events = False
    controller._in_flight_store_gpu_blocks = set()
    pages = insert(cache, (1, 2, 3, 4))
    cache.device.pool.get_new_blocks(cache.device.free_uncached())
    expected_victims = [page.block_id for page in reversed(pages)]

    assert cache.stage_evictions(4, frozenset()) == tuple(expected_victims)
    (transfer,) = controller._pending_finished_stores
    assert transfer.gpu_block_ids == expected_victims
    assert transfer.cpu_block_ids == [4, 3, 2, 1]
    assert all(page.ref_cnt == 1 for page in pages)
    controller._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
    device, host = cache.match([key(index) for index in (1, 2, 3, 4)])
    assert device == [] and [block.block_id for block in host] == [1, 2, 3, 4]
    assert all(block.ref_cnt == 0 for block in host)


@pytest.mark.parametrize("watched", [False, True])
def test_local_writeback_uses_native_retired_pages_only(cache, native_load, watched):
    offload, host = native_load
    offload._in_flight_store_gpu_blocks = set()
    offload._pending_finished_stores = []
    offload.enable_kv_cache_events = False
    pool = cache.device.pool
    (retired,) = insert(cache, (3,))
    (request_page,) = insert(cache, (4,))
    (cow_page,) = insert(cache, (5,))
    pool.touch([request_page, cow_page])
    if watched:
        pool.touch([retired])
        pool.unpin_blocks([retired], lambda _: None)

    (loading,) = pool.get_new_blocks(1)
    cache.host.pool.touch(host[:1])
    cache.reserve_load("pending-h2d", [loading.block_id], [host[0].block_id])
    pool.get_new_blocks(cache.device.free_uncached())
    assert cache.device.free_uncached() == 0

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.defer_block_free = True
    scheduler.sched_step_seq = 5
    scheduler.processed_step_seq = 3
    scheduler.deferred_frees = deque()
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=pool, pop_blocks_for_free=lambda _: [request_page])
    scheduler._free_request_blocks(SimpleNamespace(last_sched_seq=5))
    scheduler._free_cow_retained_blocks([cow_page], 6)

    assert set(cache.device.by_id) & cache.device.ready == {retired.block_id}
    assert loading.block_id not in cache.device.by_id
    calls = []

    def copy(kind, source, destination):
        assert cache.host.pool.blocks[destination[0]].ref_cnt == 1
        assert cache.host.pool.blocks[destination[0]].block_hash is None
        calls.append((kind, source.copy()))

    offload.inline_eviction_writeback = lambda gpu, cpu: copy("fenced", gpu, cpu)
    offload.inline_retired_writeback = lambda gpu, cpu: copy("retired", gpu, cpu)
    assert cache.stage_evictions(1, frozenset()) == ()
    assert calls == [("fenced" if watched else "retired", [retired.block_id])]
    assert retired.block_id not in cache.device.by_id
    assert request_page.ref_cnt == cow_page.ref_cnt == loading.ref_cnt == 1

    scheduler.processed_step_seq = 6
    scheduler._drain_deferred_frees()
    assert set(cache.device.by_id) & cache.device.ready == {request_page.block_id, cow_page.block_id}
    assert loading.ref_cnt == 1


@pytest.mark.parametrize("still_in_use", [False, True])
def test_store_ack_reuses_native_hash_and_event_publication(cache, native_load, still_in_use):
    import torch
    from vllm.distributed.kv_events import MEDIUM_CPU, BlockStored
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.simple_kv_offload.manager import TransferMeta

    offload, _ = native_load
    offload.enable_kv_cache_events = True
    offload.kv_event_medium = MEDIUM_CPU
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    offload.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    gpu, host = cache.device.pool, cache.host.pool
    gpu.bind_hiradix_index(cache.device, authoritative=True)
    host.bind_hiradix_index(cache.host, authoritative=True)
    (source,) = insert(cache, (3,))
    (destination,) = host.get_new_blocks(1)
    primary = source.block_hash
    secondary = make_block_hash_with_group_id(BlockHash(key(9)), 0)
    gpu._insert_block_hash(secondary, source, num_tokens=None)
    gpu.touch([source])
    if still_in_use:
        gpu.touch([source])
    offload._eviction_store_gpu_ids.add(source.block_id)
    offload._in_flight_store_gpu_blocks = {source.block_id}
    offload._store_event_to_blocks = {7: TransferMeta([source.block_id], [destination.block_id])}
    offload._abandoned_store_event_to_blocks = {}
    offload._store_event_to_reqs = {}

    offload._process_store_event(7)
    offload._process_store_event(7)

    assert destination.block_hash == primary and destination.ref_cnt == 0
    assert host.cached_block_hashes_by_block[destination.block_id] == {secondary}
    assert len(host.kv_event_queue) == 2
    assert all(isinstance(event, BlockStored) and event.medium == MEDIUM_CPU for event in host.kv_event_queue)
    assert cache.host.by_id[destination.block_id] is cache.pages[key(3)]
    assert source.ref_cnt == int(still_in_use) and (source.block_hash is not None) == still_in_use
    assert not offload._eviction_store_gpu_ids and not offload._in_flight_store_gpu_blocks


@pytest.mark.parametrize("issued", [False, True])
def test_reset_never_publishes_an_abandoned_writeback(cache, native_load, issued):
    from vllm.v1.simple_kv_offload.manager import BoundaryStoreStats

    offload, _ = native_load
    offload.enable_kv_cache_events = False
    offload._in_flight_store_gpu_blocks = set()
    offload._store_event_to_blocks = {}
    offload._abandoned_store_event_to_blocks = {}
    offload._abandoned_reqs_to_load = {}
    offload._store_event_to_reqs = {}
    offload._store_event_pending_counts = {}
    offload._store_event_counter = 0
    offload.boundary_store_stats = BoundaryStoreStats()
    offload._boundary_stats_snapshot = BoundaryStoreStats()
    (source,) = insert(cache, (3,))
    cache.device.pool.get_new_blocks(cache.device.free_uncached())
    assert cache.stage_evictions(1, frozenset()) == (source.block_id,)
    (cpu_id,) = offload._pending_finished_stores[0].cpu_block_ids
    destination = cache.host.pool.blocks[cpu_id]
    assert source.ref_cnt == destination.ref_cnt == 1

    if issued:
        metadata = offload.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
        assert metadata.store_gpu_blocks == [source.block_id]
    assert offload.reset() is not issued
    if issued:
        assert source.ref_cnt == destination.ref_cnt == 1
        assert cache.evictions_pending()
        offload._process_store_event(metadata.store_event)
        offload._process_store_event(metadata.store_event)

    assert source.ref_cnt == destination.ref_cnt == 0
    assert not cache.evictions_pending()
    assert cache.match([key(3)]) == ([source], [])
    assert destination.block_hash is None and cpu_id not in cache.host.by_id


@pytest.mark.parametrize("valid_tokens", [0, 16, 32])
def test_h2d_visible_for_lpm_without_publishing_early(cache, native_load, valid_tokens, step_without_worker):
    offload, host = native_load
    loading = _request("load", (1, 2), 33)
    follows = _request("same-prefix", (1, 2), 33)
    unrelated = _request("cold", (3, 4), 33)
    assert offload.get_num_new_matched_tokens(loading, 0) == (32, True)
    pool = cache.device.pool
    dest = pool.get_new_blocks(2)
    offload.update_state_after_alloc(loading, KVCacheBlocks((dest,)), 32)

    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager._radix_depth = {}
    manager._replayed_prefill_tokens = {}
    manager._eviction_stores = {}
    manager.enable_caching = True

    def index_valid_blocks(request, tokens):
        for index in range(tokens // 16):
            pool._insert_block_hash(
                make_block_hash_with_group_id(request.block_hashes[index], 0), dest[index], num_tokens=(index + 1) * 16
            )

    manager.coordinator = SimpleNamespace(
        cache_blocks=index_valid_blocks,
        get_blocks=lambda _: (dest,),
        free=lambda _: pool.free_blocks(dest),
    )
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = manager
    scheduler.tree_cache.cache_controller = offload
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    assert scheduler._order_waiting([unrelated, follows])[0] is follows
    assert cache.request_match(follows, include_loading=True) == (dest, [])
    assert manager.get_computed_blocks(follows)[1] == 0
    assert offload.get_num_new_matched_tokens(follows, 0) == (None, False)
    assert [block.ref_cnt for block in host] == [1, 1]

    scheduler.connector = offload
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.kv_cache_manager = manager
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue([follows])
    scheduler.kv_holding_waiting = FCFSRequestQueue([loading])
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.requests = {loading.request_id: loading}
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = {loading.request_id} if valid_tokens < 32 else set()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.prefix_replay_tokens = 0
    loading.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    # Invalid-block accounting has already rewound the count when ACK arrives.
    loading.num_computed_tokens = valid_tokens
    offload._reqs_to_load[loading.request_id].load_event = 0
    # Include a valid-looking suffix after an invalid first page; it cannot be published.
    invalid_ids = {dest[valid_tokens // 16].block_id} if valid_tokens < 32 else set()
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={loading.request_id}, invalid_block_ids=invalid_ids)
    )
    assert manager.get_computed_blocks(follows)[1] == 0
    assert not cache.host.leaves

    def native_schedule():
        assert loading.status == RequestStatus.WAITING
        assert manager.get_computed_blocks(follows)[1] == valid_tokens
        assert list(cache.host.by_id) == [block.block_id for block in host[valid_tokens // 16 :]]
        retry_tokens = 32 - valid_tokens
        assert offload.get_num_new_matched_tokens(follows, valid_tokens) == (retry_tokens, retry_tokens > 0)
        return SimpleNamespace(preempted_req_ids=set(), num_scheduled_tokens={})

    step_without_worker(scheduler, native_schedule)
    assert not offload._pending_cpu_hits
    assert [block.ref_cnt for block in host] == [0, 0]
    assert cache.request_match(follows)[1] == host[valid_tokens // 16 :]
    assert not cache._pending_exclusive_loads


@pytest.mark.parametrize("issued", [False, True])
def test_cancel_h2d_drops_projection_without_releasing_inflight_refs(cache, native_load, issued):
    offload, host = native_load
    request = _request("cancel", (1, 2), 33)
    assert offload.get_num_new_matched_tokens(request, 0) == (32, True)
    dest = cache.device.pool.get_new_blocks(2)
    offload.update_state_after_alloc(request, KVCacheBlocks((dest,)), 32)
    if issued:
        offload._reqs_to_load[request.request_id].load_event = 0
    offload.request_finished(request, [])
    assert cache.request_match(request, include_loading=True) == ([], host)
    assert [block.ref_cnt for block in host] == [int(issued)] * 2
    if issued:
        offload.update_connector_output(KVConnectorOutput(finished_recving={request.request_id}))
    assert [block.ref_cnt for block in host] == [0, 0]
    assert not cache._pending_exclusive_loads


@pytest.mark.parametrize("active_at_dispatch", [False, True])
def test_host_copy_drop_uses_program_activity_at_dispatch(cache, native_load, active_at_dispatch):
    offload, host = native_load
    if not active_at_dispatch:
        cache.placement.release("p")
    request = _request("load", (1, 2), 33)
    assert offload.get_num_new_matched_tokens(request, 0) == (32, True)
    dest = cache.device.pool.get_new_blocks(2)
    offload.update_state_after_alloc(request, KVCacheBlocks((dest,)), 32)
    if active_at_dispatch:
        cache.placement.release("p")
    else:
        cache.placement.register("p")
    offload._reqs_to_load[request.request_id].load_event = 0
    offload.update_connector_output(KVConnectorOutput(finished_recving={request.request_id}))
    assert set(cache._pending_exclusive_loads) == ({block.block_id for block in host} if active_at_dispatch else set())


def test_pending_host_drop_cannot_discard_a_reused_block(cache, native_load):
    offload, host = native_load
    block = host[0]
    pool = cache.host.pool
    cache._pending_exclusive_loads[block.block_id] = block.block_hash
    pool.evict_blocks({block.block_id})
    new_hash = make_block_hash_with_group_id(BlockHash(key(9)), 0)
    pool._insert_block_hash(new_hash, block, num_tokens=16)
    cache.drain_exclusive_loads()
    assert block.block_hash == new_hash
    assert not cache._pending_exclusive_loads


def test_batch_load_keeps_exclusive_policy_until_late_ack(cache, native_load):
    offload, host = native_load
    offload.mori_batch_loads = True
    request = _request("load", (1, 2), 33)
    assert offload.get_num_new_matched_tokens(request, 0) == (32, False)
    dest = cache.device.pool.get_new_blocks(2)
    offload.update_state_after_alloc(request, KVCacheBlocks((dest,)), 32)
    offload._reqs_to_load[request.request_id].load_event = 0
    cache.release_load(request.request_id)
    assert request.request_id not in cache._request_loads

    output = KVConnectorOutput(finished_recving={request.request_id})
    offload.update_connector_output(output)
    assert output.finished_recving is None
    assert not offload._reqs_to_load
    assert not cache._exclusive_load_edges
    assert set(cache._pending_exclusive_loads) == {block.block_id for block in host}


@pytest.mark.parametrize("restore", ["standalone", "planned-batch", "exclusive-async"])
@pytest.mark.parametrize("split_before_restore", [False, True])
def test_host_restore_ack_releases_the_original_edges_current_suffix(cache, native_load, restore, split_before_restore):
    offload, host = native_load
    pool = cache.device.pool
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(single_type_managers=[SimpleNamespace(req_to_blocks={}, num_cached_block={})])
    batch = restore != "exclusive-async"
    if batch:
        offload.mori_batch_loads = cache.share_batch_loads = True
        offload.publish_batch_load = manager.publish_batch_load

    owner = _request("restore", (1, 2), 33)
    borrower = _request("borrower", (1,), 17)

    def split():
        cache.request_match(borrower, include_loading=True)
        assert cache.pages[key(1)].boundary

    if split_before_restore:
        split()
    if restore == "planned-batch":
        cache.request_match(owner, include_loading=True, plan_load=True)
        if not split_before_restore:
            split()
    if restore == "standalone":
        cache.prefetch(owner, [], host)
    else:
        assert offload.get_num_new_matched_tokens(owner, 0) == (32, not batch)
        dest = pool.get_new_blocks(2)
        offload.update_state_after_alloc(owner, KVCacheBlocks((dest,)), 32)
    if not split_before_restore and restore != "planned-batch":
        split()
    (request_id,) = offload._reqs_to_load
    offload._reqs_to_load[request_id].load_event = 0
    assert [block.ref_cnt for block in host] == [1, 1]
    offload.update_connector_output(KVConnectorOutput(finished_recving={request_id}))
    assert not cache._exclusive_load_edges
    assert [block.ref_cnt for block in host] == [0, 0]
    retained = set() if split_before_restore else {host[0].block_id}
    if batch:
        assert cache.host.by_id.keys() == retained
    else:
        assert cache._pending_exclusive_loads.keys() == set(cache.host.by_id) - retained


@pytest.mark.parametrize(("cancel_owner", "invalid_tail"), [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("issued", [False, True])
def test_h2d_prefix_is_shared_and_published_on_ack(cache, native_load, cancel_owner, invalid_tail, issued):
    offload, _ = native_load
    offload.mori_batch_loads = cache.share_batch_loads = True
    cache.share_issued_loads = issued
    owner = _request("load-owner", (1, 2), 33)
    borrower = _request("load-borrower", (1, 2), 33)
    pool = cache.device.pool
    dest = pool.get_new_blocks(2)

    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.enable_caching = manager.mori_batch_loads = True
    owned = {} if cancel_owner else {owner.request_id: dest}
    native = SimpleNamespace(req_to_blocks=owned, num_cached_block={})
    manager.coordinator = SimpleNamespace(single_type_managers=[native])
    offload.publish_batch_load = manager.publish_batch_load

    assert offload.get_num_new_matched_tokens(owner, 0) == (32, False)
    offload.update_state_after_alloc(owner, KVCacheBlocks((dest,)), 32)
    assert manager.get_computed_blocks(borrower)[0].blocks == (dest,)
    assert offload.get_num_new_matched_tokens(borrower, 32) == (0, False)
    assert all(block.block_hash is None for block in dest)
    pool.touch(dest)

    cache.submit_batch_loads([block.block_id for block in dest])
    cache.finish_native_schedule()
    device, host = cache.request_match(borrower, include_loading=True)
    assert cache.waits_for_load(device, host) is not issued
    if issued:
        assert manager.get_computed_blocks(borrower)[0].blocks == (dest,)
        assert offload.get_num_new_matched_tokens(borrower, 32) == (0, False)
    else:
        assert manager.get_computed_blocks(borrower)[1] == 0
        assert offload.get_num_new_matched_tokens(borrower, 0) == (None, False)
    offload._reqs_to_load[owner.request_id].load_event = 0
    if cancel_owner:
        offload.request_finished(owner, [])
        pool.free_blocks(dest)

    invalid = {dest[-1].block_id} if invalid_tail else set()
    offload.update_connector_output(KVConnectorOutput(finished_recving={owner.request_id}, invalid_block_ids=invalid))
    depth = 1 if invalid_tail else 2
    assert manager.get_computed_blocks(borrower)[0].blocks == (dest[:depth],)
    assert not cache.issued_load_destinations
    assert [page.block_hash_num_tokens for page in dest] == [16, None if invalid_tail else 32]
    assert native.num_cached_block == ({} if cancel_owner else {owner.request_id: depth})
    pool.free_blocks(dest)


def test_writeback_pressure_requests_full_eviction_size(cache, native_load, monkeypatch):
    offload, _ = native_load
    (a,) = insert(cache, (9,))
    (b,) = insert(cache, (10,))
    (c,) = insert(cache, (11,))
    cache.device.pool.get_new_blocks(4)
    assert cache.device.free_uncached() == 1
    victims = []

    def demote(group, _):
        victims.extend(group)
        cache.device.pool.evict_blocks(set(group))
        return [], set()

    monkeypatch.setattr(cache, "_demote_device_edge", demote)
    assert cache.stage_evictions(2, frozenset()) == ()
    assert victims == [a.block_id, b.block_id]
    assert c.block_id in cache.device.by_id
