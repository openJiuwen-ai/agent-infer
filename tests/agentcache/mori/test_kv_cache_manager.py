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

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.mori.block_pool import MoriBlockPool
from agentinfer.agentcache.mori.hiradix import HiRadix
from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager
from agentinfer.agentcache.mori.scheduler import PrefillBatch
from tests.agentcache.mori.helpers import _request, _test_controller, backup, insert, key, make_scheduler
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


@pytest.mark.parametrize(
    ("published", "native_cached", "refresh", "skip"),
    [
        (1, 1, False, True),
        (0, 1, False, False),
        (1, 0, False, False),
        (1, 1, True, False),
    ],
)
def test_decode_result_reuses_confirmed_prompt(cache, monkeypatch, published, native_cached, refresh, skip):
    request = SimpleNamespace(request_id="decode", num_prompt_tokens=17, num_output_tokens=32)
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.mori_batch_loads = True
    manager.hiradix = cache
    manager._replayed_prefill_tokens = {}
    manager._radix_depth = {"decode": published}
    manager.coordinator = SimpleNamespace(
        single_type_managers=(SimpleNamespace(num_cached_block={"decode": native_cached}),)
    )
    actions = []
    monkeypatch.setattr(cache, "release_load", lambda rid: actions.append(("release", rid)))
    monkeypatch.setattr(manager, "_cache_and_publish", lambda req, tokens, **kw: actions.append(("publish", tokens)))
    manager.cache_blocks(request, 48, refresh_existing=refresh)
    assert actions == ([("release", "decode")] if skip else [("publish", 17)])


def test_confirmed_hiradix_insertion_refreshes_previously_published_edges(cache):
    (older,) = insert(cache, (1,))
    (newer,) = insert(cache, (2,))
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = cache.device.pool
    manager.hiradix = cache
    request = _request("cache-hit", (1,), 17)
    manager._radix_depth = {request.request_id: 1}

    assert cache.device.plan(1, include_uncached=False) == [[older.block_id]]
    manager.confirm_hiradix(request, 17)
    assert cache.device.plan(1, include_uncached=False) == [[older.block_id]]
    manager.confirm_hiradix(request, 17, refresh_existing=True)
    assert cache.device.plan(1, include_uncached=False) == [[newer.block_id]]


def test_inserting_terminal_prefix_touches_old_edge_before_splitting(cache):
    first, second, _, _ = insert(cache, (1, 2, 3, 4))
    assert cache.match([key(1)]) == ([first], [])
    (unrelated,) = insert(cache, (5,))
    assert not cache.pages[key(2)].boundary

    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = cache.device.pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(get_blocks=lambda _: ([first, second],))
    manager.release_replaced_hiradix_blocks = None
    request = _request("terminal", (1, 2), 32)
    manager._radix_depth = {request.request_id: 1}

    manager.publish_hiradix(request, 32, refresh_existing=True)

    assert cache.pages[key(2)].boundary
    assert cache.device.plan(1, include_uncached=False) == [[unrelated.block_id]]


@pytest.mark.parametrize(
    "case", ["suffix", "partial", "invalidated", "retracted", "aborted", "unprepared", "duplicate", "pressure"]
)
def test_first_publication_of_attached_hbm_prefix(monkeypatch, case):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

    page = 16
    spec = FullAttentionSpec(block_size=page, num_kv_heads=1, head_size=64, dtype=torch.float16)
    config = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer"], spec)])
    with mori_block_pool_constructor():
        manager = MoriKVCacheManager(config, max_model_len=256, scheduler_block_size=page, hash_block_size=page)
    gpu = manager.block_pool
    host = MoriBlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=page)
    tree = manager.hiradix = HiRadix(gpu, host, page)
    gpu.bind_hiradix_index(tree.device, authoritative=True)
    a, b = insert(tree, (1, 2), program="first")
    request = _request("second", (1, 2, 3), 49)
    matches, _ = tree.request_match(request)
    tree.stamp_prefill(key(2), "second")
    assert tree.pages[key(2)].shared
    if case != "unprepared":
        tree.remember_hbm_match(request, matches)
    hits, local, _ = manager.get_computed_blocks(request)
    if case == "duplicate":
        (other,) = gpu.get_new_blocks(1)
        gpu._insert_block_hash(b.block_hash, other, num_tokens=2 * page)
        hits = KVCacheBlocks(([a, other],))
        manager.release_replaced_hiradix_blocks = lambda blocks, _: gpu.free_blocks(blocks)
    allocated = manager.allocate_slots(
        request, 128 if case == "pressure" else page, new_computed_blocks=hits, num_new_computed_tokens=local
    )
    tree.finish_native_schedule()
    if case == "pressure":
        assert allocated is None
        assert not tree.attached_hbm_prefixes and a.ref_cnt == b.ref_cnt == 0
        return
    assert allocated is not None
    assert a.ref_cnt == 1
    assert b.ref_cnt == (0 if case == "duplicate" else 1)
    table = manager.get_blocks(request.request_id).blocks[0]
    assert table[:2] == [a, other if case == "duplicate" else b]
    assert bool(tree.attached_hbm_prefixes) is (case not in ("unprepared", "duplicate"))
    if case == "retracted":
        request.status = RequestStatus.RUNNING
        manager.free(request)
        assert not tree.attached_hbm_prefixes and a.ref_cnt == b.ref_cnt == 0
        return
    if case == "aborted":
        request.status = RequestStatus.FINISHED_ABORTED
        held = manager.pop_blocks_for_free(request)
        assert not tree.attached_hbm_prefixes and a.ref_cnt == b.ref_cnt == 1
        gpu.free_blocks(held)
        assert a.ref_cnt == b.ref_cnt == 0
        return
    if case == "invalidated":
        gpu.evict_blocks({b.block_id})
    published = []
    publish = tree.publish

    def observe(block, *args):
        published.append(block)
        publish(block, *args)

    monkeypatch.setattr(tree, "publish", observe)
    if case == "partial":
        original_recency = tree.pages[key(1)].accessed
        manager.confirm_hiradix(request, page)
        assert published == [] and manager._radix_depth[request.request_id] == 1
        assert tree.pages[key(1)].accessed > original_recency and tree.pages[key(1)].boundary
    manager.confirm_hiradix(request, 3 * page)
    if case == "invalidated":
        assert published == [a] and manager._radix_depth[request.request_id] == 1
    elif case == "duplicate":
        assert published[:2] == [a, other] and table[1] is b
        assert b.ref_cnt == other.ref_cnt == 1  # The producer still holds the duplicate.
        assert manager.take_hiradix_block_table_updates({request.request_id: 1})
        gpu.free_blocks([other])
        assert other.block_hash is None
    else:
        start = {"unprepared": 0, "partial": 1, "suffix": 2}[case]
        assert published == list(table[start:])
        assert manager._radix_depth[request.request_id] == 3
        assert tree.pages[key(2)].shared and tree.pages[key(3)].boundary
    assert not tree.attached_hbm_prefixes


def test_confirmed_active_request_reuses_canonical_pages_at_the_worker_fence(cache):
    (canonical,) = insert(cache, (1,))
    pool = cache.device.pool
    (duplicate,) = pool.get_new_blocks(1)
    pool._insert_block_hash(canonical.block_hash, duplicate, num_tokens=16)
    request = _request("duplicate", (1,), 17)
    request.last_sched_seq = 19
    blocks = [duplicate]
    releases = []
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(get_blocks=lambda _: (blocks,))
    manager._radix_depth = {}
    manager._hiradix_block_table_updates = set()
    manager.release_replaced_hiradix_blocks = lambda pages, fence: releases.append((pages, fence))
    manager.get_block_ids = lambda _: ([block.block_id for block in blocks],)

    manager.confirm_hiradix(request, 17)
    assert blocks == [canonical]
    assert canonical.ref_cnt == duplicate.ref_cnt == 1
    assert releases == [([duplicate], 19)]
    assert manager.take_hiradix_block_table_updates({}) == {}
    assert manager.take_hiradix_block_table_updates({request.request_id: 1}) == {
        request.request_id: ([canonical.block_id],)
    }
    assert manager.take_hiradix_block_table_updates({request.request_id: 1}) == {}

    pool.free_blocks(releases[0][0])
    assert cache.match([key(1)]) == ([canonical], [])
    assert duplicate.block_hash is None


@pytest.mark.parametrize("retract", [False, True])
def test_generated_pages_are_private_until_finish_or_replayed_prefill(retract):
    import torch
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    from agentinfer.agentcache.mori.scheduler import MoriAsyncScheduler
    from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

    page = 16
    spec = FullAttentionSpec(block_size=page, num_kv_heads=1, head_size=64, dtype=torch.float16)
    config = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer"], spec)])
    with mori_block_pool_constructor():
        manager = MoriKVCacheManager(config, max_model_len=128, scheduler_block_size=page, hash_block_size=page)
    gpu = manager.block_pool
    host = MoriBlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=page)
    tree = HiRadix(gpu, host, page)
    manager.hiradix = tree
    gpu.bind_hiradix_index(tree.device, authoritative=True)
    scheduler = MoriAsyncScheduler.__new__(MoriAsyncScheduler)
    scheduler.kv_cache_manager = manager
    scheduler.max_model_len = 128
    init_none_hash(sha256)
    hasher = get_request_block_hasher(page, sha256)
    request = Request(
        "decoder", list(range(page)), SamplingParams(max_tokens=19 if retract else 18), None, block_hasher=hasher
    )

    def process(tokens, next_token, **slots):
        assert manager.allocate_slots(request, tokens, **slots) is not None
        request.num_computed_tokens += tokens + slots.get("num_new_computed_tokens", 0)
        request.num_output_placeholders = 1
        request.status = RequestStatus.RUNNING
        _, stopped = scheduler._update_request_with_output(request, [next_token])
        return stopped

    assert not process(page, 30)
    for token in range(31, 47):
        assert not process(1, token)
    assert request.num_output_tokens == 17 and request.num_computed_tokens == 32
    follower = Request("follower", list(request.all_token_ids), SamplingParams(max_tokens=1), None, block_hasher=hasher)
    prompt, suffix = manager.get_blocks(request.request_id).blocks[0]
    assert manager.get_computed_blocks(follower)[1] == page
    assert gpu.get_cached_block(request.block_hashes[1], [0]) is None
    assert suffix.block_hash is None and suffix.block_id not in tree.device.by_id

    if not retract:
        assert process(1, 47)
        manager.free(request)
        assert manager.get_computed_blocks(follower)[1] == 2 * page
        assert suffix.block_hash is not None and suffix.block_id in tree.device.by_id
        return

    # The native scheduler still protects the writer's public and private KV
    # through its async free fence. Retraction does not insert the suffix.
    scheduler.defer_block_free = True
    scheduler.sched_step_seq, scheduler.processed_step_seq = 4, 3
    scheduler.deferred_frees = deque()
    scheduler.aux_output_connector = None
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _: None)
    scheduler._inflight_prefills = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.log_stats = False
    scheduler.waiting = FCFSRequestQueue()
    request.last_sched_seq = 4
    scheduler._preempt_request(request, 0)
    assert request.status == RequestStatus.PREEMPTED
    assert prompt.ref_cnt == suffix.ref_cnt == 1
    assert suffix.block_id not in tree.device.ready
    assert manager.get_computed_blocks(follower)[1] == page
    scheduler.processed_step_seq = 4
    scheduler._drain_deferred_frees()
    assert prompt.ref_cnt == suffix.ref_cnt == 0
    assert suffix.block_id in tree.device.unused and suffix.block_hash is None

    # Replay known output tokens as prefill before entering decode again.
    prefix, hit, _ = manager.get_computed_blocks(request)
    assert not process(request.num_tokens - hit, 47, num_new_computed_tokens=hit, new_computed_blocks=prefix)
    assert manager.get_computed_blocks(follower)[1] == 2 * page


@pytest.mark.parametrize(("local_worker", "connector_pending"), [(True, False), (True, True), (False, False)])
def test_next_lpm_can_match_previously_dispatched_local_prefill_chunk(
    cache, monkeypatch, local_worker, connector_pending, step_without_worker
):
    pool = cache.device.pool
    request = _request("chunk", (1, 2, 3), 49)
    request.is_prefill_chunk = True
    request.last_sched_seq = 2
    request.num_computed_tokens = 24
    request.num_in_flight_tokens = 24
    full, partial = pool.get_new_blocks(2)
    pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(key(1)), 0), full, num_tokens=16)
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(get_blocks=lambda _: ([full, partial],))
    manager._radix_depth = {}
    manager._unpublished_batch_loads = {request.request_id} if connector_pending else set()
    borrower = _request("borrower", (1, 2), 33)
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = manager
    scheduler.running = [request]
    scheduler.waiting = FCFSRequestQueue([borrower])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_active_reqs = 4
    scheduler.processed_step_seq = 1
    scheduler.sched_step_seq = 2
    _test_controller(cache).inline_eviction_writeback = object() if local_worker else None
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    seen = []

    def match_before_prefill(waiting):
        assert waiting == [borrower]
        seen.append(cache.request_match(borrower))
        return waiting

    monkeypatch.setattr(scheduler, "_order_waiting", match_before_prefill)
    monkeypatch.setattr(scheduler, "_prefer_prefill", lambda *args: PrefillBatch(tokens=16))
    monkeypatch.setattr(scheduler, "_prepare_prefill", lambda *args: True)
    step_without_worker(scheduler)
    assert seen == [([full] if local_worker and not connector_pending else [], [])]
    assert partial.block_hash is None


@pytest.mark.parametrize(
    "completion", ["ready", "hwm", "pending", "wrong_event", "wrong_hash", "additional_load", "stale_forward", "fatal"]
)
def test_mixed_chunk_stash_uses_certified_h2d_without_consuming_ack(completion):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
    from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

    page = 16
    spec = FullAttentionSpec(block_size=page, num_kv_heads=1, head_size=64, dtype=torch.float16)
    config = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer"], spec)])
    with mori_block_pool_constructor():
        manager = MoriKVCacheManager(config, max_model_len=128, scheduler_block_size=page, hash_block_size=page)
    gpu = manager.block_pool
    host = MoriBlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=page)
    tree = manager.hiradix = HiRadix(gpu, host, page)
    gpu.bind_hiradix_index(tree.device, authoritative=True)
    host.bind_hiradix_index(tree.host, authoritative=True)
    controller = _test_controller(tree)
    controller.publish_batch_load = manager.publish_batch_load
    controller.mori_batch_loads = manager.mori_batch_loads = True
    tree.share_batch_loads = tree.share_issued_loads = True

    (prefix,) = insert(tree, (1,))
    (source,) = backup(tree, [prefix])
    gpu.evict_blocks({prefix.block_id})
    request = _request("owner", (1, 2, 3), 49)
    assert controller.get_num_new_matched_tokens(request, 0) == (page, False)
    assert manager.allocate_slots(request, 2 * page, num_external_computed_tokens=page)
    blocks = manager.get_blocks(request.request_id)
    (destination, computed, _) = blocks.blocks[0]
    controller.update_state_after_alloc(request, blocks, page)
    state = controller._reqs_to_load[request.request_id]
    state.load_event = 2
    tree.submit_batch_loads([destination.block_id])
    tree.finish_native_schedule()
    request.num_computed_tokens = 3 * page
    request.status = RequestStatus.RUNNING

    worker = SimpleCPUOffloadWorker.__new__(SimpleCPUOffloadWorker)
    worker_connector = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    worker_connector.worker_handler = worker
    controller.local_load_completed = worker_connector.local_load_completed
    queried = []

    def query():
        queried.append(True)
        if completion == "fatal":
            raise RuntimeError("local CUDA copy failed")
        return completion != "pending"

    worker._load_hwm = 2 if completion == "hwm" else -1
    event = (1 if completion == "wrong_event" else 2, SimpleNamespace(query=query))
    worker._load_events = [] if completion == "hwm" else [event]
    if completion == "wrong_hash":
        request.block_hashes[0] = BlockHash(key(99))
    if completion == "additional_load":
        tree.issued_load_destinations.add(computed.block_id)

    if completion == "fatal":
        with pytest.raises(RuntimeError, match="local CUDA copy failed"):
            manager.stash_hiradix_chunk(request, latest_forward=True)
        assert computed.block_hash is None
        return

    manager.stash_hiradix_chunk(request, latest_forward=completion != "stale_forward")

    stashed = completion in ("ready", "hwm")
    assert (gpu.get_cached_block(request.block_hashes[1], [0]) is not None) is stashed
    assert len(queried) == int(completion not in ("hwm", "wrong_event", "stale_forward"))
    assert destination.ref_cnt == 2 and source.ref_cnt == 1
    assert controller._reqs_to_load[request.request_id] is state
    assert request.request_id in manager._unpublished_batch_loads
    if stashed:
        controller.update_connector_output(KVConnectorOutput(finished_recving={request.request_id}))
        assert destination.ref_cnt == 1 and source.ref_cnt == 0
        assert controller._interval_load_blocks_completed == 1
        manager.confirm_hiradix(request, request.num_computed_tokens)
        assert gpu.get_cached_block(request.block_hashes[1], [0]) == [computed]


def test_native_allocation_preserves_a_later_waiters_host_source(monkeypatch):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
    from agentinfer.agentcache.mori.schedule_batch import ScheduleBatch
    from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    page_bytes = spec.page_size_bytes
    config = KVCacheConfig(
        12,
        [KVCacheTensor(page_bytes * 12, ["layer"], page_bytes * 12, page_bytes)],
        [KVCacheGroupSpec(["layer"], spec)],
    )
    settings = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=16),
        kv_events_config=None,
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=True,
            kv_connector_extra_config={
                "cpu_bytes_to_use": page_bytes * 7,
            },
        ),
        max_in_flight_tokens=128,
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(world_size=1, decode_context_parallel_size=1, pipeline_parallel_size=1),
        speculative_config=None,
    )
    with mori_block_pool_constructor():
        manager = MoriKVCacheManager(config, max_model_len=128, scheduler_block_size=16, hash_block_size=16)
    connector = MoriCPUOffloadConnector(settings, KVConnectorRole.SCHEDULER, config)
    connector.bind_kv_cache_manager(manager)
    cold = _request("cold", (100,), 17)
    waiter = _request("restore", (1,), 17)
    state = SimpleNamespace(
        vllm_config=settings,
        kv_cache_config=config,
        kv_cache_manager=manager,
        connector=connector,
        use_v2_model_runner=True,
        defer_block_free=True,
        parallel_config=settings.parallel_config,
        recompute_kv_load_failures=False,
        needs_kv_cache_zeroing=False,
        _free_cow_retained_blocks=lambda *_: None,
        prefix_replay_tokens=0,
        need_mamba_block_aligned_split=False,
        waiting=FCFSRequestQueue([cold, waiter]),
        kv_holding_waiting=FCFSRequestQueue(),
        running=[],
        deferred_waiting=set(),
        max_num_scheduled_tokens=80,
        max_num_active_reqs=8,
        hash_block_size=16,
        block_size=16,
        num_waiting_for_streaming_input=0,
        num_lookahead_tokens=0,
        _pause_state=PauseState.UNPAUSED,
        scheduler_reserve_full_isl=True,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=80,
            enable_chunked_prefill=True,
        ),
        _handle_blocked_waiting_request=lambda _: True,
        _get_local_prefix_cache_hit=lambda request: (*manager.get_computed_blocks(request), False),
        get_request_counts=lambda: (len(scheduler.running), len(scheduler.waiting)),
        log_stats=False,
        connector_prefix_cache_stats=None,
    )
    scheduler = make_scheduler()
    vars(scheduler).update(vars(state))
    scheduler.block_size = 16
    scheduler._initialize_hiradix()
    tree, offload, gpu = scheduler.tree_cache, scheduler.tree_cache.cache_controller, manager.block_pool
    (prefix,) = insert(tree, (1,))
    (restore_source,) = backup(tree, [prefix])
    gpu.evict_blocks({prefix.block_id})
    canceled_loads = []
    for index in range(5):
        parent_key, child_key = 10 + index, 20 + index
        (parent,) = insert(tree, (parent_key,))
        (child,) = insert(tree, (child_key,), parent=parent_key)
        backup(tree, [child])
        gpu.evict_blocks({child.block_id})
        request = _request(f"canceled-{index}", (parent_key, child_key), 33)
        (destination,) = gpu.get_new_blocks(1)
        gpu.touch([parent])
        assert offload.get_num_new_matched_tokens(request, 16) == (16, False)
        offload.update_state_after_alloc(request, KVCacheBlocks(([parent, destination],)), 16)
        canceled_loads.append((request, parent, destination))
    offload.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
    for request, parent, destination in canceled_loads:
        request.status = RequestStatus.FINISHED_ABORTED
        offload.request_finished(request, [])
        gpu.free_blocks([parent, destination])
    tree.finish_native_schedule()
    assert tree.device.free_uncached() == 1 and gpu.get_num_free_blocks() == 6
    assert tree.host.free_uncached() == 0 and tree.host.pool.get_num_free_blocks() == 1
    calls = []
    offload.inline_eviction_writeback = lambda src, dst: calls.append((src, dst))
    monkeypatch.setattr(ScheduleBatch, "finish", lambda selected: selected.tokens)

    output, batch = scheduler._schedule_batch([], [], [cold, waiter], [], False)

    assert batch.waiting == [cold, waiter] and batch.planned_host == {restore_source.block_id}
    assert tree.request_match(waiter, touch=False) == ([], [restore_source])
    assert not output and not calls
    assert list(scheduler.waiting) == [cold, waiter]
    assert not tree.prefill_host_sources
