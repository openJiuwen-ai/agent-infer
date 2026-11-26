# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    from vllm.v1.core.sched.output import KVConnectorBlockState as _KVConnectorBlockState  # noqa: F401
except ImportError:
    pytest.skip("MORI engine tests require the native KVConnectorBlockState ABI", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager
from agentinfer.agentcache.mori.scheduler import PrefillBatch
from tests.agentcache.mori.helpers import _request, _test_controller, backup, insert, key, make_scheduler
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


def test_prefill_batch_reserves_shared_device_pages_once(cache):
    insert(cache, (1, 2, 3, 4))
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.max_num_active_reqs = 128
    scheduler.new_token_ratio = 0.7

    requests = [
        SimpleNamespace(
            request_id=str(tail),
            block_hashes=[key(index) for index in (1, 2, 3, 4, tail)],
            num_tokens=80,
            num_computed_tokens=0,
            num_output_tokens=0,
            max_tokens=1,
            sampling_params=None,
            skip_reading_prefix_cache=False,
            status=RequestStatus.WAITING,
        )
        for tail in (5, 6)
    ]

    # 128 tokens available. The first request reserves a shared 64-token
    # prefix and 33 more tokens, leaving enough for the second shared hit.
    batch = scheduler._prefer_prefill([], [], requests, [])
    assert (batch.tokens, batch.waiting, batch.holding) == (32, requests, [])


def test_lpm_uses_full_page_boundary_then_keeps_last_token_for_forward(cache):
    insert(cache, (1, 2, 3))
    short_hit = _request("first", (1, 2, 4), 49)
    full_hit = _request("second", (1, 2, 3), 48)
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    ranked = scheduler._order_waiting([short_hit, full_hit])
    assert ranked == [full_hit, short_hit]
    assert [len(cache.request_match(request)[0]) for request in ranked] == [2, 2]


def test_chunk_that_completes_reserves_output_capacity(cache):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    chunked = _request("chunk", (1, 2), 33, max_tokens=64)
    chunked.num_computed_tokens = 16
    small = _request("small", (3,), 16, max_tokens=1)
    assert scheduler._prefer_prefill([chunked], [], [small], []) == PrefillBatch(
        tokens=17, token_caps={chunked.request_id: 17}, full=True
    )


def test_chunk_that_exhausts_its_budget_still_checks_waiting_hbm_capacity(cache):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 16
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    chunked = _request("continuation", (1, 2), 33, max_tokens=64)
    chunked.num_computed_tokens = 16
    large = _request("waiter", (3, 4), 33, max_tokens=88)
    assert scheduler._prefer_prefill([chunked], [], [large], []) == PrefillBatch(
        tokens=16, token_caps={chunked.request_id: 16}, full=True
    )


@pytest.mark.parametrize(("max_prefill_tokens", "admitted"), [(96, 1), (128, 2)])
def test_prefill_input_limit_is_separate_from_the_chunk_limit(cache, max_prefill_tokens, admitted):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    scheduler.max_prefill_tokens = max_prefill_tokens
    waiting = [_request("first", (1, 2), 33), _request("second", (3, 4), 33)]
    assert scheduler._prefer_prefill([], [], waiting, []).waiting == waiting[:admitted]


def test_cold_program_is_registered_when_considered_for_prefill(cache):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    waiting = [_request("capacity-head", (1, 2), 33, max_tokens=128), _request("behind-head", (3,), 17)]
    for request in waiting:
        request.sampling_params.extra_args = {"mori_program_id": request.request_id}
    assert scheduler._prefer_prefill([], [], waiting, []) == PrefillBatch()
    assert cache.placement.has_program("capacity-head")
    assert not cache.placement.has_program("behind-head")


def test_considered_prefix_is_stamped_even_when_budget_rejects_request(cache):
    (previous,) = insert(cache, (1,), program="previous")
    cache.placement.register("waiting")
    req = _request("waiting", (1, 2), 33, max_tokens=128)
    req.sampling_params.extra_args = {"mori_program_id": "waiting"}
    running = _request("running", (3,), 17)
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    assert scheduler._prefer_prefill([], [running], [req], []) == PrefillBatch(full=True)
    assert cache.device.by_id[previous.block_id].shared


def test_idle_head_rejects_instead_of_admitting_a_smaller_follower(cache):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    head = _request("head", (1,), 17, max_tokens=128)
    small = _request("small", (2,), 17)
    assert scheduler._prefer_prefill([], [], [head, small], []) == PrefillBatch()


def test_prefill_budget_reaches_native_for_partial_running_chunk(cache, step_without_worker):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    cache.stage_evictions = lambda *args: None
    chunk = _request("chunk", (1, 2), 300)
    chunk.is_prefill_chunk = True
    chunk.num_computed_tokens = 16
    scheduler.running = [chunk]
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 256
    scheduler.max_num_active_reqs = 10

    selected = []

    def allocate_running(_batch, requests, limit, **_):
        if requests:
            selected.append((list(requests), limit))
        return limit

    step_without_worker(scheduler, allocate_running=allocate_running)
    assert selected == [([chunk], 128)]
    assert scheduler.max_num_scheduled_tokens == 256


def test_lpm_order_persists_when_preselection_admits_nothing(cache, step_without_worker):
    insert(cache, (1, 2, 3))
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    a = _request("first", (1,), 17)
    b = _request("long-hit", (1, 2, 3), 48, max_tokens=128)
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue([a, b])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    step_without_worker(scheduler)
    assert list(scheduler.waiting) == [b, a]


@pytest.mark.parametrize("commit_decode", [False, True])
def test_full_prefill_batch_retries_after_a_running_request_leaves(
    cache, monkeypatch, step_without_worker, commit_decode
):
    from agentinfer.agentcache.mori.schedule_batch import ScheduleBatch

    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = False
    running = _request("running", (), 17, max_tokens=24)
    running.status = RequestStatus.RUNNING
    running.num_computed_tokens = running.num_prompt_tokens
    running.num_output_placeholders = 1
    waiter = _request("waiting", (1, 2), 33, max_tokens=100)
    scheduler.running = [running]
    scheduler.waiting = FCFSRequestQueue([waiter])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.num_lookahead_tokens = 0
    scheduler.kv_cache_manager = SimpleNamespace(
        hiradix=cache,
        block_pool=cache.device.pool,
        allocate_slots=lambda *_, **__: KVCacheBlocks(([],)),
    )
    lookups = []
    monkeypatch.setattr(scheduler, "_order_waiting", lambda reqs: lookups.append(reqs) or reqs)

    allocate_running = ScheduleBatch.running if commit_decode else None
    first = step_without_worker(scheduler, allocate_running=allocate_running)
    assert first.num_scheduled_tokens == ({running.request_id: 1} if commit_decode else {})
    step_without_worker(scheduler, allocate_running=allocate_running)
    assert lookups == [[waiter]]
    scheduler.running.clear()
    step_without_worker(scheduler, allocate_running=allocate_running)
    assert lookups == [[waiter], [waiter]]
    assert scheduler._full_prefill_running is None


@pytest.mark.parametrize(("existing_decode", "partial_prefill"), [(False, False), (True, True), (True, False)])
def test_prefill_merge_preserves_full_flag_only_for_an_existing_running_batch(
    cache, monkeypatch, existing_decode, partial_prefill, step_without_worker
):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = False
    chunk = _request("prefill", (1, 2), 33)
    chunk.is_prefill_chunk = True
    decode = _request("running", (), 17)
    scheduler.running = [decode, chunk] if existing_decode else [chunk]
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    monkeypatch.setattr(scheduler, "_prefer_prefill", lambda *_: PrefillBatch(tokens=16, full=True))

    def native_schedule():
        assert scheduler.running == ([decode, chunk] if existing_decode else [chunk])
        chunk.is_prefill_chunk = partial_prefill
        return SimpleNamespace(preempted_req_ids=set(), num_scheduled_tokens={chunk.request_id: 16})

    step_without_worker(scheduler, native_schedule)
    expected = set(scheduler.running) if existing_decode and not partial_prefill else None
    assert scheduler._full_prefill_running == expected


def test_lpm_checks_the_first_32_tokens_when_page_is_larger(cache):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    cache.page_size = 64
    first = Request("first", [1] * 32 + [2] * 32, SamplingParams(), None)
    duplicate = Request("duplicate", [1] * 32 + [3] * 32, SamplingParams(), None)
    other = Request("other", [4] * 64, SamplingParams(), None)
    ranked = scheduler._order_waiting([first, duplicate, other])
    assert ranked == [first, other, duplicate]


def test_prefill_retouches_prefixes_in_ranked_order(cache):
    (short_block,) = insert(cache, (1,))
    long_blocks = insert(cache, (2, 3))
    short = _request("short", (1,), 17)
    long = _request("long", (2, 3), 33)
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    batch = scheduler._prefer_prefill([], [], [short, long], [])
    assert batch.waiting == [long, short]
    assert cache.device.plan(1, include_uncached=False) == [[block.block_id for block in reversed(long_blocks)]]
    assert cache.device.by_id[short_block.block_id].accessed > cache.device.by_id[long_blocks[-1].block_id].accessed


@pytest.mark.parametrize("failed", [False, True])
def test_waiter_restores_dram_when_prior_chunk_exhausts_prefill_budget(cache, native_load, monkeypatch, failed):
    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector

    offload, host = native_load
    (ancestor,) = insert(cache, (1,))
    pool = cache.device.pool
    cache.share_batch_loads = cache.share_issued_loads = offload.mori_batch_loads = True
    offload.inline_eviction_writeback = object()
    offload._load_event_counter = 0
    offload._pending_finished_stores = []
    offload._store_event_to_blocks = {}
    offload._abandoned_store_event_to_blocks = {}
    monkeypatch.setattr(offload, "prepare_store_specs", lambda _: ([], [], [], None))

    request = _request("waiter", (1, 2), 33)
    chunk = _request("separate-long-prefill", (), 65)
    chunk.num_computed_tokens = 16
    engine = make_scheduler()
    engine.max_num_active_reqs = 8
    engine.max_num_scheduled_tokens = 16
    engine.kv_cache_manager = manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(single_type_managers=[SimpleNamespace(req_to_blocks={}, num_cached_block={})])
    offload.publish_batch_load = manager.publish_batch_load
    engine.tree_cache.cache_controller = offload
    engine.new_token_ratio = 0.7
    engine.max_prefill_tokens = 64

    batch = engine._prefer_prefill([chunk], [], [request], [], ranked=[request])
    assert batch.tokens == 16 and not batch.waiting
    (key,) = offload._reqs_to_load
    assert key != request.request_id
    schedule = SimpleNamespace(preempted_req_ids=set(), has_sync_kv_loads=False)
    meta = offload.build_connector_meta(schedule)
    assert schedule.has_sync_kv_loads
    assert meta.load_cpu_blocks == [host[1].block_id]
    assert meta.load_event_to_reqs == {0: [key]}
    (destination,) = [pool.blocks[bid] for bid in meta.load_gpu_blocks]
    assert cache.request_match(request, include_batch_loading=True) == ([ancestor, destination], [])
    assert all(block.ref_cnt == 1 for block in (ancestor, destination, host[1]))

    connector = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    connector.scheduler_manager = offload
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.connector = connector
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = FCFSRequestQueue([request])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.running = []
    scheduler.deferred_waiting = set()
    scheduler._inflight_prefills = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.aux_output_connector = scheduler.ec_connector = None
    scheduler.defer_block_free = False
    scheduler.vllm_config = SimpleNamespace(kv_transfer_config=None)
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _: None)
    scheduler.kv_cache_manager = SimpleNamespace(
        remove_skipped_blocks=lambda **_: None,
        get_block_ids_for_computed_tokens=lambda **_: ([],),
        free=lambda _: None,
    )
    assert scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED) == [request]
    assert not scheduler.requests and scheduler.finished_req_ids == {request.request_id}
    # The first empty worker step forwards the finished ID. Its H2D event may
    # still be outstanding, so the next step must be able to poll again.
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={}))
    cache.finish_native_schedule()
    assert cache.issued_load_destinations == {destination.block_id}
    assert not scheduler.has_unfinished_requests() and not scheduler.has_finished_requests()
    assert scheduler.has_requests()
    poll = connector.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
    assert poll.load_event_to_reqs == {0: [key]}
    assert not poll.load_gpu_blocks  # The empty step polls the old transfer.
    assert all(block.ref_cnt == 1 for block in (ancestor, destination, host[1]))
    output = KVConnectorOutput(
        finished_recving={key},
        invalid_block_ids={destination.block_id} if failed else set(),
    )
    scheduler._update_from_kv_xfer_finished(output)
    assert output.finished_recving is None
    assert not offload._prefetch_pins and not cache.issued_load_destinations
    assert all(block.ref_cnt == 0 for block in (ancestor, destination, host[1]))
    assert cache.request_match(request) == (([ancestor], [host[1]]) if failed else ([ancestor, destination], []))
    assert not scheduler.has_requests()


@pytest.mark.parametrize("local_worker", [False, True])
def test_prefill_reserves_aggregate_extend_after_planned_host_restore(cache, native_load, local_worker):
    offload, restore_host = native_load
    offload._in_flight_store_gpu_blocks = set()
    offload._pending_finished_stores = []
    offload.enable_kv_cache_events = False
    copies = []

    def copy(src, dst):
        assert all(cache.device.pool.blocks[block_id].ref_cnt == 1 for block_id in src)
        assert all(cache.host.pool.blocks[block_id].block_hash is None for block_id in dst)
        copies.append((src.copy(), dst.copy()))

    if local_worker:
        offload.inline_eviction_writeback = copy
    (host_victim,) = insert(cache, (3,))
    (first,) = insert(cache, (4,))
    (second,) = insert(cache, (5,))
    (third,) = insert(cache, (6,))
    (victim_host,) = backup(cache, [host_victim])
    cache.host.pool.get_new_blocks(5)
    cache.device.pool.get_new_blocks(3)
    cache.device.pool.evict_blocks({host_victim.block_id})
    assert cache.device.free_uncached() == 2

    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.tree_cache.cache_controller = offload
    scheduler.dense_kv = True
    request = _request("restore", (1, 2), 33)
    batch = PrefillBatch(
        tokens=1, waiting=[request], planned_host={block.block_id for block in restore_host}, needs_host_load=True
    )

    # Original HiRadix sees no free pages after restore, so it evicts the
    # full two-page extend estimate. Its DRAM eviction excludes the restore.
    assert scheduler._prepare_prefill([], batch) is local_worker
    assert second.block_id not in cache.device.by_id
    assert third.block_id in cache.device.by_id
    assert [cache.host.by_id[block.block_id].key for block in restore_host] == [key(1), key(2)]
    assert key(3) not in cache.pages
    if local_worker:
        assert copies == [([first.block_id], [victim_host.block_id])]
        assert not offload._pending_finished_stores and not offload._in_flight_store_gpu_blocks
        assert not offload._eviction_store_gpu_ids
        assert first.block_id not in cache.device.by_id
        assert cache.host.by_id[victim_host.block_id] is cache.pages[key(4)]
    else:
        assert offload._eviction_store_gpu_ids == {first.block_id}
        assert offload._pending_finished_stores[0].cpu_block_ids == [victim_host.block_id]


def test_local_worker_evicts_for_host_restore_before_allocating_extend(cache, native_load):
    offload, restore_host = native_load
    cache.share_batch_loads = True
    offload._in_flight_store_gpu_blocks = set()
    offload._pending_finished_stores = []
    offload.enable_kv_cache_events = False
    copies = []
    offload.inline_eviction_writeback = lambda src, _: copies.append(src.copy())
    victims = [insert(cache, (i,))[0].block_id for i in range(3, 8)]
    cache.device.pool.get_new_blocks(2)
    assert cache.device.free_uncached() == 1

    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.tree_cache.cache_controller = offload
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 8
    scheduler.dense_kv = True
    scheduler.new_token_ratio = 0.7
    request = _request("restore", (1, 2, 8), 49)

    batch = scheduler._prefer_prefill([], [], [request], [])
    assert batch.waiting == [request] and batch.tokens == 17
    assert batch.planned_host == {block.block_id for block in restore_host}
    assert copies == [victims[:2]]
    assert scheduler._prepare_prefill([], batch)
    assert copies == [victims[:2], victims[2:]]
    assert all(block.block_id in cache.host.by_id for block in restore_host)


@pytest.mark.parametrize("pending", [False, True])
def test_decode_capacity_with_a_free_page_per_request(cache, pending):
    scheduler = make_scheduler()
    scheduler.dense_kv = True
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    _test_controller(cache)._eviction_store_gpu_ids = {1} if pending else set()
    count = cache.device.free_uncached()
    assert count > 0
    requests = [_request(str(index), (), 17) for index in range(count)]
    assert scheduler._prepare_decode(requests) is not pending
    assert cache.device.free_uncached() == count


def test_decode_waits_for_writeback_before_preempting(cache):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    cache.device.pool.get_new_blocks(8)
    req = _request("decode", (), 17)
    req.num_computed_tokens = 16
    scheduler.current_step = 10
    scheduler.max_model_len = 256
    scheduler.num_sampled_tokens_per_step = 1
    stage_calls = []

    def stage(count, protected):
        stage_calls.append(count)
        scheduler.tree_cache.cache_controller._eviction_store_gpu_ids.add(1)

    _test_controller(cache)
    cache.stage_evictions = stage
    assert not scheduler._prepare_decode([req])
    assert stage_calls == [1]
    assert not scheduler.failed_decodes


def test_decode_retracts_smallest_output_before_constructing_native_batch(cache):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    occupied = cache.device.pool.get_new_blocks(8)
    old = _request("old", (), 15, max_tokens=10)
    old.append_output_token_ids([30, 31])
    old.num_computed_tokens = 16
    new = _request("new", (), 16, max_tokens=10)
    new.append_output_token_ids(32)
    new.num_computed_tokens = 16
    oldest = _request("oldest", (), 14, max_tokens=10)
    oldest.append_output_token_ids([33, 34, 35])
    oldest.num_computed_tokens = 16
    running = [new, old, oldest]
    waiter = _request("already-waiting", (), 17)
    waiting = FCFSRequestQueue([waiter])
    preempted = []
    evictions = []

    def preempt(victim, timestamp, *, drop_stale_output):
        preempted.append(victim)
        waiting.prepend_request(victim)
        cache.device.pool.free_blocks(occupied[-2:])

    def stage(count, protected):
        evictions.append(count)

    scheduler.current_step = 10
    scheduler.max_model_len = 256
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.running = list(running)
    scheduler.waiting = waiting
    scheduler.deferred_frees = []
    scheduler.requires_kv_delivery = False
    scheduler._request_blocks_can_be_freed = lambda _: True
    scheduler._preempt_request = preempt
    _test_controller(cache)
    cache.stage_evictions = stage
    assert scheduler._prepare_decode(running)
    assert running == [oldest, old]
    assert list(waiting) == [waiter, new]
    assert preempted == [new]
    assert evictions == [3, 3]


def test_decode_aborts_last_request_when_no_page_can_be_reclaimed(cache, step_without_worker):
    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    scheduler.new_token_ratio = 0.7
    cache.device.pool.get_new_blocks(8)
    req = _request("decode", (), 17)
    req.num_computed_tokens = 16

    def finish(ids, status):
        assert ids == {req.request_id}
        req.status = status
        return [req]

    scheduler.current_step = 10
    scheduler.max_model_len = 256
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.running = [req]
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_active_reqs = 1
    scheduler.max_num_scheduled_tokens = 16
    scheduler.deferred_frees = []
    scheduler._request_blocks_can_be_freed = lambda _: True
    scheduler.finish_requests = finish
    cache.stage_evictions = lambda *args: None
    step_without_worker(scheduler)
    assert scheduler.new_token_ratio == 0.0
    outputs = {}
    scheduler._finish_failed_decodes(outputs)
    assert outputs[0].outputs[0].finish_reason == req.get_finished_reason()
    assert req.status == RequestStatus.FINISHED_ERROR
    assert not scheduler.failed_decodes


@pytest.mark.parametrize("reject_prefill", ["not_prepared", "no_slots", "pending_writeback"])
def test_direct_schedule_falls_back_to_decode_on_real_queues(cache, monkeypatch, reject_prefill):
    from vllm.v1.core.sched.output import CachedRequestData

    scheduler = make_scheduler()
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = False
    decode = _request("decode", (), 18)
    decode.status = RequestStatus.RUNNING
    decode.num_computed_tokens = 17
    chunk = _request("chunk", (), 40)
    chunk.status = RequestStatus.RUNNING
    chunk.is_prefill_chunk = True
    chunk.num_computed_tokens = 8
    waiter = _request("waiting", (), 18)
    running = [decode, chunk]
    waiting = FCFSRequestQueue([waiter])
    holding = FCFSRequestQueue()
    updates = []
    native_step = []
    controller_metadata = object()
    used = []
    kv = KVCacheBlocks(([],))

    def allocate_slots(request, tokens, **kwargs):
        assert scheduler.running is running
        assert scheduler.waiting is waiting and scheduler.kv_holding_waiting is holding
        assert scheduler.max_num_scheduled_tokens == 32
        used.append((request, tokens))
        return kv if request is decode else None

    def prepare_prefill(*_):
        if reject_prefill == "pending_writeback":
            scheduler.tree_cache.cache_controller._eviction_store_gpu_ids.add(1)
        return reject_prefill == "no_slots"

    _test_controller(cache)
    monkeypatch.setattr(
        scheduler, "_plan_prefill", lambda *args: PrefillBatch(tokens=16, token_caps={chunk.request_id: 16})
    )
    monkeypatch.setattr(scheduler, "_prepare_prefill", prepare_prefill)
    scheduler.current_step = 0
    scheduler.max_model_len = 256
    scheduler.block_size = 16
    scheduler.num_lookahead_tokens = 0
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.running = running
    scheduler.requests = {r.request_id: r for r in [decode, chunk, waiter]}
    scheduler.waiting = waiting
    scheduler.kv_holding_waiting = holding
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 32
    scheduler.max_num_running_reqs = 2
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_active_reqs = 10
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=32)
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[0])
    scheduler.kv_cache_manager = SimpleNamespace(
        hiradix=cache,
        new_step_starts=lambda: native_step.append(1),
        allocate_slots=allocate_slots,
        get_num_common_prefix_blocks=lambda _: [0],
        take_boundary_state_offloads=dict,
        get_block_ids=lambda _: ([],),
        take_kv_cache_block_copies=lambda: ([], []),
    )
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=list, get_manager_metadata=lambda: None)
    scheduler.deferred_waiting = set()
    scheduler.prefill_capacity_bound = False
    scheduler.use_v2_model_runner = True
    scheduler.model_uses_mrope = False
    scheduler.connector = object()
    scheduler.reset_preempted_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.defer_block_free = True
    scheduler.sched_step_seq = 4
    scheduler._get_new_block_ids_to_zero = lambda: None
    scheduler._build_kv_connector_meta = lambda *args: controller_metadata
    scheduler._make_cached_request_data = lambda *args: CachedRequestData.make_empty()
    scheduler._update_after_schedule = updates.append
    scheduler.get_request_counts = lambda: (len(running), len(waiting))

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"decode": 1}
    assert output.kv_connector_metadata is controller_metadata
    assert output.kv_connector_block_state is None
    assert scheduler.current_step == 1
    assert scheduler.sched_step_seq == 5
    assert updates == [output] and native_step == [1]
    assert running == [decode, chunk]
    assert list(waiting) == [waiter]
    assert [request for request, _ in used] == ([chunk, decode] if reject_prefill == "no_slots" else [decode])


@pytest.fixture
def text_scheduler(cache):
    empty = KVCacheBlocks(([],))

    def lookup(request):
        blocks, _ = cache.match_for_native_admission(request)
        return KVCacheBlocks((blocks,)), len(blocks) * 16, 0, False

    scheduler = make_scheduler()
    scheduler.dense_kv = True
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.kv_cache_manager = SimpleNamespace(
        hiradix=cache,
        block_pool=cache.device.pool,
        empty_kv_cache_blocks=empty,
        allocate_slots=lambda *_args, **_kwargs: empty,
        get_blocks=lambda _: empty,
        record_prefix_cache_stats=lambda *_: None,
        take_boundary_state_offloads=dict,
        take_kv_cache_block_copies=lambda: ([], []),
        get_num_common_prefix_blocks=lambda _: [0],
        get_block_ids=lambda _: ([],),
    )
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_active_reqs = 8
    scheduler.max_num_scheduled_tokens = 64
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=64, enable_chunked_prefill=True)
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.deferred_waiting = set()
    scheduler.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda *_: (0, False), update_state_after_alloc=lambda *_: None
    )
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.num_lookahead_tokens = 0
    scheduler.scheduler_reserve_full_isl = False
    scheduler.block_size = 16
    scheduler._get_local_prefix_cache_hit = lookup
    scheduler._handle_blocked_waiting_request = lambda _: True
    scheduler.connector_prefix_cache_stats = None
    scheduler.log_stats = False
    scheduler._inflight_prefills = set()
    scheduler.use_v2_model_runner = True
    scheduler.model_uses_mrope = False
    scheduler.max_num_running_reqs = scheduler.max_num_active_reqs
    scheduler.reset_preempted_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.requests = {}
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[None])
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=list, get_manager_metadata=lambda: None)
    scheduler._get_new_block_ids_to_zero = lambda: None
    scheduler._build_kv_connector_meta = lambda *_: None
    scheduler.defer_block_free = False
    scheduler._update_after_schedule = lambda _: None
    return scheduler


@pytest.mark.parametrize("prefix_state", ["unchanged", "evicted", "attached"])
def test_prefill_commit_uses_each_requests_planned_extend(cache, text_scheduler, monkeypatch, prefix_state):
    scheduler = text_scheduler

    prefix = insert(cache, (1, 2))
    first = _request("long-prefix", (1, 2), 33)
    second = _request("cold", (3,), 17)
    waiting = scheduler.waiting
    waiting.extend([first, second])
    holding = scheduler.kv_holding_waiting
    if prefix_state == "attached":
        cache.device.pool.touch(prefix[:1])
        first.num_computed_tokens = 16
        waiting.remove_request(first)
        holding.add_request(first)

    def prepare(_chunked, _batch):
        if prefix_state == "evicted":
            # Eviction invalidates the prepared match before slot allocation.
            cache.device.pool.evict_blocks({block.block_id for block in prefix})
            assert cache.request_match(first) == ([], [])
        return True

    monkeypatch.setattr(scheduler, "_prepare_prefill", prepare)

    output, plan = scheduler._schedule_batch([], [], list(waiting), list(holding), False)

    first_extend = 17 if prefix_state == "attached" else 1
    assert output.num_scheduled_tokens == {first.request_id: first_extend, second.request_id: 17}
    assert plan.tokens == first_extend + 17


@pytest.mark.parametrize("resident", [False, True])
def test_waiters_respect_worker_lora_slots(text_scheduler, resident):
    from vllm.lora.request import LoRARequest
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    scheduler = text_scheduler
    scheduler.lora_config = SimpleNamespace(max_loras=1)
    first = LoRARequest("first", 1, "/first")
    second = LoRARequest("second", 2, "/second")

    def request(name, adapter):
        req = _request(name, (), 6)
        req.lora_request = adapter
        scheduler.requests[name] = req
        return req

    active = request("active", first)
    if resident:
        scheduler.running.append(active)
    else:
        scheduler.waiting.add_request(active)
    deferred = request("deferred", second)
    scheduler.waiting.extend([deferred, request("same-id", first), request("base", None)])
    output, _ = scheduler._schedule_batch([], scheduler.running.copy(), list(scheduler.waiting), [], False)
    expected = {"same-id", "base"} if resident else {"active", "same-id", "base"}
    assert set(output.num_scheduled_tokens) == expected
    assert list(scheduler.waiting) == [deferred]

    activated = []
    worker = SimpleNamespace(_adapter_manager=SimpleNamespace(lora_slots=1), add_adapter=activated.append)
    adapters = {req.lora_request for req in output.scheduled_new_reqs if req.lora_request}
    LRUCacheWorkerLoRAManager._apply_adapters(worker, adapters)
    assert activated == [first]

    scheduler.running.clear()
    output, _ = scheduler._schedule_batch([], [], list(scheduler.waiting), [], False)
    assert output.num_scheduled_tokens == {"deferred": 6}
    LRUCacheWorkerLoRAManager._apply_adapters(worker, {req.lora_request for req in output.scheduled_new_reqs})
    assert activated == [first, second]
