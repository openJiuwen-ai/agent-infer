# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

import pytest

try:
    from vllm.v1.core.sched.output import KVConnectorBlockState as _KVConnectorBlockState  # noqa: F401
except ImportError:
    pytest.skip("MORI engine tests require the native KVConnectorBlockState ABI", allow_module_level=True)

from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

from agentinfer.agentcache.mori.offload import MoriHiCacheController
from tests.agentcache.mori.helpers import _request, backup, insert, key
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


def test_evict_whole_edges_and_reveal_parent(cache):
    a, b, c = insert(cache, (1, 2, 3))
    d, e = insert(cache, (4, 5), parent=1, program="other")
    leaves = cache.device.plan(8)
    assert len(leaves[0]) == 3
    assert leaves[1:] == [[c.block_id, b.block_id], [e.block_id, d.block_id], [a.block_id]]
    assert cache.pages[key(1)].shared


def test_hiradix_owns_shared_index_and_retires_before_reuse(cache):
    gpu, host = cache.device.pool, cache.host.pool
    gpu.bind_hiradix_index(cache.device, authoritative=True)
    host.bind_hiradix_index(cache.host, authoritative=True)
    (block,) = gpu.get_new_blocks(1)
    digest = BlockHash(key(1))
    grouped = make_block_hash_with_group_id(digest, 0)
    gpu._insert_block_hash(grouped, block, num_tokens=16)
    assert gpu.get_cached_block(digest, [0]) is None
    cache.publish(block, digest, None, "p")
    (copy,) = backup(cache, [block])
    gpu.free_blocks([block])

    assert gpu.cached_block_hash_to_block.get_one_block(grouped) is None
    assert host.cached_block_hash_to_block.get_one_block(grouped) is None
    assert gpu.get_cached_block(digest, [0]) == [block]
    assert host.get_indexed_copy(grouped) is copy
    assert gpu.get_usage() == 0
    with pytest.raises(RuntimeError, match="bypassed eviction planning"):
        gpu.get_new_blocks(8)

    gpu.evict_blocks({block.block_id})
    assert gpu.get_cached_block(digest, [0]) is None
    assert cache.match([digest]) == ([], [copy])
    assert len(gpu.get_new_blocks(8)) == 8


@pytest.mark.parametrize("invalid", ["pinned", "duplicate", "negative"])
def test_invalid_hiradix_allocation_leaves_the_native_queue_intact(cache, monkeypatch, invalid):
    pool = cache.device.pool
    occupied = pool.get_new_blocks(1)[0]
    before = pool.free_block_queue.get_all_free_blocks()
    free = before[0]
    wrong_id = {"pinned": occupied.block_id, "duplicate": free.block_id, "negative": -1}[invalid]
    monkeypatch.setattr(cache.device, "allocation_ids", lambda _: [free.block_id, wrong_id])

    with pytest.raises(RuntimeError, match="MORI HiRadix"):
        pool.get_new_blocks(2)

    assert pool.free_block_queue.get_all_free_blocks() == before
    assert occupied.ref_cnt == 1 and free.ref_cnt == 0
    assert cache.device.ready == {block.block_id for block in before}


def test_cache_reset_uses_native_pin_checks_then_clears_hiradix(cache):
    pool = cache.device.pool
    pool.bind_hiradix_index(cache.device, authoritative=True)
    (cached,) = insert(cache, (1,))
    pool.bind_hiradix_index(cache.device)
    assert cache.match([key(1)]) == ([cached], [])

    pool.touch([cached])
    assert not pool.reset_prefix_cache()
    assert cache.match([key(1)]) == ([cached], [])

    pool.free_blocks([cached])
    assert pool.reset_prefix_cache()
    assert not cache.pages and cached.block_hash is None
    assert len(pool.get_new_blocks(pool.get_num_free_blocks())) == pool.num_gpu_blocks - 1


def test_released_hbm_replica_does_not_take_a_second_cache_slot(cache):
    (original,) = insert(cache, (1,))
    pool = cache.device.pool
    (duplicate,) = pool.get_new_blocks(1)
    pool._insert_block_hash(original.block_hash, duplicate, num_tokens=16)
    cache.publish(duplicate, key(1), None, "p")
    assert list(cache.pages[key(1)].device.values()) == [original, duplicate]

    pool.touch([duplicate])
    pool.free_blocks([duplicate])
    assert duplicate.ref_cnt == 1
    assert duplicate.block_hash == original.block_hash

    pool.free_blocks([duplicate])
    assert duplicate.block_hash is None
    assert cache.match([key(1)]) == ([original], [])
    assert duplicate.block_id in cache.device.unused


def test_eviction_does_not_write_a_page_that_has_a_pinned_replica(cache, monkeypatch):
    (canonical,) = insert(cache, (1,))
    pool = cache.device.pool
    (pinned,) = pool.get_new_blocks(1)
    pool._insert_block_hash(canonical.block_hash, pinned, num_tokens=16)
    cache.publish(pinned, key(1), None, "p")
    offload = MoriHiCacheController.__new__(MoriHiCacheController)
    offload.bind_tree(cache)
    cache.cache_controller = offload
    offload._gpu_block_pool = pool
    allocation_sizes = []
    monkeypatch.setattr(cache, "_reserve_host_pages", lambda count, _: allocation_sizes.append(count) or [])

    assert cache._demote_device_edge([canonical.block_id], frozenset()) == ([], set())
    assert allocation_sizes == [0]
    assert cache.match([key(1)]) == ([pinned], [])
    assert pinned.ref_cnt == 1


def test_writeback_ack_preserves_dispatch_epoch(cache):
    (device,) = insert(cache, (1,))
    page = cache.device.by_id[device.block_id]
    page.write_epoch = cache.epoch
    cache.epoch += 1
    backup(cache, (device,))
    assert page.write_epoch == cache.epoch - 1


def test_match_spans_device_and_host_and_marks_epoch(cache):
    a, b, c = insert(cache, (1, 2, 3))
    host_b, host_c = backup(cache, (b, c))
    assert not cache.host.leaves
    cache.device.pool.evict_blocks({b.block_id, c.block_id})
    gpu, cpu = cache.match([key(1), key(2), key(3)], mark_load=True)
    assert gpu == [a]
    assert cpu == [host_b, host_c]
    assert cache.pages[key(3)].load_epoch == cache.epoch
    assert cache.host.plan(7)[-1] == [host_c.block_id, host_b.block_id]


def test_native_cache_manager_and_connector_share_one_prefix_lookup(cache, monkeypatch):
    (device,) = insert(cache, (1,))
    request = _request("prefix", (1,), 17)
    original = cache.request_match
    lookups = []

    def match(req, **kwargs):
        lookups.append(req.request_id)
        return original(req, **kwargs)

    monkeypatch.setattr(cache, "request_match", match)
    cache.native_schedule_active = True
    assert cache.match_for_native_admission(request) == ([device], [])
    assert cache.match_for_native_admission(request, connector=True) == ([device], [])
    assert lookups == ["prefix"]

    # The saved match never crosses the step boundary.
    cache.match_for_native_admission(request)
    cache.finish_native_schedule()
    cache.device.pool.evict_blocks({device.block_id})
    assert cache.match_for_native_admission(request, connector=True) == ([], [])
    assert lookups == ["prefix"] * 3


def test_native_admission_reuses_selected_hbm_only_until_the_cache_changes(cache, monkeypatch):
    (device,) = insert(cache, (1,))
    request = _request("prepared-prefill", (1,), 17)
    lookups = []
    match = cache.request_match

    def observe(req, **kwargs):
        lookups.append(req.request_id)
        return match(req, **kwargs)

    monkeypatch.setattr(cache, "request_match", observe)
    cache.remember_hbm_match(request, [device])
    cache.native_schedule_active = True
    assert cache.match_for_native_admission(request) == ([device], [])
    assert cache.match_for_native_admission(request, connector=True) == ([device], [])
    assert not lookups

    cache.remember_hbm_match(request, [device])
    cache.device.pool.evict_blocks({device.block_id})
    assert cache.match_for_native_admission(request) == ([], [])
    assert lookups == [request.request_id]


def test_hbm_hit_reservations_preserve_the_branch(cache):
    a, b = insert(cache, (1, 2))
    (c,) = insert(cache, (3,), parent=1)
    plan = cache.device.plan(8, {b.block_id})
    selected = {ident for edge in plan for ident in edge}
    assert b.block_id not in selected
    assert a.block_id not in selected
    assert c.block_id in selected


def test_released_owner_is_replaced_on_reuse(cache):
    (a,) = insert(cache, (1,), program="old")
    cache.placement.release("old")
    cache.placement.register("new")
    cache.stamp(a, "new")
    assert cache.pages[key(1)].program == "new"
    assert not cache.pages[key(1)].shared


def test_prefill_stamp_rechecks_ancestors_after_program_is_reregistered(cache):
    insert(cache, (1, 2), program="p")
    cache.stamp_prefill(key(2), "p")
    original = cache.pages[key(2)].stamped_for

    cache.placement.release("p")
    cache.stamp_prefill(key(1), "replacement")
    assert cache.pages[key(1)].program == "replacement"
    assert not cache.pages[key(1)].shared

    cache.stamp_prefill(key(2), "p")
    assert cache.pages[key(2)].stamped_for is not original
    assert cache.pages[key(1)].shared
    assert cache.pages[key(2)].program == "p"


def test_partial_match_touches_unmatched_suffix_before_splitting(cache):
    head, _suffix = insert(cache, (1, 2))
    (other,) = insert(cache, (3,))

    assert cache.match([key(1)])[0] == [head]

    # Five uncached blocks are allocated first, then the other edge is
    # older than the suffix of the just-accessed compressed edge.
    assert cache.device.plan(6)[-1] == [other.block_id]


def test_inserting_branch_refreshes_old_suffix_before_splitting(cache):
    head, _, old_end = insert(cache, (1, 2, 3))
    (unrelated,) = insert(cache, (4,))
    assert cache.device.plan(1, include_uncached=False)[0][0] == old_end.block_id

    # Generated tokens can diverge inside an existing edge without a
    # scheduler lookup at the split point.
    (new_end,) = insert(cache, (5,), parent=1)
    assert cache.device.plan(1, include_uncached=False) == [[unrelated.block_id]]
    assert cache.device.by_id[old_end.block_id].accessed < cache.device.by_id[head.block_id].accessed
    assert cache.device.by_id[head.block_id].accessed < cache.device.by_id[new_end.block_id].accessed


def test_stamp_visits_ancestors_when_endpoint_already_has_same_owner(cache):
    (ancestor,) = insert(cache, (1,), program="previous")
    (endpoint,) = insert(cache, (2,), parent=1, program="waiting")
    page = cache.device.by_id[ancestor.block_id]
    page.shared = False
    cache.stamp(endpoint, "waiting")
    assert page.shared
