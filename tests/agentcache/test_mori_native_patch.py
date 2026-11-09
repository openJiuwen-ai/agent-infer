# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise the shipped HBM patch against an explicitly supplied clean vLLM source tree."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.scheduling import ProgramRef

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def native_pool_class(tmp_path_factory):
    """Apply the packaged patch to pristine sources without changing the installed vLLM."""
    source = os.environ.get("MORI_VLLM_SOURCE")
    if source is None:
        pytest.skip("set MORI_VLLM_SOURCE to a clean vLLM v0.31.0 source tree to validate the native patch")
    root = tmp_path_factory.mktemp("mori-native")
    for name in ("block_pool.py", "kv_cache_manager.py"):
        target = root / "vllm/v1/core" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / "vllm/v1/core" / name, target)
    patch = Path(__file__).resolve().parents[2] / "agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("mori_native_test_pool", root / "vllm/v1/core/block_pool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BlockPool


@pytest.mark.parametrize("advice", [None, (3,), (3, 3), (3, -1), (3, 100), (3, 0), (3, True), (3, "bad"), "raise"])
def test_invalid_advice_falls_back_atomically_and_updates_provider(native_pool_class, advice) -> None:
    """Invalid second victims must not remove a valid first victim before native fallback."""
    pool = native_pool_class(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    policy = MORIHBMPolicy(MORIRuntimeState())
    pool.register_mori_block_selection_provider(policy)

    def propose(count, protected):
        if advice == "raise":
            raise RuntimeError("injected provider failure")
        return advice

    policy.propose = propose
    blocks = pool.get_new_blocks(2)
    assert [block.block_id for block in blocks] == [1, 2]
    assert all(block.ref_cnt == 1 for block in blocks)
    assert [block.block_id for block in pool.free_block_queue.get_all_free_blocks()] == [3, 4, 5, 6, 7]
    assert policy.index.propose_exact(5) == (3, 4, 5, 6, 7)
    assert pool._mori_fallback_count == 1
    pool.unregister_mori_block_selection_provider(policy)
    policy.close()


def test_allocated_advice_falls_back_and_valid_advice_preserves_native_safety(native_pool_class) -> None:
    """Reject in-use blocks, then accept valid reordered victims and retain normal exhaustion errors."""
    pool = native_pool_class(num_gpu_blocks=8, enable_caching=False, hash_block_size=16)
    policy = MORIHBMPolicy(MORIRuntimeState())
    pool.register_mori_block_selection_provider(policy)
    occupied = pool.get_new_blocks(1)[0]
    policy.propose = lambda count, protected: (4, occupied.block_id)
    assert [block.block_id for block in pool.get_new_blocks(2)] == [2, 3]
    assert occupied.ref_cnt == 1
    policy.propose = lambda count, protected: (6, 4)
    assert [block.block_id for block in pool.get_new_blocks(2)] == [6, 4]
    assert policy.index.propose_exact(2) == (5, 7)
    with pytest.raises(ValueError, match="Cannot get"):
        pool.get_new_blocks(3)
    pool.unregister_mori_block_selection_provider(policy)
    policy.close()


def test_native_reallocation_invalidates_old_footprint_before_new_owner(native_pool_class) -> None:
    """The shipped native callback prevents shared-capacity proof surviving ID reuse."""
    pool = native_pool_class(num_gpu_blocks=3, enable_caching=False, hash_block_size=16)
    state = MORIRuntimeState()
    state.footprints.enabled = True
    old, new = ProgramRef("old", 0), ProgramRef("new", 0)
    state.bind_request("old", old)
    state.bind_request("new", new)
    policy = MORIHBMPolicy(state)
    pool.register_mori_block_selection_provider(policy)
    block = pool.get_new_blocks(1)[0]
    pool.observe_mori_block_owner("old", [block])
    assert state.footprint(old, "gpu") == frozenset({block.block_id})
    pool.free_blocks([block])
    policy.propose = lambda count, protected: (block.block_id,)
    assert pool.get_new_blocks(1) == [block]
    assert not state.footprint(old, "gpu")
    pool.observe_mori_block_owner("new", [block])
    assert state.footprint(new, "gpu") == frozenset({block.block_id})
    pool.unregister_mori_block_selection_provider(policy)
    policy.close()


@pytest.fixture(scope="module")
def native_cpu_module(tmp_path_factory, native_pool_class):
    """Apply the CPU patch too; exercise actual native methods with small CPU pools."""
    import sys

    root = tmp_path_factory.mktemp("mori-native-cpu")
    name = "vllm/v1/simple_kv_offload/manager.py"
    target = root / name
    target.parent.mkdir(parents=True)
    shutil.copyfile(Path(os.environ["MORI_VLLM_SOURCE"]) / name, target)
    patch = (
        Path(__file__).resolve().parents[2] / "agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch"
    )
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("mori_native_test_cpu", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_store(native_pool_class, native_cpu_module):
    from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
    from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy

    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    policy = MORISimpleCPUOffloadPolicy(state)
    for req, tier in [("idle", MORITier.CPU), ("busy", MORITier.GPU)]:
        owner = ProgramRef(req, 0)
        state.bind_request(req, owner)
        components.strategy._set_factor(components.initial_factors, owner, MORIProgramFactors(tier=tier))
    # Construct only the scheduler-side portion; no tensors, GPU or DMA involved.
    manager = native_cpu_module.SimpleCPUOffloadScheduler.__new__(native_cpu_module.SimpleCPUOffloadScheduler)
    manager.cpu_block_pool = native_pool_class(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    manager._mori_cpu_backend = True
    manager._lazy_mode = False
    manager._mori_offload_tier_provider = None
    manager._mori_resident_block_ids = set()
    assert manager.register_mori_offload_tier_provider(3, policy) == {"hook_api_version": 3}
    yield manager, policy
    manager.unregister_mori_offload_tier_provider(policy)
    policy.close()
    state.unbind_provider(components.strategy)


def test_cpu_busy_backup_cannot_evict_idle_cache_even_when_native_calls_it_free(cpu_store):
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

    manager, policy = cpu_store
    pool = manager.cpu_block_pool
    blocks = manager._allocate_store_blocks(2, "idle")
    keys = [make_block_hash_with_group_id(BlockHash(bytes([i]) * 32), 0) for i in (1, 2)]
    for block, key in zip(blocks, keys, strict=True):
        pool._insert_block_hash(key, block, num_tokens=16)
    manager._observe_mori_owner("idle", blocks)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 2  # ref-zero cached is not empty
    assert manager._mori_store_capacity("busy") == 0
    assert manager._allocate_store_blocks(1, "busy") == []
    assert all(pool.cached_block_hash_to_block.get_one_block(key) is not None for key in keys)
    assert pool._mori_fallback_count == 0
    assert manager._mori_store_capacity("idle") == 2
    assert len(manager._allocate_store_blocks(1, "idle")) == 1


@pytest.mark.parametrize("advice", [None, (1,), (1, 1), (1, 0), (1, 99), "raise"])
def test_rejected_cpu_store_is_atomic_and_has_no_lru_fallback(cpu_store, advice):
    manager, policy = cpu_store
    pool = manager.cpu_block_pool
    before = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]

    def propose(count, request_id):
        if advice == "raise":
            raise RuntimeError("injected failure")
        return advice

    policy.propose_for_store = propose
    assert manager._allocate_store_blocks(2, "busy") == []
    assert [b.block_id for b in pool.free_block_queue.get_all_free_blocks()] == before
    assert all(pool.blocks[i].ref_cnt == 0 for i in before)
    assert pool._mori_fallback_count == 0
    assert policy.store_capacity("busy") == 2


def test_finished_flush_rejection_does_not_pin_gpu_or_publish_transfer(cpu_store, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    manager, policy = cpu_store
    request = SimpleNamespace(request_id="busy")
    manager._reqs_to_store = {"busy": object()}
    manager._gpu_block_pool = SimpleNamespace(touch=Mock(), blocks={7: object()})
    manager._pending_finished_stores = []
    manager._in_flight_store_gpu_blocks = set()
    monkeypatch.setattr(manager, "_select_eager_blocks_to_store", lambda *args: ([7], [1], None))
    monkeypatch.setattr(manager, "_find_fa_partial_tail_source", lambda *args: None)
    policy.propose_for_store = lambda *args: None
    manager._queue_finished_eager_store(request, ([7],))
    assert manager._pending_finished_stores == []
    assert manager._in_flight_store_gpu_blocks == set()
    manager._gpu_block_pool.touch.assert_not_called()


def test_positional_store_rejection_keeps_cursor_for_retry(cpu_store, native_cpu_module, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    manager, policy = cpu_store
    state = SimpleNamespace(finished=False, block_ids=([7],), num_stored_blocks=[0])
    manager._reqs_to_store = {"busy": state}
    manager._gpu_block_pool = SimpleNamespace(touch=Mock(), blocks={7: object()})
    manager._in_flight_store_gpu_blocks = set()
    manager.enable_kv_cache_events = False
    manager.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.boundary_store_stats = native_cpu_module.BoundaryStoreStats()
    output = SimpleNamespace(kv_connector_block_state=None, preempted_req_ids=set(), num_scheduled_tokens={"busy": 1})
    monkeypatch.setattr(native_cpu_module, "yield_req_data", lambda _: [("busy", None, False)])
    monkeypatch.setattr(manager, "_select_eager_blocks_to_store", lambda *args: ([7], [1], None))
    policy.propose_for_store = lambda *args: None
    assert manager._prepare_eager_store_specs(output) == ([], [], [], None)
    assert state.num_stored_blocks == [0]
    assert manager._in_flight_store_gpu_blocks == set()
    manager._gpu_block_pool.touch.assert_not_called()


def test_boundary_store_rejection_does_not_pin_gpu(cpu_store, native_cpu_module, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    manager, policy = cpu_store
    manager._reqs_to_store = {"busy": SimpleNamespace(finished=False)}
    manager._gpu_block_pool = SimpleNamespace(touch=Mock(), blocks={7: object()})
    manager._in_flight_store_gpu_blocks = set()
    manager.enable_kv_cache_events = False
    manager.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.boundary_store_stats = native_cpu_module.BoundaryStoreStats()
    output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads={"busy": [(16, 7, 0)]}),
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )
    monkeypatch.setattr(native_cpu_module, "yield_req_data", lambda _: [])
    monkeypatch.setattr(manager, "_classify_store_candidate", lambda *args: native_cpu_module._StoreAdmission.READY)
    policy.propose_for_store = lambda *args: None
    assert manager._prepare_eager_store_specs(output) == ([], [], [], None)
    assert manager.boundary_store_stats.dropped_cpu_full == 1
    assert manager._in_flight_store_gpu_blocks == set()
    manager._gpu_block_pool.touch.assert_not_called()


def test_eager_selection_uses_typed_capacity_instead_of_native_free_count(cpu_store):
    from types import SimpleNamespace

    manager, policy = cpu_store
    pool = manager.cpu_block_pool
    blocks = manager._allocate_store_blocks(2, "idle")
    manager._observe_mori_owner("idle", blocks)
    pool.free_blocks(blocks)
    manager._gpu_block_pool = object()
    manager.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.cpu_coordinator = SimpleNamespace(enable_partial_hash_hits=False)
    manager.block_size = 16
    manager.enable_kv_cache_events = False
    manager.prefix_cacheable_group_ids = {0}
    request = SimpleNamespace(request_id="busy", num_computed_tokens=16, num_output_placeholders=0)
    state = SimpleNamespace(request=request)
    assert manager._select_eager_blocks_to_store(state, ([7],)) == ([], [0], None)
    assert pool.get_num_free_blocks() == 2


def test_cpu_registration_rejects_old_blockpool_without_strict_store_support(cpu_store, monkeypatch):
    manager, policy = cpu_store
    manager.unregister_mori_offload_tier_provider(policy)
    pool = manager.cpu_block_pool
    register = pool.register_mori_block_selection_provider

    def old_register(provider):
        register(provider)
        return {"hook_api_version": 1}

    monkeypatch.setattr(pool, "register_mori_block_selection_provider", old_register)
    with pytest.raises(RuntimeError, match="unsupported MORI BlockPool"):
        manager.register_mori_offload_tier_provider(3, policy)
    assert pool._mori_block_selection_provider is None
    assert manager._mori_offload_tier_provider is None


@pytest.fixture
def finished_retry(cpu_store, native_pool_class, native_cpu_module, monkeypatch):
    """Real finish/store/pool methods with two confirmed blocks and a full IDLE CPU cache."""
    from types import SimpleNamespace

    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

    manager, policy = cpu_store
    gpu = native_pool_class(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    blocks = gpu.get_new_blocks(2)
    hashes = [BlockHash(bytes([10 + i]) * 32) for i in range(2)]
    for block, block_hash in zip(blocks, hashes, strict=True):
        gpu._insert_block_hash(make_block_hash_with_group_id(block_hash, 0), block, num_tokens=16)
    # GPU backups cannot replace this CPU working set until committed demotion.
    occupied = manager._allocate_store_blocks(2, "idle")
    for block, value in zip(occupied, (20, 21), strict=True):
        manager.cpu_block_pool._insert_block_hash(
            make_block_hash_with_group_id(BlockHash(bytes([value]) * 32), 0), block, num_tokens=16
        )
    manager._observe_mori_owner("idle", occupied)
    manager.cpu_block_pool.free_blocks(occupied)
    manager._gpu_block_pool = gpu
    manager.enable_kv_cache_events = False
    manager.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.cpu_coordinator = SimpleNamespace(
        enable_partial_hash_hits=False,
        single_type_managers=[SimpleNamespace(has_positionally_stable_blocks=True)],
    )
    manager.prefix_cacheable_group_ids = {0}
    manager.group_block_sizes = [16]
    manager.hash_block_size = manager.block_size = 16
    manager._pending_finished_stores = []
    manager._in_flight_store_gpu_blocks = set()
    manager._pending_cpu_hits = {}
    manager._reqs_to_load = {}
    manager._store_event_to_blocks = {}
    manager._abandoned_store_event_to_blocks = {}
    manager._store_event_to_reqs = {}
    manager._store_event_pending_counts = {}
    manager._expected_worker_count = 1
    manager._store_event_counter = 0
    manager._load_event_to_reqs = {}
    request = SimpleNamespace(request_id="busy", num_computed_tokens=32, num_output_placeholders=0, block_hashes=hashes)
    block_ids = ([b.block_id for b in blocks],)
    manager._reqs_to_store = {
        "busy": native_cpu_module.StoreRequestState(request=request, block_ids=block_ids, num_stored_blocks=[0])
    }
    monkeypatch.setattr(manager, "_find_fa_partial_tail_source", lambda *args: None)
    return manager, policy, request, blocks


def _finish_retry_request(finished_retry):
    manager, policy, request, blocks = finished_retry
    manager.request_finished_all_groups(request, ([b.block_id for b in blocks],))
    manager._gpu_block_pool.free_blocks(reversed(blocks))
    policy.runtime_state.unbind_request(request.request_id)
    assert "busy" not in manager._reqs_to_store
    return manager, policy, request, blocks


def _commit_retry_demotion(policy, manager=None):
    from agentinfer.scheduling.mori import MORIProgramFactors, MORITier

    strategy = policy.runtime_state._provider
    strategy._set_factor(strategy._factors, ProgramRef("busy", 0), MORIProgramFactors(tier=MORITier.CPU))
    if manager is not None:
        manager._mori_drain_store_retries()


def test_finished_rejected_store_retries_after_unbind_and_uses_native_job(finished_retry, monkeypatch):
    from types import SimpleNamespace

    manager, policy, request, blocks = _finish_retry_request(finished_retry)
    assert manager._pending_finished_stores == []
    assert all(block.ref_cnt == 0 for block in blocks)
    assert policy.runtime_state.program_for_request(request.request_id) is None
    _commit_retry_demotion(policy)
    assert manager._pending_finished_stores == []
    assert manager.has_pending_stores()
    assert all(block.ref_cnt == 0 for block in blocks)
    # A transfer-only scheduler step uses the existing dispatch path.
    monkeypatch.setattr(manager, "prepare_store_specs", lambda _: ([], [], [], None))
    meta = manager.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
    assert meta.store_gpu_blocks == [block.block_id for block in blocks]
    assert manager._pending_finished_stores == []
    transfer = manager._store_event_to_blocks[meta.store_event]
    assert all(block.ref_cnt == 1 for block in blocks)
    assert all(policy.index.owners(block_id) == {ProgramRef("busy", 0)} for block_id in transfer.cpu_block_ids)
    assert all(manager.cpu_block_pool.blocks[i].block_hash is None for i in transfer.cpu_block_ids)
    manager._process_store_event(meta.store_event)
    assert not manager.has_pending_stores()
    assert all(block.ref_cnt == 0 for block in blocks)
    assert all(
        manager.cpu_block_pool.blocks[i].block_hash == block.block_hash
        for i, block in zip(transfer.cpu_block_ids, blocks, strict=True)
    )


@pytest.mark.parametrize("invalidate", ["replace", "release"])
def test_finished_retry_invalidated_context_never_pins(finished_retry, invalidate):
    manager, policy, _request, blocks = _finish_retry_request(finished_retry)
    owner = ProgramRef("busy", 0)
    if invalidate == "replace":
        policy.runtime_state.replace_context(owner)
    else:
        policy.runtime_state.on_program_released(owner)
    assert manager._mori_store_retries == {}
    _commit_retry_demotion(policy, manager)
    assert manager._pending_finished_stores == []
    assert all(block.ref_cnt == 0 for block in blocks)


def test_finished_retry_revalidates_source_content(finished_retry):
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

    manager, policy, _request, blocks = _finish_retry_request(finished_retry)
    changed = blocks[0]
    manager._gpu_block_pool.evict_blocks({changed.block_id})
    manager._gpu_block_pool._insert_block_hash(
        make_block_hash_with_group_id(BlockHash(b"z" * 32), 0), changed, num_tokens=16
    )
    _commit_retry_demotion(policy, manager)
    assert manager._pending_finished_stores[0].gpu_block_ids == [blocks[1].block_id]
    assert changed.ref_cnt == 0


def test_finished_retry_admission_rejection_holds_no_pins(finished_retry, monkeypatch):
    manager, policy, _request, blocks = _finish_retry_request(finished_retry)
    monkeypatch.setattr(policy, "propose_for_store", lambda *args: None)
    _commit_retry_demotion(policy, manager)
    assert manager._mori_store_retries == {}
    assert manager._pending_finished_stores == []
    assert all(block.ref_cnt == 0 for block in blocks)
    assert manager.cpu_block_pool.get_num_free_blocks() == 2


@pytest.mark.parametrize("dedup", ["cached", "inflight"])
def test_finished_retry_deduplicates_at_submission(finished_retry, dedup):
    manager, policy, _request, blocks = _finish_retry_request(finished_retry)
    if dedup == "cached":
        destination = manager._allocate_store_blocks(1, "idle")[0]
        manager.cpu_block_pool._insert_block_hash(blocks[0].block_hash, destination, num_tokens=16)
        manager._observe_mori_owner("idle", [destination])
        manager.cpu_block_pool.free_blocks([destination])
    else:
        manager._in_flight_store_gpu_blocks.add(blocks[0].block_id)
    _commit_retry_demotion(policy, manager)
    assert manager._pending_finished_stores[0].gpu_block_ids == [blocks[1].block_id]
    assert blocks[0].ref_cnt == 0


def test_retry_recaptures_block_cached_at_finish_but_later_evicted(finished_retry):
    manager, policy, _request, blocks = finished_retry
    destination = manager._allocate_store_blocks(1, "idle")[0]
    manager.cpu_block_pool._insert_block_hash(blocks[0].block_hash, destination, num_tokens=16)
    manager._observe_mori_owner("idle", [destination])
    manager.cpu_block_pool.free_blocks([destination])
    _finish_retry_request(finished_retry)
    manager.cpu_block_pool.evict_blocks({destination.block_id})
    _commit_retry_demotion(policy, manager)
    assert manager._pending_finished_stores[0].gpu_block_ids == [block.block_id for block in blocks]


def test_retry_completion_after_owner_release_keeps_native_cache_ownerless(finished_retry):
    manager, policy, _request, blocks = _finish_retry_request(finished_retry)
    _commit_retry_demotion(policy, manager)
    transfer = manager._pending_finished_stores.pop()
    policy.runtime_state.on_program_released(ProgramRef("busy", 0))
    manager._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
    assert all(not policy.index.owners(block_id) for block_id in transfer.cpu_block_ids)
    assert all(manager.cpu_block_pool.blocks[i].block_hash is not None for i in transfer.cpu_block_ids)
    assert all(block.ref_cnt == 0 for block in blocks)


def test_retry_capture_rejects_wrong_position_and_unconfirmed_tail(finished_retry):
    manager, _policy, request, blocks = finished_retry
    request.num_computed_tokens = 31
    # First position's native hash belongs to another position; second block isn't fully confirmed.
    manager._mori_capture_finished_store(request, ([blocks[1].block_id, blocks[0].block_id],))
    assert manager._mori_store_retries == {}


@pytest.mark.parametrize("dispatched", [False, True])
def test_retry_reset_preserves_inflight_pins_and_never_publishes_abandoned_data(
    finished_retry, native_cpu_module, monkeypatch, dispatched
):
    from types import SimpleNamespace

    manager, policy, _request, blocks = _finish_retry_request(finished_retry)
    _commit_retry_demotion(policy, manager)
    transfer = manager._pending_finished_stores[0]
    manager._abandoned_reqs_to_load = {}
    manager.boundary_store_stats = native_cpu_module.BoundaryStoreStats()
    manager._boundary_stats_snapshot = native_cpu_module.BoundaryStoreStats()
    if dispatched:
        monkeypatch.setattr(manager, "prepare_store_specs", lambda _: ([], [], [], None))
        metadata = manager.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
    assert manager.reset() is (not dispatched)
    assert all(block.ref_cnt == int(dispatched) for block in blocks)
    assert all(manager.cpu_block_pool.blocks[i].block_hash is None for i in transfer.cpu_block_ids)
    if dispatched:
        assert manager.has_pending_stores()
        manager._process_store_event(metadata.store_event)
    assert not manager.has_pending_stores()
    assert all(block.ref_cnt == 0 for block in blocks)
    assert all(manager.cpu_block_pool.blocks[i].ref_cnt == 0 for i in transfer.cpu_block_ids)
    assert all(not policy.index.owners(i) for i in transfer.cpu_block_ids)
    assert all(manager.cpu_block_pool.blocks[i].block_hash is None for i in transfer.cpu_block_ids)


def test_cpu_registration_rejects_hook_api_without_context_retry(cpu_store):
    manager, policy = cpu_store
    with pytest.raises(ValueError, match="unsupported MORI SimpleCPU"):
        manager.register_mori_offload_tier_provider(2, policy)


def test_context_replacement_cancels_demoted_intent_before_native_allocation(finished_retry):
    manager, policy, _request, blocks = _finish_retry_request(finished_retry)
    _commit_retry_demotion(policy)
    assert manager.has_pending_stores()
    policy.runtime_state.replace_context(ProgramRef("busy", 0))
    assert not manager.has_pending_stores()
    manager._mori_drain_store_retries()
    assert manager._pending_finished_stores == []
    assert all(block.ref_cnt == 0 for block in blocks)
