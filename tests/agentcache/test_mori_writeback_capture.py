# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Check retry capture against actual vLLM block metadata, without GPU inference."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata

from agentinfer.agentcache.mori.writeback_audit import WritebackSourceAudit
from agentinfer.agentcache.mori.writeback_capture import FullPrefixWritebackCapture
from agentinfer.scheduling.mori import MORIConfig, MORIStrategy

pytestmark = pytest.mark.cpu_test


def native_metadata():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(3)
    request = SimpleNamespace(
        request_id="capture-test",
        num_computed_tokens=48,
        num_output_placeholders=0,
        block_hashes=[bytes([i]) * 32 for i in [1, 2, 3]],
    )
    pool.cache_full_blocks(request, blocks, 0, 3, 16, 0)
    context = object()
    original = Mock()
    manager = SimpleNamespace(
        _mori_capture_finished_store=original,
        _gpu_block_pool=pool,
        _mori_offload_tier_provider=SimpleNamespace(capture_store_retry_context=lambda key: context),
        _mori_store_retries={},
        prefix_cacheable_group_ids={0},
        group_block_sizes=[16],
        hash_block_size=16,
        cpu_coordinator=SimpleNamespace(single_type_managers=[SimpleNamespace(has_positionally_stable_blocks=True)]),
    )
    strategy = MORIStrategy(MORIConfig(repair_writeback_endpoints=True))
    return manager, strategy, pool, blocks, request, context, original


def test_native_absolute_endpoints_capture_the_complete_prefix():
    manager, strategy, _, blocks, request, context, original = native_metadata()
    assert [block.block_hash_num_tokens for block in blocks] == [16, 32, 48]
    capture = FullPrefixWritebackCapture(manager, strategy)
    manager._mori_capture_finished_store(request, ([block.block_id for block in blocks],))
    sources = manager._mori_store_retries[context]
    assert [item.block_id for item in sources] == [block.block_id for block in blocks]
    assert [item.num_tokens for item in sources] == [16, 32, 48]
    assert [item.block_hash for item in sources] == [block.block_hash for block in blocks]
    assert strategy.counters["writeback_capture_noninitial_blocks"] == 2
    capture.close()
    capture.close()
    assert manager._mori_capture_finished_store is original


def test_unconfirmed_output_placeholders_exclude_the_incomplete_tail():
    manager, strategy, _, blocks, request, context, _ = native_metadata()
    request.num_output_placeholders = 1
    capture = FullPrefixWritebackCapture(manager, strategy)
    manager._mori_capture_finished_store(request, ([block.block_id for block in blocks],))
    assert [item.num_tokens for item in manager._mori_store_retries[context]] == [16, 32]
    capture.close()


def test_real_native_block_id_reuse_invalidates_old_source_proof():
    manager, strategy, pool, blocks, request, context, _ = native_metadata()
    capture = FullPrefixWritebackCapture(manager, strategy)
    ids = [block.block_id for block in blocks]
    manager._mori_capture_finished_store(request, (ids,))
    assert len(manager._mori_store_retries[context]) == 3
    pool.free_blocks(blocks)
    # Native allocation evicts the old hashes before reusing these physical IDs.
    pool.get_new_blocks(7)
    manager._mori_capture_finished_store(request, (ids,))
    assert context not in manager._mori_store_retries
    capture.close()


def test_empty_native_block_table_clears_old_capture_without_preparing_work():
    manager, strategy, _, blocks, request, context, _ = native_metadata()
    capture = FullPrefixWritebackCapture(manager, strategy)
    manager._mori_capture_finished_store(request, ([block.block_id for block in blocks],))
    manager._mori_capture_finished_store(request, ())
    assert context not in manager._mori_store_retries
    capture.close()


def test_nonpositional_layout_fails_without_replacing_native_hook():
    manager, strategy, _, _, _, _, original = native_metadata()
    manager.cpu_coordinator.single_type_managers[0].has_positionally_stable_blocks = False
    with pytest.raises(RuntimeError, match="positional SimpleCPU"):
        FullPrefixWritebackCapture(manager, strategy)
    assert manager._mori_capture_finished_store is original


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_endpoint_repair_flag_is_strict_boolean(value):
    with pytest.raises(ValueError):
        MORIConfig(repair_writeback_endpoints=value)


def test_endpoint_repair_is_explicit():
    assert not MORIConfig().repair_writeback_endpoints
    assert MORIConfig(repair_writeback_endpoints=True).repair_writeback_endpoints


def native_transfer(*, capacity=3):
    """Real native pools and bookkeeping; no GPU bytes copied by this CPU test."""
    manager, strategy, pool, blocks, request, context, _ = native_metadata()
    manager.cpu_block_pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    cpu = manager.cpu_block_pool
    # A store must pass the real native provider admission/queue validation.
    cpu.register_mori_block_selection_provider(
        SimpleNamespace(
            HOOK_API_VERSION=1,
            propose_for_store=lambda count, receipt: tuple(
                b.block_id for b in cpu.free_block_queue.get_all_free_blocks()[:count]
            ),
            on_cache_reset=Mock(),
            on_block_ready=Mock(),
            on_block_owner_added=Mock(),
            on_block_evictability_changed=Mock(),
            on_block_removed=Mock(),
        )
    )
    provider = manager._mori_offload_tier_provider
    provider.store_retry_context_is_current = lambda receipt: receipt is context
    provider.store_capacity = lambda receipt: capacity
    provider.on_block_removed = Mock()
    manager._mori_pending_store_retries = {}
    manager._pending_finished_stores = []
    manager._in_flight_store_gpu_blocks = set()
    manager._store_event_to_blocks = {}
    manager._abandoned_store_event_to_blocks = {}
    manager._store_event_to_reqs = {}
    manager._store_event_pending_counts = {}
    manager._expected_worker_count = 2
    manager._mori_resident_block_ids = set()
    manager._mori_offload_tier_bridge = None
    manager._lazy_mode = False
    manager.enable_kv_cache_events = False
    for name in (
        "_mori_submit_store_retry",
        "_mori_store_capacity",
        "_allocate_store_blocks",
        "_classify_store_candidate",
        "_observe_mori_owner",
        "_process_store_completion",
        "_release_transfer_refs",
        "_process_store_event",
        "_discard_mori_entries",
        "update_connector_output",
    ):
        setattr(manager, name, MethodType(getattr(SimpleCPUOffloadScheduler, name), manager))
    capture = FullPrefixWritebackCapture(manager, strategy)
    manager._mori_capture_finished_store(request, ([block.block_id for block in blocks],))
    candidates = manager._mori_store_retries[context]
    manager._mori_pending_store_retries[context] = candidates
    pool.free_blocks(blocks)
    audit = WritebackSourceAudit(manager, strategy, track_completions=True)
    audit.snapshot_before_allocation()
    return manager, strategy, pool, blocks, context, candidates, capture, audit


def test_real_native_retry_pins_until_completion_and_registers_all_endpoints():
    manager, strategy, _, blocks, context, candidates, capture, audit = native_transfer()
    original_submit, original_complete, original_release = (
        audit._original,
        audit._original_completion,
        audit._original_release,
    )
    manager._mori_submit_store_retry(context, candidates)
    (transfer,) = manager._pending_finished_stores
    assert [block.ref_cnt for block in blocks] == [1, 1, 1]
    assert all(manager.cpu_block_pool.blocks[i].ref_cnt == 1 for i in transfer.cpu_block_ids)
    assert strategy.counters["writeback_audit_completed_blocks"] == 0
    manager._store_event_to_blocks[1] = transfer
    worker_ack = KVConnectorOutput(kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata({1: 1}))
    manager.update_connector_output(worker_ack)
    assert strategy.counters["writeback_audit_completed_blocks"] == 0
    assert [block.ref_cnt for block in blocks] == [1, 1, 1]
    manager.update_connector_output(worker_ack)
    assert not manager._in_flight_store_gpu_blocks
    for gpu, cpu_id in zip(blocks, transfer.cpu_block_ids, strict=True):
        cpu = manager.cpu_block_pool.blocks[cpu_id]
        assert cpu.block_hash == gpu.block_hash
        assert cpu.block_hash_num_tokens == gpu.block_hash_num_tokens
        assert cpu.ref_cnt == gpu.ref_cnt == 0
    assert strategy.counters["writeback_audit_completed_blocks"] == 3
    assert strategy.counters["writeback_audit_completed_noninitial_blocks"] == 2
    assert not audit._targets
    audit.close()
    capture.close()
    assert manager._mori_submit_store_retry is original_submit
    assert manager._process_store_completion is original_complete
    assert manager._release_transfer_refs is original_release


def test_native_abandoned_transfer_releases_without_completion_credit():
    manager, strategy, _, blocks, context, candidates, capture, audit = native_transfer()
    manager._mori_submit_store_retry(context, candidates)
    (transfer,) = manager._pending_finished_stores
    # Native reset removes live event bookkeeping; later worker ack only releases.
    manager._in_flight_store_gpu_blocks.clear()
    manager._abandoned_store_event_to_blocks[1] = transfer
    manager._process_store_event(1)
    assert [block.ref_cnt for block in blocks] == [0, 0, 0]
    assert all(manager.cpu_block_pool.blocks[i].ref_cnt == 0 for i in transfer.cpu_block_ids)
    assert all(manager.cpu_block_pool.blocks[i].block_hash is None for i in transfer.cpu_block_ids)
    assert strategy.counters["writeback_audit_abandoned_blocks"] == 3
    assert not strategy.counters["writeback_audit_completed_blocks"]
    assert not audit._targets
    audit.close()
    capture.close()


def test_native_retry_rejects_sources_reused_after_capture():
    manager, strategy, pool, _, context, candidates, capture, audit = native_transfer()
    pool.get_new_blocks(7)
    manager._mori_submit_store_retry(context, candidates)
    assert not manager._pending_finished_stores
    assert strategy.counters["writeback_audit_sources_lost"] == 3
    assert not strategy.counters["writeback_audit_submitted_blocks"]
    audit.close()
    capture.close()


@pytest.mark.parametrize("capacity", [0, 1])
def test_native_retry_respects_actual_destination_capacity(capacity):
    manager, strategy, _, blocks, context, candidates, capture, audit = native_transfer(capacity=capacity)
    manager._mori_submit_store_retry(context, candidates)
    assert sum(len(item.cpu_block_ids) for item in manager._pending_finished_stores) == capacity
    assert sum(block.ref_cnt for block in blocks) == capacity
    assert strategy.counters["writeback_audit_submitted_blocks"] == capacity
    for transfer in manager._pending_finished_stores:
        manager._release_transfer_refs(transfer)
    assert not strategy.counters["writeback_audit_completed_blocks"]
    audit.close()
    capture.close()


def test_duplicate_submit_has_no_duplicate_physical_pins_or_destinations():
    manager, strategy, _, blocks, context, candidates, capture, audit = native_transfer()
    manager._mori_submit_store_retry(context, candidates)
    manager._mori_submit_store_retry(context, candidates)
    assert len(manager._pending_finished_stores) == 1
    assert [block.ref_cnt for block in blocks] == [1, 1, 1]
    assert strategy.counters["writeback_audit_submitted_blocks"] == 3
    manager._release_transfer_refs(manager._pending_finished_stores[0])
    audit.close()
    capture.close()


def test_unregistered_cpu_admission_rejects_backup_without_pins_or_credit():
    manager, strategy, _, blocks, context, candidates, capture, audit = native_transfer()
    cpu = manager.cpu_block_pool
    cpu.unregister_mori_block_selection_provider(cpu._mori_block_selection_provider)
    manager._mori_submit_store_retry(context, candidates)
    assert not manager._pending_finished_stores
    assert [block.ref_cnt for block in blocks] == [0, 0, 0]
    assert not strategy.counters["writeback_audit_submitted_blocks"]
    assert not strategy.counters["writeback_audit_completed_blocks"]
    audit.close()
    capture.close()
