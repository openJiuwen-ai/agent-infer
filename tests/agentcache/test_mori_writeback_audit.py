# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Observe physical source loss without changing native retry behavior."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agentinfer.agentcache.core.controller import build_mori_controller
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori.scheduler import _VllmAdmissionHooks
from agentinfer.agentcache.mori.vllm_adapter import VllmMORIAdapter
from agentinfer.agentcache.mori.writeback_audit import WritebackSourceAudit
from agentinfer.scheduling.mori import MORIConfig, MORIStrategy
from tests.agentcache.core.test_mori_controller import backend

pytestmark = pytest.mark.cpu_test


def setup_audit(*, capacity=4):
    context = object()
    current = {context}
    blocks = [SimpleNamespace(is_null=i == 0, block_hash=str(i), block_hash_num_tokens=16) for i in range(5)]
    candidates = tuple(SimpleNamespace(block_id=i, block_hash=str(i), num_tokens=16) for i in range(1, 5))
    original = Mock(return_value="native result")
    manager = SimpleNamespace(
        _mori_submit_store_retry=original,
        _mori_store_capacity=lambda receipt: capacity,
        _mori_offload_tier_provider=SimpleNamespace(store_retry_context_is_current=lambda receipt: receipt in current),
        _gpu_block_pool=SimpleNamespace(blocks=blocks),
        cpu_block_pool=SimpleNamespace(cached_block_hash_to_block=SimpleNamespace(get_one_block={"3": object()}.get)),
        _mori_pending_store_retries={context: candidates},
        _pending_finished_stores=[],
        _in_flight_store_gpu_blocks={4},
    )
    strategy = MORIStrategy(MORIConfig(audit_writeback_sources=True))
    audit = WritebackSourceAudit(manager, strategy)
    return audit, manager, strategy, context, candidates, original, current


def test_snapshot_and_submit_are_read_only_and_keep_native_arguments_and_return():
    audit, manager, strategy, context, candidates, original, _ = setup_audit()
    before = [vars(item).copy() for item in manager._gpu_block_pool.blocks]
    pending = manager._mori_pending_store_retries.copy()
    audit.snapshot_before_allocation()
    assert strategy.counters["writeback_audit_missing_before"] == 2
    assert manager._mori_pending_store_retries == pending
    assert [vars(item) for item in manager._gpu_block_pool.blocks] == before
    assert manager._mori_submit_store_retry(context, candidates) == "native result"
    original.assert_called_once_with(context, candidates)
    assert strategy.counters["writeback_audit_matched_missing"] == 2
    assert strategy.counters["writeback_audit_sources_lost"] == 0
    assert not manager._pending_finished_stores
    assert not strategy.counters["gpu_service_completed_requests"]
    audit.close()


def test_shared_physical_sources_are_counted_once_per_native_step():
    audit, manager, strategy, context, candidates, original, current = setup_audit()
    other = object()
    current.add(other)
    manager._mori_pending_store_retries[other] = candidates
    audit.snapshot_before_allocation()
    manager._gpu_block_pool.blocks[2].block_hash = "reused by native allocation"
    manager._mori_submit_store_retry(context, candidates)
    manager._mori_submit_store_retry(other, candidates)
    assert original.call_count == 2
    assert strategy.counters["writeback_audit_candidates"] == 4
    assert strategy.counters["writeback_audit_missing_before"] == 2
    assert strategy.counters["writeback_audit_matched_missing"] == 2
    assert strategy.counters["writeback_audit_sources_lost"] == 1
    assert strategy.counters["writeback_audit_lost_with_capacity"] == 1
    audit.close()


def test_source_loss_is_distinct_from_unavailable_cpu_capacity():
    audit, manager, strategy, context, candidates, _, _ = setup_audit(capacity=0)
    audit.snapshot_before_allocation()
    manager._gpu_block_pool.blocks[1].block_hash = None
    manager._mori_submit_store_retry(context, candidates)
    assert strategy.counters["writeback_audit_sources_lost"] == 1
    assert strategy.counters["writeback_audit_lost_with_capacity"] == 0
    assert strategy.counters["writeback_audit_zero_capacity_intents"] == 1
    audit.close()


def test_invalidated_receipt_is_cancelled_rather_than_counted_as_allocation_loss():
    audit, manager, strategy, context, candidates, original, current = setup_audit()
    audit.snapshot_before_allocation()
    current.clear()
    manager._gpu_block_pool.blocks[1].block_hash = None
    manager._mori_submit_store_retry(context, candidates)
    original.assert_called_once_with(context, candidates)
    assert strategy.counters["writeback_audit_cancelled_intents"] == 1
    assert strategy.counters["writeback_audit_sources_lost"] == 0
    audit.close()


def test_changed_candidate_receipt_is_not_joined_to_an_old_snapshot():
    audit, manager, strategy, context, candidates, original, _ = setup_audit()
    audit.snapshot_before_allocation()
    replacement = (*candidates,)
    assert replacement is not candidates
    manager._mori_submit_store_retry(context, replacement)
    original.assert_called_once_with(context, replacement)
    assert strategy.counters["writeback_audit_unmatched_intents"] == 1
    assert strategy.counters["writeback_audit_matched_intents"] == 0
    audit.close()


def test_only_newly_queued_native_copies_are_recorded():
    audit, manager, strategy, context, candidates, original, _ = setup_audit()
    manager._pending_finished_stores.append(SimpleNamespace(gpu_block_ids=[9, 10, 11]))
    original.side_effect = lambda *args: manager._pending_finished_stores.append(SimpleNamespace(gpu_block_ids=[1, 2]))
    audit.snapshot_before_allocation()
    manager._mori_submit_store_retry(context, candidates)
    assert strategy.counters["writeback_audit_submitted_blocks"] == 2
    audit.close()


def test_native_failure_propagates_and_hook_restoration_is_idempotent():
    audit, manager, strategy, context, candidates, original, _ = setup_audit()
    original.side_effect = RuntimeError("native retry failure")
    audit.snapshot_before_allocation()
    with pytest.raises(RuntimeError, match="native retry failure"):
        manager._mori_submit_store_retry(context, candidates)
    assert strategy.counters["writeback_audit_native_errors"] == 1
    assert strategy.counters["writeback_audit_submitted_blocks"] == 0
    audit.close()
    audit.close()
    assert manager._mori_submit_store_retry is original
    assert not audit._snapshots


def test_loss_evidence_is_bounded_across_native_steps():
    audit, manager, strategy, context, candidates, original, _ = setup_audit()
    with patch("agentinfer.agentcache.mori.writeback_audit.logger.info") as log:
        for _ in range(20):
            manager._gpu_block_pool.blocks[1].block_hash = "1"
            audit.snapshot_before_allocation()
            manager._gpu_block_pool.blocks[1].block_hash = "new allocation"
            manager._mori_submit_store_retry(context, candidates)
    assert log.call_count == 8
    assert original.call_count == 20
    assert strategy.counters["writeback_audit_sources_lost"] == 20
    audit.close()


def test_native_allocation_is_observed_after_snapshot_with_batch_feature_disabled():
    audit, manager, strategy, context, candidates, _, _ = setup_audit()
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.controller = SimpleNamespace(strategy=strategy)
    hooks._mori_adapter = SimpleNamespace(writeback_audit=audit)

    def native():
        assert context in audit._snapshots
        manager._gpu_block_pool.blocks[2].block_hash = None
        return manager._mori_submit_store_retry(context, candidates)

    assert hooks.schedule_native(object(), native) == "native result"
    assert strategy.counters["writeback_audit_sources_lost"] == 1
    assert strategy.counters["decode_batch_capped_iterations"] == 0
    audit.close()


@pytest.mark.parametrize("field", ["_mori_store_capacity", "_gpu_block_pool", "_in_flight_store_gpu_blocks"])
def test_unsupported_native_contract_fails_without_wrapping(field):
    audit, manager, strategy, _, _, original, _ = setup_audit()
    audit.close()
    setattr(manager, field, None)
    with pytest.raises(RuntimeError, match="requires the SimpleCPU"):
        WritebackSourceAudit(manager, strategy)
    assert manager._mori_submit_store_retry is original


def test_diagnostic_cleanup_failure_still_releases_every_native_handle():
    strategy = MORIStrategy(MORIConfig())
    hbm, cpu, unregister_cpu = Mock(), Mock(), Mock()
    owner = SimpleNamespace(kv_cache_manager=SimpleNamespace(unregister_mori_block_selection_provider=Mock()))
    adapter = VllmMORIAdapter(hbm, cpu, owner, strategy, unregister_cpu)
    adapter.writeback_audit = SimpleNamespace(close=Mock(side_effect=RuntimeError("audit cleanup")))
    with (
        patch.object(MORI_RUNTIME_STATE, "unbind_provider") as unbind,
        patch.object(MORI_RUNTIME_STATE, "clear_requests") as clear,
        pytest.raises(RuntimeError, match="audit cleanup"),
    ):
        adapter.close()
    owner.kv_cache_manager.unregister_mori_block_selection_provider.assert_called_once_with(hbm)
    hbm.close.assert_called_once()
    cpu.close.assert_called_once()
    unregister_cpu.assert_called_once()
    unbind.assert_called_once_with(strategy)
    clear.assert_called_once()


def test_audit_configuration_is_explicit():
    assert not MORIConfig().audit_writeback_sources
    assert build_mori_controller(
        backend(), {"mori": {"audit_writeback_sources": True}}
    ).strategy.config.audit_writeback_sources


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_audit_configuration_rejects_non_boolean_values(value):
    with pytest.raises(ValueError):
        MORIConfig(audit_writeback_sources=value)
    with pytest.raises(ValueError):
        build_mori_controller(backend(), {"mori": {"audit_writeback_sources": value}})
