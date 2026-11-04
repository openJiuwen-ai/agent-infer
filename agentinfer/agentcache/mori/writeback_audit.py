# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Read-only evidence around native allocation and best-effort KV writeback."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from agentinfer.scheduling.mori import MORIStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SourceSnapshot:
    candidates: object
    missing: frozenset[int]
    capacity: int
    current: bool


class WritebackSourceAudit:
    """Observe real native retries without touching blocks, owners or transfers.

    Snapshot after MORI commits placement and before native allocation. Native
    metadata construction later invokes the original submit exactly once. Hash
    loss is distinguished from existing backups, pinned stores, unavailable CPU
    capacity and invalidated context receipts. Counts are diagnostic, not service.
    """

    def __init__(self, manager: Any, strategy: MORIStrategy, *, track_completions: bool = False) -> None:
        required = ("_mori_submit_store_retry", "_mori_store_capacity")
        provider = getattr(manager, "_mori_offload_tier_provider", None)
        gpu = getattr(manager, "_gpu_block_pool", None)
        cpu = getattr(manager, "cpu_block_pool", None)
        cached = getattr(cpu, "cached_block_hash_to_block", None)
        if (
            not all(callable(getattr(manager, key, None)) for key in required)
            or not isinstance(getattr(manager, "_mori_pending_store_retries", None), dict)
            or not isinstance(getattr(manager, "_pending_finished_stores", None), list)
            or not isinstance(getattr(manager, "_in_flight_store_gpu_blocks", None), set)
            or not callable(getattr(provider, "store_retry_context_is_current", None))
            or gpu is None
            or not callable(getattr(cached, "get_one_block", None))
        ):
            raise RuntimeError("MORI writeback source audit requires the SimpleCPU retry/source contract")
        if track_completions and (
            not callable(getattr(manager, "_process_store_completion", None))
            or not callable(getattr(manager, "_release_transfer_refs", None))
            or getattr(cpu, "blocks", None) is None
        ):
            raise RuntimeError("MORI writeback completion audit requires native store completion and release hooks")
        self.manager = manager
        self.strategy = strategy
        self._snapshots: dict[object, _SourceSnapshot] = {}
        self._step_lost: set[int] = set()
        self._step_lost_capacity: set[int] = set()
        self._step_matched: set[int] = set()
        self._evidence_records = 0
        self._track_completions = track_completions
        self._targets: dict[int, tuple[int, object, int]] = {}
        self._original_completion = manager._process_store_completion if track_completions else None
        self._original_release = manager._release_transfer_refs if track_completions else None
        self._completion_wrapper = self._store_completion
        self._release_wrapper = self._release_refs
        self._original = manager._mori_submit_store_retry
        self._wrapper = self._submit
        self.closed = False
        manager._mori_submit_store_retry = self._wrapper
        if track_completions:
            manager._process_store_completion = self._completion_wrapper
            manager._release_transfer_refs = self._release_wrapper

    def _valid(self, candidate: Any) -> bool:
        block = self.manager._gpu_block_pool.blocks[candidate.block_id]
        return (
            not block.is_null
            and block.block_hash == candidate.block_hash
            and block.block_hash_num_tokens == candidate.num_tokens
        )

    def snapshot_before_allocation(self) -> None:
        """Only immutable hash checks and read-only native capacity/cache lookup."""
        started = time.thread_time()
        try:
            self._snapshots.clear()
            self._step_lost.clear()
            self._step_lost_capacity.clear()
            self._step_matched.clear()
            manager = self.manager
            all_candidates: set[int] = set()
            all_missing: set[int] = set()
            for context, candidates in manager._mori_pending_store_retries.items():
                current = manager._mori_offload_tier_provider.store_retry_context_is_current(context)
                capacity = manager._mori_store_capacity(context) if current else 0
                missing = frozenset(
                    candidate.block_id
                    for candidate in candidates
                    if current
                    and self._valid(candidate)
                    and candidate.block_id not in manager._in_flight_store_gpu_blocks
                    and manager.cpu_block_pool.cached_block_hash_to_block.get_one_block(candidate.block_hash) is None
                )
                self._snapshots[context] = _SourceSnapshot(candidates, missing, capacity, current)
                all_candidates.update(candidate.block_id for candidate in candidates)
                all_missing.update(missing)
                self.strategy.counters["writeback_audit_snapshot_intents"] += 1
                self.strategy.counters["writeback_audit_zero_capacity_intents"] += int(current and capacity == 0)
            self.strategy.counters["writeback_audit_candidates"] += len(all_candidates)
            self.strategy.counters["writeback_audit_missing_before"] += len(all_missing)
        finally:
            self.strategy.writeback_audit_cpu_seconds += time.thread_time() - started

    def _submit(self, context: object, candidates: Any) -> Any:
        started = time.thread_time()
        snapshot = self._snapshots.pop(context, None)
        manager = self.manager
        current = manager._mori_offload_tier_provider.store_retry_context_is_current(context)
        if snapshot is None or snapshot.candidates is not candidates:
            self.strategy.counters["writeback_audit_unmatched_intents"] += 1
        elif not current or not snapshot.current:
            self.strategy.counters["writeback_audit_cancelled_intents"] += 1
        else:
            lost = {
                candidate.block_id
                for candidate in candidates
                if candidate.block_id in snapshot.missing and not self._valid(candidate)
            }
            capacity = manager._mori_store_capacity(context)
            self.strategy.counters["writeback_audit_matched_intents"] += 1
            self.strategy.counters["writeback_audit_matched_missing"] += len(snapshot.missing - self._step_matched)
            self._step_matched.update(snapshot.missing)
            new_lost = lost - self._step_lost
            self.strategy.counters["writeback_audit_sources_lost"] += len(new_lost)
            self._step_lost.update(lost)
            if snapshot.capacity > 0 and capacity > 0:
                self.strategy.counters["writeback_audit_lost_with_capacity"] += len(lost - self._step_lost_capacity)
                self._step_lost_capacity.update(lost)
            if new_lost and self._evidence_records < 8:
                program = getattr(context, "program", None)
                samples = [
                    {
                        "block_id": candidate.block_id,
                        "expected_hash": repr(candidate.block_hash),
                        "actual_hash": repr(manager._gpu_block_pool.blocks[candidate.block_id].block_hash),
                    }
                    for candidate in candidates
                    if candidate.block_id in new_lost
                ][:8]
                logger.info(
                    "MORI writeback source audit %s",
                    json.dumps(
                        {
                            "phase": "before_native_retry",
                            "program": getattr(program, "program_id", None),
                            "generation": getattr(program, "generation", None),
                            "missing_before": len(snapshot.missing),
                            "new_lost_sources": len(new_lost),
                            "capacity_before": snapshot.capacity,
                            "capacity_at_retry": capacity,
                            "samples": samples,
                        },
                        sort_keys=True,
                    ),
                )
                self._evidence_records += 1
        before = len(manager._pending_finished_stores)
        self.strategy.writeback_audit_cpu_seconds += time.thread_time() - started
        try:
            result = self._original(context, candidates)
        except Exception:
            self.strategy.counters["writeback_audit_native_errors"] += 1
            raise
        started = time.thread_time()
        try:
            queued = manager._pending_finished_stores[before:]
            if self._track_completions:
                for transfer in queued:
                    for gpu_id, cpu_id in zip(transfer.gpu_block_ids, transfer.cpu_block_ids, strict=True):
                        block = manager._gpu_block_pool.blocks[gpu_id]
                        self._targets[cpu_id] = (gpu_id, block.block_hash, block.block_hash_num_tokens)
            self.strategy.counters["writeback_audit_submitted_blocks"] += sum(
                len(item.gpu_block_ids) for item in queued
            )
            return result
        finally:
            self.strategy.writeback_audit_cpu_seconds += time.thread_time() - started

    def _store_completion(self, gpu_ids: list[int], cpu_ids: list[int], *args, **kwargs):
        """Count a retry target only after the real native completion validates it."""
        result = self._original_completion(gpu_ids, cpu_ids, *args, **kwargs)
        started = time.thread_time()
        try:
            size = self.manager.group_block_sizes[0]
            for gpu_id, cpu_id in zip(gpu_ids, cpu_ids, strict=True):
                expected = self._targets.pop(cpu_id, None)
                if expected is None:
                    continue
                block = self.manager.cpu_block_pool.blocks[cpu_id]
                if (
                    expected[0] != gpu_id
                    or block.block_hash != expected[1]
                    or block.block_hash_num_tokens != expected[2]
                ):
                    self.strategy.counters["writeback_audit_invalid_completions"] += 1
                    raise RuntimeError("MORI prepared writeback completion has mismatched native block proof")
                self.strategy.counters["writeback_audit_completed_blocks"] += 1
                self.strategy.counters["writeback_audit_completed_noninitial_blocks"] += int(expected[2] > size)
            return result
        finally:
            self.strategy.writeback_audit_cpu_seconds += time.thread_time() - started

    def _release_refs(self, transfer: Any):
        """Abandoned native transfers release their own refs; do not credit a copy."""
        result = self._original_release(transfer)
        started = time.thread_time()
        try:
            for cpu_id in transfer.cpu_block_ids:
                if self._targets.pop(cpu_id, None) is not None:
                    self.strategy.counters["writeback_audit_abandoned_blocks"] += 1
            return result
        finally:
            self.strategy.writeback_audit_cpu_seconds += time.thread_time() - started

    def close(self) -> None:
        """Restore only our own hook; discard metadata without touching native work."""
        if self.closed:
            return
        self.closed = True
        if self.manager._mori_submit_store_retry is self._wrapper:
            self.manager._mori_submit_store_retry = self._original
        if self._track_completions:
            if self.manager._process_store_completion is self._completion_wrapper:
                self.manager._process_store_completion = self._original_completion
            if self.manager._release_transfer_refs is self._release_wrapper:
                self.manager._release_transfer_refs = self._original_release
        self._targets.clear()
        self._snapshots.clear()
        self._step_lost.clear()
        self._step_lost_capacity.clear()
        self._step_matched.clear()
