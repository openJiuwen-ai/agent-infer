# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Capability-checked vLLM integration for full MORI cache management."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import vllm
from packaging.version import InvalidVersion, Version

from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
from agentinfer.agentcache.mori.writeback_audit import WritebackSourceAudit
from agentinfer.scheduling.mori import MORIStrategy

if TYPE_CHECKING:
    from agentinfer.agentcache.mori.writeback_capture import FullPrefixWritebackCapture


def require_mori_vllm_version() -> None:
    """Reject MORI on vLLM versions older than 0.31 before installing any hooks."""
    version = vllm.__version__
    try:
        supported = Version(version) >= Version("0.31")
    except (InvalidVersion, TypeError) as exc:
        raise RuntimeError(f"MORI requires vLLM >= 0.31; cannot parse installed version {version!r}") from exc
    if not supported:
        raise RuntimeError(f"MORI requires vLLM >= 0.31; installed version is {version}")


@dataclass(frozen=True)
class VllmOffloadObservation:
    """Rank-local CPU-offload capacity and transfer facts."""

    total_dram_kv_tokens: int | None = None
    used_dram_kv_tokens: int | None = None
    pending_h2d_kv_tokens: int | None = None
    pending_d2h_kv_tokens: int | None = None
    dram_chunk_size_tokens: int | None = None


@dataclass
class VllmMORIAdapter:
    """Installed native policy handles retained by the scheduler bridge."""

    hbm_policy: MORIHBMPolicy | None
    cpu_policy: object | None
    owner: object
    strategy: MORIStrategy
    cpu_unregister: Callable[[], None] | None = None
    closed: bool = False
    writeback_audit: WritebackSourceAudit | None = None
    writeback_capture: FullPrefixWritebackCapture | None = None

    def close(self) -> None:
        """Unregister native hooks and process-local indexes idempotently."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.writeback_audit is not None:
                self.writeback_audit.close()
        finally:
            try:
                if self.writeback_capture is not None:
                    self.writeback_capture.close()
            finally:
                self._close_native()

    def _close_native(self) -> None:
        """Release native handles even if diagnostic restoration fails."""
        kv_cache_manager = getattr(self.owner, "kv_cache_manager", None)
        try:
            if self.hbm_policy is not None:
                unregister = getattr(kv_cache_manager, "unregister_mori_block_selection_provider", None)
                if callable(unregister):
                    unregister(self.hbm_policy)
        finally:
            try:
                if self.hbm_policy is not None:
                    self.hbm_policy.close()
            finally:
                try:
                    if self.cpu_unregister is not None:
                        self.cpu_unregister()
                finally:
                    try:
                        close_cpu = getattr(self.cpu_policy, "close", None)
                        if callable(close_cpu):
                            close_cpu()
                    finally:
                        MORI_RUNTIME_STATE.unbind_provider(self.strategy)
                        MORI_RUNTIME_STATE.clear_requests()


def observe_vllm_offload(owner: object) -> VllmOffloadObservation:
    """Read vLLM 0.31 scheduler-side CPU cache facts without mutating it."""
    scheduler = _offload_scheduler(owner)
    if scheduler is None:
        return VllmOffloadObservation()
    snapshot = getattr(scheduler, "get_mori_offload_tier_snapshot", None)
    if callable(snapshot):
        return _simple_cpu_observation(snapshot())
    manager = getattr(scheduler, "manager", None)
    config = getattr(scheduler, "config", None)
    groups = tuple(getattr(config, "kv_group_configs", ()) or ())
    if manager is None or len(groups) != 1:
        return VllmOffloadObservation()
    tokens_per_chunk = _non_negative_int(getattr(groups[0], "tokens_per_chunk", None), positive=True)
    num_chunks = _non_negative_int(getattr(manager, "_num_chunks", None))
    allocated = _non_negative_int(getattr(manager, "_num_allocated_chunks", None))
    free_list = getattr(manager, "_free_list", None)
    if tokens_per_chunk is None or num_chunks is None or allocated is None or not isinstance(free_list, list):
        return VllmOffloadObservation()
    used_chunks = max(0, allocated - len(free_list))
    h2d_keys: set[object] = set()
    d2h_keys: set[object] = set()
    jobs = getattr(scheduler, "_jobs", {})
    if isinstance(jobs, dict):
        for job in jobs.values():
            keys = set(getattr(job, "keys", ()) or ())
            (d2h_keys if bool(getattr(job, "is_store", False)) else h2d_keys).update(keys)
    return VllmOffloadObservation(
        total_dram_kv_tokens=num_chunks * tokens_per_chunk,
        used_dram_kv_tokens=used_chunks * tokens_per_chunk,
        pending_h2d_kv_tokens=len(h2d_keys) * tokens_per_chunk,
        pending_d2h_kv_tokens=len(d2h_keys) * tokens_per_chunk,
        dram_chunk_size_tokens=tokens_per_chunk,
    )


def install_vllm_mori(owner: object, strategy: MORIStrategy) -> VllmMORIAdapter:
    """Bind MORI and validate every native capability required by its mode."""
    require_mori_vllm_version()
    scheduler = _offload_scheduler(owner)
    cpu_policy: object | None
    cpu_unregister: Callable[[], None] | None = None
    simple_register = getattr(scheduler, "register_mori_offload_tier_provider", None)
    simple_unregister = getattr(scheduler, "unregister_mori_offload_tier_provider", None)
    if callable(simple_register):
        if not callable(simple_unregister):
            raise RuntimeError(
                "full MORI with SimpleCPUOffloadConnector requires unregister_mori_offload_tier_provider(version=3)"
            )
        cpu_policy = MORISimpleCPUOffloadPolicy(MORI_RUNTIME_STATE)
        try:
            MORI_RUNTIME_STATE.bind_provider(strategy)
            capabilities = simple_register(cpu_policy.HOOK_API_VERSION, cpu_policy)
        except BaseException:
            with suppress(Exception):
                simple_unregister(cpu_policy)
            try:
                cpu_policy.close()
            finally:
                MORI_RUNTIME_STATE.unbind_provider(strategy)
            raise
        version = _capability_version(capabilities)
        if version != cpu_policy.HOOK_API_VERSION:
            with suppress(Exception):
                simple_unregister(cpu_policy)
            try:
                cpu_policy.close()
            finally:
                MORI_RUNTIME_STATE.unbind_provider(strategy)
            raise RuntimeError(
                f"unsupported vLLM MORI SimpleCPU hook API version: {version!r}; expected {cpu_policy.HOOK_API_VERSION}"
            )

        def unregister_simple_cpu() -> None:
            simple_unregister(cpu_policy)

        cpu_unregister = unregister_simple_cpu
    else:
        manager = getattr(scheduler, "manager", None) if scheduler is not None else None
        cpu_policy = getattr(manager, "_policy", None)
        if not bool(getattr(cpu_policy, "MORI_CPU_POLICY", False)):
            raise RuntimeError(
                "full MORI requires SimpleCPUOffloadConnector with the MORI "
                "offload-tier hook, or OffloadingConnector configured with "
                "eviction_policy=MoriCPUCachePolicy and "
                "cache_policy_module_path=agentinfer.agentcache.mori_cpu_policy"
            )
        MORI_RUNTIME_STATE.bind_provider(strategy)

    kv_cache_manager = getattr(owner, "kv_cache_manager", None)
    register = getattr(kv_cache_manager, "register_mori_block_selection_provider", None)
    if not callable(register):
        _close_failed_cpu_policy(cpu_policy, strategy, cpu_unregister)
        raise RuntimeError(
            "full MORI requires the vLLM MORI HBM hook API register_mori_block_selection_provider(version=1)"
        )
    unregister = getattr(kv_cache_manager, "unregister_mori_block_selection_provider", None)
    if not callable(unregister):
        _close_failed_cpu_policy(cpu_policy, strategy, cpu_unregister)
        raise RuntimeError(
            "full MORI requires the vLLM MORI HBM hook API unregister_mori_block_selection_provider(version=1)"
        )
    hbm_policy = MORIHBMPolicy(MORI_RUNTIME_STATE)
    try:
        capabilities = register(hbm_policy)
    except BaseException:
        with suppress(Exception):
            unregister(hbm_policy)
        hbm_policy.close()
        MORI_RUNTIME_STATE.unbind_provider(strategy)
        _close_failed_cpu_policy(cpu_policy, strategy, cpu_unregister)
        raise
    version = _capability_version(capabilities)
    if version != MORIHBMPolicy.HOOK_API_VERSION:
        with suppress(Exception):
            unregister(hbm_policy)
        hbm_policy.close()
        MORI_RUNTIME_STATE.unbind_provider(strategy)
        _close_failed_cpu_policy(cpu_policy, strategy, cpu_unregister)
        raise RuntimeError(
            f"unsupported vLLM MORI HBM hook API version: {version!r}; expected {MORIHBMPolicy.HOOK_API_VERSION}"
        )
    adapter = VllmMORIAdapter(
        hbm_policy=hbm_policy,
        cpu_policy=cpu_policy,
        owner=owner,
        strategy=strategy,
        cpu_unregister=cpu_unregister,
    )
    if strategy.config.audit_writeback_sources or strategy.config.repair_writeback_endpoints:
        try:
            if strategy.config.repair_writeback_endpoints:
                from agentinfer.agentcache.mori.writeback_capture import FullPrefixWritebackCapture

                adapter.writeback_capture = FullPrefixWritebackCapture(scheduler, strategy)
            if strategy.config.audit_writeback_sources:
                adapter.writeback_audit = WritebackSourceAudit(
                    scheduler, strategy, track_completions=strategy.config.repair_writeback_endpoints
                )
        except BaseException:
            adapter.close()
            raise
    return adapter


def _offload_scheduler(owner: object) -> object | None:
    connector = getattr(owner, "connector", None)
    if connector is None:
        return None
    direct = getattr(connector, "connector_scheduler", None)
    if direct is not None:
        return direct
    simple_cpu = getattr(connector, "scheduler_manager", None)
    if simple_cpu is not None:
        return simple_cpu
    try:
        return getattr(connector, "scheduler", None)
    except (AssertionError, AttributeError):
        return None


def _close_failed_cpu_policy(
    cpu_policy: object,
    strategy: MORIStrategy,
    unregister: Callable[[], None] | None = None,
) -> None:
    """Rollback a policy that registered process-local indexes during startup."""
    if unregister is not None:
        with suppress(Exception):
            unregister()
    try:
        close = getattr(cpu_policy, "close", None)
        if callable(close):
            close()
    finally:
        MORI_RUNTIME_STATE.unbind_provider(strategy)
        MORI_RUNTIME_STATE.clear_requests()


def _capability_version(capabilities: object) -> object:
    return (
        capabilities.get("hook_api_version")
        if isinstance(capabilities, dict)
        else getattr(capabilities, "hook_api_version", None)
    )


def _simple_cpu_observation(snapshot: object) -> VllmOffloadObservation:
    """Normalize a native SimpleCPU block snapshot into scheduler tokens."""
    block_size = _non_negative_int(
        getattr(snapshot, "scheduler_block_size", None),
        positive=True,
    )
    total = _non_negative_int(getattr(snapshot, "total_blocks", None))
    resident = _non_negative_int(getattr(snapshot, "resident_blocks", None))
    pending_store = _non_negative_int(getattr(snapshot, "pending_store_blocks", None))
    pending_load = _non_negative_int(getattr(snapshot, "pending_load_blocks", None))
    if None in (block_size, total, resident, pending_store, pending_load):
        return VllmOffloadObservation()
    assert block_size is not None
    assert total is not None
    assert resident is not None
    assert pending_store is not None
    assert pending_load is not None
    if resident > total or pending_store > total or pending_load > total:
        return VllmOffloadObservation()
    return VllmOffloadObservation(
        total_dram_kv_tokens=total * block_size,
        used_dram_kv_tokens=resident * block_size,
        pending_h2d_kv_tokens=pending_load * block_size,
        pending_d2h_kv_tokens=pending_store * block_size,
        dram_chunk_size_tokens=block_size,
    )


def _non_negative_int(value: Any, *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < (1 if positive else 0):
        return None
    return value
