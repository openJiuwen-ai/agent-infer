# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""MORI's version gate must not restrict Progress-TTL on older vLLM."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import vllm

from agentinfer.agentcache.core.scheduler import (
    AgentCacheAsyncSchedulerBridge,
    AgentCacheSyncSchedulerBridge,
    _VllmAdmissionHooks,
)
from agentinfer.agentcache.mori.scheduler import _VllmAdmissionHooks as _MORIAdmissionHooks
from agentinfer.agentcache.mori.vllm_adapter import install_vllm_mori, require_mori_vllm_version
from agentinfer.scheduling.mori import MORIConfig, MORIMode, build_mori_strategy
from agentinfer.scheduling.progress_ttl import ProgressTTLStrategy

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("bridge", "target"),
    [
        (AgentCacheAsyncSchedulerBridge, "AgentCacheAsyncSchedulerBridge"),
        (AgentCacheSyncSchedulerBridge, "AgentCacheSyncSchedulerBridge"),
    ],
)
def test_public_bridge_selects_mori_only_when_requested(bridge, target):
    config = SimpleNamespace(additional_config={"agentcache": {"strategy": "mori"}})
    sentinel = object()
    with patch(f"agentinfer.agentcache.mori.scheduler.{target}", return_value=sentinel) as factory:
        assert bridge(config) is sentinel
    factory.assert_called_once_with(config)


def test_scheduler_import_does_not_require_writeback_only_vllm_api():
    """PT imports even when vLLM lacks the optional MORI writeback API."""
    script = """
import sys
from types import SimpleNamespace
from unittest.mock import patch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
if hasattr(kv_cache_utils, 'resolve_block_hashes'):
    del kv_cache_utils.resolve_block_hashes
from agentinfer.agentcache.core.scheduler import AgentCacheAsyncSchedulerBridge, AgentCacheSyncSchedulerBridge
def initialize(scheduler, *args, **kwargs):
    scheduler.waiting = []
    scheduler.running = []
    scheduler.skipped_waiting = []
    scheduler.requests = {}
    scheduler.kv_cache_manager = None
with patch.object(AsyncScheduler, '__init__', initialize), patch.object(Scheduler, '__init__', initialize), patch.object(Scheduler, 'get_request_counts', return_value=(0, 0)):
    for bridge_type, async_enabled in ((AgentCacheAsyncSchedulerBridge, True), (AgentCacheSyncSchedulerBridge, False)):
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=async_enabled),
            parallel_config=SimpleNamespace(data_parallel_index=0),
            additional_config={'agentcache': {'strategy': 'progress_ttl'}},
        )
        bridge = bridge_type(config, SimpleNamespace(num_blocks=10), object(), 16, 16)
        assert type(bridge) is bridge_type
assert 'agentinfer.scheduling.mori' not in sys.modules
assert 'agentinfer.agentcache.mori.scheduler' not in sys.modules
assert 'agentinfer.agentcache.mori.vllm_adapter' not in sys.modules
assert 'agentinfer.agentcache.mori.writeback_capture' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("version", ["0.31.0", "0.31.0+cu129", "0.32.0", "1.0.0"])
def test_supported_mori_versions(monkeypatch, version):
    monkeypatch.setattr(vllm, "__version__", version)
    require_mori_vllm_version()


@pytest.mark.parametrize("version", ["0.23.0", "0.30.9", "0.31.0rc1", "unknown"])
def test_mori_rejects_unsupported_versions_before_controller_construction(monkeypatch, version):
    monkeypatch.setattr(vllm, "__version__", version)
    config = SimpleNamespace(additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "full"}}})
    with pytest.raises(RuntimeError, match=r"MORI requires vLLM >= 0\.31"):
        _MORIAdmissionHooks(SimpleNamespace(), config, SimpleNamespace(), 16)


def test_direct_adapter_install_checks_version_before_binding(monkeypatch):
    monkeypatch.setattr(vllm, "__version__", "0.23.0")
    strategy = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    with pytest.raises(RuntimeError, match=r"MORI requires vLLM >= 0\.31"):
        install_vllm_mori(SimpleNamespace(), strategy)


def test_progress_ttl_starts_on_older_vllm(monkeypatch):
    monkeypatch.setattr(vllm, "__version__", "0.23.0")
    owner = SimpleNamespace(kv_cache_manager=None, get_request_counts=lambda: (0, 0))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "progress_ttl"}},
    )
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=10), 16)
    assert isinstance(hooks.controller.strategy, ProgressTTLStrategy)
