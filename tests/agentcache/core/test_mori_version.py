# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""MORI's version gate must not restrict Progress-TTL on older vLLM."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import vllm

from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks
from agentinfer.agentcache.mori.vllm_adapter import install_vllm_mori, require_mori_vllm_version
from agentinfer.scheduling.mori import MORIConfig, MORIMode, build_mori_strategy
from agentinfer.scheduling.progress_ttl import ProgressTTLStrategy

pytestmark = pytest.mark.cpu_test


def test_scheduler_import_does_not_require_writeback_only_vllm_api():
    """PT imports even when vLLM lacks the optional MORI writeback API."""
    script = """
import sys
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
if hasattr(kv_cache_utils, 'resolve_block_hashes'):
    del kv_cache_utils.resolve_block_hashes
import agentinfer.agentcache.core.scheduler
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
        _VllmAdmissionHooks(SimpleNamespace(), config, SimpleNamespace(), 16)


def test_direct_adapter_install_checks_version_before_binding(monkeypatch):
    monkeypatch.setattr(vllm, "__version__", "0.23.0")
    strategy = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    with pytest.raises(RuntimeError, match=r"MORI requires vLLM >= 0\.31"):
        install_vllm_mori(SimpleNamespace(), strategy)


def test_progress_ttl_starts_on_older_vllm(monkeypatch):
    monkeypatch.setattr(vllm, "__version__", "0.23.0")
    owner = SimpleNamespace(kv_cache_manager=None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "progress_ttl"}},
    )
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=10), 16)
    try:
        assert isinstance(hooks.controller.strategy, ProgressTTLStrategy)
    finally:
        hooks.close()
