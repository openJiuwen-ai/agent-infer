# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise the MORI serving ABI checks against the installed wheel."""

from types import SimpleNamespace

import pytest
import torch
import vllm
from vllm.v1.core.sched import output, scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from agentinfer.agentcache.mori.vllm_compat import (
    MoriVllmABIError,
    mori_gpu_constructors,
    require_mori_engine_abi,
)
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


def test_mori_engine_uses_the_installed_abi_independently_of_version_metadata(monkeypatch):
    require_mori_engine_abi()
    monkeypatch.setattr(vllm, "__version__", "0.32.0+local")
    require_mori_engine_abi()


def test_mori_rejects_a_missing_native_output_contract(monkeypatch):
    native = scheduler.KVCacheManager
    monkeypatch.delattr(output, "KVConnectorBlockState")
    with pytest.raises(MoriVllmABIError, match="MORI vLLM ABI: .*KVConnectorBlockState"):
        require_mori_engine_abi()
    assert scheduler.KVCacheManager is native


def test_mori_does_not_patch_tables_for_a_windowed_full_attention_group():
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=16),
        speculative_config=None,
    )
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16, sliding_window=128)
    cache = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer"], spec)])
    native = scheduler.KVCacheManager
    with pytest.raises(ValueError, match="one dense full-attention group"), mori_gpu_constructors(vllm_config, cache):
        pytest.fail("An incompatible layout must be rejected before replacing the constructor")
    assert scheduler.KVCacheManager is native


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "single_rank",
        "deferred_frees",
        "dense_tree",
        "controller_binding",
        "issued_loads",
        "event_abi",
        "soft_error",
    ],
)
def test_retired_eviction_copy_requires_native_local_fences(monkeypatch, missing):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
    from agentinfer.agentcache.mori.vllm_compat import install_mori_inline_eviction_writeback

    monkeypatch.setattr(EngineCore, "_initialize_effective_attention_block_size", lambda _: None)
    install_mori_inline_eviction_writeback()
    connector = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    worker = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    native_worker = worker.worker_handler = SimpleCPUOffloadWorker.__new__(SimpleCPUOffloadWorker)
    native_worker._backend = DmaCopyBackend.__new__(DmaCopyBackend)
    native_worker._load_events = [] if missing != "event_abi" else None
    native_worker._load_hwm = -1
    if missing == "soft_error":
        worker.get_block_ids_with_load_errors = lambda: {1}
    device = SimpleNamespace(pool=SimpleNamespace())
    device.pool.hiradix_index = None if missing == "dense_tree" else device
    controller = SimpleNamespace(
        inline_eviction_writeback=None, inline_retired_writeback=None, local_load_completed=None
    )
    tree = SimpleNamespace(device=device, cache_controller=controller, share_issued_loads=missing != "issued_loads")
    connector.scheduler_manager = None if missing == "controller_binding" else controller

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.kv_connector = SimpleNamespace(kv_connector=worker)
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    owner = scheduler.Scheduler.__new__(scheduler.Scheduler)
    owner.connector = connector
    owner.kv_cache_manager = SimpleNamespace(hiradix=tree)
    owner.use_v2_model_runner = True
    owner.defer_block_free = missing != "deferred_frees"
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        parallel_config=SimpleNamespace(world_size=2 if missing == "single_rank" else 1),
        max_concurrent_batches=2,
    )

    EngineCore._initialize_effective_attention_block_size(
        SimpleNamespace(model_executor=executor, scheduler=owner, vllm_config=config)
    )

    expected_inline = None if missing == "controller_binding" else worker.writeback_inline
    exact_local = missing not in ("single_rank", "deferred_frees", "dense_tree", "controller_binding")
    assert controller.inline_eviction_writeback == expected_inline
    assert controller.inline_retired_writeback == (worker.writeback_retired if exact_local else None)
    assert worker.early_layerwise_loads is exact_local
    assert controller.local_load_completed == (worker.local_load_completed if missing is None else None)
