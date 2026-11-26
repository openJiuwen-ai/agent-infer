# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from types import SimpleNamespace

import pytest


@pytest.fixture
def step_without_worker(monkeypatch):
    """Run direct batch planning with a stand-in for the allocator and worker output."""
    from agentinfer.agentcache.mori.schedule_batch import ScheduleBatch

    def execute(scheduler, observe=None, *, allocate_running=None):
        scheduler.current_step = getattr(scheduler, "current_step", 0)
        scheduler.max_num_scheduled_tokens = getattr(scheduler, "max_num_scheduled_tokens", 128)
        scheduler.kv_cache_manager = getattr(scheduler, "kv_cache_manager", SimpleNamespace())
        scheduler.kv_cache_manager.new_step_starts = lambda: None
        scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=scheduler.max_num_scheduled_tokens)
        scheduler.prefill_capacity_bound = False
        scheduler.deferred_waiting = set()
        scheduler.get_request_counts = lambda: (
            len(scheduler.running),
            len(scheduler.waiting) + len(scheduler.kv_holding_waiting),
        )
        scheduler.max_model_len = getattr(scheduler, "max_model_len", 1024)
        scheduler.num_sampled_tokens_per_step = getattr(scheduler, "num_sampled_tokens_per_step", 1)
        monkeypatch.setattr(ScheduleBatch, "running", allocate_running or (lambda self, requests, limit, **_: limit))
        monkeypatch.setattr(ScheduleBatch, "waiting", lambda self, queue, requests, limit, token_caps: limit)
        monkeypatch.setattr(
            ScheduleBatch,
            "finish",
            lambda self: observe()
            if observe is not None
            else SimpleNamespace(preempted_req_ids=set(), num_scheduled_tokens=self.tokens),
        )
        return scheduler.schedule()

    return execute


@pytest.fixture
def cache():
    from agentinfer.agentcache.mori.block_pool import MoriBlockPool
    from agentinfer.agentcache.mori.hiradix import HiRadix

    gpu = MoriBlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=16)
    host = MoriBlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=16)
    tree = HiRadix(gpu, host, 16)
    gpu.bind_hiradix_index(tree.device)
    host.bind_hiradix_index(tree.host)
    return tree


@pytest.fixture
def native_load(cache):
    from tests.agentcache.mori.helpers import _test_controller, backup, insert

    device_blocks = insert(cache, (1, 2))
    host_blocks = backup(cache, device_blocks)
    cache.device.pool.evict_blocks({block.block_id for block in device_blocks})
    return _test_controller(cache), host_blocks
