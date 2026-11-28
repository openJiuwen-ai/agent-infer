# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise MORI scheduler components with real vLLM 0.31 cache objects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


@pytest.fixture(scope="module")
def native_pool_class():
    from agentinfer.agentcache.mori.block_pool import MoriBlockPool

    return MoriBlockPool


@pytest.mark.parametrize("already_bound", [False, True])
def test_hiradix_owns_the_scheduler_connector_controller(native_pool_class, already_bound):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.core.sched import scheduler
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
    from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager
    from agentinfer.agentcache.mori.scheduler import MoriScheduler
    from agentinfer.agentcache.mori.vllm_compat import mori_gpu_constructors

    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    page = spec.page_size_bytes
    groups = [KVCacheGroupSpec(["layer"], spec)]
    tensors = [KVCacheTensor(page * 8, ["layer"], page * 8, page)]
    cache_config = KVCacheConfig(8, tensors, groups)
    other_config = KVCacheConfig(8, tensors, groups)
    config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=16),
        kv_events_config=None,
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=True,
            kv_connector_extra_config={"cpu_bytes_to_use": page * 12},
        ),
        max_in_flight_tokens=64,
        model_config=SimpleNamespace(max_model_len=64),
        parallel_config=SimpleNamespace(world_size=1, decode_context_parallel_size=1),
        speculative_config=None,
    )
    connector = MoriCPUOffloadConnector(config, KVConnectorRole.SCHEDULER, cache_config)
    worker = MoriCPUOffloadConnector(config, KVConnectorRole.WORKER, cache_config)
    assert connector.scheduler_manager is connector.worker_handler is worker.scheduler_manager is None
    assert isinstance(worker.worker_handler, SimpleCPUOffloadWorker)
    assert worker.worker_handler.cpu_kv_caches is None

    native = scheduler.KVCacheManager
    with mori_gpu_constructors(config, cache_config):
        params = {"max_model_len": 64, "scheduler_block_size": 16, "hash_block_size": 16}
        native_manager = scheduler.KVCacheManager(other_config, **params)
        manager = scheduler.KVCacheManager(cache_config, **params)
    assert type(native_manager) is scheduler.KVCacheManager is native
    assert type(manager) is MoriKVCacheManager
    connector.bind_kv_cache_manager(manager)
    owner = SimpleNamespace(
        vllm_config=config,
        kv_cache_config=cache_config,
        kv_cache_manager=manager,
        connector=connector,
        prefix_replay_tokens=0,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, enable_chunked_prefill=True),
        max_num_scheduled_tokens=64,
        need_mamba_block_aligned_split=False,
        waiting=FCFSRequestQueue(),
        hash_block_size=16,
        use_v2_model_runner=False,
        needs_kv_cache_zeroing=False,
    )
    engine = MoriScheduler.__new__(MoriScheduler)
    vars(engine).update(vars(owner))
    engine.block_size = 16
    native_capacity_check = manager.coordinator.get_num_blocks_to_allocate
    if already_bound:
        previous_controller = object()
        connector.scheduler_manager = previous_controller
        with pytest.raises(RuntimeError, match="exactly once"):
            engine._initialize_hiradix()
        assert connector.scheduler_manager is previous_controller
        assert manager.hiradix is manager.block_pool.hiradix_index is None
        assert manager.coordinator.get_num_blocks_to_allocate == native_capacity_check
        assert not manager.mori_batch_loads
        return

    engine._initialize_hiradix()
    try:
        tree = engine.tree_cache
        assert tree is manager.hiradix
        with pytest.raises(AttributeError):
            engine.tree_cache = tree
        controller = tree.cache_controller
        assert controller is connector.scheduler_manager
        gpu, cpu = tree.device.pool, tree.host.pool
        assert gpu is manager.block_pool is controller._gpu_block_pool
        assert cpu is controller.cpu_coordinator.block_pool is controller.cpu_block_pool
        assert type(cpu) is type(gpu) is native_pool_class
        assert gpu is not cpu
        assert (gpu.get_num_free_blocks(), cpu.get_num_free_blocks()) == (7, 11)
        with pytest.raises(RuntimeError, match="exactly once"):
            connector.bind_controller(controller)
    finally:
        manager.close_hiradix(connector)
    assert manager.hiradix is connector.scheduler_manager is None
    assert gpu.hiradix_index is cpu.hiradix_index is None
    assert manager.coordinator.get_num_blocks_to_allocate == native_capacity_check


def test_connector_only_unbinds_its_own_tree_controller():
    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector

    connector = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    controller = object()
    connector.scheduler_manager = controller
    connector.unbind_controller(object())
    assert connector.scheduler_manager is controller
    connector.unbind_controller(controller)
    connector.unbind_controller(controller)
    assert connector.scheduler_manager is None


@pytest.mark.parametrize(
    "case", ["private", "boundary", "shared", "cached", "copy_on_write", "load", "writeback", "reserved", "lookahead"]
)
def test_running_decode_shortcut_only_uses_an_owned_slot(monkeypatch, case):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.request import RequestStatus

    from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager

    rid = "decode"
    request = SimpleNamespace(
        request_id=rid,
        status=RequestStatus.RUNNING,
        num_output_tokens=2,
        num_prompt_tokens=17,
        num_computed_tokens=32 if case == "boundary" else 18,
    )
    tail = SimpleNamespace(
        is_null=False, ref_cnt=2 if case == "shared" else 1, block_hash=b"cached" if case == "cached" else None
    )
    per_type = FullAttentionManager.__new__(FullAttentionManager)
    per_type.block_size = 16
    per_type.num_cached_block = {rid: 1}
    per_type.req_to_blocks = {rid: [None, tail]}
    per_type._partial_hit_reqs = {rid: object()} if case == "copy_on_write" else {}
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.coordinator = SimpleNamespace(single_type_managers=(per_type,))
    manager.max_model_len = 64
    manager._replayed_prefill_tokens = {}
    manager._unpublished_batch_loads = {rid} if case == "load" else set()
    manager._eviction_stores = {rid: (3,)} if case == "writeback" else {}
    manager._native_capacity_check = None
    manager.hiradix = object()
    manager.enable_caching = False
    manager.mori_batch_loads = True
    manager.empty_kv_cache_blocks = object()
    native_result = object()
    monkeypatch.setattr(KVCacheManager, "allocate_slots", lambda *args, **kw: native_result)
    options = {"num_lookahead_tokens": int(case == "lookahead")}
    if case == "reserved":
        options["reserved_blocks"] = 1
    result = manager.allocate_slots(request, 1, **options)
    assert result is (manager.empty_kv_cache_blocks if case == "private" else native_result)


def test_waiting_allocation_counts_protected_prefix_hits_and_waits_once():
    from unittest.mock import Mock

    from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager

    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 10)
    manager._eviction_stores = {}
    tree = manager.hiradix = SimpleNamespace(
        stage_evictions=Mock(side_effect=[(2, 3), ()]),
        eviction_blocks_pending=Mock(side_effect=[True, False]),
    )
    hits = ([SimpleNamespace(block_id=1, ref_cnt=0, is_null=False)],)

    assert not manager._preserve_evictions("request", 10, hits, reserved=1)
    assert manager._preserve_evictions("request", 4, hits, reserved=1)
    assert manager._preserve_evictions("request", 4, hits, reserved=1)
    assert not manager._preserve_evictions("request", 4, hits, reserved=1)
    assert tree.stage_evictions.call_args_list == [((3, frozenset({1})), {})] * 2
    assert tree.eviction_blocks_pending.call_args_list == [(((2, 3),), {})] * 2


def test_bridge_requires_the_program_strategy_before_native_initialization():
    from agentinfer.agentcache.mori.bridge import MoriSyncSchedulerBridge

    with pytest.raises(ValueError, match="requires the MORI Program strategy"):
        MoriSyncSchedulerBridge(SimpleNamespace(additional_config={}), None)


@pytest.mark.parametrize("flag", [None, "--async-scheduling", "--no-async-scheduling"])
def test_mori_serve_preserves_native_async_scheduler_dispatch(flag):
    from vllm.config import DeviceConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    from agentinfer.agentcache.entrypoints.cli import serve_profile
    from agentinfer.agentcache.mori.bridge import MoriAsyncSchedulerBridge, MoriSyncSchedulerBridge
    from agentinfer.agentcache.mori.scheduler import MoriAsyncScheduler, MoriScheduler

    parser = serve_profile._import_make_arg_parser()(serve_profile._build_parser())
    serve_profile.add_agentinfer_arguments(parser)
    argv = [
        "MODEL",
        "--agentinfer",
        "--agentinfer-strategy",
        "mori",
        "--kv-transfer-config",
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both"}',
    ]
    if flag is not None:
        argv.append(flag)
    namespace, explicit = serve_profile.parse_args_with_explicit_keys(parser, argv)
    async_enabled = flag != "--no-async-scheduling"
    assert namespace.async_scheduling is (async_enabled if flag is not None else None)

    serve_profile.inject_profile(namespace, explicit)
    args = AsyncEngineArgs.from_cli_args(namespace)
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(distributed_executor_backend="uni"),
        scheduler_config=SchedulerConfig.default_factory(
            scheduler_cls=args.scheduler_cls,
            async_scheduling=args.async_scheduling,
        ),
    )
    assert config.scheduler_config.async_scheduling is async_enabled
    concrete = config.scheduler_config.get_scheduler_cls()
    assert concrete is (MoriAsyncSchedulerBridge if async_enabled else MoriSyncSchedulerBridge)
    assert issubclass(concrete, MoriAsyncScheduler if async_enabled else MoriScheduler)
    assert concrete.schedule is MoriScheduler.schedule
    assert concrete.update_from_output is MoriScheduler.update_from_output
    native = AsyncScheduler if async_enabled else Scheduler
    assert concrete._update_after_schedule is native._update_after_schedule
    assert concrete._update_request_with_output is native._update_request_with_output


@pytest.mark.parametrize("work", ["idle", "waiting", "previous_forward", "paused"])
def test_mori_restores_admission_ratio_only_at_native_idle(work):
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue

    from agentinfer.agentcache.mori.scheduler import MoriAsyncScheduler

    scheduler = MoriAsyncScheduler.__new__(MoriAsyncScheduler)
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.finished_req_ids = set()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.connector = scheduler.ec_connector = None
    scheduler.requests = {}
    scheduler._pause_state = PauseState.PAUSED_ALL if work == "paused" else PauseState.UNPAUSED
    scheduler.sched_step_seq = int(work == "previous_forward")
    scheduler.processed_step_seq = 0
    if work == "waiting":
        scheduler.waiting.add_request(SimpleNamespace(request_id="waiting"))
    scheduler.new_token_ratio = 0.1

    assert scheduler.has_requests() == (work == "waiting")
    assert scheduler.new_token_ratio == (0.7 if work == "idle" else 0.1)


@pytest.mark.parametrize("overlap", [False, True])
def test_mori_restores_admission_ratio_after_empty_forward_steps(overlap):
    from vllm.v1.core.sched.interface import PauseState

    from agentinfer.agentcache.mori.scheduler import MoriScheduler, PrefillBatch

    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.current_step = 0
    scheduler.kv_cache_manager = SimpleNamespace(new_step_starts=lambda: None, hiradix=SimpleNamespace(epoch=0))
    scheduler.running = []
    scheduler.waiting = []
    scheduler.kv_holding_waiting = []
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_decodes = set()
    scheduler._overlap = overlap
    scheduler.new_token_ratio = 0.1
    scheduled = iter([{"running": 1}, {}, {}])
    scheduler._schedule_batch = lambda *_: (
        SimpleNamespace(num_scheduled_tokens=next(scheduled), preempted_req_ids=set()),
        PrefillBatch(),
    )

    scheduler.schedule()
    after_forward = scheduler.new_token_ratio
    assert after_forward < 0.1
    scheduler.schedule()
    assert scheduler.new_token_ratio == (after_forward if overlap else 0.7)
    scheduler.schedule()
    assert scheduler.new_token_ratio == 0.7
