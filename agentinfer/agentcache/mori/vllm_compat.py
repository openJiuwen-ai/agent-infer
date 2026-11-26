# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Bind MORI components against the supported vLLM ABI."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter, signature
from threading import RLock, get_ident

from vllm.logger import init_logger

_CONSTRUCTION_LOCK = RLock()
logger = init_logger("vllm.agentinfer.mori.compat")


class MoriVllmABIError(RuntimeError):
    """The installed wheel cannot provide a MORI runtime contract."""


def _require_members(native, *members: str) -> None:
    missing = [member for member in members if not hasattr(native, member)]
    if missing:
        owner = f"{native.__module__}.{native.__qualname__}"
        raise MoriVllmABIError(f"MORI vLLM ABI: {owner} lacks {', '.join(missing)}")


def _require_keywords(native, *keywords: str) -> None:
    try:
        parameters = signature(native).parameters
    except (TypeError, ValueError) as exc:
        raise MoriVllmABIError(f"MORI vLLM ABI: cannot inspect {native.__qualname__}") from exc
    missing = [
        name
        for name in keywords
        if name not in parameters
        or parameters[name].kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    ]
    if missing:
        raise MoriVllmABIError(f"MORI vLLM ABI: {native.__qualname__} lacks keyword parameters {', '.join(missing)}")


def require_mori_engine_abi() -> None:
    """Check the installed engine contract before loading the MORI scheduler."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
        from vllm.v1.core import kv_cache_coordinator
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched import scheduler
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
        from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata, SimpleCPUOffloadWorkerMetadata
        from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
    except ImportError as exc:
        raise MoriVllmABIError(f"MORI vLLM ABI: {exc}") from exc

    _require_members(
        scheduler.Scheduler,
        "schedule",
        "_get_local_prefix_cache_hit",
        "_make_cached_request_data",
        "_build_kv_connector_meta",
        "_get_new_block_ids_to_zero",
        "_enqueue_waiting_request",
        "_handle_blocked_waiting_request",
        "_inflight_prefill_reserved_blocks",
        "_request_blocks_can_be_freed",
        "_free_cow_retained_blocks",
    )
    _require_members(AsyncScheduler, "_update_after_schedule", "_update_request_with_output")
    _require_members(
        KVCacheManager, "take_boundary_state_offloads", "take_kv_cache_block_copies", "record_prefix_cache_stats"
    )
    _require_members(BlockPool, "_insert_block_hash", "_remove_cached_block_hashes", "reset_prefix_cache")
    _require_members(
        SimpleCPUOffloadScheduler, "_process_store_completion", "bind_gpu_block_pool", "update_connector_output"
    )
    _require_members(
        SimpleCPUOffloadWorker, "register_kv_caches", "start_load_kv", "get_finished", "clear_connector_metadata"
    )
    _require_members(KVConnectorBase_V1, "bind_kv_cache_manager")
    _require_members(EngineCore, "preprocess_add_request", "_initialize_effective_attention_block_size")
    _require_keywords(scheduler.Scheduler.schedule, "throttle_prefills")
    _require_keywords(AsyncScheduler._update_request_with_output, "is_stale")
    _require_keywords(KVConnectorBase_V1, "vllm_config", "role", "kv_cache_config")
    _require_keywords(KVCacheManager.allocate_slots, "full_sequence_must_fit", "reserved_blocks", "has_scheduled_reqs")
    _require_keywords(SimpleCPUOffloadScheduler, "scheduler_block_size", "hash_block_size")
    _require_keywords(
        SchedulerOutput,
        "has_sync_kv_loads",
        "kv_connector_block_state",
        "new_block_ids_to_zero",
        "kv_cache_block_copies",
        "block_table_updates",
        "ec_manager_metadata",
    )
    _require_keywords(KVConnectorBlockState, "req_ids", "resolve_block_ids", "boundary_state_offloads")
    _require_keywords(
        SimpleCPUOffloadMetadata, "load_event", "load_gpu_blocks", "load_cpu_blocks", "load_event_to_reqs"
    )
    _require_keywords(SimpleCPUOffloadWorkerMetadata, "completed_store_events")
    if (
        getattr(scheduler, "KVCacheManager", None) is not KVCacheManager
        or getattr(kv_cache_coordinator, "BlockPool", None) is not BlockPool
    ):
        raise MoriVllmABIError("MORI vLLM ABI: the scheduler or coordinator constructor binding has already changed")


def _require_dense_kv_layout(vllm_config, kv_cache_config) -> None:
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    groups = kv_cache_config.kv_cache_groups
    spec = groups[0].kv_cache_spec if len(groups) == 1 else None
    if (
        spec is None
        or type(spec) is not FullAttentionSpec
        or spec.sliding_window is not None
        or spec.attention_chunk_size is not None
        or spec.block_size != vllm_config.cache_config.block_size
        or not vllm_config.cache_config.enable_prefix_caching
        or vllm_config.speculative_config is not None
    ):
        raise ValueError(
            "MORI requires one dense full-attention group using native block-size prefix caching without speculation"
        )


def install_mori_program_block_hasher() -> None:
    """Install the vLLM request-hash adapter inside EngineCore's process."""
    from vllm.v1.engine.core import EngineCore

    from agentinfer.agentcache.mori.prefix_hash import MORIProgramBlockHasher

    _require_members(EngineCore, "preprocess_add_request")
    _require_keywords(EngineCore.preprocess_add_request, "request")
    native = EngineCore.preprocess_add_request
    if getattr(native, "_agentinfer_mori_program_hashes", False):
        return

    @wraps(native)
    def preprocess(core, request):
        hasher = core.request_block_hasher
        if hasher is not None and not isinstance(hasher, MORIProgramBlockHasher):
            capacity = getattr(core.scheduler, "prefix_hash_capacity", None)
            if capacity:
                core.request_block_hasher = MORIProgramBlockHasher(hasher, core.scheduler.hash_block_size, capacity)
        return native(core, request)

    vars(preprocess)["_agentinfer_mori_program_hashes"] = True
    EngineCore.preprocess_add_request = preprocess


def install_mori_inline_eviction_writeback() -> None:
    """Bind the native GPU worker when it lives in the scheduler process."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector

    _require_members(EngineCore, "_initialize_effective_attention_block_size")
    native = EngineCore._initialize_effective_attention_block_size
    if getattr(native, "_agentinfer_mori_inline_writeback", False):
        return

    @wraps(native)
    def initialize(core):
        native(core)
        executor = core.model_executor
        scheduler = core.scheduler
        connector = getattr(scheduler, "connector", None)
        if type(executor) is not UniProcExecutor or not isinstance(connector, MoriCPUOffloadConnector):
            return
        if not scheduler.use_v2_model_runner or not core.vllm_config.kv_transfer_config.is_kv_consumer:
            return
        tree = getattr(scheduler.kv_cache_manager, "hiradix", None)
        if tree is None:
            return
        controller = tree.cache_controller
        if controller is None or connector.scheduler_manager is not controller:
            return
        runner = executor.driver_worker.worker.model_runner
        worker_connector = getattr(runner.kv_connector, "kv_connector", None)
        if isinstance(worker_connector, MoriCPUOffloadConnector):
            worker = worker_connector.worker_handler
            if worker is not None and isinstance(worker._backend, DmaCopyBackend):
                controller.inline_eviction_writeback = worker_connector.writeback_inline
                logger.info_once("MORI HiRadix uses local synchronous eviction writeback")
                from vllm.v1.worker.gpu.model_runner import GPUModelRunner

                if (
                    type(runner) is GPUModelRunner
                    and core.vllm_config.parallel_config.world_size == 1
                    and core.vllm_config.max_concurrent_batches == 2
                    and scheduler.defer_block_free
                    and hasattr(scheduler, "_drain_deferred_frees")
                    and tree.device.pool.hiradix_index is tree.device
                ):
                    # Freed dense pages have passed the native step fence.
                    # The controller checks readiness again before pinning each victim.
                    controller.inline_retired_writeback = worker_connector.writeback_retired
                    worker_connector.early_layerwise_loads = True
                    if (
                        tree.share_issued_loads
                        and type(connector) is MoriCPUOffloadConnector
                        and type(worker_connector) is MoriCPUOffloadConnector
                        and type(worker) is SimpleCPUOffloadWorker
                        and type(worker._backend) is DmaCopyBackend
                        and isinstance(getattr(worker, "_load_hwm", None), int)
                        and isinstance(getattr(worker, "_load_events", None), list)
                        and getattr(worker_connector.get_transfer_results, "__func__", None)
                        is KVConnectorBase_V1.get_transfer_results
                        and getattr(worker_connector.get_block_ids_with_load_errors, "__func__", None)
                        is KVConnectorBase_V1.get_block_ids_with_load_errors
                    ):
                        # The native local DMA has no deferred soft-failure report.
                        controller.local_load_completed = worker_connector.local_load_completed

    vars(initialize)["_agentinfer_mori_inline_writeback"] = True
    EngineCore._initialize_effective_attention_block_size = initialize


def install_mori_layerwise_graph_dispatch() -> None:
    """Keep uncaptured attention hooks available for sync H2D consumers."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector

    _require_members(CUDAGraphMode, "FULL", "NONE", "PIECEWISE")
    _require_members(CudaGraphManager, "dispatch")
    _require_members(GPUModelRunner, "execute_model")
    _require_keywords(CudaGraphManager.dispatch, "num_reqs", "num_tokens", "uniform_token_count", "num_active_loras")
    _require_keywords(BatchExecutionDescriptor, "cg_mode", "num_tokens", "num_reqs", "num_active_loras")
    native = GPUModelRunner.execute_model
    if getattr(native, "_agentinfer_mori_layerwise_dispatch", False):
        return

    @wraps(native)
    def execute(runner, scheduler_output, *args, **kwargs):
        manager = runner.cudagraph_manager
        connector = getattr(runner.kv_connector, "kv_connector", None)
        if (
            manager is None
            or runner.dp_size != 1
            or not isinstance(connector, MoriCPUOffloadConnector)
            or not scheduler_output.total_num_scheduled_tokens
            or not connector.needs_layerwise_forward(scheduler_output)
        ):
            return native(runner, scheduler_output, *args, **kwargs)
        dispatch = manager.dispatch

        def layerwise(num_reqs, num_tokens, uniform_token_count, num_active_loras, *args, **kwargs):
            desc = dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, *args, **kwargs)
            if desc.cg_mode is not CUDAGraphMode.FULL:
                return desc
            logger.info_once("MORI skips FULL CUDA Graph replay for a layerwise H2D consumer")
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=desc.num_active_loras,
            )

        instance_dispatch = vars(manager).get("dispatch")
        manager.dispatch = layerwise
        try:
            dummy_run = kwargs.get("dummy_run", len(args) > 1 and args[1])
            if dummy_run or getattr(runner.kv_connector, "_disabled", True):
                return native(runner, scheduler_output, *args, **kwargs)
            with connector.prepare_layer_load(scheduler_output):
                return native(runner, scheduler_output, *args, **kwargs)
        finally:
            if instance_dispatch is None:
                del manager.dispatch
            else:
                manager.dispatch = instance_dispatch

    vars(execute)["_agentinfer_mori_layerwise_dispatch"] = True
    GPUModelRunner.execute_model = execute


@contextmanager
def mori_text_detokenizer() -> Iterator[None]:
    """Prime vLLM's ByteLevel stream with MORI's five-token context."""
    from msgspec.structs import replace
    from tokenizers.decoders import ByteLevel
    from vllm.v1.engine import EngineCoreRequest, detokenizer

    try:
        from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer
    except ImportError as exc:
        raise MoriVllmABIError("MORI vLLM ABI: vLLM lacks FastIncrementalDetokenizer") from exc
    _require_keywords(FastIncrementalDetokenizer, "tokenizer", "request")
    _require_members(EngineCoreRequest, "prompt_token_ids", "prompt_embeds", "sampling_params")

    class MORIFastIncrementalDetokenizer(FastIncrementalDetokenizer):
        def __init__(self, tokenizer, request):
            ids = request.prompt_token_ids
            backend = tokenizer._tokenizer
            if ids and len(ids) > 5 and request.prompt_embeds is None and isinstance(backend.decoder, ByteLevel):
                tail = ids[-5:]
                text = backend.decode(tail, skip_special_tokens=request.sampling_params.skip_special_tokens)
                # A clean byte boundary needs no earlier prompt bytes. Keep
                # the original decoder initialization for partial UTF-8 and
                # special-token-only suffixes.
                if text and "\ufffd" not in text:
                    request = replace(request, prompt_token_ids=tail)
            super().__init__(tokenizer, request)

    detokenizer.FastIncrementalDetokenizer = MORIFastIncrementalDetokenizer
    try:
        yield
    finally:
        detokenizer.FastIncrementalDetokenizer = FastIncrementalDetokenizer


@contextmanager
def _mori_constructors(bindings) -> Iterator[None]:
    # Native schedulers constructed concurrently in another thread still get
    # the original components, including when they never import AgentInfer.
    constructor_thread = get_ident()

    def scoped_constructor(native, mori):
        def create(*args, **kwargs):
            cls = mori if get_ident() == constructor_thread else native
            return cls(*args, **kwargs)

        return create

    with _CONSTRUCTION_LOCK:
        originals = [(module, name, getattr(module, name)) for module, name, _ in bindings]
        try:
            for (module, name, original), (_, _, replacement) in zip(originals, bindings, strict=True):
                setattr(module, name, scoped_constructor(original, replacement))
            yield
        finally:
            for module, name, original in reversed(originals):
                setattr(module, name, original)


@contextmanager
def mori_block_pool_constructor() -> Iterator[None]:
    """Use the MORI pool while a coordinator constructs its physical slots."""
    from vllm.v1.core import kv_cache_coordinator

    from agentinfer.agentcache.mori.block_pool import MoriBlockPool

    with _mori_constructors(((kv_cache_coordinator, "BlockPool", MoriBlockPool),)):
        yield


@contextmanager
def mori_gpu_constructors(vllm_config, kv_cache_config) -> Iterator[None]:
    """Adapt the native scheduler's request manager for this KV configuration."""
    from vllm.v1.core.sched import scheduler

    from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager

    with _CONSTRUCTION_LOCK:
        require_mori_engine_abi()
        _require_dense_kv_layout(vllm_config, kv_cache_config)
        native = scheduler.KVCacheManager

        def make_manager(*args, **kwargs):
            config = kwargs.get("kv_cache_config", args[0] if args else None)
            if config is kv_cache_config:
                with mori_block_pool_constructor():
                    return MoriKVCacheManager(*args, **kwargs)
            return native(*args, **kwargs)

        with _mori_constructors(((scheduler, "KVCacheManager", make_manager),)):
            yield
