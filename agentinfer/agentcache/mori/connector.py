# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""CPU offload with explicit dependencies for MORI admission and eviction."""

from __future__ import annotations

from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    DEFAULT_CPU_CAPACITY_BYTES,
    SimpleCPUOffloadConnector,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu.swap_blocks_triton import MIN_N, swap_blocks_batch
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import BatchMemcpyParams, copy_blocks
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

# vLLM installs the worker's INFO handler on its logger namespace.
logger = init_logger("vllm.agentinfer.mori.connector")

# Limit descriptor rebuilding to maps with enough potential savings.
_MIN_H2D_DESCRIPTORS_SAVED = 16


def _layer_slot_capacity(caches, bases, strides, device_type: str) -> int:
    """Return a common bound for independent dense regions in a native copy."""
    ranges = []
    capacities = []
    for tensor, base, stride in zip(caches.values(), bases, strides, strict=True):
        if (
            tensor.device.type != device_type
            or tensor.ndim != 2
            or not tensor.is_contiguous()
            or tensor.data_ptr() != base
            or tensor.size(1) * tensor.element_size() != stride
            or stride == 0
        ):
            return 0
        slots = tensor.size(0)
        ranges.append((int(base), int(base) + slots * int(stride)))
        capacities.append(slots)
    ranges.sort()
    if any(end > start for (_, end), (start, _) in zip(ranges, ranges[1:], strict=False)):
        return 0
    return min(capacities, default=0)


def _contiguous_load_runs(source: np.ndarray, destination: np.ndarray, slot_bounds: tuple[int, int]):
    """Find equally oriented consecutive pages in the existing transfer order."""
    if source.size <= _MIN_H2D_DESCRIPTORS_SAVED or source.size != destination.size:
        return None
    steps = np.diff(source.view(np.int64))
    adjacent = (steps == np.diff(destination.view(np.int64))) & ((steps == 1) | (steps == -1))
    if np.count_nonzero(adjacent) < max(_MIN_H2D_DESCRIPTORS_SAVED, (source.size + 3) // 4):
        return None
    if (
        source.min() == 0
        or destination.min() == 0
        or source.max() >= slot_bounds[0]
        or destination.max() >= slot_bounds[1]
        or np.unique(destination).size != destination.size
    ):
        return None
    boundaries = np.flatnonzero(~adjacent) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, source.size]
    # In a descending run the final slots give the lowest two byte addresses.
    return (
        np.minimum(source[starts], source[stops - 1]),
        np.minimum(destination[starts], destination[stops - 1]),
        (stops - starts).astype(np.uint64),
    )


class _PreparedLayerLoad(NamedTuple):
    metadata: SimpleCPUOffloadMetadata
    events: dict[str, torch.Event]
    done: torch.Event


class _DenseLayerLoad(NamedTuple):
    bounds: tuple[int, int]
    src_bases: np.ndarray
    dst_bases: np.ndarray
    strides: np.ndarray


class MoriCPUOffloadConnector(SimpleCPUOffloadConnector):
    """Connect the tree's scheduler-side controller to the native DMA worker."""

    mori_batch_loads = True
    _layerwise_loads = False
    _triton_writeback = False
    _fence_previous_forward = True
    _layer_load_params: dict[str, BatchMemcpyParams] | None = None
    _dense_layer_load: _DenseLayerLoad | None = None
    _layer_load_events: dict[str, torch.Event] | None = None
    early_layerwise_loads = False
    _prepared_layer_load: _PreparedLayerLoad | None = None

    def __init__(self, vllm_config, role, kv_cache_config):
        # Start at the connector protocol constructor. The SimpleCPU constructor
        # creates a scheduler manager, which belongs to HiRadix in this profile.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        self.scheduler_manager = None
        self.worker_handler = None
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("MORI CPU offloading requires prefix caching")
        transfer = self._kv_transfer_config
        options = transfer.kv_connector_extra_config or {}
        if options.get("kv_offload_backend", "cpu") != "cpu" or options.get("lazy_offload", False):
            raise ValueError("MORI CPU offloading requires non-lazy RAM transfers")
        capacity = int(options.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES))
        world_size = vllm_config.parallel_config.world_size
        self.cpu_capacity_per_rank = int(options.get("cpu_bytes_to_use_per_rank", capacity // world_size))
        if self.cpu_capacity_per_rank <= 0:
            raise ValueError("MORI CPU offloading requires a positive per-rank CPU capacity")

        self._layerwise_loads = options.get("mori_layerwise_loads", False)
        self._triton_writeback = options.get("mori_triton_writeback", False) and current_platform.is_cuda()
        # Native vLLM defers reuse of in-flight destinations for consumers.
        self._fence_previous_forward = not transfer.is_kv_consumer
        if role is KVConnectorRole.WORKER:
            self.worker_handler = SimpleCPUOffloadWorker(vllm_config, kv_cache_config, self.cpu_capacity_per_rank)

    def bind_controller(self, controller) -> None:
        if self.role is not KVConnectorRole.SCHEDULER or self.scheduler_manager is not None:
            raise RuntimeError("MORI controller must bind to the scheduler connector exactly once")
        if self._kv_cache_manager is None or controller._gpu_block_pool is not self._kv_cache_manager.block_pool:
            raise RuntimeError("MORI controller and scheduler connector require the same GPU block pool")
        self.scheduler_manager = controller

    def unbind_controller(self, controller) -> None:
        if self.scheduler_manager is controller:
            self.scheduler_manager = None

    def register_kv_caches(self, kv_caches) -> None:
        super().register_kv_caches(kv_caches)
        self._layer_load_params = None
        self._dense_layer_load = None
        worker = self.worker_handler
        if not self._layerwise_loads or worker is None:
            return
        backend = worker._backend
        # A cross-layer or aliased storage layout can only use the batch fence.
        if not isinstance(backend, DmaCopyBackend) or set(kv_caches) != set(worker.gpu_kv_caches):
            logger.info_once("MORI layerwise H2D unavailable for this KV storage layout")
            return
        params = backend._load_params
        assert params is not None
        by_name = {name: index for index, name in enumerate(worker.gpu_kv_caches)}
        order = [by_name[name] for name in kv_caches]
        self._layer_load_params = {}
        for name, index in zip(kv_caches, order, strict=True):
            span = slice(index, index + 1)
            self._layer_load_params[name] = params._replace(
                src_bases=params.src_bases[span],
                dst_bases=params.dst_bases[span],
                bpb=params.bpb[span],
                num_layers=1,
            )
        cpu_caches = worker.cpu_kv_caches
        if (
            current_platform.is_cuda()
            and type(worker) is SimpleCPUOffloadWorker
            and type(backend) is DmaCopyBackend
            and cpu_caches
            and list(cpu_caches) == list(worker.gpu_kv_caches)
            and all(
                isinstance(vector, np.ndarray) and vector.dtype == np.uint64 and vector.shape == (len(cpu_caches),)
                for vector in (params.src_bases, params.dst_bases, params.bpb)
            )
        ):
            cpu_slots = _layer_slot_capacity(cpu_caches, params.src_bases, params.bpb, "cpu")
            gpu_slots = _layer_slot_capacity(worker.gpu_kv_caches, params.dst_bases, params.bpb, "cuda")
            if cpu_slots and gpu_slots:
                self._dense_layer_load = _DenseLayerLoad(
                    (cpu_slots, gpu_slots), params.src_bases[order], params.dst_bases[order], params.bpb[order]
                )

    def has_pending_block_frees(self) -> bool:
        manager = self.scheduler_manager
        if manager is not None and (manager._eviction_store_gpu_ids or manager._prefetch_pins):
            return True
        return super().has_pending_block_frees()

    def has_pending_push_work(self) -> bool:
        # Standalone H2D restores are tracked by job ID and can outlive the
        # last request. Keep sending empty steps until the worker returns ACK.
        return bool(getattr(self.scheduler_manager, "_prefetch_pins", None)) or super().has_pending_push_work()

    def _submit_layer_load(self, metadata: SimpleCPUOffloadMetadata) -> _PreparedLayerLoad:
        worker = self.worker_handler
        assert worker is not None and self._layer_load_params
        source = np.asarray(metadata.load_cpu_blocks, dtype=np.uint64)
        destination = np.asarray(metadata.load_gpu_blocks, dtype=np.uint64)
        layout = self._dense_layer_load
        runs = _contiguous_load_runs(source, destination, layout.bounds) if layout is not None else None
        if runs is not None:
            assert layout is not None
            src_run, dst_run, lengths = runs
            strides = layout.strides[:, None]
            src_bases = layout.src_bases[:, None] + src_run * strides
            dst_bases = layout.dst_bases[:, None] + dst_run * strides
            sizes = lengths * strides
            logger.info_once("MORI H2D merges contiguous ranges in the native per-layer DMA")
        events = {}
        try:
            for index, (layer, params) in enumerate(self._layer_load_params.items()):
                if runs is None:
                    copy_blocks(source, destination, params)
                else:
                    # Treat each contiguous range as a single native byte-copy
                    # region. Keep this layer's stream and source access order.
                    copy_blocks(
                        [0],
                        [0],
                        params._replace(
                            src_bases=src_bases[index],
                            dst_bases=dst_bases[index],
                            bpb=sizes[index],
                            num_layers=len(lengths),
                        ),
                    )
                done = torch.Event()
                done.record(worker.load_stream)
                events[layer] = done
        except BaseException:
            # An error in a later layer may leave earlier copies in flight.
            worker.load_stream.synchronize()
            raise
        return _PreparedLayerLoad(metadata, events, done)

    @contextmanager
    def prepare_layer_load(self, scheduler_output):
        """Start copies before input preparation, then let pre_forward consume them."""
        metadata = scheduler_output.kv_connector_metadata
        if (
            not self.early_layerwise_loads
            or self._fence_previous_forward
            or not isinstance(metadata, SimpleCPUOffloadMetadata)
            or metadata.load_event < 0
            or not self.needs_layerwise_forward(scheduler_output)
            or metadata.need_flush
            or scheduler_output.new_block_ids_to_zero
            or scheduler_output.kv_cache_block_copies
        ):
            yield
            return

        assert self._prepared_layer_load is None
        prepared = self._submit_layer_load(metadata)
        self._prepared_layer_load = prepared
        logger.info_once("MORI H2D submits layer copies before V2 input preparation")
        try:
            yield
            if self._prepared_layer_load is prepared:
                raise RuntimeError("MORI V2 forward did not consume its prepared H2D load")
        finally:
            if self._prepared_layer_load is prepared:
                self._prepared_layer_load = None
                prepared.done.synchronize()

    def start_load_kv(self, forward_context, **kwargs) -> None:
        worker = self.worker_handler
        if worker is None or worker._connector_metadata is None:
            return
        meta = worker._connector_metadata
        if not meta.load_cpu_blocks:
            return

        backend = worker._backend
        if not isinstance(backend, DmaCopyBackend) or backend._load_params is None:
            raise RuntimeError("MORI batch loads require the vLLM CPU DMA backend")

        # The normal backend submits after forward through a background
        # thread. An event must be recorded before this step's consumer can
        # wait on it, including when forward replays a CUDA graph.
        compute = torch.cuda.current_stream()
        if self._fence_previous_forward:
            worker.load_stream.wait_stream(compute)
        else:
            logger.info_once("MORI H2D overlaps previous forwards using native destination fences")
        # vLLM runner V2 passes attn_metadata separately for FULL replay and
        # supplies a placeholder context, whose mode alone looks eager.
        layerwise = (
            self._layer_load_params
            and "attn_metadata" not in kwargs
            and forward_context.attn_metadata is not None
            and forward_context.cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
        )
        prepared = self._prepared_layer_load
        if prepared is not None and prepared.metadata is not meta:
            raise RuntimeError("MORI V2 forward bound a different H2D job after preparation")
        if layerwise:
            logger.info_once("MORI H2D uses per-layer CUDA stream dependencies")
            transfer = prepared or self._submit_layer_load(meta)
            self._layer_load_events = transfer.events
            done = transfer.done
        else:
            if self._layerwise_loads:
                logger.info_once(
                    "MORI H2D uses a batch fence (mapped layers=%d, forward mode=%s, full replay=%s)",
                    len(self._layer_load_params or ()),
                    forward_context.cudagraph_runtime_mode,
                    "attn_metadata" in kwargs,
                )
            if prepared is None:
                copy_blocks(meta.load_cpu_blocks, meta.load_gpu_blocks, backend._load_params)
                done = torch.Event()
                done.record(worker.load_stream)
            else:
                done = prepared.done
            compute.wait_event(done)
        worker._load_events.append((meta.load_event, done))
        self._prepared_layer_load = None

    def local_load_completed(self, event_id: int) -> bool:
        """Check the copy's completion without consuming the native ACK."""
        worker = self.worker_handler
        assert worker is not None
        if event_id <= worker._load_hwm:
            return True
        return any(index == event_id and event.query() for index, event in worker._load_events)

    def needs_layerwise_forward(self, scheduler_output) -> bool:
        metadata = scheduler_output.kv_connector_metadata
        return bool(
            self._layer_load_params
            and scheduler_output.has_sync_kv_loads
            and metadata is not None
            and metadata.load_cpu_blocks
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._layer_load_events is not None and (ready := self._layer_load_events.pop(layer_name, None)):
            torch.cuda.current_stream().wait_event(ready)

    def clear_connector_metadata(self) -> None:
        if self._layer_load_events:
            raise RuntimeError(f"MORI layerwise H2D did not see attention for {sorted(self._layer_load_events)}")
        self._layer_load_events = None
        super().clear_connector_metadata()

    def wait_for_save(self) -> None:
        worker = self.worker_handler
        if worker is None or worker._connector_metadata is None:
            return
        meta = worker._connector_metadata
        if not meta.store_gpu_blocks:
            return
        done = self.writeback_inline(meta.store_gpu_blocks, meta.store_cpu_blocks)
        worker._store_events.append((meta.store_event, done))

    def writeback_retired(self, gpu_blocks: list[int], cpu_blocks: list[int]) -> torch.Event:
        """Copy after native scheduler and transfer refs have retired."""
        logger.info_once("MORI D2H copies retired eviction sources independently of the compute stream")
        return self.writeback_inline(gpu_blocks, cpu_blocks, fence_compute_stream=False)

    def writeback_inline(
        self, gpu_blocks: list[int], cpu_blocks: list[int], *, fence_compute_stream: bool = True
    ) -> torch.Event:
        """Copy preserved HBM pages and return only after the DMA is visible."""
        worker = self.worker_handler
        assert worker is not None
        backend = worker._backend
        if not isinstance(backend, DmaCopyBackend) or backend._store_params is None:
            raise RuntimeError("MORI writeback requires the vLLM CPU DMA backend")

        stream = worker.store_stream
        if fence_compute_stream:
            stream.wait_stream(torch.cuda.current_stream())
        params = backend._store_params
        if self._triton_writeback and len(gpu_blocks) * params.num_layers >= MIN_N and np.all(params.bpb % 8 == 0):
            # The native kernel writes pinned DRAM through CUDA unified
            # addressing on the same stream as the DMA fallback.
            logger.info_once("MORI D2H writeback uses the native vLLM Triton copy kernel")
            gpu = np.asarray(gpu_blocks, dtype=np.uint64)
            cpu = np.asarray(cpu_blocks, dtype=np.uint64)
            strides = params.bpb[:, None]
            source = params.src_bases[:, None] + gpu * strides
            destination = params.dst_bases[:, None] + cpu * strides
            sizes = np.repeat(params.bpb, len(gpu))
            with torch.cuda.stream(stream):
                swap_blocks_batch(
                    torch.from_numpy(source.ravel().view(np.int64)),
                    torch.from_numpy(destination.ravel().view(np.int64)),
                    torch.from_numpy(sizes.view(np.int64)),
                    bytes_per_chunk=4096,
                )
        else:
            copy_blocks(gpu_blocks, cpu_blocks, params)
        done = torch.Event()
        done.record(stream)
        done.synchronize()
        return done
