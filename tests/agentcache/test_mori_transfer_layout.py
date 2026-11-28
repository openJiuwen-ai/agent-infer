# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise the existing vLLM descriptor builder with MORI H2D ranges."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

from agentinfer.agentcache.mori import connector as mori
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


class _CudaRegion:
    """CPU storage bearing the device type used for layout validation."""

    device = torch.device("cuda")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def _bind_layerwise_worker(monkeypatch, source, destination, stream, *, cuda=True, layerwise=True, use_v2=False):
    worker = mori.SimpleCPUOffloadWorker(None, None, 1024)
    worker.load_stream = stream
    worker._backend = mori.DmaCopyBackend()
    worker._backend._load_params = cuda_mem_ops.build_params(source, destination, stream)
    worker.gpu_kv_caches = {name: _CudaRegion(tensor) for name, tensor in destination.items()}
    worker.cpu_kv_caches = source
    monkeypatch.setattr(worker, "register_kv_caches", Mock())
    monkeypatch.setattr(mori.current_platform, "is_cuda", lambda: cuda)
    connector = mori.MoriCPUOffloadConnector.__new__(mori.MoriCPUOffloadConnector)
    connector.worker_handler = worker
    connector._vllm_config = SimpleNamespace(use_v2_model_runner=use_v2)
    connector._layerwise_loads = layerwise
    connector.register_kv_caches(dict(reversed(list(worker.gpu_kv_caches.items()))))
    return connector


@pytest.mark.parametrize(("use_v2", "layerwise"), [(True, True), (False, True), (True, False)])
def test_registered_layers_select_v2_graph_dispatch(monkeypatch, use_v2, layerwise):
    from agentinfer.agentcache.mori import vllm_compat

    install = Mock()
    monkeypatch.setattr(vllm_compat, "install_mori_layerwise_graph_dispatch", install)
    source = {"layer": torch.zeros((4, 32), dtype=torch.uint8)}
    destination = {"layer": torch.empty_like(source["layer"])}
    connector = _bind_layerwise_worker(
        monkeypatch, source, destination, SimpleNamespace(cuda_stream=23), layerwise=layerwise, use_v2=use_v2
    )
    assert bool(connector._layer_load_params) is layerwise
    assert install.call_count == int(layerwise and use_v2)


@pytest.mark.parametrize(
    ("source", "destination", "bounds", "runs"),
    [
        (list(range(1, 18)), list(range(41, 58)), (128, 128), [17]),
        (list(range(17, 0, -1)), list(range(57, 40, -1)), (128, 128), [17]),
        (
            list(range(1, 20)) + list(range(70, 50, -1)) + [99],
            list(range(1, 20)) + list(range(100, 80, -1)) + [122],
            (128, 128),
            [19, 20, 1],
        ),
        (list(range(1, 21)) + list(range(20, 40)), list(range(1, 41)), (128, 128), [20, 20]),
        (list(range(2, 96, 2)), list(range(1, 95, 2)), (128, 128), [1] * 47),
        (list(range(1, 33)), list(range(72, 40, -1)), (128, 128), [1] * 32),
        (list(range(1, 17)), list(range(41, 57)), (128, 128), [1] * 16),
        (list(range(1, 18)) + list(range(20, 114, 2)), list(range(1, 65)), (128, 128), [17] + [1] * 47),
        (list(range(1, 33)), list(range(41, 73)), None, [1] * 32),
    ],
)
def test_layerwise_h2d_uses_native_copy_and_records_event_after_each_layer(
    monkeypatch, source, destination, bounds, runs
):
    cpu = {
        name: torch.arange(128 * width, dtype=torch.int64).mul(71).remainder(251).to(torch.uint8).view(128, width)
        for name, width in (("first", 32), ("last", 56))
    }
    device = {name: torch.full_like(layer, 0xCA) for name, layer in cpu.items()}
    expected = {name: layer.clone() for name, layer in device.items()}
    for name in cpu:
        expected[name][destination] = cpu[name][source]
    calls = []

    def copy(dst, src, sizes, count, attrs, attrs_idx, num_attrs, fail_idx, stream):
        assert stream == load_stream.cuda_stream
        assert num_attrs == 1
        access = ctypes.cast(attrs, ctypes.POINTER(cuda_mem_ops._CUmemcpyAttributes)).contents.srcAccessOrder
        assert access == cuda_mem_ops.CU_MEMCPY_SRC_ACCESS_ORDER_ANY
        pointer = ctypes.POINTER(ctypes.c_uint64)
        destinations, sources, lengths = (ctypes.cast(array, pointer) for array in (dst, src, sizes))
        calls.append(("copy", [lengths[index] for index in range(count)]))
        for index in range(count):
            ctypes.memmove(destinations[index], sources[index], lengths[index])
        return 0

    def new_event():
        def record(stream):
            assert stream is load_stream
            calls.append(("record",))

        return SimpleNamespace(record=record)

    load_stream = SimpleNamespace(cuda_stream=73)
    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy", (copy, 1))
    monkeypatch.setattr(cuda_mem_ops, "_resolve_max_batch_descriptors", lambda: 0)
    monkeypatch.setattr(mori.torch, "Event", new_event)
    connector = _bind_layerwise_worker(monkeypatch, cpu, device, load_stream, cuda=bounds is not None)
    layout = connector._dense_layer_load
    assert (layout.bounds if layout is not None else None) == bounds
    metadata = SimpleCPUOffloadMetadata(load_cpu_blocks=source, load_gpu_blocks=destination, load_event=4)

    loaded = connector._submit_layer_load(metadata)

    assert loaded.metadata is metadata
    assert metadata.load_cpu_blocks == source and metadata.load_gpu_blocks == destination
    assert loaded.done is loaded.events["first"]
    assert not connector.worker_handler._load_events
    assert calls == [
        ("copy", [run * 56 for run in runs]),
        ("record",),
        ("copy", [run * 32 for run in runs]),
        ("record",),
    ]
    assert all(torch.equal(device[name], expected[name]) for name in cpu)


@pytest.mark.parametrize(
    ("source", "destination"),
    [
        (list(range(1, 34)), [41, *range(41, 73)]),  # destination alias
        ([128, *range(1, 33)], list(range(41, 74))),  # outside physical source
        (list(range(1, 34)), [0, *range(41, 73)]),  # null destination
        (list(range(1, 18)) + list(range(20, 120, 2)), list(range(1, 68))),  # below 25% savings
    ],
)
def test_h2d_range_planner_declines_ambiguous_page_mappings(source, destination):
    assert (
        mori._contiguous_load_runs(
            np.array(source, dtype=np.uint64),
            np.array(destination, dtype=np.uint64),
            (128, 128),
        )
        is None
    )


def test_layerwise_load_drains_previous_copy_if_a_later_native_call_fails(monkeypatch):
    load_stream = SimpleNamespace(cuda_stream=73, synchronize=Mock())
    driver = Mock(side_effect=[0, 91])
    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy", (driver, 1))
    monkeypatch.setattr(cuda_mem_ops, "_resolve_max_batch_descriptors", lambda: 0)
    event = SimpleNamespace(record=Mock())
    monkeypatch.setattr(mori.torch, "Event", lambda: event)
    caches = {name: torch.empty(64, 16, dtype=torch.uint8) for name in ("first", "last")}
    device = {name: torch.empty_like(tensor) for name, tensor in caches.items()}
    connector = _bind_layerwise_worker(monkeypatch, caches, device, load_stream)

    with pytest.raises(RuntimeError, match="batch memcpy failed"):
        connector._submit_layer_load(
            SimpleCPUOffloadMetadata(load_cpu_blocks=list(range(1, 33)), load_gpu_blocks=list(range(24, 56)))
        )

    assert driver.call_count == 2
    assert all(call.args[3] == 1 for call in driver.call_args_list)
    event.record.assert_called_once_with(load_stream)
    load_stream.synchronize.assert_called_once_with()
    assert not connector.worker_handler._load_events


def test_layerwise_regions_must_match_native_dense_bases_strides_and_not_overlap():
    storage = torch.empty(16, 32, dtype=torch.uint8)
    first = storage[:8]
    second = storage[8:]
    assert (
        mori._layer_slot_capacity({"a": first, "b": second}, [first.data_ptr(), second.data_ptr()], [32, 32], "cpu")
        == 8
    )
    overlapping = storage[4:12]
    assert (
        mori._layer_slot_capacity(
            {"a": first, "b": overlapping}, [first.data_ptr(), overlapping.data_ptr()], [32, 32], "cpu"
        )
        == 0
    )
    assert mori._layer_slot_capacity({"a": first}, [first.data_ptr()], [31], "cpu") == 0
    padded = first[:, :16]
    assert mori._layer_slot_capacity({"a": padded}, [padded.data_ptr()], [16], "cpu") == 0
