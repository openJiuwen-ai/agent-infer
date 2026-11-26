# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

try:
    from vllm.v1.core.sched.output import KVConnectorBlockState as _KVConnectorBlockState  # noqa: F401
except ImportError:
    pytest.skip("MORI engine tests require the native KVConnectorBlockState ABI", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from tests.agentcache.mori.helpers import make_scheduler
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


@pytest.mark.parametrize(
    ("query_len", "prefilling", "expected"),
    [(2, [True, True], "prefill"), (1, [True], "native"), (8, [False, True], "native"), (2, None, "native")],
)
def test_split_attention_preserves_native_decode_graph_metadata(monkeypatch, query_len, prefilling, expected):
    import torch

    from agentinfer.agentcache.mori.attention import FlashInferMetadataBuilder, MORIAttentionMetadataBuilder

    monkeypatch.setattr(FlashInferMetadataBuilder, "build", lambda *_: "native")
    builder = MORIAttentionMetadataBuilder.__new__(MORIAttentionMetadataBuilder)
    builder.prefill_builder = SimpleNamespace(build=lambda *_: "prefill")
    host_flags = None if prefilling is None else torch.tensor(prefilling)
    meta = SimpleNamespace(max_query_len=query_len, is_prefilling=host_flags)
    assert builder.build(0, meta) == expected


def test_full_graph_is_skipped_only_for_a_layerwise_h2d_consumer(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
    from agentinfer.agentcache.mori.vllm_compat import install_mori_layerwise_graph_dispatch

    def execute(runner, output):
        return runner.cudagraph_manager.dispatch(1, 1, 1, 0)

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)
    install_mori_layerwise_graph_dispatch()
    graph = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=4, num_reqs=4)

    def dispatch(*_):
        return graph

    connector = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    connector._layer_load_params = {"layer_a": "params"}
    manager = SimpleNamespace(dispatch=dispatch)
    runner = SimpleNamespace(dp_size=1, cudagraph_manager=manager, kv_connector=SimpleNamespace(kv_connector=connector))
    output = SimpleNamespace(
        has_sync_kv_loads=True,
        total_num_scheduled_tokens=1,
        kv_connector_metadata=SimpleNamespace(load_cpu_blocks=[1]),
    )
    desc = GPUModelRunner.execute_model(runner, output)
    assert (desc.cg_mode, desc.num_reqs, desc.num_tokens) == (CUDAGraphMode.NONE, 1, 1)
    assert manager.dispatch is dispatch
    output.kv_connector_metadata.load_cpu_blocks = []
    assert GPUModelRunner.execute_model(runner, output) is graph


@pytest.mark.parametrize(
    ("forward", "blocked"),
    [
        ("eager", None),
        ("piecewise", None),
        ("full", None),
        ("eager", "flush"),
        ("eager", "zero"),
        ("eager", "cow"),
        ("aborted", None),
    ],
)
def test_local_worker_submits_one_h2d_job_before_prepare_and_adopts_its_events(monkeypatch, forward, blocked):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
    from vllm.v1.worker.gpu import kv_connector
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from agentinfer.agentcache.mori import connector as connector_module
    from agentinfer.agentcache.mori.vllm_compat import install_mori_layerwise_graph_dispatch

    calls = []
    events = []
    worker = SimpleCPUOffloadWorker(None, None, 1024)
    worker.load_stream = object()
    worker._backend = connector_module.DmaCopyBackend.__new__(connector_module.DmaCopyBackend)
    worker._backend._load_params = "batch"
    connector = connector_module.MoriCPUOffloadConnector.__new__(connector_module.MoriCPUOffloadConnector)
    connector.worker_handler = worker
    connector._fence_previous_forward = False
    connector.early_layerwise_loads = True
    connector._layer_load_params = {"first": "first", "last": "last"}
    active = kv_connector.ActiveKVConnector.__new__(kv_connector.ActiveKVConnector)
    active.kv_connector = connector
    active._disabled = False
    runner = SimpleNamespace(
        dp_size=1, cudagraph_manager=SimpleNamespace(dispatch=lambda *_: None), kv_connector=active
    )
    mode = CUDAGraphMode.PIECEWISE if forward == "piecewise" else CUDAGraphMode.NONE
    context = SimpleNamespace(cudagraph_runtime_mode=mode, attn_metadata={"first": object(), "last": object()})
    monkeypatch.setattr(kv_connector, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(kv_connector, "get_forward_context", lambda: context)
    compute = SimpleNamespace(wait_event=lambda event: calls.append(("wait", events.index(event))))
    monkeypatch.setattr(connector_module.torch.cuda, "current_stream", lambda: compute)

    def make_event():
        def record(stream):
            assert stream is worker.load_stream
            calls.append(("record", len(events) - 1))

        event = SimpleNamespace(record=record, query=lambda: True, synchronize=lambda: calls.append("drain"))
        events.append(event)
        return event

    def copy(source, destination, params):
        assert list(source) == [1, 2] and list(destination) == [3, 4]
        calls.append(("copy", params))

    monkeypatch.setattr(connector_module.torch, "Event", make_event)
    monkeypatch.setattr(connector_module, "copy_blocks", copy)
    metadata = SimpleCPUOffloadMetadata(
        load_event=7,
        load_cpu_blocks=[1, 2],
        load_gpu_blocks=[3, 4],
        load_event_to_reqs={7: ["request"]},
        need_flush=blocked == "flush",
    )
    output = SimpleNamespace(
        total_num_scheduled_tokens=18,
        num_scheduled_tokens={"request": 17, "decode": 1},
        has_sync_kv_loads=True,
        kv_connector_metadata=metadata,
        new_block_ids_to_zero=[3] if blocked == "zero" else None,
        kv_cache_block_copies=[(1, 3)] if blocked == "cow" else None,
    )

    def execute(_runner, output):
        assert not worker._load_events
        calls.append("prepare")
        if forward == "aborted":
            raise ValueError("Worker input preparation failed")
        # FULL replay passes external attn metadata with a placeholder context
        # and bypasses the live attention hooks.
        kwargs = {"attn_metadata": {}} if forward == "full" else {}
        active.pre_forward(output, **kwargs)
        assert worker._load_events == [(7, events[-1])]
        if forward != "full":
            for layer in connector._layer_load_params:
                connector.wait_for_layer_load(layer)
        return active.post_forward(set())

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)
    install_mori_layerwise_graph_dispatch()

    transfer = [("copy", "first"), ("record", 0), ("copy", "last"), ("record", 1)]
    if forward == "aborted":
        with pytest.raises(ValueError, match="Worker input preparation failed"):
            GPUModelRunner.execute_model(runner, output)
        assert calls == [*transfer, "prepare", "drain"]
        assert connector._prepared_layer_load is None and not worker._load_events
        return

    acknowledgment = GPUModelRunner.execute_model(runner, output)

    waits = [("wait", 1)] if forward == "full" else [("wait", 0), ("wait", 1)]
    expected = ["prepare", *transfer] if blocked else [*transfer, "prepare"]
    assert calls == [*expected, *waits]
    assert acknowledgment.finished_recving == {"request"}
    assert connector._prepared_layer_load is connector._layer_load_events is None
    assert not worker._load_events


@pytest.mark.parametrize("retired", [False, True])
def test_writeback_stream_dependency_is_scoped_to_retired_evictions(monkeypatch, retired):
    from agentinfer.agentcache.mori import connector as connector_module

    calls = []
    compute = object()
    stream = SimpleNamespace(wait_stream=lambda previous: calls.append(("wait", previous)))
    event = SimpleNamespace(
        record=lambda stream: calls.append(("record", stream)), synchronize=lambda: calls.append("complete")
    )
    backend = connector_module.DmaCopyBackend.__new__(connector_module.DmaCopyBackend)
    backend._store_params = object()
    connector = connector_module.MoriCPUOffloadConnector.__new__(connector_module.MoriCPUOffloadConnector)
    worker = SimpleNamespace(
        _backend=backend,
        _connector_metadata=SimpleNamespace(store_gpu_blocks=[1], store_cpu_blocks=[2], store_event=42),
        store_stream=stream,
        _store_events=[],
    )
    connector.worker_handler = worker
    monkeypatch.setattr(connector_module.torch.cuda, "current_stream", lambda: compute)
    monkeypatch.setattr(connector_module.torch, "Event", lambda: event)
    monkeypatch.setattr(connector_module, "copy_blocks", lambda gpu, cpu, _: calls.append(("copy", gpu, cpu)))

    if retired:
        assert connector.writeback_retired([1], [2]) is event
        assert worker._store_events == []
    else:
        connector.wait_for_save()
        assert worker._store_events == [(42, event)]
    expected = [("copy", [1], [2]), ("record", stream), "complete"]
    assert calls == (expected if retired else [("wait", compute), *expected])


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("joint_enabled", [False, True])
def test_mori_flashinfer_sampling_uses_upstream_joint_filter(monkeypatch, use_v2, joint_enabled):
    import torch
    from flashinfer import sampling
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.sample.ops import topk_topp_sampler as v1_sampler
    from vllm.v1.worker.gpu.sample import sampler as v2_sampler

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector

    sampler, other = (v2_sampler, v1_sampler) if use_v2 else (v1_sampler, v2_sampler)
    other_native = other.flashinfer_sample
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        additional_config={"agentcache": {"mori_joint_flashinfer_sampling": joint_enabled}},
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        parallel_config=SimpleNamespace(world_size=1),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True, kv_connector_extra_config={}),
    )
    logits = torch.tensor([[1.0, 2.0]])
    k = torch.tensor([2])
    p = torch.tensor([0.8])

    def joint(actual_logits, actual_k, actual_p, *, filter_apply_order):
        assert actual_logits is logits and actual_k is k and actual_p is p
        assert filter_apply_order == "joint"
        return torch.tensor([[1]])

    def native(actual_logits, actual_k, actual_p, *args):
        assert actual_logits is logits and actual_k is k
        assert args == (() if use_v2 else ({},))
        return actual_k

    monkeypatch.setattr(sampling, "top_k_top_p_sampling_from_logits", joint)
    monkeypatch.setattr(sampler, "flashinfer_sample", native)
    scheduler_connector = MoriCPUOffloadConnector(config, KVConnectorRole.SCHEDULER, None)
    assert sampler.flashinfer_sample is native
    worker_connector = MoriCPUOffloadConnector(config, KVConnectorRole.WORKER, None)
    args = () if use_v2 else ({},)
    try:
        assert other.flashinfer_sample is other_native
        assert sampler.flashinfer_sample(logits, k, p, *args).tolist() == ([1] if joint_enabled else [2])
        assert sampler.flashinfer_sample(logits, k, None, *args) is k
    finally:
        worker_connector.shutdown()
        scheduler_connector.shutdown()
    assert sampler.flashinfer_sample is native


@pytest.mark.parametrize("partial_utf8", [False, True])
def test_mori_detokenizer_uses_bounded_context_at_clean_byte_boundary(monkeypatch, partial_utf8):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from vllm.v1.engine import EngineCoreRequest, detokenizer

    from agentinfer.agentcache.mori.vllm_compat import mori_text_detokenizer

    vocabulary = {symbol: index for index, symbol in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = SimpleNamespace(_tokenizer=backend)
    text = "Long prefix " * 100 + "é"
    tokens = backend.encode(text).ids
    if partial_utf8:
        tokens = tokens[:-1]
    request = EngineCoreRequest("test", tokens, None, SamplingParams(temperature=0), None, 0, None, None, None)
    output = backend.encode(" hello ☀").ids
    native = detokenizer.FastIncrementalDetokenizer
    full = native(tokenizer, request)
    original_stream = decoders.DecodeStream
    contexts = []

    def record_context(*, ids, skip_special_tokens):
        contexts.append(ids)
        return original_stream(ids=ids, skip_special_tokens=skip_special_tokens)

    monkeypatch.setattr(decoders, "DecodeStream", record_context)
    with mori_text_detokenizer():
        aligned = detokenizer.FastIncrementalDetokenizer(tokenizer, request)
        assert [aligned.decode_next(token) for token in output] == [full.decode_next(token) for token in output]
    assert detokenizer.FastIncrementalDetokenizer is native
    assert request.prompt_token_ids is tokens
    assert contexts == [tokens if partial_utf8 else tokens[-5:]]


def test_mori_engine_core_reuses_exact_hashes_and_keeps_native_decode_chain(monkeypatch):
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.core import EngineCore

    from agentinfer.agentcache.mori.programs import PROGRAM_ARG
    from agentinfer.agentcache.mori.vllm_compat import install_mori_request_preprocessing

    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", bytes(32), raising=False)
    native = kv_cache_utils.get_request_block_hasher(16, sha256)
    starts = []

    def observe(request):
        starts.append(len(request.block_hashes))
        return native(request)

    scheduler = make_scheduler()
    scheduler.hash_block_size = 16
    scheduler.prefix_hash_capacity = 192
    core = SimpleNamespace(mm_receiver_cache=None, request_block_hasher=observe, scheduler=scheduler)
    monkeypatch.setattr(EngineCore, "preprocess_add_request", EngineCore.preprocess_add_request)
    install_mori_request_preprocessing()

    def admit(tokens, program, salt=None):
        params = SamplingParams(temperature=0, extra_args={PROGRAM_ARG: program})
        request = EngineCoreRequest("r", tokens, None, params, None, 0, None, salt, None)
        expected = Request.from_engine_core_request(request, native)
        actual, _ = EngineCore.preprocess_add_request(core, request)
        assert actual.block_hashes == expected.block_hashes
        return actual, expected

    ids = list(range(64))
    admit(ids, "a")
    actual, expected = admit(ids[:43] + [7] * 21, "a")
    assert starts == [0, 2]
    actual.append_output_token_ids([1] * 16)
    expected.append_output_token_ids([1] * 16)
    assert actual.block_hashes == expected.block_hashes
    admit(ids, "a", salt="salt")
    admit(ids, "b")
    assert starts[-3:] == [4, 0, 0]
    admit(ids, "c")
    admit(ids, "d")
    admit(ids, "a")
    assert starts[-3:] == [0, 0, 0]


@pytest.mark.parametrize("full_runner_v2", [False, True])
@pytest.mark.parametrize("fence_previous_forward", [False, True])
def test_batch_load_worker_fences_the_consumer(monkeypatch, full_runner_v2, fence_previous_forward):
    from agentinfer.agentcache.mori import connector as module

    connector = module.MoriCPUOffloadConnector.__new__(module.MoriCPUOffloadConnector)
    connector._fence_previous_forward = fence_previous_forward
    calls = []
    event = SimpleNamespace(record=lambda stream: calls.append(("record", stream)))
    compute = SimpleNamespace(wait_event=lambda ready: calls.append(("compute_wait", ready)))
    load = SimpleNamespace(wait_stream=lambda stream: calls.append(("load_wait", stream)))
    backend = module.DmaCopyBackend()
    backend._load_params = object()
    connector.worker_handler = SimpleNamespace(
        _backend=backend,
        _connector_metadata=SimpleNamespace(load_cpu_blocks=[1], load_gpu_blocks=[2], load_event=3),
        _load_events=[],
        load_stream=load,
    )
    monkeypatch.setattr(module.torch, "Event", lambda: event)
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda: compute)
    monkeypatch.setattr(module, "copy_blocks", lambda cpu, gpu, params: calls.append(("copy", cpu, gpu, params)))

    if full_runner_v2:
        connector._layer_load_params = {"layer_a": "param_a"}
        context = SimpleNamespace(cudagraph_runtime_mode=module.CUDAGraphMode.NONE)
        connector.start_load_kv(context, attn_metadata=object())
    else:
        connector.start_load_kv(None)
    before = [("load_wait", compute)] if fence_previous_forward else []
    assert calls == [
        *before,
        ("copy", [1], [2], backend._load_params),
        ("record", load),
        ("compute_wait", event),
    ]
    assert connector.worker_handler._load_events == [(3, event)]


def test_layerwise_load_waits_only_at_the_consuming_layer(monkeypatch):
    from agentinfer.agentcache.mori import connector as module

    connector = module.MoriCPUOffloadConnector.__new__(module.MoriCPUOffloadConnector)
    calls = []
    compute = SimpleNamespace(wait_event=lambda event: calls.append(("compute_wait", event.name)))
    load = SimpleNamespace(wait_stream=lambda _: calls.append(("load_wait",)))
    backend = module.DmaCopyBackend()
    backend._load_params = object()
    connector.worker_handler = SimpleNamespace(
        _backend=backend,
        _connector_metadata=SimpleNamespace(load_cpu_blocks=[1], load_gpu_blocks=[2], load_event=3),
        _load_events=[],
        load_stream=load,
    )
    connector._layer_load_params = {"layer_a": "param_a", "layer_b": "param_b"}
    events = [
        SimpleNamespace(name=name, record=lambda _, key=name: calls.append(("record", key))) for name in ("a", "b")
    ]
    iterator = iter(events)
    monkeypatch.setattr(module.torch, "Event", lambda: next(iterator))
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda: compute)
    transfers = []

    def copy_blocks(cpu, gpu, params):
        transfers.append((cpu, gpu))
        calls.append(("copy", params))

    monkeypatch.setattr(module, "copy_blocks", copy_blocks)
    monkeypatch.setattr(module.SimpleCPUOffloadConnector, "clear_connector_metadata", lambda _: None)

    connector.start_load_kv(
        SimpleNamespace(cudagraph_runtime_mode=module.CUDAGraphMode.PIECEWISE, attn_metadata={"layer_a": object()})
    )
    assert calls == [("load_wait",), ("copy", "param_a"), ("record", "a"), ("copy", "param_b"), ("record", "b")]
    assert all(first is second for first, second in zip(*transfers, strict=True))
    assert [pages.tolist() for pages in transfers[0]] == [[1], [2]]
    connector.wait_for_layer_load("layer_a")
    with pytest.raises(RuntimeError, match="layer_b"):
        connector.clear_connector_metadata()
    connector.wait_for_layer_load("layer_b")
    connector.clear_connector_metadata()
    assert calls[-2:] == [("compute_wait", "a"), ("compute_wait", "b")]
    assert connector.worker_handler._load_events == [(3, events[-1])]


def test_batch_writeback_finishes_before_worker_reports_eviction(monkeypatch):
    from agentinfer.agentcache.mori import connector as module

    connector = module.MoriCPUOffloadConnector.__new__(module.MoriCPUOffloadConnector)
    calls = []
    event = SimpleNamespace(
        record=lambda stream: calls.append(("record", stream)),
        synchronize=lambda: calls.append(("completed",)),
    )
    compute = object()
    store = SimpleNamespace(wait_stream=lambda stream: calls.append(("store_wait", stream)))
    backend = module.DmaCopyBackend()
    backend._store_params = object()
    connector.worker_handler = SimpleNamespace(
        _backend=backend,
        _connector_metadata=SimpleNamespace(store_gpu_blocks=[2], store_cpu_blocks=[1], store_event=4),
        _store_events=[],
        store_stream=store,
    )
    monkeypatch.setattr(module.torch, "Event", lambda: event)
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda: compute)
    monkeypatch.setattr(module, "copy_blocks", lambda gpu, cpu, params: calls.append(("copy", gpu, cpu, params)))

    connector.wait_for_save()
    assert calls == [
        ("store_wait", compute),
        ("copy", [2], [1], backend._store_params),
        ("record", store),
        ("completed",),
    ]
    assert connector.worker_handler._store_events == [(4, event)]


def test_triton_writeback_keeps_native_page_order_and_ack_fence(monkeypatch):
    from agentinfer.agentcache.mori import connector as module

    connector = module.MoriCPUOffloadConnector.__new__(module.MoriCPUOffloadConnector)
    connector._triton_writeback = True
    calls = []
    event = SimpleNamespace(
        record=lambda _: calls.append(("record",)),
        synchronize=lambda: calls.append(("completed",)),
    )
    backend = module.DmaCopyBackend()
    backend._store_params = SimpleNamespace(
        num_layers=2,
        src_bases=module.np.array([1000, 2000], dtype=module.np.uint64),
        dst_bases=module.np.array([3000, 4000], dtype=module.np.uint64),
        bpb=module.np.array([8, 16], dtype=module.np.uint64),
    )
    connector.worker_handler = SimpleNamespace(
        _backend=backend,
        _connector_metadata=SimpleNamespace(store_gpu_blocks=[2, 4], store_cpu_blocks=[1, 3], store_event=7),
        _store_events=[],
        store_stream=SimpleNamespace(wait_stream=lambda _: calls.append(("store_wait",))),
    )
    monkeypatch.setattr(module, "MIN_N", 4)
    monkeypatch.setattr(module.torch, "Event", lambda: event)
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(module.torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(
        module,
        "swap_blocks_batch",
        lambda src, dst, sizes, **_: calls.append(("copy", src.tolist(), dst.tolist(), sizes.tolist())),
    )

    connector.wait_for_save()
    assert calls == [
        ("store_wait",),
        ("copy", [1016, 1032, 2032, 2064], [3008, 3024, 4016, 4048], [8, 8, 16, 16]),
        ("record",),
        ("completed",),
    ]
    assert connector.worker_handler._store_events == [(7, event)]


def test_engine_rejects_native_encoder_features_before_add(monkeypatch):
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.core import EngineCore

    from agentinfer.agentcache.mori.vllm_compat import install_mori_request_preprocessing

    feature = MultiModalFeatureSpec(data=None, modality="image", identifier="img", mm_position=PlaceholderRange(0, 2))
    req = EngineCoreRequest("image", [1, 2], [feature], SamplingParams(), None, 0, None, None, None)
    core = SimpleNamespace(scheduler=make_scheduler(), request_block_hasher=None)
    monkeypatch.setattr(EngineCore, "preprocess_add_request", lambda self, request: request)
    install_mori_request_preprocessing()
    with pytest.raises(ValueError, match="text inputs only"):
        EngineCore.preprocess_add_request(core, req)
    core.scheduler = SimpleNamespace()
    assert EngineCore.preprocess_add_request(core, req) is req
