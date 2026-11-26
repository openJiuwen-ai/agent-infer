# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

try:
    from vllm.v1.core.sched.output import KVConnectorBlockState as _KVConnectorBlockState  # noqa: F401
except ImportError:
    pytest.skip("MORI engine tests require the native KVConnectorBlockState ABI", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.mori.block_pool import MoriBlockPool
from agentinfer.agentcache.mori.hiradix import HiRadix
from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager
from agentinfer.agentcache.mori.offload import MoriHiCacheController
from agentinfer.agentcache.mori.scheduler import MoriScheduler, PrefillBatch
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


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


def test_mori_flashinfer_sampling_uses_upstream_joint_filter(monkeypatch):
    import torch
    from flashinfer import sampling

    from agentinfer.agentcache.mori import attention

    logits = torch.tensor([[1.0, 2.0]])
    k = torch.tensor([2])
    p = torch.tensor([0.8])

    def joint(actual_logits, actual_k, actual_p, *, filter_apply_order):
        assert actual_logits is logits and actual_k is k and actual_p is p
        assert filter_apply_order == "joint"
        return torch.tensor([[1]])

    monkeypatch.setattr(sampling, "top_k_top_p_sampling_from_logits", joint)
    monkeypatch.setattr(attention, "flashinfer_sample", lambda actual_logits, actual_k, actual_p: actual_k)
    assert attention._sample_flashinfer_joint(logits, k, p).tolist() == [1]
    assert attention._sample_flashinfer_joint(logits, k, None) is k


def key(index: int) -> bytes:
    return index.to_bytes(32, "big")


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
    from agentinfer.agentcache.mori.vllm_compat import install_mori_program_block_hasher

    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", bytes(32), raising=False)
    native = kv_cache_utils.get_request_block_hasher(16, sha256)
    starts = []

    def observe(request):
        starts.append(len(request.block_hashes))
        return native(request)

    core = SimpleNamespace(
        mm_receiver_cache=None,
        request_block_hasher=observe,
        scheduler=SimpleNamespace(hash_block_size=16, prefix_hash_capacity=192),
    )
    monkeypatch.setattr(EngineCore, "preprocess_add_request", EngineCore.preprocess_add_request)
    install_mori_program_block_hasher()

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


@pytest.fixture
def cache() -> HiRadix:
    gpu = MoriBlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=16)
    host = MoriBlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=16)
    tree = HiRadix(gpu, host, 16)
    gpu.bind_hiradix_index(tree.device)
    host.bind_hiradix_index(tree.host)
    return tree


def insert(tree: HiRadix, keys: Iterable[int], parent: int | None = None, program: str = "p"):
    pool = tree.device.pool
    ids = list(keys)
    blocks = pool.get_new_blocks(len(ids))
    tree.placement.register(program)
    depth = 0 if parent is None else next(iter(tree.pages[key(parent)].device.values())).block_hash_num_tokens
    for index, block in zip(ids, blocks, strict=True):
        depth += tree.page_size
        pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(key(index)), 0), block, num_tokens=depth)
        tree.publish(block, key(index), key(parent) if parent is not None else None, program)
        parent = index
    tree.pages[key(ids[-1])].boundary = True
    tree.stamp(blocks[-1], program)
    pool.free_blocks(blocks)
    return blocks


def backup(tree: HiRadix, blocks):
    pool = tree.host.pool
    copies = pool.get_new_blocks(len(blocks))
    for source, target in zip(blocks, copies, strict=True):
        pool._insert_block_hash(source.block_hash, target, num_tokens=source.block_hash_num_tokens)
        tree.backup(source.block_id, target)
    pool.free_blocks(copies)
    return copies


def test_evict_whole_edges_and_reveal_parent(cache):
    a, b, c = insert(cache, (1, 2, 3))
    d, e = insert(cache, (4, 5), parent=1, program="other")
    leaves = cache.device.plan(8)
    assert len(leaves[0]) == 3
    assert leaves[1:] == [[c.block_id, b.block_id], [e.block_id, d.block_id], [a.block_id]]
    assert cache.pages[key(1)].shared


def test_hiradix_owns_shared_index_and_retires_before_reuse(cache):
    gpu, host = cache.device.pool, cache.host.pool
    gpu.bind_hiradix_index(cache.device, authoritative=True)
    host.bind_hiradix_index(cache.host, authoritative=True)
    (block,) = gpu.get_new_blocks(1)
    digest = BlockHash(key(1))
    grouped = make_block_hash_with_group_id(digest, 0)
    gpu._insert_block_hash(grouped, block, num_tokens=16)
    assert gpu.get_cached_block(digest, [0]) is None
    cache.publish(block, digest, None, "p")
    (copy,) = backup(cache, [block])
    gpu.free_blocks([block])

    assert gpu.cached_block_hash_to_block.get_one_block(grouped) is None
    assert host.cached_block_hash_to_block.get_one_block(grouped) is None
    assert gpu.get_cached_block(digest, [0]) == [block]
    assert host.get_indexed_copy(grouped) is copy
    assert gpu.get_usage() == 0
    with pytest.raises(RuntimeError, match="bypassed eviction planning"):
        gpu.get_new_blocks(8)

    gpu.evict_blocks({block.block_id})
    assert gpu.get_cached_block(digest, [0]) is None
    assert cache.match([digest]) == ([], [copy])
    assert len(gpu.get_new_blocks(8)) == 8


@pytest.mark.parametrize("invalid", ["pinned", "duplicate", "negative"])
def test_invalid_hiradix_allocation_leaves_the_native_queue_intact(cache, monkeypatch, invalid):
    pool = cache.device.pool
    occupied = pool.get_new_blocks(1)[0]
    before = pool.free_block_queue.get_all_free_blocks()
    free = before[0]
    wrong_id = {"pinned": occupied.block_id, "duplicate": free.block_id, "negative": -1}[invalid]
    monkeypatch.setattr(cache.device, "allocation_ids", lambda _: [free.block_id, wrong_id])

    with pytest.raises(RuntimeError, match="MORI HiRadix"):
        pool.get_new_blocks(2)

    assert pool.free_block_queue.get_all_free_blocks() == before
    assert occupied.ref_cnt == 1 and free.ref_cnt == 0
    assert cache.device.ready == {block.block_id for block in before}


def test_cache_reset_uses_native_pin_checks_then_clears_hiradix(cache):
    pool = cache.device.pool
    pool.bind_hiradix_index(cache.device, authoritative=True)
    (cached,) = insert(cache, (1,))
    pool.bind_hiradix_index(cache.device)
    assert cache.match([key(1)]) == ([cached], [])

    pool.touch([cached])
    assert not pool.reset_prefix_cache()
    assert cache.match([key(1)]) == ([cached], [])

    pool.free_blocks([cached])
    assert pool.reset_prefix_cache()
    assert not cache.pages and cached.block_hash is None
    assert len(pool.get_new_blocks(pool.get_num_free_blocks())) == pool.num_gpu_blocks - 1


def test_released_hbm_replica_does_not_take_a_second_cache_slot(cache):
    (original,) = insert(cache, (1,))
    pool = cache.device.pool
    (duplicate,) = pool.get_new_blocks(1)
    pool._insert_block_hash(original.block_hash, duplicate, num_tokens=16)
    cache.publish(duplicate, key(1), None, "p")
    assert list(cache.pages[key(1)].device.values()) == [original, duplicate]

    pool.touch([duplicate])
    pool.free_blocks([duplicate])
    assert duplicate.ref_cnt == 1
    assert duplicate.block_hash == original.block_hash

    pool.free_blocks([duplicate])
    assert duplicate.block_hash is None
    assert cache.match([key(1)]) == ([original], [])
    assert duplicate.block_id in cache.device.unused


@pytest.mark.parametrize(
    ("published", "native_cached", "refresh", "skip"),
    [
        (1, 1, False, True),
        (0, 1, False, False),
        (1, 0, False, False),
        (1, 1, True, False),
    ],
)
def test_decode_result_reuses_confirmed_prompt(cache, monkeypatch, published, native_cached, refresh, skip):
    request = SimpleNamespace(request_id="decode", num_prompt_tokens=17, num_output_tokens=32)
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.mori_batch_loads = True
    manager.hiradix = cache
    manager._replayed_prefill_tokens = {}
    manager._radix_depth = {"decode": published}
    manager.coordinator = SimpleNamespace(
        single_type_managers=(SimpleNamespace(num_cached_block={"decode": native_cached}),)
    )
    actions = []
    monkeypatch.setattr(cache, "release_load", lambda rid: actions.append(("release", rid)))
    monkeypatch.setattr(manager, "_cache_and_publish", lambda req, tokens, **kw: actions.append(("publish", tokens)))
    manager.cache_blocks(request, 48, refresh_existing=refresh)
    assert actions == ([("release", "decode")] if skip else [("publish", 17)])


def test_confirmed_hiradix_insertion_refreshes_previously_published_edges(cache):
    (older,) = insert(cache, (1,))
    (newer,) = insert(cache, (2,))
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = cache.device.pool
    manager.hiradix = cache
    request = _request("cache-hit", (1,), 17)
    manager._radix_depth = {request.request_id: 1}

    assert cache.device.plan(1, include_uncached=False) == [[older.block_id]]
    manager.confirm_hiradix(request, 17)
    assert cache.device.plan(1, include_uncached=False) == [[older.block_id]]
    manager.confirm_hiradix(request, 17, refresh_existing=True)
    assert cache.device.plan(1, include_uncached=False) == [[newer.block_id]]


def test_inserting_terminal_prefix_touches_old_edge_before_splitting(cache):
    first, second, _, _ = insert(cache, (1, 2, 3, 4))
    assert cache.match([key(1)]) == ([first], [])
    (unrelated,) = insert(cache, (5,))
    assert not cache.pages[key(2)].boundary

    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = cache.device.pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(get_blocks=lambda _: ([first, second],))
    manager.release_replaced_hiradix_blocks = None
    request = _request("terminal", (1, 2), 32)
    manager._radix_depth = {request.request_id: 1}

    manager.publish_hiradix(request, 32, refresh_existing=True)

    assert cache.pages[key(2)].boundary
    assert cache.device.plan(1, include_uncached=False) == [[unrelated.block_id]]


def test_eviction_does_not_write_a_page_that_has_a_pinned_replica(cache, monkeypatch):
    (canonical,) = insert(cache, (1,))
    pool = cache.device.pool
    (pinned,) = pool.get_new_blocks(1)
    pool._insert_block_hash(canonical.block_hash, pinned, num_tokens=16)
    cache.publish(pinned, key(1), None, "p")
    offload = MoriHiCacheController.__new__(MoriHiCacheController)
    offload.bind_tree(cache)
    cache.cache_controller = offload
    offload._gpu_block_pool = pool
    allocation_sizes = []
    monkeypatch.setattr(cache, "_reserve_host_pages", lambda count, _: allocation_sizes.append(count) or [])

    assert cache._demote_device_edge([canonical.block_id], frozenset()) == ([], set())
    assert allocation_sizes == [0]
    assert cache.match([key(1)]) == ([pinned], [])
    assert pinned.ref_cnt == 1


def test_confirmed_active_request_reuses_canonical_pages_at_the_worker_fence(cache):
    (canonical,) = insert(cache, (1,))
    pool = cache.device.pool
    (duplicate,) = pool.get_new_blocks(1)
    pool._insert_block_hash(canonical.block_hash, duplicate, num_tokens=16)
    request = _request("duplicate", (1,), 17)
    request.last_sched_seq = 19
    blocks = [duplicate]
    releases = []
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(get_blocks=lambda _: (blocks,))
    manager._radix_depth = {}
    manager._hiradix_block_table_updates = set()
    manager.release_replaced_hiradix_blocks = lambda pages, fence: releases.append((pages, fence))
    manager.get_block_ids = lambda _: ([block.block_id for block in blocks],)

    manager.confirm_hiradix(request, 17)
    assert blocks == [canonical]
    assert canonical.ref_cnt == duplicate.ref_cnt == 1
    assert releases == [([duplicate], 19)]
    assert manager.take_hiradix_block_table_updates({}) == {}
    assert manager.take_hiradix_block_table_updates({request.request_id: 1}) == {
        request.request_id: ([canonical.block_id],)
    }
    assert manager.take_hiradix_block_table_updates({request.request_id: 1}) == {}

    pool.free_blocks(releases[0][0])
    assert cache.match([key(1)]) == ([canonical], [])
    assert duplicate.block_hash is None


@pytest.mark.parametrize("retract", [False, True])
def test_generated_pages_are_private_until_finish_or_replayed_prefill(retract):
    import torch
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    from agentinfer.agentcache.mori.scheduler import MoriAsyncScheduler
    from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

    page = 16
    spec = FullAttentionSpec(block_size=page, num_kv_heads=1, head_size=64, dtype=torch.float16)
    config = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer"], spec)])
    with mori_block_pool_constructor():
        manager = MoriKVCacheManager(config, max_model_len=128, scheduler_block_size=page, hash_block_size=page)
    gpu = manager.block_pool
    host = MoriBlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=page)
    tree = HiRadix(gpu, host, page)
    manager.hiradix = tree
    gpu.bind_hiradix_index(tree.device, authoritative=True)
    scheduler = MoriAsyncScheduler.__new__(MoriAsyncScheduler)
    scheduler.kv_cache_manager = manager
    scheduler.max_model_len = 128
    init_none_hash(sha256)
    hasher = get_request_block_hasher(page, sha256)
    request = Request(
        "decoder", list(range(page)), SamplingParams(max_tokens=19 if retract else 18), None, block_hasher=hasher
    )

    def process(tokens, next_token, **slots):
        assert manager.allocate_slots(request, tokens, **slots) is not None
        request.num_computed_tokens += tokens + slots.get("num_new_computed_tokens", 0)
        request.num_output_placeholders = 1
        request.status = RequestStatus.RUNNING
        _, stopped = scheduler._update_request_with_output(request, [next_token])
        return stopped

    assert not process(page, 30)
    for token in range(31, 47):
        assert not process(1, token)
    assert request.num_output_tokens == 17 and request.num_computed_tokens == 32
    follower = Request("follower", list(request.all_token_ids), SamplingParams(max_tokens=1), None, block_hasher=hasher)
    prompt, suffix = manager.get_blocks(request.request_id).blocks[0]
    assert manager.get_computed_blocks(follower)[1] == page
    assert gpu.get_cached_block(request.block_hashes[1], [0]) is None
    assert suffix.block_hash is None and suffix.block_id not in tree.device.by_id

    if not retract:
        assert process(1, 47)
        manager.free(request)
        assert manager.get_computed_blocks(follower)[1] == 2 * page
        assert suffix.block_hash is not None and suffix.block_id in tree.device.by_id
        return

    # The native scheduler still protects the writer's public and private KV
    # through its async free fence. Retraction does not insert the suffix.
    scheduler.defer_block_free = True
    scheduler.sched_step_seq, scheduler.processed_step_seq = 4, 3
    scheduler.deferred_frees = deque()
    scheduler.aux_output_connector = None
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _: None)
    scheduler._inflight_prefills = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.log_stats = False
    scheduler.waiting = FCFSRequestQueue()
    request.last_sched_seq = 4
    scheduler._preempt_request(request, 0)
    assert request.status == RequestStatus.PREEMPTED
    assert prompt.ref_cnt == suffix.ref_cnt == 1
    assert suffix.block_id not in tree.device.ready
    assert manager.get_computed_blocks(follower)[1] == page
    scheduler.processed_step_seq = 4
    scheduler._drain_deferred_frees()
    assert prompt.ref_cnt == suffix.ref_cnt == 0
    assert suffix.block_id in tree.device.unused and suffix.block_hash is None

    # Replay known output tokens as prefill before entering decode again.
    prefix, hit, _ = manager.get_computed_blocks(request)
    assert not process(request.num_tokens - hit, 47, num_new_computed_tokens=hit, new_computed_blocks=prefix)
    assert manager.get_computed_blocks(follower)[1] == 2 * page


@pytest.mark.parametrize(("local_worker", "connector_pending"), [(True, False), (True, True), (False, False)])
def test_next_lpm_can_match_previously_dispatched_local_prefill_chunk(
    cache, monkeypatch, local_worker, connector_pending, step_without_worker
):
    pool = cache.device.pool
    request = _request("chunk", (1, 2, 3), 49)
    request.is_prefill_chunk = True
    request.last_sched_seq = 2
    request.num_computed_tokens = 24
    request.num_in_flight_tokens = 24
    full, partial = pool.get_new_blocks(2)
    pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(key(1)), 0), full, num_tokens=16)
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(get_blocks=lambda _: ([full, partial],))
    manager._radix_depth = {}
    manager._unpublished_batch_loads = {request.request_id} if connector_pending else set()
    borrower = _request("borrower", (1, 2), 33)
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = manager
    scheduler.running = [request]
    scheduler.waiting = FCFSRequestQueue([borrower])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_active_reqs = 4
    scheduler.processed_step_seq = 1
    scheduler.sched_step_seq = 2
    _test_controller(cache).inline_eviction_writeback = object() if local_worker else None
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    seen = []

    def match_before_prefill(waiting):
        assert waiting == [borrower]
        seen.append(cache.request_match(borrower))
        return waiting

    monkeypatch.setattr(scheduler, "_order_waiting", match_before_prefill)
    monkeypatch.setattr(scheduler, "_prefer_prefill", lambda *args: PrefillBatch(tokens=16))
    monkeypatch.setattr(scheduler, "_prepare_prefill", lambda *args: True)
    step_without_worker(scheduler)
    assert seen == [([full] if local_worker and not connector_pending else [], [])]
    assert partial.block_hash is None


def test_writeback_ack_preserves_dispatch_epoch(cache):
    (device,) = insert(cache, (1,))
    page = cache.device.by_id[device.block_id]
    page.write_epoch = cache.epoch
    cache.epoch += 1
    backup(cache, (device,))
    assert page.write_epoch == cache.epoch - 1


def test_match_spans_device_and_host_and_marks_epoch(cache):
    a, b, c = insert(cache, (1, 2, 3))
    host_b, host_c = backup(cache, (b, c))
    assert not cache.host.leaves
    cache.device.pool.evict_blocks({b.block_id, c.block_id})
    gpu, cpu = cache.match([key(1), key(2), key(3)], mark_load=True)
    assert gpu == [a]
    assert cpu == [host_b, host_c]
    assert cache.pages[key(3)].load_epoch == cache.epoch
    assert cache.host.plan(7)[-1] == [host_c.block_id, host_b.block_id]


def test_native_cache_manager_and_connector_share_one_prefix_lookup(cache, monkeypatch):
    (device,) = insert(cache, (1,))
    request = _request("prefix", (1,), 17)
    original = cache.request_match
    lookups = []

    def match(req, **kwargs):
        lookups.append(req.request_id)
        return original(req, **kwargs)

    monkeypatch.setattr(cache, "request_match", match)
    cache.native_schedule_active = True
    assert cache.match_for_native_admission(request) == ([device], [])
    assert cache.match_for_native_admission(request, connector=True) == ([device], [])
    assert lookups == ["prefix"]

    # The saved match never crosses the step boundary.
    cache.match_for_native_admission(request)
    cache.finish_native_schedule()
    cache.device.pool.evict_blocks({device.block_id})
    assert cache.match_for_native_admission(request, connector=True) == ([], [])
    assert lookups == ["prefix"] * 3


def test_native_admission_reuses_selected_hbm_only_until_the_cache_changes(cache, monkeypatch):
    (device,) = insert(cache, (1,))
    request = _request("prepared-prefill", (1,), 17)
    lookups = []
    match = cache.request_match

    def observe(req, **kwargs):
        lookups.append(req.request_id)
        return match(req, **kwargs)

    monkeypatch.setattr(cache, "request_match", observe)
    cache.remember_hbm_match(request, [device])
    cache.native_schedule_active = True
    assert cache.match_for_native_admission(request) == ([device], [])
    assert cache.match_for_native_admission(request, connector=True) == ([device], [])
    assert not lookups

    cache.remember_hbm_match(request, [device])
    cache.device.pool.evict_blocks({device.block_id})
    assert cache.match_for_native_admission(request) == ([], [])
    assert lookups == [request.request_id]


def test_hbm_hit_reservations_preserve_the_branch(cache):
    a, b = insert(cache, (1, 2))
    (c,) = insert(cache, (3,), parent=1)
    plan = cache.device.plan(8, {b.block_id})
    selected = {ident for edge in plan for ident in edge}
    assert b.block_id not in selected
    assert a.block_id not in selected
    assert c.block_id in selected


def test_released_owner_is_replaced_on_reuse(cache):
    (a,) = insert(cache, (1,), program="old")
    cache.placement.release("old")
    cache.placement.register("new")
    cache.stamp(a, "new")
    assert cache.pages[key(1)].program == "new"
    assert not cache.pages[key(1)].shared


def test_prefill_stamp_rechecks_ancestors_after_program_is_reregistered(cache):
    insert(cache, (1, 2), program="p")
    cache.stamp_prefill(key(2), "p")
    original = cache.pages[key(2)].stamped_for

    cache.placement.release("p")
    cache.stamp_prefill(key(1), "replacement")
    assert cache.pages[key(1)].program == "replacement"
    assert not cache.pages[key(1)].shared

    cache.stamp_prefill(key(2), "p")
    assert cache.pages[key(2)].stamped_for is not original
    assert cache.pages[key(1)].shared
    assert cache.pages[key(2)].program == "p"


def test_partial_match_touches_unmatched_suffix_before_splitting(cache):
    head, _suffix = insert(cache, (1, 2))
    (other,) = insert(cache, (3,))

    assert cache.match([key(1)])[0] == [head]

    # Five uncached blocks are allocated first, then the other edge is
    # older than the suffix of the just-accessed compressed edge.
    assert cache.device.plan(6)[-1] == [other.block_id]


def test_inserting_branch_refreshes_old_suffix_before_splitting(cache):
    head, _, old_end = insert(cache, (1, 2, 3))
    (unrelated,) = insert(cache, (4,))
    assert cache.device.plan(1, include_uncached=False)[0][0] == old_end.block_id

    # Generated tokens can diverge inside an existing edge without a
    # scheduler lookup at the split point.
    (new_end,) = insert(cache, (5,), parent=1)
    assert cache.device.plan(1, include_uncached=False) == [[unrelated.block_id]]
    assert cache.device.by_id[old_end.block_id].accessed < cache.device.by_id[head.block_id].accessed
    assert cache.device.by_id[head.block_id].accessed < cache.device.by_id[new_end.block_id].accessed


def test_prefill_batch_reserves_shared_device_pages_once(cache):
    insert(cache, (1, 2, 3, 4))
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.max_num_active_reqs = 128
    scheduler.new_token_ratio = 0.7

    requests = [
        SimpleNamespace(
            request_id=str(tail),
            block_hashes=[key(index) for index in (1, 2, 3, 4, tail)],
            num_tokens=80,
            num_computed_tokens=0,
            num_output_tokens=0,
            max_tokens=1,
            sampling_params=None,
            skip_reading_prefix_cache=False,
            status=RequestStatus.WAITING,
        )
        for tail in (5, 6)
    ]

    # 128 tokens available. The first request reserves a shared 64-token
    # prefix and 33 more tokens, leaving enough for the second shared hit.
    batch = scheduler._prefer_prefill([], [], requests, [])
    assert (batch.tokens, batch.waiting, batch.holding) == (32, requests, [])


def _request(name, hashes, length, max_tokens=1):
    request = Request(name, [0] * length, SamplingParams(max_tokens=max_tokens), None)
    request.block_hashes.extend(BlockHash(key(index)) for index in hashes)
    return request


def test_lpm_uses_full_page_boundary_then_keeps_last_token_for_forward(cache):
    insert(cache, (1, 2, 3))
    short_hit = _request("first", (1, 2, 4), 49)
    full_hit = _request("second", (1, 2, 3), 48)
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    ranked = scheduler._order_waiting([short_hit, full_hit])
    assert ranked == [full_hit, short_hit]
    assert [len(cache.request_match(request)[0]) for request in ranked] == [2, 2]


def test_chunk_that_completes_reserves_output_capacity(cache):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    chunked = _request("chunk", (1, 2), 33, max_tokens=64)
    chunked.num_computed_tokens = 16
    small = _request("small", (3,), 16, max_tokens=1)
    assert scheduler._prefer_prefill([chunked], [], [small], []) == PrefillBatch(
        tokens=17, token_caps={chunked.request_id: 17}, full=True
    )


def test_chunk_that_exhausts_its_budget_still_checks_waiting_hbm_capacity(cache):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 16
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    chunked = _request("continuation", (1, 2), 33, max_tokens=64)
    chunked.num_computed_tokens = 16
    large = _request("waiter", (3, 4), 33, max_tokens=88)
    assert scheduler._prefer_prefill([chunked], [], [large], []) == PrefillBatch(
        tokens=16, token_caps={chunked.request_id: 16}, full=True
    )


@pytest.mark.parametrize(("max_prefill_tokens", "admitted"), [(96, 1), (128, 2)])
def test_prefill_input_limit_is_separate_from_the_chunk_limit(cache, max_prefill_tokens, admitted):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    scheduler.max_prefill_tokens = max_prefill_tokens
    waiting = [_request("first", (1, 2), 33), _request("second", (3, 4), 33)]
    assert scheduler._prefer_prefill([], [], waiting, []).waiting == waiting[:admitted]


def test_cold_program_is_registered_when_considered_for_prefill(cache):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    waiting = [_request("capacity-head", (1, 2), 33, max_tokens=128), _request("behind-head", (3,), 17)]
    for request in waiting:
        request.sampling_params.extra_args = {"mori_program_id": request.request_id}
    assert scheduler._prefer_prefill([], [], waiting, []) == PrefillBatch()
    assert cache.placement.has_program("capacity-head")
    assert not cache.placement.has_program("behind-head")


def test_considered_prefix_is_stamped_even_when_budget_rejects_request(cache):
    (previous,) = insert(cache, (1,), program="previous")
    cache.placement.register("waiting")
    req = _request("waiting", (1, 2), 33, max_tokens=128)
    req.sampling_params.extra_args = {"mori_program_id": "waiting"}
    running = _request("running", (3,), 17)
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    assert scheduler._prefer_prefill([], [running], [req], []) == PrefillBatch(full=True)
    assert cache.device.by_id[previous.block_id].shared


def test_stamp_visits_ancestors_when_endpoint_already_has_same_owner(cache):
    (ancestor,) = insert(cache, (1,), program="previous")
    (endpoint,) = insert(cache, (2,), parent=1, program="waiting")
    page = cache.device.by_id[ancestor.block_id]
    page.shared = False
    cache.stamp(endpoint, "waiting")
    assert page.shared


def test_idle_head_rejects_instead_of_admitting_a_smaller_follower(cache):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    head = _request("head", (1,), 17, max_tokens=128)
    small = _request("small", (2,), 17)
    assert scheduler._prefer_prefill([], [], [head, small], []) == PrefillBatch()


def test_prefill_budget_reaches_native_for_partial_running_chunk(cache, step_without_worker):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    cache.stage_evictions = lambda *args: None
    chunk = _request("chunk", (1, 2), 300)
    chunk.is_prefill_chunk = True
    chunk.num_computed_tokens = 16
    scheduler.running = [chunk]
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 256
    scheduler.max_num_active_reqs = 10

    selected = []

    def allocate_running(_batch, requests, limit, **_):
        if requests:
            selected.append((list(requests), limit))
        return limit

    step_without_worker(scheduler, allocate_running=allocate_running)
    assert selected == [([chunk], 128)]
    assert scheduler.max_num_scheduled_tokens == 256


def test_lpm_order_persists_when_preselection_admits_nothing(cache, step_without_worker):
    insert(cache, (1, 2, 3))
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    a = _request("first", (1,), 17)
    b = _request("long-hit", (1, 2, 3), 48, max_tokens=128)
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue([a, b])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    step_without_worker(scheduler)
    assert list(scheduler.waiting) == [b, a]


@pytest.mark.parametrize("commit_decode", [False, True])
def test_full_prefill_batch_retries_after_a_running_request_leaves(
    cache, monkeypatch, step_without_worker, commit_decode
):
    from agentinfer.agentcache.mori.schedule_batch import ScheduleBatch

    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = False
    running = _request("running", (), 17, max_tokens=24)
    running.status = RequestStatus.RUNNING
    running.num_computed_tokens = running.num_prompt_tokens
    running.num_output_placeholders = 1
    waiter = _request("waiting", (1, 2), 33, max_tokens=100)
    scheduler.running = [running]
    scheduler.waiting = FCFSRequestQueue([waiter])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.num_lookahead_tokens = 0
    scheduler.kv_cache_manager = SimpleNamespace(
        hiradix=cache,
        block_pool=cache.device.pool,
        allocate_slots=lambda *_, **__: KVCacheBlocks(([],)),
    )
    lookups = []
    monkeypatch.setattr(scheduler, "_order_waiting", lambda reqs: lookups.append(reqs) or reqs)

    allocate_running = ScheduleBatch.running if commit_decode else None
    first = step_without_worker(scheduler, allocate_running=allocate_running)
    assert first.num_scheduled_tokens == ({running.request_id: 1} if commit_decode else {})
    step_without_worker(scheduler, allocate_running=allocate_running)
    assert lookups == [[waiter]]
    scheduler.running.clear()
    step_without_worker(scheduler, allocate_running=allocate_running)
    assert lookups == [[waiter], [waiter]]
    assert scheduler._full_prefill_running is None


@pytest.mark.parametrize(("existing_decode", "partial_prefill"), [(False, False), (True, True), (True, False)])
def test_prefill_merge_preserves_full_flag_only_for_an_existing_running_batch(
    cache, monkeypatch, existing_decode, partial_prefill, step_without_worker
):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = False
    chunk = _request("prefill", (1, 2), 33)
    chunk.is_prefill_chunk = True
    decode = _request("running", (), 17)
    scheduler.running = [decode, chunk] if existing_decode else [chunk]
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    monkeypatch.setattr(scheduler, "_prefer_prefill", lambda *_: PrefillBatch(tokens=16, full=True))

    def native_schedule():
        assert scheduler.running == ([decode, chunk] if existing_decode else [chunk])
        chunk.is_prefill_chunk = partial_prefill
        return SimpleNamespace(preempted_req_ids=set(), num_scheduled_tokens={chunk.request_id: 16})

    step_without_worker(scheduler, native_schedule)
    expected = set(scheduler.running) if existing_decode and not partial_prefill else None
    assert scheduler._full_prefill_running == expected


def test_lpm_checks_the_first_32_tokens_when_page_is_larger(cache):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    cache.page_size = 64
    first = Request("first", [1] * 32 + [2] * 32, SamplingParams(), None)
    duplicate = Request("duplicate", [1] * 32 + [3] * 32, SamplingParams(), None)
    other = Request("other", [4] * 64, SamplingParams(), None)
    ranked = scheduler._order_waiting([first, duplicate, other])
    assert ranked == [first, other, duplicate]


def test_prefill_retouches_prefixes_in_ranked_order(cache):
    (short_block,) = insert(cache, (1,))
    long_blocks = insert(cache, (2, 3))
    short = _request("short", (1,), 17)
    long = _request("long", (2, 3), 33)
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.new_token_ratio = 0.7
    batch = scheduler._prefer_prefill([], [], [short, long], [])
    assert batch.waiting == [long, short]
    assert cache.device.plan(1, include_uncached=False) == [[block.block_id for block in reversed(long_blocks)]]
    assert cache.device.by_id[short_block.block_id].accessed > cache.device.by_id[long_blocks[-1].block_id].accessed


def _test_controller(cache):
    offload = MoriHiCacheController.__new__(MoriHiCacheController)
    offload.bind_tree(cache)
    cache.cache_controller = offload
    offload.cpu_block_pool = cache.host.pool
    offload._gpu_block_pool = cache.device.pool
    offload._lazy_mode = False
    offload.hash_block_size = offload.block_size = 16
    offload.group_block_sizes = [16]
    offload.prefix_cacheable_group_ids = {0}
    offload._pending_cpu_hits = {}
    offload._reqs_to_load = {}
    offload._reqs_to_store = {}
    offload._load_event_to_reqs = {}
    offload._interval_load_blocks_completed = 0
    offload._eviction_store_gpu_ids = set()
    offload._prefetch_pins = {}
    offload._pending_finished_stores = []
    return offload


def test_writeback_keeps_gpu_victim_order_and_assigns_host_slots_in_prefix_order(cache):
    controller = _test_controller(cache)
    controller.enable_kv_cache_events = False
    controller._in_flight_store_gpu_blocks = set()
    pages = insert(cache, (1, 2, 3, 4))
    cache.device.pool.get_new_blocks(cache.device.free_uncached())
    expected_victims = [page.block_id for page in reversed(pages)]

    assert cache.stage_evictions(4, frozenset()) == tuple(expected_victims)
    (transfer,) = controller._pending_finished_stores
    assert transfer.gpu_block_ids == expected_victims
    assert transfer.cpu_block_ids == [4, 3, 2, 1]
    assert all(page.ref_cnt == 1 for page in pages)
    controller._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
    device, host = cache.match([key(index) for index in (1, 2, 3, 4)])
    assert device == [] and [block.block_id for block in host] == [1, 2, 3, 4]
    assert all(block.ref_cnt == 0 for block in host)


@pytest.fixture
def native_load(cache):
    device_blocks = insert(cache, (1, 2))
    host_blocks = backup(cache, device_blocks)
    cache.device.pool.evict_blocks({block.block_id for block in device_blocks})
    return _test_controller(cache), host_blocks


@pytest.mark.parametrize("watched", [False, True])
def test_local_writeback_uses_native_retired_pages_only(cache, native_load, watched):
    from collections import deque

    from vllm.v1.core.sched.scheduler import Scheduler

    offload, host = native_load
    offload._in_flight_store_gpu_blocks = set()
    offload._pending_finished_stores = []
    offload.enable_kv_cache_events = False
    pool = cache.device.pool
    (retired,) = insert(cache, (3,))
    (request_page,) = insert(cache, (4,))
    (cow_page,) = insert(cache, (5,))
    pool.touch([request_page, cow_page])
    if watched:
        pool.touch([retired])
        pool.unpin_blocks([retired], lambda _: None)

    (loading,) = pool.get_new_blocks(1)
    cache.host.pool.touch(host[:1])
    cache.reserve_load("pending-h2d", [loading.block_id], [host[0].block_id])
    pool.get_new_blocks(cache.device.free_uncached())
    assert cache.device.free_uncached() == 0

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.defer_block_free = True
    scheduler.sched_step_seq = 5
    scheduler.processed_step_seq = 3
    scheduler.deferred_frees = deque()
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=pool, pop_blocks_for_free=lambda _: [request_page])
    scheduler._free_request_blocks(SimpleNamespace(last_sched_seq=5))
    scheduler._free_cow_retained_blocks([cow_page], 6)

    assert set(cache.device.by_id) & cache.device.ready == {retired.block_id}
    assert loading.block_id not in cache.device.by_id
    calls = []

    def copy(kind, source, destination):
        assert cache.host.pool.blocks[destination[0]].ref_cnt == 1
        assert cache.host.pool.blocks[destination[0]].block_hash is None
        calls.append((kind, source.copy()))

    offload.inline_eviction_writeback = lambda gpu, cpu: copy("fenced", gpu, cpu)
    offload.inline_retired_writeback = lambda gpu, cpu: copy("retired", gpu, cpu)
    assert cache.stage_evictions(1, frozenset()) == ()
    assert calls == [("fenced" if watched else "retired", [retired.block_id])]
    assert retired.block_id not in cache.device.by_id
    assert request_page.ref_cnt == cow_page.ref_cnt == loading.ref_cnt == 1

    scheduler.processed_step_seq = 6
    scheduler._drain_deferred_frees()
    assert set(cache.device.by_id) & cache.device.ready == {request_page.block_id, cow_page.block_id}
    assert loading.ref_cnt == 1


@pytest.mark.parametrize("still_in_use", [False, True])
def test_store_ack_reuses_native_hash_and_event_publication(cache, native_load, still_in_use):
    import torch
    from vllm.distributed.kv_events import MEDIUM_CPU, BlockStored
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.simple_kv_offload.manager import TransferMeta

    offload, _ = native_load
    offload.enable_kv_cache_events = True
    offload.kv_event_medium = MEDIUM_CPU
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    offload.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    gpu, host = cache.device.pool, cache.host.pool
    gpu.bind_hiradix_index(cache.device, authoritative=True)
    host.bind_hiradix_index(cache.host, authoritative=True)
    (source,) = insert(cache, (3,))
    (destination,) = host.get_new_blocks(1)
    primary = source.block_hash
    secondary = make_block_hash_with_group_id(BlockHash(key(9)), 0)
    gpu._insert_block_hash(secondary, source, num_tokens=None)
    gpu.touch([source])
    if still_in_use:
        gpu.touch([source])
    offload._eviction_store_gpu_ids.add(source.block_id)
    offload._in_flight_store_gpu_blocks = {source.block_id}
    offload._store_event_to_blocks = {7: TransferMeta([source.block_id], [destination.block_id])}
    offload._abandoned_store_event_to_blocks = {}
    offload._store_event_to_reqs = {}

    offload._process_store_event(7)
    offload._process_store_event(7)

    assert destination.block_hash == primary and destination.ref_cnt == 0
    assert host.cached_block_hashes_by_block[destination.block_id] == {secondary}
    assert len(host.kv_event_queue) == 2
    assert all(isinstance(event, BlockStored) and event.medium == MEDIUM_CPU for event in host.kv_event_queue)
    assert cache.host.by_id[destination.block_id] is cache.pages[key(3)]
    assert source.ref_cnt == int(still_in_use) and (source.block_hash is not None) == still_in_use
    assert not offload._eviction_store_gpu_ids and not offload._in_flight_store_gpu_blocks


@pytest.mark.parametrize("issued", [False, True])
def test_reset_never_publishes_an_abandoned_writeback(cache, native_load, issued):
    from vllm.v1.simple_kv_offload.manager import BoundaryStoreStats

    offload, _ = native_load
    offload.enable_kv_cache_events = False
    offload._in_flight_store_gpu_blocks = set()
    offload._store_event_to_blocks = {}
    offload._abandoned_store_event_to_blocks = {}
    offload._abandoned_reqs_to_load = {}
    offload._store_event_to_reqs = {}
    offload._store_event_pending_counts = {}
    offload._store_event_counter = 0
    offload.boundary_store_stats = BoundaryStoreStats()
    offload._boundary_stats_snapshot = BoundaryStoreStats()
    (source,) = insert(cache, (3,))
    cache.device.pool.get_new_blocks(cache.device.free_uncached())
    assert cache.stage_evictions(1, frozenset()) == (source.block_id,)
    (cpu_id,) = offload._pending_finished_stores[0].cpu_block_ids
    destination = cache.host.pool.blocks[cpu_id]
    assert source.ref_cnt == destination.ref_cnt == 1

    if issued:
        metadata = offload.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
        assert metadata.store_gpu_blocks == [source.block_id]
    assert offload.reset() is not issued
    if issued:
        assert source.ref_cnt == destination.ref_cnt == 1
        assert cache.evictions_pending()
        offload._process_store_event(metadata.store_event)
        offload._process_store_event(metadata.store_event)

    assert source.ref_cnt == destination.ref_cnt == 0
    assert not cache.evictions_pending()
    assert cache.match([key(3)]) == ([source], [])
    assert destination.block_hash is None and cpu_id not in cache.host.by_id


@pytest.mark.parametrize("valid_tokens", [0, 16, 32])
def test_h2d_visible_for_lpm_without_publishing_early(cache, native_load, valid_tokens, step_without_worker):
    offload, host = native_load
    loading = _request("load", (1, 2), 33)
    follows = _request("same-prefix", (1, 2), 33)
    unrelated = _request("cold", (3, 4), 33)
    assert offload.get_num_new_matched_tokens(loading, 0) == (32, True)
    pool = cache.device.pool
    dest = pool.get_new_blocks(2)
    offload.update_state_after_alloc(loading, KVCacheBlocks((dest,)), 32)

    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager._radix_depth = {}
    manager._replayed_prefill_tokens = {}
    manager._eviction_stores = {}
    manager.enable_caching = True

    def index_valid_blocks(request, tokens):
        for index in range(tokens // 16):
            pool._insert_block_hash(
                make_block_hash_with_group_id(request.block_hashes[index], 0), dest[index], num_tokens=(index + 1) * 16
            )

    manager.coordinator = SimpleNamespace(
        cache_blocks=index_valid_blocks,
        get_blocks=lambda _: (dest,),
        free=lambda _: pool.free_blocks(dest),
    )
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = manager
    scheduler.tree_cache.cache_controller = offload
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    assert scheduler._order_waiting([unrelated, follows])[0] is follows
    assert cache.request_match(follows, include_loading=True) == (dest, [])
    assert manager.get_computed_blocks(follows)[1] == 0
    assert offload.get_num_new_matched_tokens(follows, 0) == (None, False)
    assert [block.ref_cnt for block in host] == [1, 1]

    scheduler.connector = offload
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.kv_cache_manager = manager
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue([follows])
    scheduler.kv_holding_waiting = FCFSRequestQueue([loading])
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 10
    scheduler.requests = {loading.request_id: loading}
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = {loading.request_id} if valid_tokens < 32 else set()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.prefix_replay_tokens = 0
    loading.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    # Invalid-block accounting has already rewound the count when ACK arrives.
    loading.num_computed_tokens = valid_tokens
    offload._reqs_to_load[loading.request_id].load_event = 0
    # Include a valid-looking suffix after an invalid first page; it cannot be published.
    invalid_ids = {dest[valid_tokens // 16].block_id} if valid_tokens < 32 else set()
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={loading.request_id}, invalid_block_ids=invalid_ids)
    )
    assert manager.get_computed_blocks(follows)[1] == 0
    assert not cache.host.leaves

    def native_schedule():
        assert loading.status == RequestStatus.WAITING
        assert manager.get_computed_blocks(follows)[1] == valid_tokens
        assert list(cache.host.by_id) == [block.block_id for block in host[valid_tokens // 16 :]]
        retry_tokens = 32 - valid_tokens
        assert offload.get_num_new_matched_tokens(follows, valid_tokens) == (retry_tokens, retry_tokens > 0)
        return SimpleNamespace(preempted_req_ids=set(), num_scheduled_tokens={})

    step_without_worker(scheduler, native_schedule)
    assert not offload._pending_cpu_hits
    assert [block.ref_cnt for block in host] == [0, 0]
    assert cache.request_match(follows)[1] == host[valid_tokens // 16 :]
    assert not cache._pending_exclusive_loads


@pytest.mark.parametrize("issued", [False, True])
def test_cancel_h2d_drops_projection_without_releasing_inflight_refs(cache, native_load, issued):
    offload, host = native_load
    request = _request("cancel", (1, 2), 33)
    assert offload.get_num_new_matched_tokens(request, 0) == (32, True)
    dest = cache.device.pool.get_new_blocks(2)
    offload.update_state_after_alloc(request, KVCacheBlocks((dest,)), 32)
    if issued:
        offload._reqs_to_load[request.request_id].load_event = 0
    offload.request_finished(request, [])
    assert cache.request_match(request, include_loading=True) == ([], host)
    assert [block.ref_cnt for block in host] == [int(issued)] * 2
    if issued:
        offload.update_connector_output(KVConnectorOutput(finished_recving={request.request_id}))
    assert [block.ref_cnt for block in host] == [0, 0]
    assert not cache._pending_exclusive_loads


@pytest.mark.parametrize("active_at_dispatch", [False, True])
def test_host_copy_drop_uses_program_activity_at_dispatch(cache, native_load, active_at_dispatch):
    offload, host = native_load
    if not active_at_dispatch:
        cache.placement.release("p")
    request = _request("load", (1, 2), 33)
    assert offload.get_num_new_matched_tokens(request, 0) == (32, True)
    dest = cache.device.pool.get_new_blocks(2)
    offload.update_state_after_alloc(request, KVCacheBlocks((dest,)), 32)
    if active_at_dispatch:
        cache.placement.release("p")
    else:
        cache.placement.register("p")
    offload._reqs_to_load[request.request_id].load_event = 0
    offload.update_connector_output(KVConnectorOutput(finished_recving={request.request_id}))
    assert set(cache._pending_exclusive_loads) == ({block.block_id for block in host} if active_at_dispatch else set())


def test_pending_host_drop_cannot_discard_a_reused_block(cache, native_load):
    offload, host = native_load
    block = host[0]
    pool = cache.host.pool
    cache._pending_exclusive_loads[block.block_id] = block.block_hash
    pool.evict_blocks({block.block_id})
    new_hash = make_block_hash_with_group_id(BlockHash(key(9)), 0)
    pool._insert_block_hash(new_hash, block, num_tokens=16)
    cache.drain_exclusive_loads()
    assert block.block_hash == new_hash
    assert not cache._pending_exclusive_loads


def test_batch_load_keeps_exclusive_policy_until_late_ack(cache, native_load):
    offload, host = native_load
    offload.mori_batch_loads = True
    request = _request("load", (1, 2), 33)
    assert offload.get_num_new_matched_tokens(request, 0) == (32, False)
    dest = cache.device.pool.get_new_blocks(2)
    offload.update_state_after_alloc(request, KVCacheBlocks((dest,)), 32)
    offload._reqs_to_load[request.request_id].load_event = 0
    cache.release_load(request.request_id)
    assert request.request_id not in cache._request_loads

    output = KVConnectorOutput(finished_recving={request.request_id})
    offload.update_connector_output(output)
    assert output.finished_recving is None
    assert not offload._reqs_to_load
    assert not cache._exclusive_load_edges
    assert set(cache._pending_exclusive_loads) == {block.block_id for block in host}


@pytest.mark.parametrize("restore", ["standalone", "planned-batch", "exclusive-async"])
@pytest.mark.parametrize("split_before_restore", [False, True])
def test_host_restore_ack_releases_the_original_edges_current_suffix(cache, native_load, restore, split_before_restore):
    offload, host = native_load
    pool = cache.device.pool
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(single_type_managers=[SimpleNamespace(req_to_blocks={}, num_cached_block={})])
    batch = restore != "exclusive-async"
    if batch:
        offload.mori_batch_loads = cache.share_batch_loads = True
        offload.publish_batch_load = manager.publish_batch_load

    owner = _request("restore", (1, 2), 33)
    borrower = _request("borrower", (1,), 17)

    def split():
        cache.request_match(borrower, include_loading=True)
        assert cache.pages[key(1)].boundary

    if split_before_restore:
        split()
    if restore == "planned-batch":
        cache.request_match(owner, include_loading=True, plan_load=True)
        if not split_before_restore:
            split()
    if restore == "standalone":
        cache.prefetch(owner, [], host)
    else:
        assert offload.get_num_new_matched_tokens(owner, 0) == (32, not batch)
        dest = pool.get_new_blocks(2)
        offload.update_state_after_alloc(owner, KVCacheBlocks((dest,)), 32)
    if not split_before_restore and restore != "planned-batch":
        split()
    (request_id,) = offload._reqs_to_load
    offload._reqs_to_load[request_id].load_event = 0
    assert [block.ref_cnt for block in host] == [1, 1]
    offload.update_connector_output(KVConnectorOutput(finished_recving={request_id}))
    assert not cache._exclusive_load_edges
    assert [block.ref_cnt for block in host] == [0, 0]
    retained = set() if split_before_restore else {host[0].block_id}
    if batch:
        assert cache.host.by_id.keys() == retained
    else:
        assert cache._pending_exclusive_loads.keys() == set(cache.host.by_id) - retained


@pytest.mark.parametrize(("cancel_owner", "invalid_tail"), [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("issued", [False, True])
def test_h2d_prefix_is_shared_and_published_on_ack(cache, native_load, cancel_owner, invalid_tail, issued):
    offload, _ = native_load
    offload.mori_batch_loads = cache.share_batch_loads = True
    cache.share_issued_loads = issued
    owner = _request("load-owner", (1, 2), 33)
    borrower = _request("load-borrower", (1, 2), 33)
    pool = cache.device.pool
    dest = pool.get_new_blocks(2)

    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.enable_caching = manager.mori_batch_loads = True
    owned = {} if cancel_owner else {owner.request_id: dest}
    native = SimpleNamespace(req_to_blocks=owned, num_cached_block={})
    manager.coordinator = SimpleNamespace(single_type_managers=[native])
    offload.publish_batch_load = manager.publish_batch_load

    assert offload.get_num_new_matched_tokens(owner, 0) == (32, False)
    offload.update_state_after_alloc(owner, KVCacheBlocks((dest,)), 32)
    assert manager.get_computed_blocks(borrower)[0].blocks == (dest,)
    assert offload.get_num_new_matched_tokens(borrower, 32) == (0, False)
    assert all(block.block_hash is None for block in dest)
    pool.touch(dest)

    cache.submit_batch_loads([block.block_id for block in dest])
    cache.finish_native_schedule()
    device, host = cache.request_match(borrower, include_loading=True)
    assert cache.waits_for_load(device, host) is not issued
    if issued:
        assert manager.get_computed_blocks(borrower)[0].blocks == (dest,)
        assert offload.get_num_new_matched_tokens(borrower, 32) == (0, False)
    else:
        assert manager.get_computed_blocks(borrower)[1] == 0
        assert offload.get_num_new_matched_tokens(borrower, 0) == (None, False)
    offload._reqs_to_load[owner.request_id].load_event = 0
    if cancel_owner:
        offload.request_finished(owner, [])
        pool.free_blocks(dest)

    invalid = {dest[-1].block_id} if invalid_tail else set()
    offload.update_connector_output(KVConnectorOutput(finished_recving={owner.request_id}, invalid_block_ids=invalid))
    depth = 1 if invalid_tail else 2
    assert manager.get_computed_blocks(borrower)[0].blocks == (dest[:depth],)
    assert not cache.issued_load_destinations
    assert [page.block_hash_num_tokens for page in dest] == [16, None if invalid_tail else 32]
    assert native.num_cached_block == ({} if cancel_owner else {owner.request_id: depth})
    pool.free_blocks(dest)


@pytest.mark.parametrize("failed", [False, True])
def test_waiter_restores_dram_when_prior_chunk_exhausts_prefill_budget(cache, native_load, monkeypatch, failed):
    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector

    offload, host = native_load
    (ancestor,) = insert(cache, (1,))
    pool = cache.device.pool
    cache.share_batch_loads = cache.share_issued_loads = offload.mori_batch_loads = True
    offload.inline_eviction_writeback = object()
    offload._load_event_counter = 0
    offload._pending_finished_stores = []
    offload._store_event_to_blocks = {}
    offload._abandoned_store_event_to_blocks = {}
    monkeypatch.setattr(offload, "prepare_store_specs", lambda _: ([], [], [], None))

    request = _request("waiter", (1, 2), 33)
    chunk = _request("separate-long-prefill", (), 65)
    chunk.num_computed_tokens = 16
    engine = MoriScheduler.__new__(MoriScheduler)
    engine.max_num_active_reqs = 8
    engine.max_num_scheduled_tokens = 16
    engine.kv_cache_manager = manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.block_pool = pool
    manager.hiradix = cache
    manager.coordinator = SimpleNamespace(single_type_managers=[SimpleNamespace(req_to_blocks={}, num_cached_block={})])
    offload.publish_batch_load = manager.publish_batch_load
    engine.tree_cache.cache_controller = offload
    engine.new_token_ratio = 0.7
    engine.max_prefill_tokens = 64

    batch = engine._prefer_prefill([chunk], [], [request], [], ranked=[request])
    assert batch.tokens == 16 and not batch.waiting
    (key,) = offload._reqs_to_load
    assert key != request.request_id
    schedule = SimpleNamespace(preempted_req_ids=set(), has_sync_kv_loads=False)
    meta = offload.build_connector_meta(schedule)
    assert schedule.has_sync_kv_loads
    assert meta.load_cpu_blocks == [host[1].block_id]
    assert meta.load_event_to_reqs == {0: [key]}
    (destination,) = [pool.blocks[bid] for bid in meta.load_gpu_blocks]
    assert cache.request_match(request, include_batch_loading=True) == ([ancestor, destination], [])
    assert all(block.ref_cnt == 1 for block in (ancestor, destination, host[1]))

    connector = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    connector.scheduler_manager = offload
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.connector = connector
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = FCFSRequestQueue([request])
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.running = []
    scheduler.deferred_waiting = set()
    scheduler._inflight_prefills = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.aux_output_connector = scheduler.ec_connector = None
    scheduler.defer_block_free = False
    scheduler.vllm_config = SimpleNamespace(kv_transfer_config=None)
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _: None)
    scheduler.kv_cache_manager = SimpleNamespace(
        remove_skipped_blocks=lambda **_: None,
        get_block_ids_for_computed_tokens=lambda **_: ([],),
        free=lambda _: None,
    )
    assert scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED) == [request]
    assert not scheduler.requests and scheduler.finished_req_ids == {request.request_id}
    # The first empty worker step forwards the finished ID. Its H2D event may
    # still be outstanding, so the next step must be able to poll again.
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={}))
    cache.finish_native_schedule()
    assert cache.issued_load_destinations == {destination.block_id}
    assert not scheduler.has_unfinished_requests() and not scheduler.has_finished_requests()
    assert scheduler.has_requests()
    poll = connector.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
    assert poll.load_event_to_reqs == {0: [key]}
    assert not poll.load_gpu_blocks  # The empty step polls the old transfer.
    assert all(block.ref_cnt == 1 for block in (ancestor, destination, host[1]))
    output = KVConnectorOutput(
        finished_recving={key},
        invalid_block_ids={destination.block_id} if failed else set(),
    )
    scheduler._update_from_kv_xfer_finished(output)
    assert output.finished_recving is None
    assert not offload._prefetch_pins and not cache.issued_load_destinations
    assert all(block.ref_cnt == 0 for block in (ancestor, destination, host[1]))
    assert cache.request_match(request) == (([ancestor], [host[1]]) if failed else ([ancestor, destination], []))
    assert not scheduler.has_requests()


@pytest.mark.parametrize(
    "completion", ["ready", "hwm", "pending", "wrong_event", "wrong_hash", "additional_load", "stale_forward", "fatal"]
)
def test_mixed_chunk_stash_uses_certified_h2d_without_consuming_ack(completion):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
    from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

    page = 16
    spec = FullAttentionSpec(block_size=page, num_kv_heads=1, head_size=64, dtype=torch.float16)
    config = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer"], spec)])
    with mori_block_pool_constructor():
        manager = MoriKVCacheManager(config, max_model_len=128, scheduler_block_size=page, hash_block_size=page)
    gpu = manager.block_pool
    host = MoriBlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=page)
    tree = manager.hiradix = HiRadix(gpu, host, page)
    gpu.bind_hiradix_index(tree.device, authoritative=True)
    host.bind_hiradix_index(tree.host, authoritative=True)
    controller = _test_controller(tree)
    controller.publish_batch_load = manager.publish_batch_load
    controller.mori_batch_loads = manager.mori_batch_loads = True
    tree.share_batch_loads = tree.share_issued_loads = True

    (prefix,) = insert(tree, (1,))
    (source,) = backup(tree, [prefix])
    gpu.evict_blocks({prefix.block_id})
    request = _request("owner", (1, 2, 3), 49)
    assert controller.get_num_new_matched_tokens(request, 0) == (page, False)
    assert manager.allocate_slots(request, 2 * page, num_external_computed_tokens=page)
    blocks = manager.get_blocks(request.request_id)
    (destination, computed, _) = blocks.blocks[0]
    controller.update_state_after_alloc(request, blocks, page)
    state = controller._reqs_to_load[request.request_id]
    state.load_event = 2
    tree.submit_batch_loads([destination.block_id])
    tree.finish_native_schedule()
    request.num_computed_tokens = 3 * page
    request.status = RequestStatus.RUNNING

    worker = SimpleCPUOffloadWorker.__new__(SimpleCPUOffloadWorker)
    worker_connector = MoriCPUOffloadConnector.__new__(MoriCPUOffloadConnector)
    worker_connector.worker_handler = worker
    controller.local_load_completed = worker_connector.local_load_completed
    queried = []

    def query():
        queried.append(True)
        if completion == "fatal":
            raise RuntimeError("local CUDA copy failed")
        return completion != "pending"

    worker._load_hwm = 2 if completion == "hwm" else -1
    event = (1 if completion == "wrong_event" else 2, SimpleNamespace(query=query))
    worker._load_events = [] if completion == "hwm" else [event]
    if completion == "wrong_hash":
        request.block_hashes[0] = BlockHash(key(99))
    if completion == "additional_load":
        tree.issued_load_destinations.add(computed.block_id)

    if completion == "fatal":
        with pytest.raises(RuntimeError, match="local CUDA copy failed"):
            manager.stash_hiradix_chunk(request, latest_forward=True)
        assert computed.block_hash is None
        return

    manager.stash_hiradix_chunk(request, latest_forward=completion != "stale_forward")

    stashed = completion in ("ready", "hwm")
    assert (gpu.get_cached_block(request.block_hashes[1], [0]) is not None) is stashed
    assert len(queried) == int(completion not in ("hwm", "wrong_event", "stale_forward"))
    assert destination.ref_cnt == 2 and source.ref_cnt == 1
    assert controller._reqs_to_load[request.request_id] is state
    assert request.request_id in manager._unpublished_batch_loads
    if stashed:
        controller.update_connector_output(KVConnectorOutput(finished_recving={request.request_id}))
        assert destination.ref_cnt == 1 and source.ref_cnt == 0
        assert controller._interval_load_blocks_completed == 1
        manager.confirm_hiradix(request, request.num_computed_tokens)
        assert gpu.get_cached_block(request.block_hashes[1], [0]) == [computed]


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


def test_writeback_pressure_requests_full_eviction_size(cache, native_load, monkeypatch):
    offload, _ = native_load
    (a,) = insert(cache, (9,))
    (b,) = insert(cache, (10,))
    (c,) = insert(cache, (11,))
    cache.device.pool.get_new_blocks(4)
    assert cache.device.free_uncached() == 1
    victims = []

    def demote(group, _):
        victims.extend(group)
        cache.device.pool.evict_blocks(set(group))
        return [], set()

    monkeypatch.setattr(cache, "_demote_device_edge", demote)
    assert cache.stage_evictions(2, frozenset()) == ()
    assert victims == [a.block_id, b.block_id]
    assert c.block_id in cache.device.by_id


@pytest.mark.parametrize("local_worker", [False, True])
def test_prefill_reserves_aggregate_extend_after_planned_host_restore(cache, native_load, local_worker):
    offload, restore_host = native_load
    offload._in_flight_store_gpu_blocks = set()
    offload._pending_finished_stores = []
    offload.enable_kv_cache_events = False
    copies = []

    def copy(src, dst):
        assert all(cache.device.pool.blocks[block_id].ref_cnt == 1 for block_id in src)
        assert all(cache.host.pool.blocks[block_id].block_hash is None for block_id in dst)
        copies.append((src.copy(), dst.copy()))

    if local_worker:
        offload.inline_eviction_writeback = copy
    (host_victim,) = insert(cache, (3,))
    (first,) = insert(cache, (4,))
    (second,) = insert(cache, (5,))
    (third,) = insert(cache, (6,))
    (victim_host,) = backup(cache, [host_victim])
    cache.host.pool.get_new_blocks(5)
    cache.device.pool.get_new_blocks(3)
    cache.device.pool.evict_blocks({host_victim.block_id})
    assert cache.device.free_uncached() == 2

    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.tree_cache.cache_controller = offload
    scheduler.dense_kv = True
    request = _request("restore", (1, 2), 33)
    batch = PrefillBatch(
        tokens=1, waiting=[request], planned_host={block.block_id for block in restore_host}, needs_host_load=True
    )

    # Original HiRadix sees no free pages after restore, so it evicts the
    # full two-page extend estimate. Its DRAM eviction excludes the restore.
    assert scheduler._prepare_prefill([], batch) is local_worker
    assert second.block_id not in cache.device.by_id
    assert third.block_id in cache.device.by_id
    assert [cache.host.by_id[block.block_id].key for block in restore_host] == [key(1), key(2)]
    assert key(3) not in cache.pages
    if local_worker:
        assert copies == [([first.block_id], [victim_host.block_id])]
        assert not offload._pending_finished_stores and not offload._in_flight_store_gpu_blocks
        assert not offload._eviction_store_gpu_ids
        assert first.block_id not in cache.device.by_id
        assert cache.host.by_id[victim_host.block_id] is cache.pages[key(4)]
    else:
        assert offload._eviction_store_gpu_ids == {first.block_id}
        assert offload._pending_finished_stores[0].cpu_block_ids == [victim_host.block_id]


def test_local_worker_evicts_for_host_restore_before_allocating_extend(cache, native_load):
    offload, restore_host = native_load
    cache.share_batch_loads = True
    offload._in_flight_store_gpu_blocks = set()
    offload._pending_finished_stores = []
    offload.enable_kv_cache_events = False
    copies = []
    offload.inline_eviction_writeback = lambda src, _: copies.append(src.copy())
    victims = [insert(cache, (i,))[0].block_id for i in range(3, 8)]
    cache.device.pool.get_new_blocks(2)
    assert cache.device.free_uncached() == 1

    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.tree_cache.cache_controller = offload
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache, block_pool=cache.device.pool)
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_num_active_reqs = 8
    scheduler.dense_kv = True
    scheduler.new_token_ratio = 0.7
    request = _request("restore", (1, 2, 8), 49)

    batch = scheduler._prefer_prefill([], [], [request], [])
    assert batch.waiting == [request] and batch.tokens == 17
    assert batch.planned_host == {block.block_id for block in restore_host}
    assert copies == [victims[:2]]
    assert scheduler._prepare_prefill([], batch)
    assert copies == [victims[:2], victims[2:]]
    assert all(block.block_id in cache.host.by_id for block in restore_host)


def test_native_allocation_preserves_a_later_waiters_host_source(monkeypatch):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

    from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
    from agentinfer.agentcache.mori.schedule_batch import ScheduleBatch
    from agentinfer.agentcache.mori.vllm_compat import mori_block_pool_constructor

    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    page_bytes = spec.page_size_bytes
    config = KVCacheConfig(
        12,
        [KVCacheTensor(page_bytes * 12, ["layer"], page_bytes * 12, page_bytes)],
        [KVCacheGroupSpec(["layer"], spec)],
    )
    settings = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=16),
        kv_events_config=None,
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=True,
            kv_connector_extra_config={
                "cpu_bytes_to_use": page_bytes * 7,
            },
        ),
        max_in_flight_tokens=128,
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(world_size=1, decode_context_parallel_size=1, pipeline_parallel_size=1),
        speculative_config=None,
    )
    with mori_block_pool_constructor():
        manager = MoriKVCacheManager(config, max_model_len=128, scheduler_block_size=16, hash_block_size=16)
    connector = MoriCPUOffloadConnector(settings, KVConnectorRole.SCHEDULER, config)
    connector.bind_kv_cache_manager(manager)
    cold = _request("cold", (100,), 17)
    waiter = _request("restore", (1,), 17)
    state = SimpleNamespace(
        vllm_config=settings,
        kv_cache_config=config,
        kv_cache_manager=manager,
        connector=connector,
        use_v2_model_runner=True,
        defer_block_free=True,
        parallel_config=settings.parallel_config,
        recompute_kv_load_failures=False,
        needs_kv_cache_zeroing=False,
        _free_cow_retained_blocks=lambda *_: None,
        prefix_replay_tokens=0,
        need_mamba_block_aligned_split=False,
        waiting=FCFSRequestQueue([cold, waiter]),
        kv_holding_waiting=FCFSRequestQueue(),
        running=[],
        deferred_waiting=set(),
        max_num_scheduled_tokens=80,
        max_num_active_reqs=8,
        hash_block_size=16,
        block_size=16,
        num_waiting_for_streaming_input=0,
        num_lookahead_tokens=0,
        _pause_state=PauseState.UNPAUSED,
        scheduler_reserve_full_isl=True,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=80,
            enable_chunked_prefill=True,
        ),
        _handle_blocked_waiting_request=lambda _: True,
        _get_local_prefix_cache_hit=lambda request: (*manager.get_computed_blocks(request), False),
        get_request_counts=lambda: (len(scheduler.running), len(scheduler.waiting)),
        log_stats=False,
        connector_prefix_cache_stats=None,
    )
    scheduler = MoriScheduler.__new__(MoriScheduler)
    vars(scheduler).update(vars(state))
    scheduler.block_size = 16
    scheduler._initialize_hiradix()
    tree, offload, gpu = scheduler.tree_cache, scheduler.tree_cache.cache_controller, manager.block_pool
    (prefix,) = insert(tree, (1,))
    (restore_source,) = backup(tree, [prefix])
    gpu.evict_blocks({prefix.block_id})
    canceled_loads = []
    for index in range(5):
        parent_key, child_key = 10 + index, 20 + index
        (parent,) = insert(tree, (parent_key,))
        (child,) = insert(tree, (child_key,), parent=parent_key)
        backup(tree, [child])
        gpu.evict_blocks({child.block_id})
        request = _request(f"canceled-{index}", (parent_key, child_key), 33)
        (destination,) = gpu.get_new_blocks(1)
        gpu.touch([parent])
        assert offload.get_num_new_matched_tokens(request, 16) == (16, False)
        offload.update_state_after_alloc(request, KVCacheBlocks(([parent, destination],)), 16)
        canceled_loads.append((request, parent, destination))
    offload.build_connector_meta(SimpleNamespace(preempted_req_ids=set()))
    for request, parent, destination in canceled_loads:
        request.status = RequestStatus.FINISHED_ABORTED
        offload.request_finished(request, [])
        gpu.free_blocks([parent, destination])
    tree.finish_native_schedule()
    assert tree.device.free_uncached() == 1 and gpu.get_num_free_blocks() == 6
    assert tree.host.free_uncached() == 0 and tree.host.pool.get_num_free_blocks() == 1
    calls = []
    offload.inline_eviction_writeback = lambda src, dst: calls.append((src, dst))
    monkeypatch.setattr(ScheduleBatch, "finish", lambda selected: selected.tokens)

    output, batch = scheduler._schedule_batch([], [], [cold, waiter], [], False)

    assert batch.waiting == [cold, waiter] and batch.planned_host == {restore_source.block_id}
    assert tree.request_match(waiter, touch=False) == ([], [restore_source])
    assert not output and not calls
    assert list(scheduler.waiting) == [cold, waiter]
    assert not tree.prefill_host_sources


@pytest.mark.parametrize("pending", [False, True])
def test_decode_capacity_with_a_free_page_per_request(cache, pending):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.dense_kv = True
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    _test_controller(cache)._eviction_store_gpu_ids = {1} if pending else set()
    count = cache.device.free_uncached()
    assert count > 0
    requests = [_request(str(index), (), 17) for index in range(count)]
    assert scheduler._prepare_decode(requests) is not pending
    assert cache.device.free_uncached() == count


def test_decode_waits_for_writeback_before_preempting(cache):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    cache.device.pool.get_new_blocks(8)
    req = _request("decode", (), 17)
    req.num_computed_tokens = 16
    scheduler.current_step = 10
    scheduler.max_model_len = 256
    scheduler.num_sampled_tokens_per_step = 1
    stage_calls = []

    def stage(count, protected):
        stage_calls.append(count)
        scheduler.tree_cache.cache_controller._eviction_store_gpu_ids.add(1)

    _test_controller(cache)
    cache.stage_evictions = stage
    assert not scheduler._prepare_decode([req])
    assert stage_calls == [1]
    assert not scheduler.failed_decodes


def test_decode_retracts_smallest_output_before_constructing_native_batch(cache):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    occupied = cache.device.pool.get_new_blocks(8)
    old = _request("old", (), 15, max_tokens=10)
    old.append_output_token_ids([30, 31])
    old.num_computed_tokens = 16
    new = _request("new", (), 16, max_tokens=10)
    new.append_output_token_ids(32)
    new.num_computed_tokens = 16
    oldest = _request("oldest", (), 14, max_tokens=10)
    oldest.append_output_token_ids([33, 34, 35])
    oldest.num_computed_tokens = 16
    running = [new, old, oldest]
    waiter = _request("already-waiting", (), 17)
    waiting = FCFSRequestQueue([waiter])
    preempted = []
    evictions = []

    def preempt(victim, timestamp, *, drop_stale_output):
        preempted.append(victim)
        waiting.prepend_request(victim)
        cache.device.pool.free_blocks(occupied[-2:])

    def stage(count, protected):
        evictions.append(count)

    scheduler.current_step = 10
    scheduler.max_model_len = 256
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.running = list(running)
    scheduler.waiting = waiting
    scheduler.deferred_frees = []
    scheduler.requires_kv_delivery = False
    scheduler._request_blocks_can_be_freed = lambda _: True
    scheduler._preempt_request = preempt
    _test_controller(cache)
    cache.stage_evictions = stage
    assert scheduler._prepare_decode(running)
    assert running == [oldest, old]
    assert list(waiting) == [waiter, new]
    assert preempted == [new]
    assert evictions == [3, 3]


def test_decode_aborts_last_request_when_no_page_can_be_reclaimed(cache, step_without_worker):
    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.failed_decodes = set()
    scheduler.dense_kv = True
    scheduler.new_token_ratio = 0.7
    cache.device.pool.get_new_blocks(8)
    req = _request("decode", (), 17)
    req.num_computed_tokens = 16

    def finish(ids, status):
        assert ids == {req.request_id}
        req.status = status
        return [req]

    scheduler.current_step = 10
    scheduler.max_model_len = 256
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.running = [req]
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_active_reqs = 1
    scheduler.max_num_scheduled_tokens = 16
    scheduler.deferred_frees = []
    scheduler._request_blocks_can_be_freed = lambda _: True
    scheduler.finish_requests = finish
    cache.stage_evictions = lambda *args: None
    step_without_worker(scheduler)
    assert scheduler.new_token_ratio == 0.0
    outputs = {}
    scheduler._finish_failed_decodes(outputs)
    assert outputs[0].outputs[0].finish_reason == req.get_finished_reason()
    assert req.status == RequestStatus.FINISHED_ERROR
    assert not scheduler.failed_decodes


@pytest.mark.parametrize("reject_prefill", ["not_prepared", "no_slots", "pending_writeback"])
def test_direct_schedule_falls_back_to_decode_on_real_queues(cache, monkeypatch, reject_prefill):
    from vllm.v1.core.sched.output import CachedRequestData

    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.dense_kv = False
    decode = _request("decode", (), 18)
    decode.status = RequestStatus.RUNNING
    decode.num_computed_tokens = 17
    chunk = _request("chunk", (), 40)
    chunk.status = RequestStatus.RUNNING
    chunk.is_prefill_chunk = True
    chunk.num_computed_tokens = 8
    waiter = _request("waiting", (), 18)
    running = [decode, chunk]
    waiting = FCFSRequestQueue([waiter])
    holding = FCFSRequestQueue()
    updates = []
    native_step = []
    controller_metadata = object()
    used = []
    kv = KVCacheBlocks(([],))

    def allocate_slots(request, tokens, **kwargs):
        assert scheduler.running is running
        assert scheduler.waiting is waiting and scheduler.kv_holding_waiting is holding
        assert scheduler.max_num_scheduled_tokens == 32
        used.append((request, tokens))
        return kv if request is decode else None

    def prepare_prefill(*_):
        if reject_prefill == "pending_writeback":
            scheduler.tree_cache.cache_controller._eviction_store_gpu_ids.add(1)
        return reject_prefill == "no_slots"

    _test_controller(cache)
    monkeypatch.setattr(
        scheduler, "_plan_prefill", lambda *args: PrefillBatch(tokens=16, token_caps={chunk.request_id: 16})
    )
    monkeypatch.setattr(scheduler, "_prepare_prefill", prepare_prefill)
    scheduler.current_step = 0
    scheduler.max_model_len = 256
    scheduler.block_size = 16
    scheduler.num_lookahead_tokens = 0
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.running = running
    scheduler.requests = {r.request_id: r for r in [decode, chunk, waiter]}
    scheduler.waiting = waiting
    scheduler.kv_holding_waiting = holding
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.max_num_scheduled_tokens = 32
    scheduler.max_num_running_reqs = 2
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_active_reqs = 10
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=32)
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[0])
    scheduler.kv_cache_manager = SimpleNamespace(
        hiradix=cache,
        new_step_starts=lambda: native_step.append(1),
        allocate_slots=allocate_slots,
        get_num_common_prefix_blocks=lambda _: [0],
        take_boundary_state_offloads=dict,
        get_block_ids=lambda _: ([],),
        take_kv_cache_block_copies=lambda: ([], []),
    )
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=list, get_manager_metadata=lambda: None)
    scheduler.deferred_waiting = set()
    scheduler.prefill_capacity_bound = False
    scheduler.use_v2_model_runner = True
    scheduler.model_uses_mrope = False
    scheduler.connector = object()
    scheduler.reset_preempted_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.defer_block_free = True
    scheduler.sched_step_seq = 4
    scheduler._get_new_block_ids_to_zero = lambda: None
    scheduler._build_kv_connector_meta = lambda *args: controller_metadata
    scheduler._make_cached_request_data = lambda *args: CachedRequestData.make_empty()
    scheduler._update_after_schedule = updates.append
    scheduler.get_request_counts = lambda: (len(running), len(waiting))

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"decode": 1}
    assert output.kv_connector_metadata is controller_metadata
    assert output.kv_connector_block_state is None
    assert scheduler.current_step == 1
    assert scheduler.sched_step_seq == 5
    assert updates == [output] and native_step == [1]
    assert running == [decode, chunk]
    assert list(waiting) == [waiter]
    assert [request for request, _ in used] == ([chunk, decode] if reject_prefill == "no_slots" else [decode])


@pytest.mark.parametrize("prefix_state", ["unchanged", "evicted", "attached"])
def test_prefill_commit_uses_each_requests_planned_extend(cache, monkeypatch, prefix_state):
    from agentinfer.agentcache.mori.schedule_batch import ScheduleBatch

    prefix = insert(cache, (1, 2))
    first = _request("long-prefix", (1, 2), 33)
    second = _request("cold", (3,), 17)
    waiting = FCFSRequestQueue([first, second])
    holding = FCFSRequestQueue()
    if prefix_state == "attached":
        cache.device.pool.touch(prefix[:1])
        first.num_computed_tokens = 16
        waiting.remove_request(first)
        holding.add_request(first)

    empty = KVCacheBlocks(([],))

    def lookup(request):
        blocks, _ = cache.match_for_native_admission(request)
        return KVCacheBlocks((blocks,)), len(blocks) * 16, 0, False

    def prepare(_chunked, _batch):
        if prefix_state == "evicted":
            # Eviction invalidates the prepared match before slot allocation.
            cache.device.pool.evict_blocks({block.block_id for block in prefix})
            assert cache.request_match(first) == ([], [])
        return True

    scheduler = MoriScheduler.__new__(MoriScheduler)
    scheduler.kv_cache_manager = SimpleNamespace(hiradix=cache)
    scheduler.new_token_ratio = 0.7
    scheduler.failed_decodes = set()
    scheduler.kv_cache_manager = SimpleNamespace(
        hiradix=cache,
        block_pool=cache.device.pool,
        empty_kv_cache_blocks=empty,
        allocate_slots=lambda *_args, **_kwargs: empty,
        get_blocks=lambda _: empty,
        record_prefix_cache_stats=lambda *_: None,
    )
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_active_reqs = 8
    scheduler.max_num_scheduled_tokens = 64
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=64, enable_chunked_prefill=True)
    scheduler.running = []
    scheduler.waiting = waiting
    scheduler.kv_holding_waiting = holding
    scheduler.deferred_waiting = set()
    scheduler.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda *_: (0, False), update_state_after_alloc=lambda *_: None
    )
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.num_lookahead_tokens = 0
    scheduler.scheduler_reserve_full_isl = False
    scheduler.block_size = 16
    scheduler._get_local_prefix_cache_hit = lookup
    scheduler._handle_blocked_waiting_request = lambda _: True
    scheduler.get_request_counts = lambda: (len(scheduler.running), len(waiting) + len(holding))
    scheduler.connector_prefix_cache_stats = None
    scheduler.log_stats = False
    scheduler._inflight_prefills = set()
    monkeypatch.setattr(scheduler, "_prepare_prefill", prepare)
    monkeypatch.setattr(ScheduleBatch, "finish", lambda selected: selected.tokens)

    output, plan = scheduler._schedule_batch([], [], list(waiting), list(holding), False)

    first_extend = 17 if prefix_state == "attached" else 1
    assert output == {first.request_id: first_extend, second.request_id: 17}
    assert plan.tokens == first_extend + 17
