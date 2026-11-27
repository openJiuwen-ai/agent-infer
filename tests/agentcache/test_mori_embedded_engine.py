# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise the embedded owner-thread control path on R11's two real CPU-backed test pools."""

from types import SimpleNamespace

import pytest

try:
    from vllm.v1.core.sched.output import KVConnectorBlockState as _KVConnectorBlockState  # noqa: F401
except ImportError:
    pytest.skip("MORI engine tests require the native KVConnectorBlockState ABI", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.mori.admission import InvalidMoriAdmission, ProgramOperation
from agentinfer.agentcache.mori.block_pool import MoriBlockPool
from agentinfer.agentcache.mori.bridge import MoriAsyncSchedulerBridge, MoriSyncSchedulerBridge
from agentinfer.agentcache.mori.hiradix import HiRadix
from agentinfer.agentcache.mori.ipc import ATTEMPT_ARG, CHILD_COUNT_ARG, CONTEXT_CHARS_ARG, PREFIX_ARG, UTILITY_METHOD
from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager
from agentinfer.agentcache.mori.programs import PROGRAM_ARG, CacheClass, request_program
from agentinfer.agentcache.mori.scheduler import MoriScheduler
from agentinfer.scheduling import runtime
from agentinfer.scheduling.transitions import ProgramPlacement
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


@pytest.fixture(params=[MoriSyncSchedulerBridge, MoriAsyncSchedulerBridge])
def scheduler(request, monkeypatch):
    device = MoriBlockPool(num_gpu_blocks=76, enable_caching=True, hash_block_size=4)
    host = MoriBlockPool(num_gpu_blocks=41, enable_caching=True, hash_block_size=4)
    tree = HiRadix(device, host, 4)
    device.bind_hiradix_index(tree.device)
    host.bind_hiradix_index(tree.host)
    manager = MoriKVCacheManager.__new__(MoriKVCacheManager)
    manager.hiradix = tree
    manager._radix_depth = {}
    manager.block_pool = device

    def initialize_engine(owner, config, kv_config):
        owner.kv_cache_manager = manager
        owner.block_size = 4
        owner.vllm_config = config
        owner.kv_cache_config = kv_config
        owner.requests = {}
        owner.waiting = FCFSRequestQueue()
        owner.log_stats = False
        owner.spec_decode_metrics_level = "none"
        owner.connector = None

    monkeypatch.setattr(MoriScheduler, "__init__", initialize_engine)
    return request.param(
        SimpleNamespace(additional_config={"agentcache": {"strategy": "mori"}}), SimpleNamespace(num_blocks=9999)
    )


def native(rid, extra):
    return Request(rid, list(range(12)), SamplingParams(extra_args=extra), None)


def chat(rid, attempt, program_id, chars=100, n=1):
    return native(
        rid,
        {
            ATTEMPT_ARG: attempt,
            PROGRAM_ARG: program_id,
            CONTEXT_CHARS_ARG: chars,
            PREFIX_ARG: "_empty_",
            CHILD_COUNT_ARG: n,
        },
    )


def publish(owner, request, page_key):
    manager = owner.kv_cache_manager
    pool = manager.block_pool
    (block,) = pool.get_new_blocks(1)
    key = BlockHash(page_key)
    pool._insert_block_hash(make_block_hash_with_group_id(key, 0), block, num_tokens=4)
    manager.coordinator = SimpleNamespace(get_blocks=lambda rid: ([block],))
    request.block_hashes = [key]
    manager.publish_hiradix(request, 4, refresh_existing=True)
    pool.free_blocks([block])
    return block


def test_old_dispatch_after_release_and_retry_cannot_claim_reused_pages(scheduler):
    core = SimpleNamespace(scheduler=scheduler)
    handle = getattr(EngineCore, UTILITY_METHOD)
    tree = scheduler.tree_cache
    adapter = scheduler.program_cache_adapter
    first = chat("first-child", "old", "p", n=2)
    scheduler.add_request(first)
    old = request_program(first)
    tree.stamp_prefill(None, old)
    publish(scheduler, first, b"first")
    assert handle(core, "release", ["p"])
    replacement = chat("new-child", "new", "p")
    scheduler.add_request(replacement)
    current = request_program(replacement)
    tree.stamp_prefill(None, current)
    assert current.generation == old.generation + 1
    late = chat("late-child", "old", "p", n=2)
    scheduler.add_request(late)
    assert request_program(late) == old
    assert list(scheduler.waiting) == [first, replacement, late]
    publish(scheduler, replacement, b"new")
    tree.stamp_prefill(b"new", old)
    publish(scheduler, late, b"late")
    operations = (ProgramOperation(old, None), ProgramOperation(old, ProgramPlacement.INTERACTIVE))
    for _ in range(2):
        adapter.apply_program_ops(operations)
    assert not tree.pages[b"new"].shared
    assert tree.placement.type_for_id(old) is CacheClass.INACTIVE
    assert tree.placement.type_for_id(current) is CacheClass.BUSY
    snapshot = EngineCore.agentinfer_mori_diagnostics(core, "cache")
    assert snapshot["placements"] == [{"program_id": "p", "generation": current.generation, "class": "busy"}]
    assert [(row["generation"], row["hbm_pages"]) for row in snapshot["owners"]] == [
        (old.generation, 2),
        (current.generation, 1),
    ]
    tree.stamp_prefill(b"first", current)
    assert tree.pages[b"first"].program == current
    assert not tree.pages[b"first"].shared
    handle(core, "finish", ["old", 1000, 1000, 0])
    assert EngineCore.agentinfer_mori_diagnostics(core, "programs") == {"p": "active"}


def test_unmanaged_native_add_and_expired_dispatch_do_not_change_ownership(scheduler):
    bridge = scheduler._agentcache.program_admission
    unsigned = native("unsigned", {PROGRAM_ARG: "victim", "generation": 8})
    scheduler.add_request(unsigned)
    publish(scheduler, unsigned, b"unsigned")
    assert scheduler.tree_cache.pages[b"unsigned"].program is None
    actual = chat("matching", "attempt", "owner", n=2)
    scheduler.add_request(actual)
    forged = chat("wrong", "attempt", "other", n=2)
    with pytest.raises(InvalidMoriAdmission):
        scheduler.add_request(forged)
    sibling = chat("sibling", "attempt", "owner", n=2)
    scheduler.add_request(sibling)
    assert request_program(actual) == request_program(sibling) == bridge.runtime.registry.current_ref("owner")

    aborts = []
    core = SimpleNamespace(
        _reject_add_in_shutdown=lambda req: False,
        add_request=lambda req, wave: scheduler.add_request(req),
        _send_abort_outputs_to_client=lambda ids, index: aborts.append((ids, index)),
    )
    expired = chat("expired", "attempt", "owner", n=2)
    EngineCoreProc._handle_client_request(core, EngineCoreRequestType.ADD, (expired, 0))
    assert aborts == [(["expired"], 0)]
    assert set(scheduler.requests) == {"matching", "sibling", "unsigned"}


def test_periodic_placement_and_retained_abort_use_real_pool_capacity(scheduler, monkeypatch):
    core = SimpleNamespace(scheduler=scheduler)
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock.now))
    config = EngineCore.agentinfer_mori_diagnostics(core, "metrics")["backends"]["vllm-local"]["cache_config"]
    assert config == {"total_tokens_capacity": 300, "cpu_total_tokens_capacity": 160}
    bridge = scheduler._agentcache.program_admission
    scheduler.add_request(chat("first", "first", "A", 500))
    clock.now = 2.0
    bridge.finish("first", 160, 100, 0)
    req = chat("next", "next", "B", 500)
    scheduler.add_request(req)
    assert "next" not in scheduler.requests
    clock.now = 4.0
    assert bridge.tick()
    assert scheduler.requests["next"] is req
    assert scheduler.tree_cache.placement.type_for_id("A") is CacheClass.IDLE
    assert scheduler.tree_cache.placement.type_for_id("B") is CacheClass.BUSY

    pending = chat("queued", "queued", "C", 10000)
    scheduler.add_request(pending)
    assert "queued" not in scheduler.requests
    # Report a retained v0.31 Request to the native pause/abort consumer.
    assert scheduler.finish_requests(["queued"], RequestStatus.FINISHED_ABORTED) == [pending]
    assert not bridge.take_aborted_requests()
    assert not bridge.tick()
    assert "queued" not in scheduler.requests


def test_partial_placement_retry_on_real_tree_and_release_after_cold_demotion(scheduler, monkeypatch):
    clock = SimpleNamespace(now=1.0)
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock.now))
    bridge = scheduler._agentcache.program_admission
    scheduler.add_request(chat("a", "a", "A", 500))
    clock.now = 2.0
    bridge.finish("a", 160, 100, 0)
    pending = chat("b", "b", "B", 500)
    scheduler.add_request(pending)
    assert "b" not in scheduler.requests
    cache = scheduler.program_cache_adapter
    apply = cache.apply_program_ops
    applied = []

    def apply_with_interruption(ops):
        applied.append(ops)
        if len(applied) == 1:
            apply(ops[:1])
            raise OSError("interrupted after first mark")
        apply(ops)

    monkeypatch.setattr(cache, "apply_program_ops", apply_with_interruption)
    clock.now = 4.0
    assert not bridge.tick() and "b" not in scheduler.requests
    assert scheduler.tree_cache.placement.type_for_id("A") is CacheClass.IDLE
    assert bridge.tick() and scheduler.requests["b"] is pending
    assert applied[0] == applied[1]
    assert scheduler.tree_cache.placement.type_for_id("B") is CacheClass.BUSY

    # Losing one host page moves A from the warm list to cold waiting.
    monkeypatch.setattr(cache.tree.host.pool, "num_gpu_blocks", 40)
    bridge.observe_backend()
    assert bridge.tick()
    assert cache.tree.placement.has_program("A")
    assert cache.tree.placement.type_for_id("A") is CacheClass.INACTIVE
    assert bridge.release("A")
    assert not cache.tree.placement.has_program("A")


def test_shutdown_resolves_pending_admission_before_releasing_engine(scheduler, monkeypatch):
    bridge = scheduler._agentcache.program_admission
    scheduler.add_request(chat("a", "a", "A", 500))
    waiting = chat("b", "b", "B", 10000)
    scheduler.add_request(waiting)
    assert "b" not in scheduler.requests
    released = []

    def release_engine(owner):
        assert bridge.finish_requests(None, RequestStatus.FINISHED_ABORTED) == []
        assert owner.tree_cache.placement.current_ref is None
        released.append(owner)

    monkeypatch.setattr(MoriScheduler, "shutdown", release_engine)
    scheduler.shutdown()
    assert released == [scheduler]


def test_failed_admission_setup_closes_the_initialized_engine(scheduler, monkeypatch):
    from agentinfer.agentcache.mori import bridge as bridge_module

    released = []

    def fail(*args):
        raise ValueError("invalid admission config")

    monkeypatch.setattr(bridge_module, "_VllmAdmissionHooks", fail)
    monkeypatch.setattr(MoriScheduler, "shutdown", lambda owner: released.append(owner))
    with pytest.raises(ValueError, match="invalid admission config"):
        type(scheduler)(scheduler.vllm_config, scheduler.kv_cache_config)
    (owner,) = released
    assert owner is not scheduler
    assert owner._agentcache is owner.program_cache_adapter is None
