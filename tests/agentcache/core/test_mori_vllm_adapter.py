# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Capability and cleanup tests for MORI's vLLM integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from agentinfer.agentcache.core.scheduler import (
    AgentCacheAsyncSchedulerBridge as PublicAsyncSchedulerBridge,
    AgentCacheSyncSchedulerBridge as PublicSyncSchedulerBridge,
)
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori.scheduler import (
    AgentCacheAsyncSchedulerBridge,
    AgentCacheSyncSchedulerBridge,
    _VllmAdmissionHooks,
)
from agentinfer.agentcache.mori.vllm_adapter import (
    VllmMORIAdapter,
    VllmOffloadObservation,
    install_vllm_mori,
    observe_vllm_offload,
)
from agentinfer.scheduling import ProgramRef, ProgramStatus
from agentinfer.scheduling.mori import MORIConfig, MORIMode, build_mori_strategy

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def reset_runtime_state():
    MORI_RUNTIME_STATE.reset()
    yield
    MORI_RUNTIME_STATE.reset()


def full_native_owner():
    """Provide the full adapter's required CPU and HBM registration contracts."""
    cpu_policy = SimpleNamespace(MORI_CPU_POLICY=True, close=Mock())
    return SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=SimpleNamespace(manager=SimpleNamespace(_policy=cpu_policy))),
        kv_cache_manager=SimpleNamespace(
            register_mori_block_selection_provider=lambda provider: {"hook_api_version": provider.HOOK_API_VERSION},
            unregister_mori_block_selection_provider=Mock(),
        ),
    )


@pytest.fixture
def mock_bridge_native(monkeypatch):
    """Exercise full-mode request handling with a zero-capacity fake CPU tier."""

    def install(owner, strategy):
        MORI_RUNTIME_STATE.bind_provider(strategy)
        return VllmMORIAdapter(hbm_policy=None, cpu_policy=None, owner=owner, strategy=strategy)

    monkeypatch.setattr("agentinfer.agentcache.mori.scheduler.install_vllm_mori", install)
    monkeypatch.setattr(
        "agentinfer.agentcache.mori.scheduler.observe_vllm_offload",
        lambda owner: VllmOffloadObservation(total_dram_kv_tokens=0, dram_chunk_size_tokens=16),
    )


@pytest.mark.parametrize(
    ("public_bridge", "mori_bridge", "async_enabled"),
    [
        (PublicAsyncSchedulerBridge, AgentCacheAsyncSchedulerBridge, True),
        (PublicSyncSchedulerBridge, AgentCacheSyncSchedulerBridge, False),
    ],
)
def test_public_bridge_constructs_real_mori_scheduler(public_bridge, mori_bridge, async_enabled, mock_bridge_native):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_enabled),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori"}},
    )

    def initialize(owner, *args, **kwargs):
        owner.requests = {}
        owner.kv_cache_manager = None

    with (
        patch.object(Scheduler, "__init__", initialize),
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "get_request_counts", return_value=(0, 0)),
    ):
        bridge = public_bridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)
    assert type(bridge) is mori_bridge
    assert bridge._agentcache._mori_adapter is not None
    bridge._agentcache.close()


def test_full_adapter_can_close_and_rebind_in_process() -> None:
    first = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    second = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    owner = full_native_owner()

    adapter = install_vllm_mori(owner, first)
    MORI_RUNTIME_STATE.bind_request("shutdown-request", ProgramRef("shutdown-program", 0))
    with pytest.raises(RuntimeError, match="already bound"):
        install_vllm_mori(owner, second)
    adapter.close()
    assert MORI_RUNTIME_STATE.program_for_request("shutdown-request") is None
    rebound = install_vllm_mori(owner, second)
    rebound.close()


def test_failed_mori_startup_does_not_install_prefix_observer(mock_bridge_native) -> None:
    """Failed capability negotiation must leave the native lookup unchanged."""

    def lookup(request):
        return None

    owner = SimpleNamespace(kv_cache_manager=SimpleNamespace(get_computed_blocks=lookup))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "full"}}},
    )
    with patch("agentinfer.agentcache.mori.scheduler.install_vllm_mori", side_effect=RuntimeError("missing hook")):
        with pytest.raises(RuntimeError, match="missing hook"):
            _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=10), 16)
    assert owner.kv_cache_manager.get_computed_blocks is lookup


def test_full_adapter_rejects_missing_hbm_hook_without_leaking_provider() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    replacement = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy

    class CPUPolicy:
        MORI_CPU_POLICY = True

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    cpu_policy = CPUPolicy()
    owner = SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=SimpleNamespace(manager=SimpleNamespace(_policy=cpu_policy))),
        kv_cache_manager=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="register_mori_block_selection_provider"):
        install_vllm_mori(owner, full)
    assert cpu_policy.closed is True

    adapter = install_vllm_mori(full_native_owner(), replacement)
    adapter.close()


def test_full_adapter_unregisters_hbm_and_cpu_indexes_on_close() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    calls: list[tuple[str, object]] = []

    class CPUPolicy:
        MORI_CPU_POLICY = True

        def close(self) -> None:
            calls.append(("cpu_close", self))

    cpu_policy = CPUPolicy()

    class KVManager:
        def register_mori_block_selection_provider(self, provider):
            calls.append(("register", provider))
            return {"hook_api_version": 1}

        def unregister_mori_block_selection_provider(self, provider):
            calls.append(("unregister", provider))

    owner = SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=SimpleNamespace(manager=SimpleNamespace(_policy=cpu_policy))),
        kv_cache_manager=KVManager(),
    )

    adapter = install_vllm_mori(owner, full)
    adapter.close()
    adapter.close()

    assert [name for name, _ in calls] == ["register", "unregister", "cpu_close"]


def test_full_adapter_installs_and_uninstalls_simple_cpu_provider() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    calls: list[tuple[str, object]] = []

    class SimpleCPUScheduler:
        def register_mori_offload_tier_provider(self, version, provider):
            calls.append(("cpu_register", (version, provider)))
            return {"hook_api_version": version}

        def unregister_mori_offload_tier_provider(self, provider):
            calls.append(("cpu_unregister", provider))

    class KVManager:
        def register_mori_block_selection_provider(self, provider):
            calls.append(("hbm_register", provider))
            return {"hook_api_version": provider.HOOK_API_VERSION}

        def unregister_mori_block_selection_provider(self, provider):
            calls.append(("hbm_unregister", provider))

    owner = SimpleNamespace(
        connector=SimpleNamespace(scheduler_manager=SimpleCPUScheduler()),
        kv_cache_manager=KVManager(),
    )

    adapter = install_vllm_mori(owner, full)
    cpu_policy = adapter.cpu_policy
    assert getattr(cpu_policy, "HOOK_API_VERSION", None) == 3
    adapter.close()
    adapter.close()

    assert [name for name, _ in calls] == [
        "cpu_register",
        "hbm_register",
        "hbm_unregister",
        "cpu_unregister",
    ]


def test_simple_cpu_snapshot_is_normalized_to_tokens() -> None:
    snapshot = SimpleNamespace(
        total_blocks=100,
        resident_blocks=40,
        pending_store_blocks=3,
        pending_load_blocks=2,
        scheduler_block_size=16,
    )
    owner = SimpleNamespace(
        connector=SimpleNamespace(
            scheduler_manager=SimpleNamespace(
                get_mori_offload_tier_snapshot=lambda: snapshot,
            )
        )
    )

    observed = observe_vllm_offload(owner)

    assert observed.total_dram_kv_tokens == 1600
    assert observed.used_dram_kv_tokens == 640
    assert observed.pending_d2h_kv_tokens == 48
    assert observed.pending_h2d_kv_tokens == 32
    assert observed.dram_chunk_size_tokens == 16


def test_simple_cpu_snapshot_rejects_impossible_residency() -> None:
    snapshot = SimpleNamespace(
        total_blocks=4,
        resident_blocks=5,
        pending_store_blocks=0,
        pending_load_blocks=0,
        scheduler_block_size=16,
    )
    owner = SimpleNamespace(
        connector=SimpleNamespace(
            scheduler_manager=SimpleNamespace(
                get_mori_offload_tier_snapshot=lambda: snapshot,
            )
        )
    )

    assert observe_vllm_offload(owner).total_dram_kv_tokens is None


def test_simple_cpu_registration_rolls_back_on_hbm_failure() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    calls: list[str] = []

    class SimpleCPUScheduler:
        def register_mori_offload_tier_provider(self, version, provider):
            calls.append("cpu_register")
            return {"hook_api_version": version}

        def unregister_mori_offload_tier_provider(self, provider):
            calls.append("cpu_unregister")

    owner = SimpleNamespace(
        connector=SimpleNamespace(scheduler_manager=SimpleCPUScheduler()),
        kv_cache_manager=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="register_mori_block_selection_provider"):
        install_vllm_mori(owner, full)

    assert calls == ["cpu_register", "cpu_unregister"]
    rebound = install_vllm_mori(
        full_native_owner(),
        build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy,
    )
    rebound.close()


def test_full_adapter_cleans_process_state_when_native_unregister_fails() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy

    class CPUPolicy:
        MORI_CPU_POLICY = True

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class KVManager:
        @staticmethod
        def register_mori_block_selection_provider(provider):
            return {"hook_api_version": provider.HOOK_API_VERSION}

        @staticmethod
        def unregister_mori_block_selection_provider(provider):
            raise RuntimeError("native unregister failed")

    cpu_policy = CPUPolicy()
    owner = SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=SimpleNamespace(manager=SimpleNamespace(_policy=cpu_policy))),
        kv_cache_manager=KVManager(),
    )
    adapter = install_vllm_mori(owner, full)
    MORI_RUNTIME_STATE.bind_request("request", ProgramRef("program", 0))

    with pytest.raises(RuntimeError, match="native unregister failed"):
        adapter.close()

    assert cpu_policy.closed is True
    assert MORI_RUNTIME_STATE.program_for_request("request") is None
    rebound = install_vllm_mori(
        full_native_owner(),
        build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy,
    )
    rebound.close()


@pytest.mark.parametrize("bridge_type", [AgentCacheSyncSchedulerBridge, AgentCacheAsyncSchedulerBridge])
@pytest.mark.parametrize("invalid", ["missing", "malformed", "oversized"])
def test_mori_rejects_only_invalid_request_and_delivers_terminal_output(
    bridge_type, invalid, mock_bridge_native
) -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=bridge_type is AgentCacheAsyncSchedulerBridge),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "full"}}},
    )

    def initialize(owner, *args, **kwargs):
        owner.requests = {}
        owner.kv_cache_manager = None

    def native_add(owner, request):
        owner.requests[request.request_id] = request

    def request(name, extra):
        return SimpleNamespace(
            request_id=name,
            client_index=7,
            num_prompt_tokens=100,
            num_tokens=100,
            num_output_tokens=0,
            sampling_params=SimpleNamespace(extra_args=extra),
            trace_headers={},
            events=[],
        )

    identity = {"agentic_context": {"program_id": "program"}}
    good = request("good", identity)
    bad = request("bad", None if invalid == "missing" else {"agentic_context": "{"})
    if invalid == "oversized":
        bad = request("bad", identity)
        bad.num_prompt_tokens = 1601
        bad.num_tokens = 1601
    with (
        patch.object(Scheduler, "__init__", initialize),
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "add_request", native_add),
        patch.object(Scheduler, "get_request_counts", return_value=(0, 0)),
        patch.object(Scheduler, "get_num_unfinished_requests", return_value=0),
        patch.object(Scheduler, "update_from_output", side_effect=[{}, {}]),
        patch.object(AsyncScheduler, "update_from_output", side_effect=[{}, {}]),
    ):
        bridge = bridge_type(config, SimpleNamespace(num_blocks=100), object(), 16, 16)
        bridge.add_request(bad)
        assert bad.status is RequestStatus.FINISHED_ERROR
        assert "bad" not in bridge.requests
        assert bridge._agentcache.controller.program_ref_for_request("bad") is None
        assert MORI_RUNTIME_STATE.program_for_request("bad") is None
        # A lone rejected request must keep EngineCore polling until its error is sent.
        assert bridge.has_unfinished_requests()
        outputs = bridge.update_from_output(SimpleNamespace(), SimpleNamespace(req_ids=[]))
        assert outputs[7].outputs[0].finish_reason is FinishReason.ERROR
        assert outputs[7].finished_requests == {"bad"}
        assert bridge.update_from_output(SimpleNamespace(), SimpleNamespace(req_ids=[])) == {}
        assert bridge._agentcache._rejected_requests == []
        bridge.add_request(good)
        assert bridge.requests["good"] is good
        terminal = {7: EngineCoreOutputs(outputs=[EngineCoreOutput("good", [], finish_reason=FinishReason.STOP)])}
        bridge._agentcache.after_output(bridge, terminal, {})
        assert MORI_RUNTIME_STATE.program_for_request("good") is None
        assert bridge._agentcache.controller.program_ref_for_request("good") is None
        bridge._agentcache.close()


@pytest.mark.parametrize("bridge_type", [AgentCacheSyncSchedulerBridge, AgentCacheAsyncSchedulerBridge])
def test_mori_bridge_accepts_overlapping_requests_for_one_program(bridge_type, mock_bridge_native) -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=bridge_type is AgentCacheAsyncSchedulerBridge),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "full"}}},
    )

    def initialize(owner, *args, **kwargs):
        owner.requests = {}
        owner.kv_cache_manager = None

    def native_add(owner, request):
        owner.requests[request.request_id] = request

    def request(name):
        return SimpleNamespace(
            request_id=name,
            client_index=7,
            num_prompt_tokens=100,
            num_tokens=100,
            num_output_tokens=0,
            sampling_params=SimpleNamespace(extra_args={"agentic_context": {"program_id": "program"}}),
            trace_headers={},
            events=[],
        )

    with (
        patch.object(Scheduler, "__init__", initialize),
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "add_request", native_add),
    ):
        bridge = bridge_type(config, SimpleNamespace(num_blocks=100), object(), 16, 16)
        first = request("first")
        second = request("second")
        bridge.add_request(first)
        bridge.add_request(second)

        first_ref = MORI_RUNTIME_STATE.program_for_request("first")
        assert first_ref is not None
        second_ref = MORI_RUNTIME_STATE.program_for_request("second")
        assert second_ref is not None and second_ref != first_ref
        assert bridge.requests == {"first": first, "second": second}

        first_output = {7: EngineCoreOutputs(outputs=[EngineCoreOutput("first", [], finish_reason=FinishReason.STOP)])}
        bridge._agentcache.after_output(bridge, first_output, {})
        assert MORI_RUNTIME_STATE.program_for_request("first") is None
        assert MORI_RUNTIME_STATE.program_for_request("second") == second_ref
        assert bridge._agentcache.controller.registry.require(second_ref).status is ProgramStatus.REASONING
        assert bridge._agentcache.controller.registry.require(first_ref).status is ProgramStatus.ACTING

        second_output = {
            7: EngineCoreOutputs(outputs=[EngineCoreOutput("second", [], finish_reason=FinishReason.STOP)])
        }
        bridge._agentcache.after_output(bridge, second_output, {})
        assert MORI_RUNTIME_STATE.program_for_request("second") is None
        assert bridge._agentcache.controller.registry.require(first_ref).status is ProgramStatus.ACTING
        bridge._agentcache.close()


def test_simple_cpu_bind_failure_does_not_leak_an_index_or_unbind_existing_owner() -> None:
    first = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    adapter = install_vllm_mori(full_native_owner(), first)
    existing_indexes = tuple(MORI_RUNTIME_STATE._indexes)
    MORI_RUNTIME_STATE.bind_request("existing", ProgramRef("existing", 0))
    second = build_mori_strategy().strategy
    owner = SimpleNamespace(
        connector=SimpleNamespace(
            scheduler_manager=SimpleNamespace(
                register_mori_offload_tier_provider=lambda *args: {"hook_api_version": 1},
                unregister_mori_offload_tier_provider=lambda provider: None,
            )
        )
    )
    with pytest.raises(RuntimeError, match="already bound"):
        install_vllm_mori(owner, second)
    assert tuple(MORI_RUNTIME_STATE._indexes) == existing_indexes
    assert MORI_RUNTIME_STATE._provider is first
    assert MORI_RUNTIME_STATE.program_for_request("existing") is not None
    adapter.close()


@pytest.mark.parametrize("returns_requests", [False, True])
def test_retained_abort_uses_host_contract_and_releases_program_binding(
    monkeypatch, returns_requests, mock_bridge_native
) -> None:
    """Retained-only aborts preserve the installed host's return contract and clean bindings."""
    monkeypatch.setattr("agentinfer.agentcache.mori.scheduler._NATIVE_FINISH_RETURNS_REQUESTS", returns_requests)
    owner = SimpleNamespace(requests={}, kv_cache_manager=None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "full"}}},
    )
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=13), 16)
    request = SimpleNamespace(
        request_id="retained",
        client_index=7,
        num_prompt_tokens=100,
        sampling_params=SimpleNamespace(extra_args={"agentic_context": {"program_id": "program"}}),
        trace_headers={},
        events=[],
    )
    native = []
    resident = SimpleNamespace(
        request_id="resident",
        num_prompt_tokens=100,
        sampling_params=SimpleNamespace(extra_args={"agentic_context": {"program_id": "resident"}}),
        trace_headers={},
        events=[],
    )
    # Retain because capacity is occupied, not because the prompt can never fit.
    hooks.add_request(owner, resident, native.append)
    hooks.add_request(owner, request, native.append)
    assert native == [resident]
    # Requests outside native scheduling deliberately have no native cache
    # binding; the previous Program context stays protected until dispatch.
    assert MORI_RUNTIME_STATE.program_for_request("retained") is None
    finished = hooks.finish_requests(owner, "retained", RequestStatus.FINISHED_ABORTED, lambda ids, status: [])
    assert finished == ([request] if returns_requests else [("retained", 7)])
    assert request.status is RequestStatus.FINISHED_ABORTED
    assert hooks.controller.retained_request_count == 0
    assert hooks.controller.program_ref_for_request("retained") is None
    assert MORI_RUNTIME_STATE.program_for_request("retained") is None
    hooks.close()


@pytest.mark.parametrize("returns_requests", [False, True])
def test_running_abort_keeps_lease_until_native_finish_confirmation(
    monkeypatch, returns_requests, mock_bridge_native
) -> None:
    """A cancellation intent alone cannot convert an executing lease into Acting."""
    monkeypatch.setattr("agentinfer.agentcache.mori.scheduler._NATIVE_FINISH_RETURNS_REQUESTS", returns_requests)
    owner = SimpleNamespace(requests={}, kv_cache_manager=None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "full"}}},
    )
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=33), 16)
    request = SimpleNamespace(
        request_id="running",
        client_index=7,
        num_prompt_tokens=100,
        num_tokens=120,
        sampling_params=SimpleNamespace(extra_args={"agentic_context": {"program_id": "program"}}),
        trace_headers={},
        events=[],
    )
    hooks.add_request(owner, request, lambda req: owner.requests.update({req.request_id: req}))
    ref = hooks.controller.program_ref_for_request("running")
    assert hooks.finish_requests(owner, "running", RequestStatus.FINISHED_ABORTED, lambda ids, status: []) == []
    assert hooks.controller.program_ref_for_request("running") == ref
    assert hooks.controller.registry.require(ref).status is ProgramStatus.REASONING
    receipt = [request] if returns_requests else [("running", 7)]
    hooks.finish_requests(owner, "running", RequestStatus.FINISHED_ABORTED, lambda ids, status: receipt)
    assert hooks.controller.program_ref_for_request("running") is None
    assert hooks.controller.registry.require(ref).status is ProgramStatus.ACTING
    hooks.close()


@pytest.mark.parametrize(("prompt", "output", "accepted"), [(191, 1, True), (192, 1, False), (193, 1, False)])
def test_mori_capacity_excludes_null_block_and_checks_decode_upper_bound(
    prompt, output, accepted, mock_bridge_native
) -> None:
    """A single request must fit the allocatable pool through its maximum decode."""
    owner = SimpleNamespace(requests={}, kv_cache_manager=None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "full"}}},
    )
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=13), 16)
    request = SimpleNamespace(
        request_id="request",
        client_index=7,
        num_prompt_tokens=prompt,
        sampling_params=SimpleNamespace(max_tokens=output, extra_args={"agentic_context": {"program_id": "program"}}),
        trace_headers={},
        events=[],
    )
    native = []
    hooks.add_request(owner, request, native.append)
    assert hooks.local_hbm_kv_tokens == 192
    assert bool(native) is accepted
    if not accepted:
        assert request.status is RequestStatus.FINISHED_ERROR
        assert hooks.controller.program_ref_for_request("request") is None
    hooks.close()


@pytest.mark.parametrize(
    "cache_hit",
    [((), 96), ((), 96, 0), ((), 96, 32)],
    ids=["legacy-pair", "v031-triple", "v031-uncached-prefix"],
)
def test_retained_prefix_observation_accepts_vllm_cache_hit_results(cache_hit) -> None:
    """Both interfaces report the hit length independently of the uncached prefix."""
    request = SimpleNamespace(request_id="retained", block_hashes=[], num_tokens=129)
    controller = SimpleNamespace(
        prefix_cache_candidates=lambda: (("retained", request),),
        on_prefix_cache_observation=Mock(),
    )
    lookup = Mock(return_value=cache_hit)
    coordinator = SimpleNamespace(find_longest_cache_hit=lookup)
    owner = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(coordinator=coordinator, prefix_cache_lookup_enabled=lambda candidate: True)
    )
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.controller = controller

    with patch("agentinfer.agentcache.mori.scheduler.logger.warning") as warning:
        hooks._observe_retained_prefix_cache(owner)

    lookup.assert_called_once_with(request.block_hashes, 128)
    controller.on_prefix_cache_observation.assert_called_once_with("retained", 96)
    warning.assert_not_called()


@pytest.mark.parametrize("cache_hit", [(), ((),)])
def test_retained_prefix_observation_continues_after_incomplete_cache_hit(cache_hit) -> None:
    """An incomplete result must not prevent observing subsequent retained requests."""
    request = SimpleNamespace(block_hashes=[], num_tokens=129)
    controller = SimpleNamespace(
        prefix_cache_candidates=lambda: (("invalid", request), ("valid", request)),
        on_prefix_cache_observation=Mock(),
    )
    lookup = Mock(side_effect=[cache_hit, ((), 96, 0)])
    owner = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(find_longest_cache_hit=lookup))
    )
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.controller = controller

    with patch("agentinfer.agentcache.mori.scheduler.logger.warning") as warning:
        hooks._observe_retained_prefix_cache(owner)

    controller.on_prefix_cache_observation.assert_called_once_with("valid", 96)
    warning.assert_called_once_with("AgentInfer prefix-cache observation failed request=%s", "invalid", exc_info=True)


def test_context_handoff_preserves_old_owner_until_exact_native_lookup() -> None:
    """The lookup must see the old prompt, while only exact hits survive replacement."""
    program = ProgramRef("program", 0)
    MORI_RUNTIME_STATE.bind_request("request", program)
    old_context = MORI_RUNTIME_STATE.capture_context("request")
    events: list[object] = []
    blocks = SimpleNamespace(
        blocks=([SimpleNamespace(block_id=11, ref_cnt=0), SimpleNamespace(block_id=12, ref_cnt=1)],)
    )

    def native_lookup(request):
        if not events:
            assert MORI_RUNTIME_STATE.program_for_context(old_context) == program
        events.append("lookup")
        return blocks, 32

    def handoff(request_id):
        events.append("handoff")
        assert request_id == "request"
        MORI_RUNTIME_STATE.replace_context(program)

    policy = SimpleNamespace(
        on_block_owner_added=Mock(side_effect=lambda block_id, request_id: events.append((block_id, request_id)))
    )
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.controller = SimpleNamespace(
        strategy=build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy,
        program_ref_for_request=lambda request_id: program,
        on_request_dispatch=Mock(side_effect=handoff),
    )
    hooks._mori_adapter = SimpleNamespace(hbm_policy=policy)
    hooks.tracked_native = {"request"}
    hooks._local_prefix_observations = {}
    hooks._mori_context_handed_off = set()
    hooks._context_handoff_at_lookup = False
    owner = SimpleNamespace(kv_cache_manager=SimpleNamespace(get_computed_blocks=native_lookup))
    request = SimpleNamespace(request_id="request")

    hooks._install_prefix_lookup_observer(owner)
    assert hooks._context_handoff_at_lookup is True
    assert owner.kv_cache_manager.get_computed_blocks(request) == (blocks, 32)

    assert events == ["lookup", "handoff", (11, "request"), (12, "request")]
    assert MORI_RUNTIME_STATE.program_for_context(old_context) is None
    assert MORI_RUNTIME_STATE.program_for_request("request") == program
    assert hooks._local_prefix_observations["request"] == (32, 16)
    assert hooks._mori_context_handed_off == {"request"}

    # A preempted request may query again, but prompt ownership changes once.
    owner.kv_cache_manager.get_computed_blocks(request)
    hooks.controller.on_request_dispatch.assert_called_once_with("request")
    assert policy.on_block_owner_added.call_count == 2
