# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Behavioral tests for vLLM request retention and native release hooks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEvent, EngineCoreEventType
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.core.scheduler import (
    AgentCacheAsyncSchedulerBridge,
    AgentCacheSyncSchedulerBridge,
    _metadata_from_request,
    _VllmAdmissionHooks,
)
from agentinfer.scheduling.backend import DispatchTarget
from agentinfer.scheduling.request_pool import RequestPool, RequestPoolEntry
from agentinfer.scheduling.runtime import ProgramScheduler
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu")


class _Controller:
    """Minimal Scheduler test double that admits retained requests on the next cycle."""

    def __init__(self) -> None:
        self.pool: RequestPool[object] = RequestPool()
        self.schedule_calls = 0
        self.completed: list[tuple[str, int]] = []
        self.cancelled: list[str] = []
        self.prefix_observations: list[tuple[str, int, int | None]] = []
        self.recovery_observations: list[tuple[int, float]] = []
        self.prefix_candidates: list[tuple[str, object]] = []

    @property
    def retained_request_count(self) -> int:
        return self.pool.unfinished_count

    @property
    def retained_request_ids(self) -> tuple[str, ...]:
        return tuple(entry.request_id for entry in self.pool.entries)

    def on_request_arrival(self, request_id, metadata, prompt_tokens, backend_pool_info, retained_request) -> bool:
        self.pool.add(RequestPoolEntry(request_id, None, 1.0, retained_request))
        return False

    def schedule_cycle(self, backend_pool_info) -> None:
        self.schedule_calls += 1
        backend = backend_pool_info.backends[0]
        target = DispatchTarget(backend.backend_id, backend.dp_ranks[0].dp_rank)
        for request_id in self.pool.waiting_request_ids:
            self.pool.admit(request_id, target)

    def needs_schedule_cycle(self, now_monotonic_s) -> bool:
        return self.retained_request_count > 0

    def run_due_lightweight_checks(self, now_monotonic_s) -> None:
        return None

    def consume_admitted_requests(self):
        return self.pool.consume_admitted()

    def prefix_cache_candidates(self):
        return tuple(self.prefix_candidates)

    def on_stream_output(self, request_id, total_output_tokens) -> None:
        return None

    def on_prefix_cache_observation(self, request_id, cached_prefix_tokens, shared_prefix_tokens=None) -> None:
        self.prefix_observations.append((request_id, cached_prefix_tokens, shared_prefix_tokens))

    def on_cache_recovery_observation(self, recovered_tokens, elapsed_seconds) -> None:
        self.recovery_observations.append((recovered_tokens, elapsed_seconds))

    def on_request_completion(self, request_id, total_tokens) -> None:
        self.completed.append((request_id, total_tokens))

    def cancel_request(self, request_id):
        self.cancelled.append(request_id)
        return self.pool.cancel(request_id)


def build_test_controller(backend_pool_info, settings):
    """Test double installed in place of the direct ``build_progress_ttl_controller`` call."""
    assert settings["backend_id"] == "vllm-local"
    return _Controller()


def _patch_controller_build():
    """Patch the bridge's direct controller build to install the test double."""
    return patch("agentinfer.agentcache.core.scheduler.build_progress_ttl_controller", build_test_controller)


def test_async_bridge_builds_progress_ttl_controller_directly() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"backend_id": "vllm-local"}},
    )

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.waiting = []
        scheduler.running = []
        scheduler.kv_holding_waiting = []
        scheduler.requests = {}
        scheduler.kv_cache_manager = None

    with (
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "get_request_counts", return_value=(0, 0)),
    ):
        bridge = AgentCacheAsyncSchedulerBridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)

    assert isinstance(bridge._agentcache.controller, ProgramScheduler)


def test_metadata_from_request_parses_claude_headers_without_extra_args() -> None:
    """Header identity must not depend on the optional vLLM body-extension mapping."""
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=None),
        trace_headers={
            "x-claude-code-session-id": "session-a",
            "x-claude-code-agent-id": "child-a",
        },
    )

    metadata = _metadata_from_request(request)

    assert metadata is not None
    assert metadata.program_id == "session-a:child-a"
    assert metadata.parent_program_id == "session-a:lead"
    assert metadata.blocks_parent is True
    assert metadata.expected_resume is False


def test_metadata_from_request_returns_none_without_body_or_header_identity() -> None:
    """A request with neither identity source remains compatibility traffic."""
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=None),
        trace_headers=None,
    )

    assert _metadata_from_request(request) is None


def test_async_bridge_retains_then_releases_to_native_waiting() -> None:
    native_requests: list[object] = []
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=3),
        additional_config={
            "agentcache": {
                "backend_id": "vllm-local",
            }
        },
    )

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.waiting = object()
        scheduler.requests = {}

    def native_add(scheduler, request) -> None:
        request.events.append(EngineCoreEvent.new_event(EngineCoreEventType.QUEUED, 10.0))
        native_requests.append(request)

    request = SimpleNamespace(
        request_id="request-1",
        client_index=3,
        num_prompt_tokens=100,
        sampling_params=SimpleNamespace(extra_args={"agentic_context": '{"task_id":"task","agent_id":"lead"}'}),
        trace_headers={},
        events=[],
    )
    with (
        _patch_controller_build(),
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "add_request", native_add),
        patch.object(Scheduler, "get_request_counts", return_value=(0, 0)),
        patch.object(Scheduler, "get_num_unfinished_requests", return_value=0),
        patch.object(
            Scheduler,
            "schedule",
            side_effect=(
                SimpleNamespace(scheduled_new_reqs=(SimpleNamespace(req_id="request-1", num_computed_tokens=64),)),
                SimpleNamespace(scheduled_new_reqs=()),
            ),
        ),
    ):
        bridge = AgentCacheAsyncSchedulerBridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)
        controller = bridge._agentcache.controller
        assert isinstance(controller, _Controller)
        native_waiting = bridge.waiting
        bridge.add_request(request)

        assert native_requests == []
        assert bridge.get_num_unfinished_requests() == 1
        assert bridge.schedule().scheduled_new_reqs[0].req_id == "request-1"
        assert bridge.schedule().scheduled_new_reqs == ()
        rank = bridge._agentcache.backend_pool_info(bridge).backends[0].dp_ranks[0]

    assert native_requests == [request]
    assert request.events == [EngineCoreEvent.new_event(EngineCoreEventType.QUEUED, 1.0)]
    assert bridge.waiting is native_waiting
    assert controller.retained_request_count == 0
    assert controller.schedule_calls == 1
    assert controller.prefix_observations == [("request-1", 64, None)]
    assert rank.dp_rank == 3
    assert rank.total_hbm_kv_tokens == 160


def test_ref_count_shared_tokens_uses_conservative_common_prefix_across_groups() -> None:
    computed = SimpleNamespace(
        blocks=(
            [SimpleNamespace(ref_cnt=1), SimpleNamespace(ref_cnt=1), SimpleNamespace(ref_cnt=0)],
            [SimpleNamespace(ref_cnt=2), SimpleNamespace(ref_cnt=0), SimpleNamespace(ref_cnt=0)],
        )
    )

    assert _VllmAdmissionHooks._ref_count_shared_tokens(computed, 96) == 32


def test_prefix_lookup_observer_captures_ref_counts_before_native_allocation() -> None:
    computed = SimpleNamespace(
        blocks=([SimpleNamespace(ref_cnt=1), SimpleNamespace(ref_cnt=0)],),
    )
    native_calls: list[str] = []

    def native_get_computed_blocks(request):
        native_calls.append(request.request_id)
        return computed, 64, 128

    owner = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(get_computed_blocks=native_get_computed_blocks),
    )
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.tracked_native = {"request-1"}
    hooks._local_prefix_observations = {}
    hooks._install_prefix_lookup_observer(owner)
    hooks._install_prefix_lookup_observer(owner)

    returned = owner.kv_cache_manager.get_computed_blocks(SimpleNamespace(request_id="request-1"))

    assert returned == (computed, 64, 128)
    assert native_calls == ["request-1"]
    assert hooks._local_prefix_observations == {"request-1": (64, 32)}


@pytest.mark.parametrize("uncached_common_prefix_tokens", [0, 32])
def test_retained_prefix_observation_uses_only_cached_tokens(uncached_common_prefix_tokens: int) -> None:
    request = SimpleNamespace(block_hashes=[], num_tokens=97)
    controller = _Controller()
    controller.prefix_candidates.append(("request-1", request))
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.controller = controller
    owner = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(
                find_longest_cache_hit=lambda hashes, maximum: ((), 64, uncached_common_prefix_tokens),
            ),
        ),
    )

    hooks._observe_retained_prefix_cache(owner)

    assert controller.prefix_observations == [("request-1", 64, None)]


def test_async_bridge_rejects_obsolete_configured_rank_count() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"num_ranks": 2}},
    )

    with patch.object(AsyncScheduler, "__init__", return_value=None):
        with pytest.raises(ValueError, match="num_ranks is obsolete"):
            AgentCacheAsyncSchedulerBridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)


def test_async_bridge_does_not_multiply_resolved_context_parallel_block_size_twice() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={},
    )

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.dcp_world_size = 4
        scheduler.pcp_world_size = 2

    with (
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "get_request_counts", return_value=(0, 0)),
        patch("agentinfer.agentcache.core.scheduler.attach_agentinfer_to_vllm_logging") as attach_logging,
    ):
        resolved_block_size = 16 * 4 * 2
        bridge = AgentCacheAsyncSchedulerBridge(
            config,
            SimpleNamespace(num_blocks=10),
            object(),
            resolved_block_size,
            resolved_block_size,
        )
        rank = bridge._agentcache.backend_pool_info(bridge).backends[0].dp_ranks[0]

    attach_logging.assert_called_once_with()
    assert rank.total_hbm_kv_tokens == 10 * resolved_block_size


def test_async_bridge_reports_running_usage_and_native_waiting_token_estimate() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={},
    )

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.waiting = [SimpleNamespace(num_tokens=120), SimpleNamespace(num_tokens=80)]
        scheduler.kv_holding_waiting = [SimpleNamespace(num_tokens=40)]
        scheduler.deferred_waiting = tuple(scheduler.waiting)
        scheduler.kv_cache_manager = SimpleNamespace(usage=lambda: 0.5)

    with (
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "get_request_counts", return_value=(1, 3)),
    ):
        bridge = AgentCacheAsyncSchedulerBridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)
        rank = bridge._agentcache.backend_pool_info(bridge).backends[0].dp_ranks[0]

    assert rank.used_hbm_kv_tokens == 80
    assert rank.waiting_hbm_kv_tokens == 240


def test_async_bridge_accepts_numeric_kv_usage_property() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={},
    )

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.kv_cache_manager = SimpleNamespace(usage=0.5)

    with (
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "get_request_counts", return_value=(1, 0)),
    ):
        bridge = AgentCacheAsyncSchedulerBridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)
        rank = bridge._agentcache.backend_pool_info(bridge).backends[0].dp_ranks[0]

    assert rank.used_hbm_kv_tokens == 80


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"backend_id": 1}, "backend_id"),
        ({"expected_reasoning_agent_nums": True}, "expected_reasoning_agent_nums"),
        ({"expected_reasoning_agent_nums": -1}, "expected_reasoning_agent_nums"),
    ],
)
def test_async_bridge_rejects_invalid_adapter_settings(settings: dict[str, object], message: str) -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": settings},
    )

    with patch.object(AsyncScheduler, "__init__", return_value=None):
        with pytest.raises(ValueError, match=message):
            AgentCacheAsyncSchedulerBridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("abort_all", [False, True], ids=["selected", "all"])
def test_bridge_finishes_native_and_retained_requests(async_scheduling: bool, abort_all: bool) -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={
            "agentcache": {
                "backend_id": "vllm-local",
            }
        },
    )
    native_request = Request("native", [1, 2, 3], SamplingParams(max_tokens=8), None, client_index=3)
    retained_request = Request("retained", [1, 2], SamplingParams(max_tokens=8), None, client_index=7)
    finished_status = RequestStatus.FINISHED_ABORTED
    selected_ids = ("retained", "native", "retained")
    native_request_ids = None if abort_all else selected_ids
    native_parent = AsyncScheduler if async_scheduling else Scheduler
    bridge_cls = AgentCacheAsyncSchedulerBridge if async_scheduling else AgentCacheSyncSchedulerBridge

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.waiting = [native_request]
        scheduler.running = []
        scheduler.kv_holding_waiting = []
        scheduler.requests = {native_request.request_id: native_request}

    def finish_native(scheduler, request_ids, status) -> list[Request]:
        assert status == finished_status
        assert request_ids == native_request_ids
        request = scheduler.requests.pop(native_request.request_id, None)
        if request is None:
            return []
        request.status = status
        return [request]

    with (
        _patch_controller_build(),
        patch.object(native_parent, "__init__", initialize),
        patch.object(Scheduler, "get_request_counts", return_value=(0, 1)),
        patch.object(Scheduler, "finish_requests", finish_native),
    ):
        bridge = bridge_cls(config, SimpleNamespace(num_blocks=10), object(), 16, 16)
        controller = bridge._agentcache.controller
        assert isinstance(controller, _Controller)
        controller.pool.add(RequestPoolEntry("retained", None, 1.0, retained_request))
        bridge._agentcache.tracked_native.add("native")
        bridge._agentcache._local_prefix_observations["native"] = (2, 1)
        finished = bridge.finish_requests(None if abort_all else iter(selected_ids), finished_status)
        assert bridge.finish_requests(None if abort_all else iter(selected_ids), finished_status) == []

    assert finished == [retained_request, native_request]
    assert [(request.request_id, request.client_index) for request in finished] == [("retained", 7), ("native", 3)]
    assert all(request.status == finished_status for request in finished)
    assert controller.completed == [("native", 3)]
    assert controller.cancelled == ["retained"]
    assert controller.retained_request_count == 0
    assert bridge._agentcache.tracked_native == set()
    assert bridge._agentcache._local_prefix_observations == {}
