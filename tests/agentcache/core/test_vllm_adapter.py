# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Behavioral tests for vLLM request retention and native release hooks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEvent, EngineCoreEventType

from agentinfer.agentcache.core.scheduler import AgentCacheAsyncSchedulerBridge, _metadata_from_request, _resolve_object
from agentinfer.scheduling.backend import DispatchTarget
from agentinfer.scheduling.request_pool import RequestPool, RequestPoolEntry

pytestmark = pytest.mark.cpu_test


class _Controller:
    """Minimal Scheduler test double that admits retained requests on the next cycle."""

    def __init__(self) -> None:
        self.pool: RequestPool[object] = RequestPool()
        self.schedule_calls = 0
        self.completed: list[tuple[str, int]] = []
        self.prefix_observations: list[tuple[str, int]] = []
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

    def consume_admitted_requests(self):
        return self.pool.consume_admitted()

    def prefix_cache_candidates(self):
        return tuple(self.prefix_candidates)

    def on_stream_output(self, request_id, total_output_tokens) -> None:
        return None

    def on_prefix_cache_observation(self, request_id, cached_prefix_tokens) -> None:
        self.prefix_observations.append((request_id, cached_prefix_tokens))

    def on_request_completion(self, request_id, total_tokens) -> None:
        self.completed.append((request_id, total_tokens))

    def cancel_request(self, request_id):
        return self.pool.cancel(request_id)


def build_test_controller(backend_pool_info, settings):
    """Factory resolved by the bridge exactly as an installed controller would be."""
    assert settings["backend_id"] == "vllm-local"
    return _Controller()


def test_resolve_object_reports_controller_factory_module_import() -> None:
    with pytest.raises(ImportError, match=r"controller_factory='missing_agentinfer_package\.module\.factory'"):
        _resolve_object("missing_agentinfer_package.module.factory")


def test_resolve_object_reports_missing_controller_factory_attribute() -> None:
    path = f"{__name__}.missing_factory"

    with pytest.raises(AttributeError, match=rf"controller_factory='{path}'"):
        _resolve_object(path)


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
                "controller_factory": f"{__name__}.build_test_controller",
            }
        },
    )

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.waiting = object()
        scheduler.requests = {}
        scheduler.kv_cache_manager = SimpleNamespace(
            prefix_cache_lookup_enabled=lambda request: True,
            coordinator=SimpleNamespace(find_longest_cache_hit=lambda hashes, maximum: ((), 48)),
        )

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
        controller.prefix_candidates.append(("request-1", request))

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
    assert controller.prefix_observations == [("request-1", 64)]
    assert rank.dp_rank == 3
    assert rank.total_hbm_kv_tokens == 160


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
        scheduler.skipped_waiting = [SimpleNamespace(num_tokens=40)]
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


def test_async_bridge_reports_native_abort_as_completion_without_failure_semantics() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={
            "agentcache": {
                "backend_id": "vllm-local",
                "controller_factory": f"{__name__}.build_test_controller",
            }
        },
    )
    native_request = SimpleNamespace(num_tokens=123)

    def initialize(scheduler, *args, **kwargs) -> None:
        scheduler.waiting = []
        scheduler.running = []
        scheduler.skipped_waiting = []
        scheduler.requests = {"request-1": native_request}

    with (
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(
            Scheduler,
            "finish_requests",
            return_value=[("request-1", 3)],
        ),
    ):
        bridge = AgentCacheAsyncSchedulerBridge(config, SimpleNamespace(num_blocks=10), object(), 16, 16)
        controller = bridge._agentcache.controller
        assert isinstance(controller, _Controller)
        bridge._agentcache.tracked_native.add("request-1")
        finished = bridge.finish_requests("request-1", object())

    assert finished == [("request-1", 3)]
    assert controller.completed == [("request-1", 123)]
    assert bridge._agentcache.tracked_native == set()
