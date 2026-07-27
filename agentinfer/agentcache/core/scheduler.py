# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Packaged vLLM scheduler compatibility and explicit AgentCache admission bridges.

The packaged ``AgentAwareScheduler`` preserves the existing FCFS-compatible extension point. Progress-TTL instead
uses the explicit async or sync bridge and a composed helper, while native waiting/running/KV state and every
token-level scheduling operation remain owned by vLLM.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.core.api_adapter import LIFECYCLE_SOCKET_ENV, UnixLifecycleReceiver
from agentinfer.agentcache.core.request_queue import AgentAwareQueue
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DispatchTarget, DpRankInfo
from agentinfer.scheduling.identity import AgentIdentity, JsonMapping, JsonObject, parse_agent_identity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.request_pool import RequestPoolEntry, RetainedRequestT

logger = logging.getLogger(__name__)


class AgentAwareScheduler(Scheduler):
    """Preserve the packaged AgentAwareQueue scheduler extension point."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting = AgentAwareQueue()
        logger.warning("AgentAwareQueue initialized as waiting queue")


class EmbeddedSchedulerController(Protocol[RetainedRequestT]):
    """AgentCache Scheduler surface consumed by an embedded engine bridge."""

    @property
    def retained_request_count(self) -> int:
        """Return requests held before native admission."""

    @property
    def retained_request_ids(self) -> tuple[str, ...]:
        """Return retained request attempts for host-wide abort."""

    def on_request_arrival(
        self,
        request_id: str,
        metadata: AgentIdentity,
        prompt_tokens: int,
        backend_pool_info: BackendPoolInfo,
        retained_request: RetainedRequestT,
    ) -> bool:
        """Retain a request and return whether it may enter native waiting immediately."""

    def schedule_cycle(self, backend_pool_info: BackendPoolInfo) -> None:
        """Refresh decisions and populate the controller's recent-admit results."""

    def needs_schedule_cycle(self, now_monotonic_s: float) -> bool:
        """Return whether this native poll needs AgentCache strategy work."""

    def consume_admitted_requests(
        self,
    ) -> tuple[tuple[RequestPoolEntry[RetainedRequestT], DispatchTarget], ...]:
        """Return and remove newly admitted retained requests."""

    def on_stream_output(self, request_id: str, total_output_tokens: int) -> None:
        """Observe one native output update."""

    def on_request_completion(self, request_id: str, total_tokens: int) -> None:
        """Observe terminal request cleanup without success/failure semantics."""

    def on_response_completion(self, program_id: str, lifecycle: ProgramLifecycle) -> bool:
        """Apply an optional lifecycle fact observed after API tool parsing."""

    def cancel_request(self, request_id: str) -> RequestPoolEntry[RetainedRequestT] | None:
        """Remove and return a request retained before native admission."""


ControllerFactory = Callable[[BackendPoolInfo, JsonMapping], EmbeddedSchedulerController[Request]]


class _VllmAdmissionHooks:
    """Composition helper shared by the two single-inheritance native bridges."""

    def __init__(self, owner: Scheduler, vllm_config: object, kv_cache_config: object, block_size: int) -> None:
        additional = getattr(vllm_config, "additional_config", {})
        settings = additional.get("agentcache", {}) if isinstance(additional, dict) else {}
        if not isinstance(settings, dict):
            raise ValueError("additional_config.agentcache must be an object")
        settings = cast(JsonMapping, settings)
        if "num_ranks" in settings:
            raise ValueError("additional_config.agentcache.num_ranks is obsolete; rank identity comes from EngineCore")
        backend_id = settings.get("backend_id", "vllm-local")
        if not isinstance(backend_id, str) or not backend_id:
            raise ValueError("agentcache.backend_id must be a non-empty string")
        self.backend_id = backend_id
        parallel_config = getattr(vllm_config, "parallel_config", None)
        rank = getattr(parallel_config, "data_parallel_index", None)
        self.dp_rank = int(rank if rank is not None else getattr(parallel_config, "data_parallel_rank", 0))
        if self.dp_rank < 0:
            raise ValueError("EngineCore data_parallel_index must be non-negative")
        expected = settings.get("expected_reasoning_agent_nums")
        if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int) or expected < 0):
            raise ValueError("agentcache.expected_reasoning_agent_nums must be a non-negative integer")
        self.expected_agents = expected
        # EngineCore passes resolve_kv_cache_block_sizes() as block_size, which already includes DCP and PCP.
        self.local_hbm_kv_tokens = int(kv_cache_config.num_blocks) * int(block_size)
        self.tracked_native: set[str] = set()
        self.controller: EmbeddedSchedulerController[Request] | None = None
        self.lifecycle_receiver: UnixLifecycleReceiver | None = None
        factory_path = settings.get("controller_factory")
        if factory_path is not None:
            if not isinstance(factory_path, str) or not factory_path:
                raise ValueError("agentcache.controller_factory must be a non-empty import path")
            factory = cast(ControllerFactory, _resolve_object(factory_path))
            self.controller = factory(self.backend_pool_info(owner), settings)
            socket_path = settings.get("lifecycle_socket_path") or os.environ.get(LIFECYCLE_SOCKET_ENV)
            if socket_path is not None:
                if not isinstance(socket_path, str) or not socket_path:
                    raise ValueError("agentcache.lifecycle_socket_path must be a non-empty string")
                self.lifecycle_receiver = UnixLifecycleReceiver(socket_path, self.dp_rank)

    def add_request(self, owner: Scheduler, request: Request, native_add: Callable[[Request], None]) -> None:
        """Retain agentic work or pass compatibility traffic directly to vLLM."""
        self._drain_lifecycle_signals()
        if self.controller is None:
            native_add(request)
            return
        metadata = _metadata_from_request(request)
        if metadata is None:
            native_add(request)
            return
        dispatch_now = self.controller.on_request_arrival(
            request.request_id,
            metadata,
            request.num_prompt_tokens,
            self.backend_pool_info(owner),
            request,
        )
        if dispatch_now:
            self.tracked_native.add(request.request_id)
            native_add(request)

    def before_schedule(self, owner: Scheduler, native_add: Callable[[Request], None]) -> None:
        """Run one AgentCache cycle and transfer admitted requests to native waiting."""
        self._drain_lifecycle_signals()
        if self.controller is None or not self.controller.needs_schedule_cycle(time.monotonic()):
            return
        self.controller.schedule_cycle(self.backend_pool_info(owner))
        for entry, target in self.controller.consume_admitted_requests():
            if target != DispatchTarget(self.backend_id, self.dp_rank):
                raise RuntimeError("embedded vLLM bridge received a target for another backend or DP rank")
            self.tracked_native.add(entry.request_id)
            native_add(entry.retained_request)

    def after_output(
        self,
        owner: Scheduler,
        outputs: Mapping[int, object],
        token_counts_before_update: Mapping[str, tuple[int, int]],
    ) -> None:
        """Publish token deltas and terminal facts after native state is consistent."""
        if self.controller is None:
            return
        for batch in outputs.values():
            for output in getattr(batch, "outputs", ()):
                request_id = str(output.request_id)
                if request_id not in self.tracked_native:
                    continue
                request = owner.requests.get(request_id)
                total_output_tokens = (
                    request.num_output_tokens
                    if request is not None
                    else token_counts_before_update.get(request_id, (0, 0))[1] + len(output.new_token_ids)
                )
                self.controller.on_stream_output(request_id, total_output_tokens)
                if output.finished:
                    total_tokens = (
                        request.num_tokens
                        if request is not None
                        else token_counts_before_update.get(request_id, (0, 0))[0] + len(output.new_token_ids)
                    )
                    self.controller.on_request_completion(request_id, total_tokens)
                    self.tracked_native.discard(request_id)

    def cancel_retained(self, request_ids: str | Iterable[str] | None) -> list[tuple[str, int]]:
        """Cancel attempts that have not entered native vLLM yet."""
        if self.controller is None:
            return []
        normalized = (
            self.controller.retained_request_ids
            if request_ids is None
            else ((request_ids,) if isinstance(request_ids, str) else tuple(request_ids))
        )
        cancelled: list[tuple[str, int]] = []
        for request_id in normalized:
            entry = self.controller.cancel_request(request_id)
            if entry is not None:
                cancelled.append((request_id, entry.retained_request.client_index))
        return cancelled

    def finish_requests(
        self,
        owner: Scheduler,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
        native_finish: Callable[[str | Iterable[str] | None, RequestStatus], list[tuple[str, int]]],
    ) -> list[tuple[str, int]]:
        """Finish retained and native attempts while reporting only terminal lifecycle facts."""
        normalized_request_ids = (
            request_ids if request_ids is None or isinstance(request_ids, str) else tuple(request_ids)
        )
        retained = self.cancel_retained(normalized_request_ids)
        token_counts = {
            request_id: request.num_tokens
            for request_id, request in owner.requests.items()
            if request_id in self.tracked_native
        }
        native = native_finish(normalized_request_ids, finished_status)
        if self.controller is not None:
            for request_id, _ in native:
                if request_id in self.tracked_native:
                    self.controller.on_request_completion(request_id, token_counts.get(request_id, 0))
                    self.tracked_native.discard(request_id)
        return retained + native

    def unfinished_count(self, native_count: int) -> int:
        """Merge retained and native liveness without exposing request objects to EngineCore."""
        retained = self.controller.retained_request_count if self.controller is not None else 0
        return native_count + retained

    def backend_pool_info(self, owner: Scheduler) -> BackendPoolInfo:
        """Build rank-local logical KV-token and request-load facts from native vLLM state."""
        now = time.monotonic()
        running, waiting = owner.get_request_counts()
        backend = BackendInfo(
            backend_id=self.backend_id,
            backend_url="embedded://vllm",
            healthy=True,
            dp_ranks=(
                DpRankInfo(
                    dp_rank=self.dp_rank,
                    healthy=True,
                    schedulable=True,
                    running_requests=running,
                    waiting_requests=waiting,
                    total_hbm_kv_tokens=self.local_hbm_kv_tokens,
                    expected_reasoning_agent_nums=self.expected_agents,
                ),
            ),
            observed_at_monotonic_s=now,
        )
        return BackendPoolInfo((backend,), observed_at_monotonic_s=now)

    def _drain_lifecycle_signals(self) -> None:
        """Apply API-derived lifecycle facts before any new admission or periodic decision."""
        if self.controller is None or self.lifecycle_receiver is None:
            return
        for signal in self.lifecycle_receiver.receive():
            released = self.controller.on_response_completion(signal.program_id, signal.lifecycle)
            logger.info(
                "AgentCache API lifecycle program=%s lifecycle=%s released=%s",
                signal.program_id,
                signal.lifecycle.value,
                released,
            )


class AgentCacheAsyncSchedulerBridge(AsyncScheduler):
    """Explicit AgentCache bridge for configurations with async scheduling enabled."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs.get("vllm_config") or args[0]
        if getattr(vllm_config.scheduler_config, "async_scheduling", None) is not True:
            raise ValueError("AgentCacheAsyncSchedulerBridge requires async_scheduling=true")
        super().__init__(*args, **kwargs)
        kv_cache_config = kwargs.get("kv_cache_config") or args[1]
        block_size = kwargs.get("block_size") or args[3]
        self._agentcache = _VllmAdmissionHooks(self, vllm_config, kv_cache_config, block_size)

    def add_request(self, request: Request) -> None:
        self._agentcache.add_request(self, request, super().add_request)

    def schedule(self):
        if self._agentcache.controller is not None:
            self._agentcache.before_schedule(self, super().add_request)
        return super().schedule()

    def update_from_output(self, scheduler_output, model_runner_output):
        if self._agentcache.controller is None:
            return super().update_from_output(scheduler_output, model_runner_output)
        before = {
            request_id: (request.num_tokens, request.num_output_tokens)
            for request_id in model_runner_output.req_ids
            if request_id in self._agentcache.tracked_native and (request := self.requests.get(request_id)) is not None
        }
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self._agentcache.after_output(self, outputs, before)
        return outputs

    def finish_requests(self, request_ids, finished_status: RequestStatus):
        if self._agentcache.controller is None:
            return super().finish_requests(request_ids, finished_status)
        return self._agentcache.finish_requests(self, request_ids, finished_status, super().finish_requests)

    def get_num_unfinished_requests(self) -> int:
        if self._agentcache.controller is None:
            return super().get_num_unfinished_requests()
        return self._agentcache.unfinished_count(super().get_num_unfinished_requests())


class AgentCacheSyncSchedulerBridge(Scheduler):
    """Explicit fallback bridge for configurations where vLLM disables async scheduling."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs.get("vllm_config") or args[0]
        if getattr(vllm_config.scheduler_config, "async_scheduling", None) is not False:
            raise ValueError("AgentCacheSyncSchedulerBridge requires async_scheduling=false")
        super().__init__(*args, **kwargs)
        kv_cache_config = kwargs.get("kv_cache_config") or args[1]
        block_size = kwargs.get("block_size") or args[3]
        self._agentcache = _VllmAdmissionHooks(self, vllm_config, kv_cache_config, block_size)

    def add_request(self, request: Request) -> None:
        self._agentcache.add_request(self, request, super().add_request)

    def schedule(self):
        if self._agentcache.controller is not None:
            self._agentcache.before_schedule(self, super().add_request)
        return super().schedule()

    def update_from_output(self, scheduler_output, model_runner_output):
        if self._agentcache.controller is None:
            return super().update_from_output(scheduler_output, model_runner_output)
        before = {
            request_id: (request.num_tokens, request.num_output_tokens)
            for request_id in model_runner_output.req_ids
            if request_id in self._agentcache.tracked_native and (request := self.requests.get(request_id)) is not None
        }
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self._agentcache.after_output(self, outputs, before)
        return outputs

    def finish_requests(self, request_ids, finished_status: RequestStatus):
        if self._agentcache.controller is None:
            return super().finish_requests(request_ids, finished_status)
        return self._agentcache.finish_requests(self, request_ids, finished_status, super().finish_requests)

    def get_num_unfinished_requests(self) -> int:
        if self._agentcache.controller is None:
            return super().get_num_unfinished_requests()
        return self._agentcache.unfinished_count(super().get_num_unfinished_requests())


def _metadata_from_request(request: Request) -> AgentIdentity | None:
    """Normalize independent vLLM body-extension and request-header inputs."""
    sampling_params = request.sampling_params
    extra_args = sampling_params.extra_args if sampling_params is not None else None
    headers = request.trace_headers if request.trace_headers is not None else {}
    return parse_agent_identity(
        vllm_xargs=cast(JsonObject, extra_args) if isinstance(extra_args, dict) else None,
        headers=headers,
    )


def _resolve_object(path: str) -> object:
    """Resolve one explicit ``module.attribute`` factory path."""
    module_name, separator, attribute = path.rpartition(".")
    if not separator:
        raise ValueError("controller_factory must use module.attribute form")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"failed to import controller factory module {module_name!r} from controller_factory={path!r}"
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(f"controller_factory={path!r} does not define attribute {attribute!r}") from exc
