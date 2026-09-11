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
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.core.api_adapter import LIFECYCLE_SOCKET_ENV, UnixLifecycleReceiver
from agentinfer.agentcache.core.request_queue import AgentAwareQueue
from agentinfer.agentcache.core.vllm_logging import attach_agentinfer_to_vllm_logging
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

    def prefix_cache_candidates(self) -> tuple[tuple[str, RetainedRequestT], ...]:
        """Return retained paused reasoning requests needing a refreshed prefix-cache observation."""

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

    def run_due_lightweight_checks(self, now_monotonic_s: float) -> None:
        """Apply due factor-only checks without entering a full scheduling cycle."""

    def consume_admitted_requests(
        self,
    ) -> tuple[tuple[RequestPoolEntry[RetainedRequestT], DispatchTarget], ...]:
        """Return and remove newly admitted retained requests."""

    def on_stream_output(self, request_id: str, total_output_tokens: int) -> None:
        """Observe one native output update."""

    def on_prefix_cache_observation(
        self,
        request_id: str,
        cached_prefix_tokens: int,
        shared_prefix_tokens: int | None = None,
        hbm_cached_prefix_tokens: int | None = None,
        hbm_hit_ref_zero_tokens: int | None = None,
    ) -> None:
        """Observe total and local cache hits plus the pre-allocation ref-count-zero subset."""

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
        attach_agentinfer_to_vllm_logging()
        self._owner = owner
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
        self._local_prefix_observations: dict[str, tuple[int, int]] = {}
        self._prefix_refresh_pending = False
        self.controller: EmbeddedSchedulerController[Request] | None = None
        self.lifecycle_receiver: UnixLifecycleReceiver | None = None
        factory_path = settings.get("controller_factory")
        if factory_path is not None:
            if not isinstance(factory_path, str) or not factory_path:
                raise ValueError("agentcache.controller_factory must be a non-empty import path")
            factory = cast(ControllerFactory, _resolve_object(factory_path))
            self.controller = factory(self.backend_pool_info(owner), settings)
            self._install_prefix_lookup_observer(owner)
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
        retained_at_monotonic_s = time.monotonic()
        dispatch_now = self.controller.on_request_arrival(
            request.request_id,
            metadata,
            request.num_prompt_tokens,
            self.backend_pool_info(owner),
            request,
        )
        if dispatch_now:
            self.tracked_native.add(request.request_id)
            self._add_to_native_waiting(request, retained_at_monotonic_s, native_add)

    def before_schedule(self, owner: Scheduler, native_add: Callable[[Request], None]) -> None:
        """Run one AgentCache cycle and transfer admitted requests to native waiting."""
        self._drain_lifecycle_signals()
        if self.controller is None:
            return
        now = time.monotonic()
        lightweight_check = getattr(self.controller, "run_due_lightweight_checks", None)
        if callable(lightweight_check):
            lightweight_check(now)
        needs_cycle = self.controller.needs_schedule_cycle(now)
        if not needs_cycle and not self._prefix_refresh_pending:
            return
        if self._prefix_refresh_pending:
            self._observe_retained_prefix_cache(owner)
            self._prefix_refresh_pending = False
        if not needs_cycle:
            return
        self.controller.schedule_cycle(self.backend_pool_info(owner))
        for entry, target in self.controller.consume_admitted_requests():
            if target != DispatchTarget(self.backend_id, self.dp_rank):
                raise RuntimeError("embedded vLLM bridge received a target for another backend or DP rank")
            self.tracked_native.add(entry.request_id)
            self._add_to_native_waiting(entry.retained_request, entry.arrived_at_monotonic_s, native_add)

    @staticmethod
    def _add_to_native_waiting(
        request: Request,
        queued_at_monotonic_s: float,
        native_add: Callable[[Request], None],
    ) -> None:
        """Insert a request natively while preserving its original EngineCore queue start.

        vLLM records ``QUEUED`` inside ``Scheduler.add_request``. Requests retained by AgentInfer reach that call
        later, so the native queue metric would otherwise omit RequestPool residence. Retimestamping only the newly
        emitted event keeps vLLM's scheduling state and ``SCHEDULED - QUEUED`` metric calculation unchanged.
        """
        existing_event_count = len(request.events)
        native_add(request)
        for event in request.events[existing_event_count:]:
            if event.type == EngineCoreEventType.QUEUED:
                event.timestamp = min(event.timestamp, queued_at_monotonic_s)
                return

    def _observe_retained_prefix_cache(self, owner: Scheduler) -> None:
        """Refresh retained paused reasoning requests after a completion-triggered deferred observation.

        The coordinator lookup is deliberately used instead of ``get_computed_blocks``: it does not allocate blocks,
        record prefix-cache statistics, or emit KV-cache events for a synthetic Scheduler observation.
        ``segment_share_tokens`` intentionally records all immediately reusable prefix tokens without attempting to
        distinguish residual self KV from another Program's shared prefix.
        """
        if self.controller is None:
            return
        kv_cache_manager = getattr(owner, "kv_cache_manager", None)
        prefix_lookup_enabled = getattr(kv_cache_manager, "prefix_cache_lookup_enabled", None)
        coordinator = getattr(kv_cache_manager, "coordinator", None)
        find_longest_cache_hit = getattr(coordinator, "find_longest_cache_hit", None)
        if find_longest_cache_hit is None:
            logger.warning("AgentInfer prefix-cache observation unavailable: vLLM coordinator lookup is missing")
            return
        for request_id, request in self.controller.prefix_cache_candidates():
            if callable(prefix_lookup_enabled) and not prefix_lookup_enabled(request):
                self.controller.on_prefix_cache_observation(request_id, 0)
                continue
            try:
                _, cached_prefix_tokens = find_longest_cache_hit(
                    request.block_hashes,
                    max(0, int(request.num_tokens) - 1),
                )
            except (AttributeError, TypeError, ValueError):
                logger.warning("AgentInfer prefix-cache observation failed request=%s", request_id, exc_info=True)
                continue
            self.controller.on_prefix_cache_observation(request_id, max(0, int(cached_prefix_tokens)))

    def after_output(
        self,
        owner: Scheduler,
        outputs: Mapping[int, object],
        token_counts_before_update: Mapping[str, tuple[int, int]],
    ) -> None:
        """Publish token deltas and terminal facts after native state is consistent."""
        if self.controller is None:
            return
        completed_request = False
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
                    self._local_prefix_observations.pop(request_id, None)
                    completed_request = True
        if completed_request:
            self._prefix_refresh_pending = True

    def observe_prefix_cache(self, scheduler_output: object) -> None:
        """Forward vLLM's post-prefix-lookup computed-token boundary without touching KV ownership."""
        if self.controller is None:
            return
        for request in getattr(scheduler_output, "scheduled_new_reqs", ()):
            request_id = str(getattr(request, "req_id", ""))
            if request_id in self.tracked_native:
                cached_tokens = max(0, int(getattr(request, "num_computed_tokens", 0)))
                local_observation = self._local_prefix_observations.pop(request_id, None)
                if local_observation is None:
                    self.controller.on_prefix_cache_observation(request_id, cached_tokens)
                else:
                    self.controller.on_prefix_cache_observation(
                        request_id,
                        cached_tokens,
                        local_observation[1],
                        local_observation[0],
                        max(0, local_observation[0] - local_observation[1]),
                    )

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
                    self._local_prefix_observations.pop(request_id, None)
        return retained + native

    def _install_prefix_lookup_observer(self, owner: Scheduler) -> None:
        """Wrap the real vLLM lookup to observe block ownership before allocation mutates ``ref_cnt``."""
        kv_cache_manager = getattr(owner, "kv_cache_manager", None)
        native_get_computed_blocks = getattr(kv_cache_manager, "get_computed_blocks", None)
        if kv_cache_manager is None or not callable(native_get_computed_blocks):
            logger.warning("AgentInfer exact shared-prefix observation unavailable: get_computed_blocks is missing")
            return

        def observed_get_computed_blocks(request: Request):
            computed_blocks, cached_tokens = native_get_computed_blocks(request)
            if request.request_id in self.tracked_native:
                cached = max(0, int(cached_tokens))
                self._local_prefix_observations[request.request_id] = (
                    cached,
                    self._ref_count_shared_tokens(computed_blocks, cached),
                )
            return computed_blocks, cached_tokens

        kv_cache_manager.get_computed_blocks = observed_get_computed_blocks

    @staticmethod
    def _ref_count_shared_tokens(computed_blocks: object, cached_tokens: int) -> int:
        """Return a conservative token-equivalent count of hit blocks already referenced by native requests."""
        if cached_tokens <= 0:
            return 0
        shared_lengths: list[int] = []
        for group in getattr(computed_blocks, "blocks", ()):
            blocks = tuple(group)
            if not blocks:
                continue
            referenced_blocks = sum(int(getattr(block, "ref_cnt", 0)) > 0 for block in blocks)
            shared_lengths.append(cached_tokens * referenced_blocks // len(blocks))
        return min(cached_tokens, min(shared_lengths)) if shared_lengths else 0

    def unfinished_count(self, native_count: int) -> int:
        """Merge retained and native liveness without exposing request objects to EngineCore."""
        retained = self.controller.retained_request_count if self.controller is not None else 0
        return native_count + retained

    def backend_pool_info(self, owner: Scheduler) -> BackendPoolInfo:
        """Build rank-local logical KV-token and request-load facts from native vLLM state."""
        now = time.monotonic()
        running, waiting = owner.get_request_counts()
        native_used_kv_tokens = self._native_used_kv_tokens(owner)
        native_waiting_kv_tokens = self._native_waiting_kv_tokens(owner)
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
                    used_hbm_kv_tokens=native_used_kv_tokens,
                    waiting_hbm_kv_tokens=native_waiting_kv_tokens,
                    expected_reasoning_agent_nums=self.expected_agents,
                ),
            ),
            observed_at_monotonic_s=now,
        )
        return BackendPoolInfo((backend,), observed_at_monotonic_s=now)

    def _native_used_kv_tokens(self, owner: Scheduler) -> int | None:
        """Return native running-request KV use as a conservative logical-token count, when available."""
        kv_cache_manager = getattr(owner, "kv_cache_manager", None)
        usage = getattr(kv_cache_manager, "usage", None)
        if usage is None:
            return None
        try:
            observed = float(usage() if callable(usage) else usage)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(observed) or not 0 <= observed <= 1:
            return None
        return min(self.local_hbm_kv_tokens, max(0, math.ceil(observed * self.local_hbm_kv_tokens)))

    @staticmethod
    def _native_waiting_kv_tokens(owner: Scheduler) -> int | None:
        """Estimate native waiting demand from its queued request contexts without Program-level share attribution."""
        waiting_queues = tuple(
            queue
            for queue in (getattr(owner, "waiting", None), getattr(owner, "skipped_waiting", None))
            if queue is not None
        )
        if not waiting_queues:
            return None
        try:
            return sum(max(0, int(request.num_tokens)) for queue in waiting_queues for request in queue)
        except (AttributeError, TypeError, ValueError):
            return None

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
        output = super().schedule()
        self._agentcache.observe_prefix_cache(output)
        return output

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
        output = super().schedule()
        self._agentcache.observe_prefix_cache(output)
        return output

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
