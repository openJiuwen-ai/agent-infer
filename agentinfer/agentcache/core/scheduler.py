# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Explicit AgentCache admission bridges for vLLM.

Progress-TTL and MORI use the explicit async or sync bridge and a composed
helper, while native waiting/running/KV state and every token-level scheduling
operation remain owned by vLLM.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, cast, get_args, get_type_hints

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.core.api_adapter import LIFECYCLE_SOCKET_ENV, UnixLifecycleReceiver
from agentinfer.agentcache.core.controller import build_controller, build_progress_ttl_controller
from agentinfer.agentcache.core.vllm_logging import attach_agentinfer_to_vllm_logging
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori.vllm_adapter import install_vllm_mori, observe_vllm_offload, require_mori_vllm_version
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DispatchTarget, DpRankInfo
from agentinfer.scheduling.identity import AgentIdentity, JsonMapping, JsonObject, MetadataError, parse_agent_identity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.mori import MORIStrategy
from agentinfer.scheduling.request_pool import RequestPoolEntry, RetainedRequestT
from agentinfer.scheduling.runtime import RequestAdmissionError

logger = logging.getLogger(__name__)

# vLLM 0.23 returns (request_id, client_index); 0.31 returns Request objects.
# Inspect the native public contract once, so retained-only aborts also use the
# host's format even when native_finish returns an empty list.
_NATIVE_FINISH_RETURNS_REQUESTS = get_args(get_type_hints(Scheduler.finish_requests)["return"]) == (Request,)
NativeFinishedRequest = Request | tuple[str, int]


class EmbeddedSchedulerController(Protocol[RetainedRequestT]):
    """AgentCache Scheduler surface consumed by an embedded engine bridge."""

    @property
    def retained_request_count(self) -> int:
        """Return requests held before native admission."""

    @property
    def retained_request_ids(self) -> tuple[str, ...]:
        """Return retained request attempts for host-wide abort."""

    @property
    def requires_program_identity(self) -> bool:
        """Return whether a request without server-resolved identity must fail."""

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

    def on_cache_pressure(self, now_monotonic_s: float) -> None:
        """Request policy evaluation without forcing any placement transition."""

    def run_due_lightweight_checks(self, now_monotonic_s: float) -> None:
        """Apply due factor-only checks without entering a full scheduling cycle."""

    def consume_admitted_requests(
        self,
    ) -> tuple[tuple[RequestPoolEntry[RetainedRequestT], DispatchTarget], ...]:
        """Return and remove newly admitted retained requests."""

    def on_stream_output(self, request_id: str, total_output_tokens: int) -> None:
        """Observe one native output update."""

    def program_ref_for_request(self, request_id: str):
        """Return the exact Program generation for one live request."""

    def on_request_dispatch(self, request_id: str) -> None:
        """Commit prompt cache ownership immediately before native admission."""

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
        if settings.get("strategy") == "mori":
            require_mori_vllm_version()
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
        self.block_size = int(block_size)
        if settings.get("strategy") == "mori" and len(getattr(kv_cache_config, "kv_cache_groups", ())) > 1:
            raise ValueError("MORI capacity accounting requires one native KV cache group")
        reserved_blocks = 1 if settings.get("strategy") == "mori" else 0
        self.local_hbm_kv_tokens = max(0, int(kv_cache_config.num_blocks) - reserved_blocks) * self.block_size
        self.tracked_native: set[str] = set()
        self._rejected_requests: list[Request] = []
        self._local_prefix_observations: dict[str, tuple[int, int]] = {}
        self._mori_context_handed_off: set[str] = set()
        self._context_handoff_at_lookup = False
        self._prefix_refresh_pending = False
        self.controller = (
            build_progress_ttl_controller(self.backend_pool_info(owner), settings)
            if settings.get("strategy", "progress_ttl") == "progress_ttl"
            else build_controller(self.backend_pool_info(owner), settings)
        )
        strategy = getattr(self.controller, "strategy", None)
        if isinstance(strategy, MORIStrategy) and strategy.config.decode_batch_tokens > 0:
            native_budget = getattr(owner, "max_num_scheduled_tokens", None)
            if isinstance(native_budget, bool) or not isinstance(native_budget, int) or native_budget <= 0:
                raise ValueError("MORI decode_batch_tokens requires a positive native max_num_scheduled_tokens")
            if not getattr(owner.scheduler_config, "enable_chunked_prefill", False):
                raise ValueError("MORI decode_batch_tokens requires native chunked prefill")
        self._mori_adapter = install_vllm_mori(owner, strategy) if isinstance(strategy, MORIStrategy) else None
        if self._mori_adapter is not None:
            # Token-equivalent union accounting is exact only for one native KV group.
            MORI_RUNTIME_STATE.footprints.enabled = len(getattr(kv_cache_config, "kv_cache_groups", ())) == 1
        self._install_prefix_lookup_observer(owner)
        self.lifecycle_receiver: UnixLifecycleReceiver | None = None
        socket_path = settings.get("lifecycle_socket_path") or os.environ.get(LIFECYCLE_SOCKET_ENV)
        if socket_path is not None:
            if not isinstance(socket_path, str) or not socket_path:
                raise ValueError("agentcache.lifecycle_socket_path must be a non-empty string")
            self.lifecycle_receiver = UnixLifecycleReceiver(socket_path, self.dp_rank)

    def add_request(self, owner: Scheduler, request: Request, native_add: Callable[[Request], None]) -> None:
        """Retain agentic work or pass compatibility traffic directly to vLLM."""
        self._drain_lifecycle_signals()
        try:
            metadata = _metadata_from_request(request)
        except MetadataError:
            self._reject_request(request, "invalid_program_identity")
            return
        if metadata is None:
            if bool(getattr(self.controller, "requires_program_identity", False)):
                self._reject_request(request, "missing_program_identity")
                return
            native_add(request)
            return
        if (
            isinstance(getattr(self.controller, "strategy", None), MORIStrategy)
            and request.num_prompt_tokens > self.local_hbm_kv_tokens
        ):
            self._reject_request(request, "mori_prompt_exceeds_hbm_capacity")
            return
        max_output = getattr(request.sampling_params, "max_tokens", None)
        if (
            isinstance(getattr(self.controller, "strategy", None), MORIStrategy)
            and isinstance(max_output, int)
            and request.num_prompt_tokens + max_output > self.local_hbm_kv_tokens
        ):
            self._reject_request(request, "mori_request_exceeds_hbm_capacity")
            return
        retained_at_monotonic_s = time.monotonic()
        try:
            dispatch_now = self.controller.on_request_arrival(
                request.request_id,
                metadata,
                request.num_prompt_tokens,
                self.backend_pool_info(owner),
                request,
            )
        except RequestAdmissionError:
            self._reject_request(request, "invalid_program_admission")
            return
        is_mori = isinstance(getattr(self.controller, "strategy", None), MORIStrategy)
        if is_mori:
            program = self.controller.program_ref_for_request(request.request_id)
            if program is None:
                raise RuntimeError("AgentInfer controller did not retain the request-to-Program binding")
        if dispatch_now:
            if is_mori:
                self._prepare_mori_dispatch(request.request_id)
            self.tracked_native.add(request.request_id)
            self._add_to_native_waiting(request, retained_at_monotonic_s, native_add)

    def _reject_request(self, request: Request, reason: str) -> None:
        """Finish an unadmitted request and retain its error until EngineCore polls outputs.

        No native queue, KV allocation, or connector has observed this request,
        so native finish_requests would be a no-op. Keep the engine live until
        the normal output channel delivers a terminal result to this client.
        """
        request.status = RequestStatus.FINISHED_ERROR
        self._rejected_requests.append(request)
        logger.warning("AgentInfer rejected request=%s reason=%s", request.request_id, reason)

    def publish_rejections(self, outputs: dict[int, EngineCoreOutputs]) -> None:
        """Append terminal admission errors to native output batches, exactly once."""
        for request in self._rejected_requests:
            batch = outputs.setdefault(request.client_index, EngineCoreOutputs())
            batch.outputs.append(
                EngineCoreOutput(request_id=request.request_id, new_token_ids=[], finish_reason=FinishReason.ERROR)
            )
            if batch.finished_requests is None:
                batch.finished_requests = set()
            batch.finished_requests.add(request.request_id)
        self._rejected_requests.clear()

    def before_schedule(self, owner: Scheduler, native_add: Callable[[Request], None]) -> None:
        """Run one AgentCache cycle and transfer admitted requests to native waiting."""
        self._drain_lifecycle_signals()
        now = time.monotonic()
        if (
            isinstance(getattr(self.controller, "strategy", None), MORIStrategy)
            and MORI_RUNTIME_STATE.consume_hbm_pressure()
        ):
            self.controller.on_cache_pressure(now)
        self.controller.run_due_lightweight_checks(now)
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
            if isinstance(getattr(self.controller, "strategy", None), MORIStrategy):
                self._prepare_mori_dispatch(entry.request_id)
            self.tracked_native.add(entry.request_id)
            self._add_to_native_waiting(entry.retained_request, entry.arrived_at_monotonic_s, native_add)

    def _prepare_mori_dispatch(self, request_id: str) -> None:
        """Bind native identity while preserving the prior prompt through cache lookup."""
        program = self.controller.program_ref_for_request(request_id)
        if program is None:
            raise RuntimeError("AgentInfer controller lost a request selected for native dispatch")
        if not self._context_handoff_at_lookup:
            self.controller.on_request_dispatch(request_id)
            self._mori_context_handed_off.add(request_id)
        MORI_RUNTIME_STATE.bind_request(request_id, program)

    def schedule_native(self, owner: Scheduler, native_schedule: Callable, *args, **kwargs):
        """Bound mixed iterations only while a native request is actually decoding.

        This chooses a batch within the configured native ceiling. It neither
        reads output bounds nor pauses/cancels requests. Existing decoders are
        visited before prefills within this call; native queue mutations survive
        restoring the remaining requests' original relative order afterwards.
        """
        audit = getattr(getattr(self, "_mori_adapter", None), "writeback_audit", None)
        if audit is not None:
            audit.snapshot_before_allocation()
        strategy = getattr(self.controller, "strategy", None)
        if not isinstance(strategy, MORIStrategy) or strategy.config.decode_batch_tokens <= 0:
            return native_schedule(*args, **kwargs)
        started = time.thread_time()
        native_cpu = 0.0

        def run_native():
            nonlocal native_cpu
            native_started = time.thread_time()
            try:
                return native_schedule(*args, **kwargs)
            finally:
                native_cpu += time.thread_time() - native_started

        try:
            return self._schedule_mori_batch(owner, strategy, run_native)
        finally:
            strategy.batch_policy_cpu_seconds += max(0.0, time.thread_time() - started - native_cpu)

    @staticmethod
    def _schedule_mori_batch(owner: Scheduler, strategy: MORIStrategy, native_schedule: Callable):
        """Apply one transient decode-first budget over the current native queue."""
        running = owner.running
        decoding = {
            id(request)
            for request in running
            if request.num_output_tokens > 0 and request.num_computed_tokens >= request.num_prompt_tokens
        }
        if not decoding:
            return native_schedule()
        original = owner.max_num_scheduled_tokens
        budget = min(original, max(strategy.config.decode_batch_tokens, len(running)))
        if budget == original:
            return native_schedule()
        ordered = sorted(running, key=lambda request: id(request) not in decoding)
        reordered = any(before is not after for before, after in zip(running, ordered, strict=False))
        previous_order = {id(request): index for index, request in enumerate(running)} if reordered else {}
        if reordered:
            running[:] = ordered
        owner.max_num_scheduled_tokens = budget
        try:
            output = native_schedule()
        except Exception:
            strategy.counters["decode_batch_native_errors"] += 1
            raise
        finally:
            owner.max_num_scheduled_tokens = original
            if reordered:
                # Preserve actual removals/preemptions and append new arrivals
                # in native order. Never resurrect a removed request.
                owner.running.sort(key=lambda request: previous_order.get(id(request), len(previous_order)))
        assigned = sum(output.num_scheduled_tokens.values())
        if assigned:
            strategy.counters["decode_batch_capped_iterations"] += 1
            strategy.counters["decode_batch_reordered_iterations"] += int(reordered)
            strategy.counters["decode_batch_assigned_tokens"] += assigned
            strategy.counters["decode_batch_max_budget_tokens"] = max(
                strategy.counters["decode_batch_max_budget_tokens"], budget
            )
            strategy.counters["decode_batch_native_ceiling_tokens"] = max(
                strategy.counters["decode_batch_native_ceiling_tokens"], original
            )
            strategy.counters["decode_batch_max_assigned_tokens"] = max(
                strategy.counters["decode_batch_max_assigned_tokens"], assigned
            )
        return output

    def _commit_mori_context_handoff(self, request_id: str, computed_blocks: object) -> None:
        """Replace old ownership after lookup, then immediately reclaim exact HBM hits."""
        if request_id in self._mori_context_handed_off:
            return
        program = self.controller.program_ref_for_request(request_id)
        if program is None:
            raise RuntimeError("AgentInfer lost a request during native prefix lookup")
        self.controller.on_request_dispatch(request_id)
        MORI_RUNTIME_STATE.bind_request(request_id, program)
        hbm_policy = self._mori_adapter.hbm_policy if self._mori_adapter is not None else None
        if hbm_policy is not None:
            for group in getattr(computed_blocks, "blocks", ()):
                for block in group:
                    block_id = getattr(block, "block_id", None)
                    if isinstance(block_id, int):
                        hbm_policy.on_block_owner_added(block_id, request_id)
        self._mori_context_handed_off.add(request_id)

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
                cache_hit = find_longest_cache_hit(
                    request.block_hashes,
                    max(0, int(request.num_tokens) - 1),
                )
                # Hit length is the second item in both the legacy pair and
                # vLLM 0.31's triple (whose third item is an uncached prefix).
                cached_prefix_tokens = cache_hit[1]
            except (AttributeError, IndexError, TypeError, ValueError):
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
                    self._mori_context_handed_off.discard(request_id)
                    MORI_RUNTIME_STATE.unbind_request(request_id)
                    completed_request = True
        if completed_request:
            self._prefix_refresh_pending = True

    def observe_prefix_cache(self, scheduler_output: object) -> None:
        """Forward vLLM's post-prefix-lookup computed-token boundary without touching KV ownership."""
        observe_execution = getattr(self.controller, "on_native_execution", None)
        if callable(observe_execution):
            for request_id in getattr(scheduler_output, "preempted_req_ids", ()) or ():
                observe_execution(request_id, running=False)
            for request_id, count in (getattr(scheduler_output, "num_scheduled_tokens", {}) or {}).items():
                if count > 0:
                    observe_execution(request_id, running=True)
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

    def cancel_retained(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[NativeFinishedRequest]:
        """Cancel attempts that have not entered native vLLM yet."""
        normalized = (
            self.controller.retained_request_ids
            if request_ids is None
            else ((request_ids,) if isinstance(request_ids, str) else tuple(request_ids))
        )
        cancelled: list[NativeFinishedRequest] = []
        retained_ids = set(self.controller.retained_request_ids)
        for request_id in normalized:
            if request_id not in retained_ids:
                continue
            entry = self.controller.cancel_request(request_id)
            if entry is not None:
                MORI_RUNTIME_STATE.unbind_request(request_id)
                entry.retained_request.status = finished_status
                cancelled.append(
                    entry.retained_request
                    if _NATIVE_FINISH_RETURNS_REQUESTS
                    else (request_id, entry.retained_request.client_index)
                )
        return cancelled

    def finish_requests(
        self,
        owner: Scheduler,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
        native_finish: Callable[[str | Iterable[str] | None, RequestStatus], list[NativeFinishedRequest]],
    ) -> list[NativeFinishedRequest]:
        """Finish retained and native attempts while reporting only terminal lifecycle facts."""
        normalized_request_ids = (
            request_ids if request_ids is None or isinstance(request_ids, str) else tuple(request_ids)
        )
        retained = self.cancel_retained(normalized_request_ids, finished_status)
        token_counts = {
            request_id: request.num_tokens
            for request_id, request in owner.requests.items()
            if request_id in self.tracked_native
        }
        native = native_finish(normalized_request_ids, finished_status)
        for request in native:
            request_id = request[0] if isinstance(request, tuple) else request.request_id
            if request_id in self.tracked_native:
                self.controller.on_request_completion(request_id, token_counts.get(request_id, 0))
                self.tracked_native.discard(request_id)
                self._local_prefix_observations.pop(request_id, None)
                self._mori_context_handed_off.discard(request_id)
                MORI_RUNTIME_STATE.unbind_request(request_id)
        return retained + native

    def _install_prefix_lookup_observer(self, owner: Scheduler) -> None:
        """Wrap the real vLLM lookup to observe block ownership before allocation mutates ``ref_cnt``."""
        kv_cache_manager = getattr(owner, "kv_cache_manager", None)
        native_get_computed_blocks = getattr(kv_cache_manager, "get_computed_blocks", None)
        if kv_cache_manager is None or not callable(native_get_computed_blocks):
            logger.warning("AgentInfer exact shared-prefix observation unavailable: get_computed_blocks is missing")
            return
        if getattr(native_get_computed_blocks, "_agentinfer_prefix_lookup_observer", False):
            return

        def observed_get_computed_blocks(request: Request):
            computed = native_get_computed_blocks(request)
            computed_blocks, cached_tokens = computed[0], computed[1]
            if request.request_id in self.tracked_native:
                controller = getattr(self, "controller", None)
                if isinstance(getattr(controller, "strategy", None), MORIStrategy):
                    self._commit_mori_context_handoff(request.request_id, computed_blocks)
                cached = max(0, int(cached_tokens))
                self._local_prefix_observations[request.request_id] = (
                    cached,
                    self._ref_count_shared_tokens(computed_blocks, cached),
                )
            return computed

        observed_get_computed_blocks._agentinfer_prefix_lookup_observer = True
        kv_cache_manager.get_computed_blocks = observed_get_computed_blocks
        self._context_handoff_at_lookup = True

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
        return native_count + self.controller.retained_request_count + len(self._rejected_requests)

    def close(self) -> None:
        """Release process-local MORI hooks and the lifecycle socket idempotently."""
        if self.lifecycle_receiver is not None:
            self.lifecycle_receiver.close()
            self.lifecycle_receiver = None
        if self._mori_adapter is not None:
            self._mori_adapter.close()
            self._mori_adapter = None

    def backend_pool_info(self, owner: Scheduler) -> BackendPoolInfo:
        """Build rank-local logical KV-token and request-load facts from native vLLM state."""
        now = time.monotonic()
        try:
            running, waiting = owner.get_request_counts()
        except AttributeError:
            # Test doubles and older/newer vLLM layouts may not materialize
            # every auxiliary waiting queue. The primary queues still provide
            # an exact lower-level count for controller construction.
            running = len(getattr(owner, "running", ()))
            waiting = sum(
                len(getattr(owner, name, ())) for name in ("waiting", "skipped_waiting", "kv_holding_waiting")
            )
        native_used_kv_tokens = self._native_used_kv_tokens(owner)
        native_waiting_kv_tokens = self._native_waiting_kv_tokens(owner)
        offload = observe_vllm_offload(owner)
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
                    total_dram_kv_tokens=offload.total_dram_kv_tokens,
                    used_dram_kv_tokens=offload.used_dram_kv_tokens,
                    pending_h2d_kv_tokens=offload.pending_h2d_kv_tokens,
                    pending_d2h_kv_tokens=offload.pending_d2h_kv_tokens,
                    hbm_block_size_tokens=self.block_size,
                    dram_chunk_size_tokens=offload.dram_chunk_size_tokens,
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
        if self.lifecycle_receiver is None:
            return
        for signal in self.lifecycle_receiver.receive():
            scoped = getattr(self.controller, "on_scoped_response_completion", None)
            if isinstance(getattr(self.controller, "strategy", None), MORIStrategy) and callable(scoped):
                released = scoped(signal.program_id, signal.lifecycle, signal.lifecycle_token)
            else:
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

    def schedule(self, *args, **kwargs):
        self._agentcache.before_schedule(self, super().add_request)
        output = self._agentcache.schedule_native(self, super().schedule, *args, **kwargs)
        self._agentcache.observe_prefix_cache(output)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        before = {
            request_id: (request.num_tokens, request.num_output_tokens)
            for request_id in model_runner_output.req_ids
            if request_id in self._agentcache.tracked_native and (request := self.requests.get(request_id)) is not None
        }
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self._agentcache.after_output(self, outputs, before)
        self._agentcache.publish_rejections(outputs)
        return outputs

    def finish_requests(self, request_ids, finished_status: RequestStatus):
        return self._agentcache.finish_requests(self, request_ids, finished_status, super().finish_requests)

    def get_num_unfinished_requests(self) -> int:
        return self._agentcache.unfinished_count(super().get_num_unfinished_requests())

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self._agentcache.close()


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

    def schedule(self, *args, **kwargs):
        self._agentcache.before_schedule(self, super().add_request)
        output = self._agentcache.schedule_native(self, super().schedule, *args, **kwargs)
        self._agentcache.observe_prefix_cache(output)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        before = {
            request_id: (request.num_tokens, request.num_output_tokens)
            for request_id in model_runner_output.req_ids
            if request_id in self._agentcache.tracked_native and (request := self.requests.get(request_id)) is not None
        }
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self._agentcache.after_output(self, outputs, before)
        self._agentcache.publish_rejections(outputs)
        return outputs

    def finish_requests(self, request_ids, finished_status: RequestStatus):
        return self._agentcache.finish_requests(self, request_ids, finished_status, super().finish_requests)

    def get_num_unfinished_requests(self) -> int:
        return self._agentcache.unfinished_count(super().get_num_unfinished_requests())

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self._agentcache.close()


def _metadata_from_request(request: Request) -> AgentIdentity | None:
    """Normalize independent vLLM body-extension and request-header inputs."""
    sampling_params = request.sampling_params
    extra_args = sampling_params.extra_args if sampling_params is not None else None
    headers = request.trace_headers if request.trace_headers is not None else {}
    return parse_agent_identity(
        vllm_xargs=cast(JsonObject, extra_args) if isinstance(extra_args, dict) else None,
        headers=headers,
    )
