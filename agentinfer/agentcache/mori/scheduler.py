# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""MORI request selection and two-tier cache scheduling on native vLLM queues."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import Request, RequestStatus

from agentinfer.agentcache.mori.connector import MoriCPUOffloadConnector
from agentinfer.agentcache.mori.decision_trace import DecisionTrace
from agentinfer.agentcache.mori.hiradix import LOAD_BACK_THRESHOLD, HiRadix
from agentinfer.agentcache.mori.kv_cache_manager import MoriKVCacheManager
from agentinfer.agentcache.mori.programs import request_program
from agentinfer.agentcache.mori.schedule_batch import ScheduleBatch
from agentinfer.agentcache.mori.vllm_compat import (
    install_mori_inline_eviction_writeback,
    install_mori_request_preprocessing,
    mori_gpu_constructors,
)

logger = init_logger("vllm.agentinfer.mori.scheduler")

_MAX_NEW_TOKENS = 4096
_INITIAL_NEW_TOKEN_RATIO = 0.7
_MIN_NEW_TOKEN_RATIO = _INITIAL_NEW_TOKEN_RATIO * 0.14
_RATIO_DECAY = (_INITIAL_NEW_TOKEN_RATIO - _MIN_NEW_TOKEN_RATIO) / 600
_COLD_PREFIX_TOKENS = 32
_ACTIVE_LPM_LIMIT = 128
_MAX_PREFILL_TOKENS = 16384
_RETRACT_DECODE_STEPS = 20
_WAITING_READY = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


@dataclass
class PrefillBatch:
    tokens: int = 0
    token_caps: dict[str, int] = field(default_factory=dict)
    waiting: list[Request] = field(default_factory=list)
    holding: list[Request] = field(default_factory=list)
    protected: set[int] = field(default_factory=set)
    planned_host: set[int] = field(default_factory=set)
    needs_host_load: bool = False
    full: bool = False

    def add_extend(self, request: Request, tokens: int) -> None:
        self.token_caps[request.request_id] = tokens
        self.tokens += tokens


class MoriScheduler(Scheduler):
    """Choose MORI batches from native vLLM queues using the manager's HiRadix."""

    max_prefill_tokens = _MAX_PREFILL_TOKENS
    _full_prefill_running: set[Request] | None = None
    _overlap = True
    _previous_step_had_forward = False
    _rebind_confirmed_pages = False
    decision_trace = None
    _agentcache = None

    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        if vllm_config.additional_config.get("agentcache", {}).get("strategy") != "mori":
            raise ValueError("MoriScheduler requires the MORI Program strategy")
        async_enabled = isinstance(self, AsyncScheduler)
        if vllm_config.scheduler_config.async_scheduling is not async_enabled:
            raise ValueError("The MORI scheduler class must match vLLM's async scheduling mode")
        if vllm_config.parallel_config.data_parallel_size != 1 or len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("MORI requires a single data-parallel rank and a single dense KV group")

        with mori_gpu_constructors(vllm_config, kv_cache_config):
            super().__init__(vllm_config, kv_cache_config, *args, **kwargs)

        manager = self.kv_cache_manager
        if not isinstance(manager, MoriKVCacheManager):
            raise RuntimeError("MORI scheduler requires its own KV cache manager")
        self._overlap = async_enabled
        try:
            self._initialize_hiradix()
            tree = self.tree_cache
            self.prefix_hash_capacity = (
                tree.device.pool.num_gpu_blocks + tree.host.pool.num_gpu_blocks - 2
            ) * self.block_size
            install_mori_request_preprocessing()
            install_mori_inline_eviction_writeback()
            self._initialize_program_scheduling()
        except BaseException:
            self._agentcache = self.program_cache_adapter = None
            manager.close_hiradix(self.connector)
            raise

    def _initialize_program_scheduling(self) -> None:
        from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks
        from agentinfer.agentcache.mori.program_cache import HiRadixProgramCache

        self.program_cache_adapter = HiRadixProgramCache(self.tree_cache)
        self._agentcache = _VllmAdmissionHooks(self, self.vllm_config, self.kv_cache_config, self.block_size)
        self.tree_cache.placement.current_ref = self._agentcache.program_scheduling.runtime.registry.current_ref

    def add_request(self, request):
        if self._agentcache is None:
            return super().add_request(request)
        return self._agentcache.add_request(self, request, super().add_request)

    def finish_requests(self, request_ids, finished_status):
        if self._agentcache is None:
            return super().finish_requests(request_ids, finished_status)
        return self._agentcache.finish_requests(self, request_ids, finished_status, super().finish_requests)

    @property
    def tree_cache(self) -> HiRadix:
        tree = self.kv_cache_manager.hiradix
        assert tree is not None
        return tree

    def _initialize_hiradix(self) -> None:
        self.new_token_ratio = _INITIAL_NEW_TOKEN_RATIO
        self.failed_decodes: set[str] = set()
        settings = self.vllm_config.additional_config.get("agentcache", {})
        self.max_prefill_tokens = settings.get("mori_max_prefill_tokens", _MAX_PREFILL_TOKENS)
        if type(self.max_prefill_tokens) is not int or self.max_prefill_tokens <= 0:
            raise ValueError("mori_max_prefill_tokens must be a positive integer")
        # The batch-level boundary arithmetic is for a single dense KV group.
        self.dense_kv = not (self.prefix_replay_tokens or self.need_mamba_block_aligned_split)
        if not isinstance(self.waiting, FCFSRequestQueue) or self.hash_block_size != self.block_size:
            raise ValueError("MORI HiRadix requires FCFS queues and one hash per allocation page")
        if self.vllm_config.speculative_config is not None:
            raise ValueError("MORI HiRadix currently supports non-speculative generation")
        connector = self.connector
        if not isinstance(connector, MoriCPUOffloadConnector):
            raise ValueError("MORI HiRadix requires MoriCPUOffloadConnector")
        if connector.mori_batch_loads and (not self.dense_kv or self.needs_kv_cache_zeroing):
            raise ValueError("MORI batch loads require dense KV without block zeroing")
        # EngineCore dispatches each forward before scheduling the next. Only
        # this local dense V2 path can borrow pages before the first H2D ACK.
        share_issued_loads = connector.mori_batch_loads and (
            self.use_v2_model_runner
            and self.defer_block_free
            and self.parallel_config.pipeline_parallel_size == 1
            and not self.recompute_kv_load_failures
        )
        self.kv_cache_manager.bind_hiradix(
            self.vllm_config,
            self.kv_cache_config,
            connector,
            page_size=self.block_size,
            dense=self.dense_kv,
            use_v2_model_runner=self.use_v2_model_runner,
            share_issued_loads=share_issued_loads,
            release_replaced=self._free_cow_retained_blocks,
        )
        self._rebind_confirmed_pages = self.kv_cache_manager.release_replaced_hiradix_blocks is not None
        self.decision_trace = DecisionTrace.from_environment()

    def _reset_idle_ratio(self) -> None:
        self.new_token_ratio = _INITIAL_NEW_TOKEN_RATIO
        self._previous_step_had_forward = False

    def _order_waiting(self, requests):
        """SGLang LPM ranks the device prefix, then defers duplicate cold prefixes."""
        if len(requests) > _ACTIVE_LPM_LIMIT:
            return requests
        cold = set()
        tree = self.tree_cache
        page_size = tree.page_size
        cold_page = _COLD_PREFIX_TOKENS // page_size - 1
        ranked = []
        for request in requests:
            device_tokens = 0
            if not request.skip_reading_prefix_cache:
                depth = request.num_tokens // page_size
                device_tokens = tree.touch_prefix(request.block_hashes, depth, for_lpm=True) * page_size
            duplicate = False
            if device_tokens <= _COLD_PREFIX_TOKENS <= request.num_tokens and not request.skip_reading_prefix_cache:
                if _COLD_PREFIX_TOKENS % page_size == 0 and cold_page < len(request.block_hashes):
                    key = request.block_hashes[cold_page]
                else:
                    lora = request.lora_request
                    key = (
                        request.cache_salt,
                        lora.lora_int_id if lora is not None else None,
                        tuple(request.all_token_ids[:_COLD_PREFIX_TOKENS]),
                    )
                duplicate = key in cold
                cold.add(key)
            ranked.append((duplicate, 0 if duplicate else -device_tokens, request))
        ranked.sort(key=lambda row: row[:2])
        return [request for _, _, request in ranked]

    def _ready(self, request) -> bool:
        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            grammar = request.structured_output_request
            return grammar is not None and grammar.grammar is not None
        return request.status in _WAITING_READY

    def _prefer_prefill(self, chunked, decode, waiting, holding, ranked=None):
        tree = self.tree_cache
        page = tree.page_size
        max_steps = self.max_num_scheduled_tokens
        batch = PrefillBatch()
        if not chunked and (not waiting and not holding or len(decode) >= self.max_num_active_reqs):
            return batch

        reserve = sum(
            min(max(request.max_tokens - request.num_output_tokens, 0), _MAX_NEW_TOKENS) * self.new_token_ratio
            for request in decode
        )
        current = self.kv_cache_manager.block_pool.get_num_free_blocks() * page
        available = current - reserve
        budget = max_steps
        input_budget = self.max_prefill_tokens
        for request in chunked:
            remaining = request.num_tokens - request.num_computed_tokens
            chunk_space = min(budget, int(available))
            step = min(remaining, chunk_space if chunk_space > 0 else budget)
            if step > 0:
                batch.add_extend(request, step)
                padded = round_up(step, page)
                budget -= padded
                input_budget -= padded
                current -= padded + page
                available -= padded + page
                if step == remaining:
                    available -= min(request.max_tokens, _MAX_NEW_TOKENS)

        if ranked is None:
            ranked = self._order_waiting([request for request in waiting if self._ready(request)])
        holding_ranked = [request for request in holding if self._ready(request)]
        remaining_slots = self.max_num_active_reqs - len(decode) - len(chunked)
        # These prefills can join resident requests on the next decode step.
        lora_config = self.lora_config
        if lora_config is not None:
            active_loras = {req.lora_request.lora_int_id for req in self.running if req.lora_request is not None}
        for request in (*holding_ranked, *ranked):
            if remaining_slots <= 0:
                batch.full = bool(decode or batch.tokens)
                break
            if lora_config is not None and request.lora_request is not None:
                if len(active_loras) >= lora_config.max_loras and request.lora_request.lora_int_id not in active_loras:
                    continue
            # A waiter with attached KV resumes from its existing frontier.
            # For fresh admissions, the second lookup after LPM sets recency.
            computed = request.num_computed_tokens
            device, host = (
                ([], [])
                if computed
                else tree.request_match(request, mark_load=True, include_loading=True, plan_load=True)
            )
            if tree.waits_for_load(device, host):
                continue
            borrowed = 0
            for block in host:
                if block.block_id not in batch.planned_host:
                    break
                borrowed += 1
            if borrowed:
                host = host[borrowed:]
            prefix_tokens = computed or (len(device) + borrowed) * page
            prefix_pages = min(prefix_tokens // page, len(request.block_hashes))
            program_id = request_program(request)
            if program_id is not None:
                tree.stamp_prefill(request.block_hashes[prefix_pages - 1] if prefix_pages else None, program_id)
            remaining = request.num_tokens - prefix_tokens
            host_hit = len(host) * page
            output = min(max(request.max_tokens - request.num_output_tokens, 0), _MAX_NEW_TOKENS)
            if remaining + output >= available:
                batch.full = bool(decode or batch.tokens)
                break
            has_prefill = bool(batch.tokens)
            if round_up(remaining - host_hit, page) >= input_budget and has_prefill:
                break
            new_pins = {
                block.block_id for block in device if block.ref_cnt == 0 and block.block_id not in batch.protected
            }
            pinned = len(new_pins) * page
            if remaining + output >= available - pinned:
                batch.full = bool(decode or batch.tokens)
                break
            if host_hit and not self._can_plan_host_restore(batch, device, host, new_pins):
                host_hit = 0
            extend = remaining - host_hit
            padded = round_up(extend, page)
            if padded >= input_budget and has_prefill:
                break
            allocated = padded if padded <= budget else budget // page * page
            if allocated <= 0:
                if host_hit and tree.share_issued_loads and tree.has_inline_writeback():
                    # HiRadix load_back runs before PrefillAdder checks the
                    # last chunk-budget constraint, even for this waiter.
                    tree.prefetch(request, device, host)
                break
            if request in holding:
                batch.holding.append(request)
            else:
                batch.waiting.append(request)
            if device and not host and not borrowed:
                tree.remember_hbm_match(request, device)
            batch.protected.update(new_pins)
            batch.needs_host_load |= bool(host_hit)
            if host_hit and tree.share_batch_loads:
                batch.planned_host.update(block.block_id for block in host)
            remaining_slots -= 1
            budget -= allocated
            batch.add_extend(request, min(extend, allocated))
            if lora_config is not None and request.lora_request is not None:
                active_loras.add(request.lora_request.lora_int_id)
            input_budget -= allocated
            charge = allocated + host_hit + pinned + page
            current -= charge
            available -= charge
            if extend <= allocated:
                available -= min(request.max_tokens, _MAX_NEW_TOKENS)
            if current <= 0 or available <= 0:
                batch.full = True
                break
            if input_budget <= 0 or budget <= 0:
                break
        return batch

    def _can_plan_host_restore(self, batch, device, host, new_pins) -> bool:
        tree = self.tree_cache
        if not tree.has_inline_writeback() or not tree.share_batch_loads:
            return tree.can_load(device, host)
        pages = len(host)
        if pages * tree.page_size < LOAD_BACK_THRESHOLD:
            return False
        reserved = len(batch.planned_host)
        tier = tree.device
        free = tier.free_uncached()
        if pages + reserved > free:
            # HiCache first retries the whole restore after evicting for the
            # host path. Earlier planned loads occupy space and protect their
            # backing pages until the native slots and transfers are built.
            logger.debug(
                "MORI host restore requires eviction: %d new restore, %d reserved restore, %d uncached pages",
                pages,
                reserved,
                free,
            )
            tree.stage_evictions(
                pages,
                frozenset(batch.protected | new_pins),
                reserved_hbm=reserved,
                protected_host=frozenset(batch.planned_host.union(block.block_id for block in host)),
            )
        return pages + reserved <= tier.free_uncached()

    def _reserve_device_pages(self, pages: int, protected: frozenset[int] = frozenset()) -> bool:
        tree = self.tree_cache
        if pages > tree.device.free_uncached():
            tree.stage_evictions(pages, protected)
        return not tree.evictions_pending()

    def _prepare_prefill(self, chunked, batch: PrefillBatch) -> bool:
        if not self.dense_kv:
            return True
        tree = self.tree_cache
        pages = cdiv(batch.tokens, tree.page_size)
        if tree.page_size > 1:
            pages += len(chunked) + len(batch.waiting) + len(batch.holding)
        if batch.needs_host_load:
            free = tree.device.free_uncached()
            # HiRadix first allocates the restored prefix, then checks the
            # aggregate paged extend allocation against the remaining HBM.
            # If the restore itself needs eviction, the native slot path must
            # still stage it before we know which restores can be reserved.
            host_pages = len(batch.planned_host)
            if host_pages and host_pages <= free < host_pages + pages:
                logger.debug(
                    "MORI paged extend requires eviction: %d extend, %d planned restore, %d uncached pages",
                    pages,
                    host_pages,
                    free,
                )
                tree.stage_evictions(
                    pages,
                    frozenset(batch.protected),
                    reserved_hbm=host_pages,
                    protected_host=frozenset(batch.planned_host),
                )
                return not tree.evictions_pending()
            return True
        return self._reserve_device_pages(pages, frozenset(batch.protected))

    def _decode_requests(self, requests):
        return [
            request
            for request in requests
            if request.num_tokens + request.num_output_placeholders > request.num_computed_tokens
            and request.num_computed_tokens < self.max_model_len - self.num_sampled_tokens_per_step
            and self.current_step >= request.next_decode_eligible_step
            and (
                not request.num_output_placeholders
                or request.num_computed_tokens + 2 - request.num_output_placeholders
                < request.num_prompt_tokens + request.max_tokens
            )
        ]

    def _decode_pages(self, requests) -> int:
        page_size = self.tree_cache.page_size
        return sum(request.num_computed_tokens % page_size == 0 for request in requests)

    def _prepare_decode(self, decode) -> bool:
        """Evict for the complete decode batch before native slot allocation."""
        if not self.dense_kv:
            return True
        tree = self.tree_cache
        tier = tree.device
        # A dense non-speculative decode needs at most one new page per
        # request. Actual slot allocation still checks each request below.
        if len(decode) <= tier.free_uncached():
            return not tree.evictions_pending()
        eligible = self._decode_requests(decode)
        needed = self._decode_pages(eligible)
        if not self._reserve_device_pages(needed):
            return False
        if needed <= tier.free_uncached():
            return self._reserve_device_pages(len(eligible))
        if self.deferred_frees:
            return False

        decode.sort(key=lambda request: (request.num_output_tokens, -request.num_prompt_tokens), reverse=True)
        timestamp = monotonic()
        while needed > tier.free_uncached():
            victim = decode[-1]
            if not self._request_blocks_can_be_freed(victim):
                return False
            if len(decode) == 1:
                self.failed_decodes.add(victim.request_id)
                return False
            decode.pop()
            self.running.remove(victim)
            self._preempt_request(victim, timestamp, drop_stale_output=self.requires_kv_delivery)
            # Native vLLM prepends preempted requests. MORI enqueues each
            # retracted request at the tail, which breaks ties in the next LPM.
            self.waiting.remove_request(victim)
            self.waiting.add_request(victim)
            surviving = iter(decode)
            self.running[:] = [request if request.is_prefill_chunk else next(surviving) for request in self.running]
            eligible = self._decode_requests(decode)
            needed = self._decode_pages(eligible)
            # Releasing the victim evicts for future decode steps first. A
            # separate current-step check then sees the resulting leaf set.
            future = cdiv(_RETRACT_DECODE_STEPS * len(decode), self.tree_cache.page_size)
            if not self._reserve_device_pages(future) or not self._reserve_device_pages(needed):
                return False
        return self._reserve_device_pages(len(eligible))

    def _finish_failed_decodes(self, outputs) -> None:
        if not self.failed_decodes:
            return
        failed = self.finish_requests(self.failed_decodes, RequestStatus.FINISHED_ERROR)
        self.failed_decodes.clear()
        for request in failed:
            batch = outputs.setdefault(request.client_index, EngineCoreOutputs())
            batch.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )

    def _plan_prefill(self, chunked, decode, waiting, holding) -> PrefillBatch:
        if self.pause_state == PauseState.UNPAUSED and (
            chunked or self._full_prefill_running is None and len(decode) < self.max_num_active_reqs
        ):
            ranked = self._order_waiting([request for request in waiting if self._ready(request)])
            ready = set(ranked)
            ordered = iter(ranked)
            self.waiting.clear()
            self.waiting.extend(next(ordered) if request in ready else request for request in waiting)
        else:
            waiting, holding, ranked = [], [], []
        return self._prefer_prefill(chunked, decode, waiting, holding, ranked)

    def _schedule_batch(self, chunked, decode, waiting, holding, throttle_prefills):
        tree = self.tree_cache
        selected = ScheduleBatch(self, throttle_prefills)
        batch = PrefillBatch()
        tree.native_schedule_active = True
        try:
            if selected.budget and not self.failed_decodes:
                if (chunked or waiting or holding) and not selected.defer_prefills and not tree.evictions_pending():
                    batch = self._plan_prefill(chunked, decode, waiting, holding)
                    if batch.tokens and self._prepare_prefill(chunked, batch):
                        tree.prefill_host_sources = frozenset(batch.planned_host)
                        try:
                            remaining = selected.running(
                                chunked, selected.remaining(batch.tokens), token_caps=batch.token_caps
                            )
                            if self.pause_state == PauseState.UNPAUSED:
                                remaining = selected.waiting(
                                    self.kv_holding_waiting, batch.holding, remaining, batch.token_caps
                                )
                                selected.waiting(self.waiting, batch.waiting, remaining, batch.token_caps)
                        finally:
                            tree.prefill_host_sources = frozenset()
                        _, outstanding = self.get_request_counts()
                        self.prefill_capacity_bound = outstanding > len(self.deferred_waiting)
                if not selected.tokens:
                    # Async prefill can have staged a transfer even when it
                    # committed no compute. Decode can still use free pages
                    # without reallocating pinned blocks or preempting a peer.
                    pending = tree.evictions_pending()
                    ready = decode if not pending and self._prepare_decode(decode) else []
                    pending = tree.evictions_pending()
                    if pending:
                        ready = self._decode_requests(decode)
                    selected.running(
                        ready,
                        selected.remaining(selected.budget),
                        existing_block_only=pending,
                    )
                    # Keep a zero-token capacity rejection while the same
                    # running batch decodes. Discard a failed nonempty plan.
                    if selected.tokens and batch.tokens:
                        batch = PrefillBatch()
            output = selected.finish()
        finally:
            tree.finish_native_schedule()
            tree.release_unused_hits()
        return output, batch

    def schedule(self, throttle_prefills: bool = False):
        tree = self.tree_cache
        trace = self.decision_trace
        before = trace.snapshot(self) if trace is not None else None
        tree.epoch += 1
        self.current_step += 1
        self.kv_cache_manager.new_step_starts()
        if self._full_prefill_running is not None and not self._full_prefill_running.issubset(self.running):
            self._full_prefill_running = None
        chunked: list[Request] = []
        decode: list[Request] = []
        for request in self.running:
            target = chunked if request.is_prefill_chunk else decode
            target.append(request)
        if chunked and tree.has_inline_writeback():
            for request in chunked:
                if request.last_sched_seq > self.processed_step_seq:
                    self.kv_cache_manager.stash_hiradix_chunk(
                        request, latest_forward=request.last_sched_seq == self.sched_step_seq
                    )
        waiting = list(self.waiting)
        holding = list(self.kv_holding_waiting)
        if self.finished_recving_kv_req_ids:
            # Complete native ACK handling before LPM and batch budgeting.
            for request in (*holding, *waiting):
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    self._handle_blocked_waiting_request(request)
            tree.drain_exclusive_loads()
        output, batch = self._schedule_batch(chunked, decode, waiting, holding, throttle_prefills)
        if self._rebind_confirmed_pages:
            updates = self.kv_cache_manager.take_hiradix_block_table_updates(output.num_scheduled_tokens)
            if updates:
                output.block_table_updates = {**(output.block_table_updates or {}), **updates}
        # SGLang clears batch_is_full when it filters the unfinished chunk
        # out of last_batch or replaces an empty running batch with prefill.
        prefill_refreshes_running = (
            batch.tokens
            and output.num_scheduled_tokens
            and (
                not decode
                or any(
                    request.is_prefill_chunk and request.request_id in output.num_scheduled_tokens
                    for request in self.running
                )
            )
        )
        if output.preempted_req_ids or prefill_refreshes_running:
            self._full_prefill_running = None
        elif batch.full and self.running and not tree.evictions_pending():
            self._full_prefill_running = set(self.running)
        has_forward = bool(output.num_scheduled_tokens)
        if self.failed_decodes:
            self.new_token_ratio = 0.0
        elif not batch.tokens and output.preempted_req_ids:
            survivors = [request for request in self.running if not request.is_prefill_chunk]
            generated = sum(request.num_output_tokens for request in survivors)
            maximum = sum(request.max_tokens for request in survivors)
            self.new_token_ratio = min(1.0, (generated + _RETRACT_DECODE_STEPS * len(survivors)) / (maximum + 1))
        elif not batch.tokens and has_forward:
            self.new_token_ratio = max(self.new_token_ratio - _RATIO_DECAY, _MIN_NEW_TOKEN_RATIO)
        elif (
            not has_forward
            and (not self._overlap or not self._previous_step_had_forward)
            and self.pause_state == PauseState.UNPAUSED
        ):
            # SGLang's overlap loop also considers the prior forward batch.
            self._reset_idle_ratio()
        self._previous_step_had_forward = has_forward
        if trace is not None:
            trace.emit(
                "schedule",
                step=tree.epoch,
                before=before,
                decision={
                    "scheduled": [
                        [trace.request_keys.get(request_id, request_id), tokens]
                        for request_id, tokens in output.num_scheduled_tokens.items()
                    ],
                    "retracted": sorted(
                        trace.request_keys.get(request_id, request_id) for request_id in output.preempted_req_ids or ()
                    ),
                },
                prefill_candidates=[trace.key(request) for request in (*batch.holding, *batch.waiting)],
                prefill_tokens=batch.tokens,
                after=trace.snapshot(self),
            )
        return output

    def has_requests(self) -> bool:
        pending = super().has_requests()
        if (
            not pending
            and not self.requests
            and self.pause_state == PauseState.UNPAUSED
            and self.processed_step_seq == self.sched_step_seq
        ):
            # A quiescent EngineCore can block for input without scheduling
            # another empty forward. Reset the policy at that idle boundary.
            self._reset_idle_ratio()
        return pending

    def update_from_output(self, scheduler_output, model_runner_output):
        previous_outputs = {
            request_id: request.num_output_tokens
            for request_id in model_runner_output.req_ids
            if (request := self.requests.get(request_id)) is not None
        }
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self._finish_failed_decodes(outputs)
        for request_id, previous_output_tokens in previous_outputs.items():
            request = self.requests.get(request_id)
            if request is not None and not request.is_finished():
                confirmed = request.num_computed_tokens - request.num_in_flight_tokens
                self.kv_cache_manager.confirm_hiradix(
                    request,
                    confirmed,
                    refresh_existing=previous_output_tokens == 0 and request.num_output_tokens > 0,
                )
        if (trace := self.decision_trace) is not None:
            trace.emit(
                "result",
                results=[
                    [
                        trace.request_keys.get(output.request_id, output.request_id),
                        output.new_token_ids,
                        output.finished,
                    ]
                    for batch in outputs.values()
                    for output in batch.outputs
                ],
            )
        return outputs

    def shutdown(self) -> None:
        try:
            if self._agentcache is not None:
                self._agentcache.program_scheduling.cancel_gates()
        finally:
            self._agentcache = self.program_cache_adapter = None
            try:
                super().shutdown()
            finally:
                self.kv_cache_manager.close_hiradix(self.connector)


class MoriAsyncScheduler(MoriScheduler, AsyncScheduler):
    """Share MORI selection and keep vLLM's async request bookkeeping."""
