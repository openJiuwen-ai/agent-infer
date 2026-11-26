# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Commit one MORI batch using vLLM's dense text request and worker contracts."""

from collections.abc import Iterable
from time import monotonic

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import KVConnectorBlockState, NewRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus


class ScheduleBatch:
    """Accumulate actual allocations before publishing one SchedulerOutput."""

    def __init__(self, scheduler, throttle_prefills: bool):
        self.scheduler = scheduler
        self.timestamp = monotonic()
        self.new_requests: list[Request] = []
        self.resumed_requests: list[Request] = []
        self.running_requests: list[Request] = []
        self.blocks: dict[str, KVCacheBlocks] = {}
        self.tokens: dict[str, int] = {}
        self.has_sync_kv_loads = False
        self.input_budget = scheduler.scheduler_config.max_num_batched_tokens
        self.budget = scheduler.max_num_scheduled_tokens
        if scheduler.pause_state == PauseState.PAUSED_ALL:
            self.budget = 0
        self.defer_prefills = (
            throttle_prefills
            and not scheduler.prefill_capacity_bound
            and any(not request.is_prefill_chunk for request in scheduler.running)
        )

    def remaining(self, limit: int) -> int:
        return min(self.budget - sum(self.tokens.values()), self.input_budget - sum(self.tokens.values()), limit)

    def running(
        self,
        requests: Iterable[Request],
        limit: int,
        *,
        token_caps: dict[str, int] | None = None,
        existing_block_only: bool = False,
    ) -> int:
        """Allocate the selected resident requests without changing queue membership."""
        owner = self.scheduler
        for request in requests:
            if limit <= 0:
                break
            if request.num_output_placeholders and (
                request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                continue
            if owner.current_step < request.next_decode_eligible_step:
                continue
            new_tokens = min(
                request.num_tokens + request.num_output_placeholders - request.num_computed_tokens,
                owner.max_model_len - request.num_computed_tokens - owner.num_sampled_tokens_per_step,
                limit,
            )
            if token_caps is not None:
                new_tokens = min(new_tokens, token_caps.get(request.request_id, 0))
            if existing_block_only:
                used = request.num_computed_tokens % owner.block_size
                if not used:
                    continue
                new_tokens = min(new_tokens, owner.block_size - used)
            if new_tokens <= 0:
                continue
            blocks = owner.kv_cache_manager.allocate_slots(
                request,
                new_tokens,
                num_lookahead_tokens=owner.num_lookahead_tokens,
            )
            if blocks is None:
                # The MORI controller may have staged a writeback at the
                # capacity check. Other resident requests can still proceed.
                continue
            self.running_requests.append(request)
            self.blocks[request.request_id] = blocks
            self.tokens[request.request_id] = new_tokens
            limit -= new_tokens
        return limit

    def waiting(self, queue, requests: Iterable[Request], limit: int, token_caps: dict[str, int]) -> int:
        """Admit only the MORI candidates. Unselected queue entries stay in place."""
        owner = self.scheduler
        for request in requests:
            if limit <= 0 or len(owner.running) + owner.num_waiting_for_streaming_input >= owner.max_num_active_reqs:
                break
            if not owner._handle_blocked_waiting_request(request):
                continue
            if request.num_stale_output_tokens and not request.drop_stale_output:
                continue
            accepted = self._admit(queue, request, min(limit, token_caps[request.request_id]))
            if accepted is None:
                break
            limit -= accepted
        return limit

    def _admit(self, queue, request: Request, limit: int) -> int | None:
        owner = self.scheduler
        manager = owner.kv_cache_manager
        connector = owner.connector
        request_id = request.request_id
        local = external = 0
        asynchronously = False
        local_blocks = manager.empty_kv_cache_blocks
        cache_lookup = request.num_computed_tokens == 0
        if cache_lookup:
            local_blocks, local, request.shared_prefix_boundary, divergent = owner._get_local_prefix_cache_hit(request)
            assert not divergent and local % owner.block_size == 0
            if not request.skip_reading_prefix_cache:
                external, asynchronously = connector.get_num_new_matched_tokens(request, local)
                if external is None:
                    return 0
        computed = local + external if cache_lookup else request.num_computed_tokens
        if asynchronously:
            assert external > 0
            scheduled = 0
        else:
            remaining = request.num_tokens - computed
            if remaining <= 0 or not owner.scheduler_config.enable_chunked_prefill and remaining > limit:
                return None
            scheduled = min(remaining, limit)

        reserved = owner._inflight_prefill_reserved_blocks() if asynchronously else 0
        blocks = manager.allocate_slots(
            request,
            scheduled,
            num_new_computed_tokens=local,
            new_computed_blocks=local_blocks,
            num_lookahead_tokens=0 if asynchronously else owner.num_lookahead_tokens,
            num_external_computed_tokens=external,
            delay_cache_blocks=asynchronously,
            full_sequence_must_fit=owner.scheduler_reserve_full_isl,
            reserved_blocks=reserved,
            has_scheduled_reqs=bool(owner.running),
        )
        if blocks is None:
            return None
        full_blocks = manager.get_blocks(request_id)
        connector.update_state_after_alloc(request, full_blocks, external)
        if cache_lookup:
            manager.record_prefix_cache_stats(request, local)
            if owner.connector_prefix_cache_stats is not None:
                owner.connector_prefix_cache_stats.record(
                    num_tokens=request.num_tokens - local,
                    num_hits=external,
                    preempted=request.num_preemptions > 0,
                )
            if request.prefill_stats and request.num_preemptions <= 0:
                request.prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=local,
                    num_external_cached_tokens=external,
                )

        queue.remove_request(request)
        request.num_computed_tokens = computed
        if asynchronously:
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            owner._inflight_prefills.add(request)
            owner._enqueue_waiting_request(request)
            return 0

        owner.deferred_waiting.discard(request)
        owner.running.append(request)
        if external:
            self.has_sync_kv_loads = True
        if owner.log_stats:
            request.record_event(EngineCoreEventType.SCHEDULED, self.timestamp)
        if request.status == RequestStatus.WAITING:
            self.new_requests.append(request)
        elif request.status == RequestStatus.PREEMPTED:
            self.resumed_requests.append(request)
        else:
            raise RuntimeError(f"Invalid MORI prefill request status: {request.status}")
        self.blocks[request_id] = full_blocks
        self.tokens[request_id] = scheduled
        request.status = RequestStatus.RUNNING
        if computed + scheduled < request.num_tokens:
            owner._inflight_prefills.add(request)
        return scheduled

    def finish(self) -> SchedulerOutput:
        # vLLM constructs the worker's input snapshot before advancing
        # num_computed_tokens and async placeholders.
        owner = self.scheduler
        manager = owner.kv_cache_manager
        connector = owner.connector
        new, resumed = self.new_requests, self.resumed_requests
        if owner.use_v2_model_runner:
            new = [*new, *resumed]
            resumed = []
        new_data = [
            NewRequestData.from_request(
                request,
                self.blocks[request.request_id].get_block_ids(),
                request._all_token_ids if owner.use_v2_model_runner else None,
                uses_mrope=owner.model_uses_mrope,
            )
            for request in new
        ]
        cached_data = owner._make_cached_request_data(self.running_requests, resumed, self.tokens, {}, self.blocks)
        if not owner.use_v2_model_runner:
            owner.prev_step_scheduled_req_ids.clear()
            owner.prev_step_scheduled_req_ids.update(self.tokens)
        total = sum(self.tokens.values())
        assert total <= self.budget and total <= self.input_budget
        assert len(owner.running) <= owner.max_num_running_reqs

        boundary_offloads = manager.take_boundary_state_offloads()
        block_state = KVConnectorBlockState(
            req_ids=set(self.tokens) | (boundary_offloads.keys() & owner.requests.keys()),
            resolve_block_ids=manager.get_block_ids,
            boundary_state_offloads=boundary_offloads,
        )
        copies, retained = manager.take_kv_cache_block_copies()
        if copies:
            owner._free_cow_retained_blocks(retained, owner.sched_step_seq + 1)
        common_prefix = (
            manager.get_num_common_prefix_blocks(owner.running[0].request_id)
            if owner.running
            else [0] * len(owner.kv_cache_config.kv_cache_groups)
        )
        output = SchedulerOutput(
            scheduled_new_reqs=new_data,
            scheduled_cached_reqs=cached_data,
            num_scheduled_tokens=self.tokens,
            total_num_scheduled_tokens=total,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=common_prefix,
            preempted_req_ids=owner.reset_preempted_req_ids,
            finished_req_ids=owner.finished_req_ids,
            free_encoder_mm_hashes=owner.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=owner._get_new_block_ids_to_zero(),
            has_sync_kv_loads=self.has_sync_kv_loads,
            kv_cache_block_copies=copies or None,
            kv_connector_block_state=block_state,
            ec_manager_metadata=owner.encoder_cache_manager.get_manager_metadata(),
        )
        output.kv_connector_metadata = owner._build_kv_connector_meta(connector, output)
        output.kv_connector_block_state = None
        if owner.defer_block_free and total:
            owner.sched_step_seq += 1
        owner._update_after_schedule(output)
        return output
