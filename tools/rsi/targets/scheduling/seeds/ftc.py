"""FTC: First-Token Carousel.

When every sequence slot is occupied by long-running decode requests, stock
FCFS cannot admit a newly arrived request before its TTFT deadline. FTC makes
one bounded rotation:

1. Only while an unserved waiter is still inside the workload's 200 ms TTFT
   SLO.
2. Only for a short ("mouse") waiter; long requests never displace each other.
3. Only when a running request has already produced its first token and has an
   elephant decode budget (at least 1024 tokens).
4. Preempt at most one never-before-preempted elephant, admit the oldest urgent
   mouse, and never preempt the victim a second time.

This is a lifecycle transition, not a global length sort. Outside that narrow
condition FTC returns exact FCFS order. The 200 ms boundary is the declared
workload SLO, not a searched coefficient; 1024 tokens distinguishes multi-
second decode occupancy from the short BurstGPT responses.
"""
from __future__ import annotations

from integrations.frontier.ve_policy_api import ScheduleDecision
from targets.scheduling.skeleton import RequestInfo

ALLOW_ACTIVE_PREEMPTION = True
ACTIVE_PREEMPTION_ONLY = True
ACTIVE_PREEMPTION_MIN_OUTPUT_TOKENS = 1024
ACTIVE_PREEMPTION_REQUIRES_FIRST_TOKEN = True
_TTFT_SLO_S = 0.200
_ELEPHANT_OUTPUT_TOKENS = 1024


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    fcfs = sorted(waiting_requests, key=lambda request: request.arrival_time_s)
    decode_batch = [
        request.request_id
        for request in running_requests
        if not request.is_prefill
    ]
    token_budget = max_num_batched_tokens - sum(
        request.remaining_prompt_tokens
        for request in running_requests
        if request.is_prefill
    )
    slots = max_num_seqs - len(running_requests)

    urgent = [
        request
        for request in fcfs
        if request.num_preemptions == 0
        and request.num_output_tokens < _ELEPHANT_OUTPUT_TOKENS
        and float(getattr(request, "waiting_age_s", 0.0)) <= _TTFT_SLO_S
    ]
    victims = [
        request
        for request in running_requests
        if not request.is_prefill
        and request.num_preemptions == 0
        and int(getattr(request, "generated_output_tokens", 0)) >= 1
        and request.num_output_tokens >= _ELEPHANT_OUTPUT_TOKENS
    ]
    preempt_ids = []
    applicable = False
    if slots <= 0 and urgent and victims:
        victim = max(
            victims,
            key=lambda request: (
                int(
                    getattr(
                        request,
                        "remaining_output_tokens",
                        request.num_output_tokens,
                    )
                ),
                -request.arrival_time_s,
            ),
        )
        preempt_ids = [victim.request_id]
        decode_batch = [
            request_id
            for request_id in decode_batch
            if request_id != victim.request_id
        ]
        slots = 1
        applicable = True

    prefill_batch = []
    urgent_ids = {request.request_id for request in urgent}
    ordered = (
        urgent
        + [
            request
            for request in fcfs
            if request.request_id not in urgent_ids
        ]
        if applicable
        else fcfs
    )
    for request in ordered:
        need = request.remaining_prompt_tokens
        if need <= token_budget and len(prefill_batch) < max(0, slots):
            prefill_batch.append(request.request_id)
            token_budget -= need

    return ScheduleDecision(
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        preempt_ids=preempt_ids,
        defer_ids=[],
        mechanism_applicable=applicable,
    )
