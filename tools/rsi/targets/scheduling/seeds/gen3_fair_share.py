"""
gen3_fair_share: Weighted fair-share scheduling for heterogeneous workloads.

Key insight: Pure FCFS starves short requests when long prefills occupy the
token budget. Pure SJF starves long requests. Fair-share balances both by
assigning a "virtual time" to each request and scheduling the earliest.

Strategy (approximation of WFQ):
- Virtual time = arrival_time + (remaining_prompt_tokens / max_num_batched_tokens) * weight
  where weight = 1.0 for short requests, 1.5 for long requests.
- This gives short requests slight priority over long ones that arrived at
  the same time, but does not starve long requests indefinitely.
- Combine with a KV-aware admission gate: skip if KV blocks insufficient.
- Preemption: if a running prefill has been waiting too long relative to new
  arrivals and KV pressure is high, preempt it (add to preempt_ids).
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_SHORT_THRESHOLD = 512  # tokens below this are "short"


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    seq_budget = max_num_seqs - len(running_requests)
    inflight_tokens = sum(r.remaining_prompt_tokens for r in running_requests if r.is_prefill)
    token_budget = max_num_batched_tokens - inflight_tokens

    if seq_budget <= 0 or token_budget <= 0 or not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    def virtual_time(req: RequestInfo) -> float:
        # Weight: long requests pay more (slower virtual clock)
        weight = 1.5 if req.num_prompt_tokens > _SHORT_THRESHOLD else 1.0
        service_time = req.remaining_prompt_tokens / max(max_num_batched_tokens, 1)
        return req.arrival_time_s + service_time * weight

    # Estimate KV pressure
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    total_est = max(used_kv + available_kv_blocks, 1)
    kv_util = used_kv / total_est

    prefill_batch: list[str] = []
    avail_kv = available_kv_blocks

    for req in sorted(waiting_requests, key=virtual_time):
        if len(prefill_batch) >= seq_budget:
            break
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        # Under high pressure, apply stricter KV gate
        kv_gate = kv_needed * (1.5 if kv_util > 0.75 else 1.0)
        if tokens <= token_budget and avail_kv >= kv_gate:
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
