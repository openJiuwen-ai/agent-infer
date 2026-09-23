"""KV-pressure-aware admission scheduler.

Rationale: when KV cache blocks are scarce, throttle new prefills aggressively
to avoid preemptions and keep decode throughput high. When KV is plentiful,
admit generously in arrival order. Also preempts the largest KV consumer
when pressure is critical.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    """KV-pressure-aware: gate admission based on free KV block ratio."""
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    running_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - running_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    # Estimate total KV capacity from used + available
    total_kv_used = sum(r.kv_blocks_used for r in running_requests)
    total_kv_capacity = max(total_kv_used + available_kv_blocks, 1)
    kv_free_ratio = available_kv_blocks / total_kv_capacity

    # Preemption: if KV pressure is critical, preempt the biggest consumer
    preempt_ids: list[str] = []
    if kv_free_ratio < 0.05 and len(running_requests) > 1:
        # Find the running decode request using the most KV blocks
        decode_running = [r for r in running_requests if not r.is_prefill]
        if decode_running:
            worst = max(decode_running, key=lambda r: r.kv_blocks_used)
            preempt_ids.append(worst.request_id)
            decode_batch = [rid for rid in decode_batch if rid != worst.request_id]
            available_kv_blocks += worst.kv_blocks_used
            seq_budget += 1

    # Throttle admission based on KV pressure
    # high pressure -> admit fewer; low pressure -> admit more
    if kv_free_ratio < 0.15:
        max_new_prefills = 1  # very conservative
    elif kv_free_ratio < 0.35:
        max_new_prefills = 2
    else:
        max_new_prefills = seq_budget  # no throttle

    prefill_batch: list[str] = []
    sorted_waiting = sorted(waiting_requests, key=lambda r: r.arrival_time_s)

    for req in sorted_waiting:
        if len(prefill_batch) >= max_new_prefills:
            break
        tokens_needed = req.remaining_prompt_tokens
        kv_needed = max(1, tokens_needed // 16)
        if (tokens_needed <= token_budget
                and len(prefill_batch) < seq_budget
                and available_kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            token_budget -= tokens_needed
            available_kv_blocks -= kv_needed

    return ScheduleDecision(
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        preempt_ids=preempt_ids,
    )
