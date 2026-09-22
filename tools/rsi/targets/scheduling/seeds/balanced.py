"""Balanced scheduler: SJF when queue is long, FCFS when short, with KV gating.

Rationale: SJF improves throughput under load by clearing small jobs quickly,
but under light load FCFS is fairer and avoids starvation. KV gating prevents
over-admission when memory pressure is high.
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
    """Balanced: SJF under load, FCFS when idle, with KV pressure gating."""
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    running_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - running_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    # Determine queue pressure: long queue -> SJF, short queue -> FCFS
    queue_len = len(waiting_requests)
    use_sjf = queue_len > 4

    if use_sjf:
        sorted_waiting = sorted(
            waiting_requests, key=lambda r: r.remaining_prompt_tokens
        )
    else:
        sorted_waiting = sorted(
            waiting_requests, key=lambda r: r.arrival_time_s
        )

    # KV gating: compute free ratio to limit admission
    total_kv_used = sum(r.kv_blocks_used for r in running_requests)
    total_kv_capacity = max(total_kv_used + available_kv_blocks, 1)
    kv_free_ratio = available_kv_blocks / total_kv_capacity

    # Scale admission limit by KV pressure
    if kv_free_ratio < 0.15:
        max_new_prefills = 1
    elif kv_free_ratio < 0.30:
        max_new_prefills = max(2, seq_budget // 2)
    else:
        max_new_prefills = seq_budget

    prefill_batch: list[str] = []
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

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
