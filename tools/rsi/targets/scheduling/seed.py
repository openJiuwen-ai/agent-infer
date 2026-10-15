"""
FCFS seed scheduler — the baseline for fitness normalization.
All evolved schedulers are compared against this.
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
    """FCFS: First-Come-First-Served baseline."""
    # Decode all currently running (non-prefill) requests
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    # Admit new prefills in arrival order
    running_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - running_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    prefill_batch = []
    sorted_waiting = sorted(waiting_requests, key=lambda r: r.arrival_time_s)

    for req in sorted_waiting:
        tokens_needed = req.remaining_prompt_tokens
        if (tokens_needed <= token_budget and
                len(prefill_batch) < seq_budget and
                available_kv_blocks >= max(1, tokens_needed // 16)):
            prefill_batch.append(req.request_id)
            token_budget -= tokens_needed
            available_kv_blocks -= max(1, tokens_needed // 16)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
