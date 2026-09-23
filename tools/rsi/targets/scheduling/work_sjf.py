"""SJF variant — admit shortest-remaining-prompt prefills first.

Differs from the FCFS seed only in the waiting-queue sort key: shortest prompt
first (SJF) instead of arrival order. Used to drive a real seed-vs-variant
``ar compare`` round end-to-end.
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
    """SJF: shortest remaining prompt first."""
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    running_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - running_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    prefill_batch = []
    sorted_waiting = sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens)

    for req in sorted_waiting:
        tokens_needed = req.remaining_prompt_tokens
        if (tokens_needed <= token_budget and
                len(prefill_batch) < seq_budget and
                available_kv_blocks >= max(1, tokens_needed // 16)):
            prefill_batch.append(req.request_id)
            token_budget -= tokens_needed
            available_kv_blocks -= max(1, tokens_needed // 16)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
