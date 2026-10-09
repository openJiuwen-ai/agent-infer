"""SJF: Shortest Job First — prioritize requests with fewest remaining prompt tokens.

Rationale: shorter prefills finish faster, freeing token budget and KV blocks
sooner, which improves throughput and reduces head-of-line blocking.
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
    """Shortest Job First: admit smallest remaining_prompt_tokens first."""
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    running_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - running_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    prefill_batch: list[str] = []
    # Sort ascending by remaining prompt tokens (shortest first)
    sorted_waiting = sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens)

    for req in sorted_waiting:
        tokens_needed = req.remaining_prompt_tokens
        kv_needed = max(1, tokens_needed // 16)
        if (tokens_needed <= token_budget
                and len(prefill_batch) < seq_budget
                and available_kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            token_budget -= tokens_needed
            available_kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
