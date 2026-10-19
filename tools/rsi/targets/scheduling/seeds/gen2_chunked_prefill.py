"""
Gen2 Strategy 1: Chunked Prefill with Priority Scoring and Decode Preservation

Insight: Chunked prefill lets multiple requests share the token budget each step.
Evolution: Combine chunked prefill with priority scoring AND decode preservation.

Key idea: Always schedule all running decode requests first. Then use remaining
budget for chunked prefill of top-priority waiting requests. Requests already
mid-prefill (num_computed_tokens > 0) are prioritized, followed by shortest
remaining prefill (SJF within chunk size).
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

CHUNK_SIZE = 256  # smaller chunks = better interleaving


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    """
    Chunked prefill scheduler with priority scoring and decode preservation.

    Always decodes all running decode requests first, then fills remaining
    token budget with chunked prefill of high-priority waiting requests.
    Priority: (1) in-progress prefills, (2) shortest remaining, (3) FCFS.
    """
    # Always decode ALL running decode requests
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    # Chunked prefill: each request contributes at most CHUNK_SIZE tokens
    seq_budget = max_num_seqs - len(running_requests)
    token_budget = max_num_batched_tokens
    kv_blocks = available_kv_blocks
    prefill_batch = []

    # Priority: (1) requests already mid-prefill, (2) shortest remaining, (3) arrival order
    def priority(r: RequestInfo):
        remaining = r.num_prompt_tokens - r.num_computed_tokens
        in_progress = 1 if r.num_computed_tokens > 0 else 0
        chunk = min(remaining, CHUNK_SIZE)
        return (-in_progress, chunk, r.arrival_time_s)

    for req in sorted(waiting_requests, key=priority):
        remaining = req.num_prompt_tokens - req.num_computed_tokens
        chunk = min(remaining, CHUNK_SIZE)
        kv_needed = max(1, chunk // 16)
        if (chunk <= token_budget
                and len(prefill_batch) < seq_budget
                and kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            token_budget -= chunk
            kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
