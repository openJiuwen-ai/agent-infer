"""
Gen2 Strategy 3: Two-Level Scheduler (Short vs Long Request Bifurcation)

Insight: Long-context requests have different optimal scheduling than short requests.
Evolution: Two-level scheduling that detects "short" vs "long" requests and applies
different policies to each tier.

Short requests (prompt <= SHORT_THRESHOLD): SJF to minimize TTFT.
Long requests (prompt > SHORT_THRESHOLD): Chunked prefill at LONG_CHUNK_SIZE tokens/step
to avoid monopolizing the token budget.

Budget split: 20% reserved for long-context chunking, 80% for short-request prefill.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

SHORT_THRESHOLD = 512   # tokens; requests at or below this are "short"
LONG_CHUNK_SIZE = 256   # max tokens per step for long requests


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    """
    Two-level scheduler: SJF for short requests, chunked prefill for long requests.

    Allocates 80% of the token budget to short requests (lowest TTFT) and up to
    20% (capped at 4 chunks) to long requests served via chunked prefill.
    Decode requests are always preserved.
    """
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    seq_budget = max_num_seqs - len(running_requests)
    kv_blocks = available_kv_blocks
    prefill_batch: list[str] = []

    # Partition waiting requests into short and long tiers
    short_requests = sorted(
        [r for r in waiting_requests if r.num_prompt_tokens <= SHORT_THRESHOLD],
        key=lambda r: (r.num_prompt_tokens - r.num_computed_tokens, r.arrival_time_s),
    )
    long_requests = sorted(
        [r for r in waiting_requests if r.num_prompt_tokens > SHORT_THRESHOLD],
        key=lambda r: r.arrival_time_s,
    )

    # Allocate budgets: long tier gets at most 20% of tokens, capped at 4 chunks
    long_budget = min(max_num_batched_tokens // 5, LONG_CHUNK_SIZE * 4)
    short_budget = max_num_batched_tokens - long_budget

    # --- Tier 1: Short requests (SJF) ---
    for req in short_requests:
        tokens = req.num_prompt_tokens - req.num_computed_tokens
        kv_needed = max(1, tokens // 16)
        if (tokens <= short_budget
                and len(prefill_batch) < seq_budget
                and kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            short_budget -= tokens
            kv_blocks -= kv_needed

    # --- Tier 2: Long requests (chunked prefill) ---
    for req in long_requests:
        remaining = req.num_prompt_tokens - req.num_computed_tokens
        chunk = min(remaining, LONG_CHUNK_SIZE)
        kv_needed = max(1, chunk // 16)
        if (chunk <= long_budget
                and len(prefill_batch) < seq_budget
                and kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            long_budget -= chunk
            kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
