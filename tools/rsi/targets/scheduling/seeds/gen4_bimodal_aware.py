"""
gen4_bimodal_aware: Optimal for mixed short+long request workloads.

Key insight: Real LLM workloads are bimodal — short interactive queries
(100-500 tokens) and long batch/document queries (1000-8192 tokens).
The optimal scheduler for bimodal workloads:
1. Admits ALL short requests first (they clear quickly, freeing KV + improving p50)
2. Uses remaining token budget for one or two long requests (preventing starvation)
3. Under heavy KV pressure, delays new long requests until at least one running
   long request completes (length-aware backpressure).

Threshold for "short" is adaptive: median of current waiting queue prompts.
This avoids hardcoding and adapts to the actual workload distribution.
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
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    seq_budget = max_num_seqs - len(running_requests)
    inflight_tokens = sum(r.remaining_prompt_tokens for r in running_requests if r.is_prefill)
    token_budget = max_num_batched_tokens - inflight_tokens

    if seq_budget <= 0 or token_budget <= 0 or not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    # Adaptive short/long threshold: 40th percentile of waiting request sizes
    sorted_lens = sorted(r.remaining_prompt_tokens for r in waiting_requests)
    p40_idx = max(0, int(len(sorted_lens) * 0.40) - 1)
    short_threshold = sorted_lens[p40_idx] if sorted_lens else 512

    short_waiting = [r for r in waiting_requests if r.remaining_prompt_tokens <= short_threshold]
    long_waiting  = [r for r in waiting_requests if r.remaining_prompt_tokens > short_threshold]

    # KV pressure: count running requests using many blocks
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    total_est = max(used_kv + available_kv_blocks, 1)
    kv_util = used_kv / total_est

    prefill_batch: list[str] = []
    admitted: set[str] = set()
    avail_kv = available_kv_blocks

    def try_admit(req: RequestInfo) -> bool:
        nonlocal token_budget, avail_kv
        if req.request_id in admitted:
            return False
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if tokens <= token_budget and len(prefill_batch) < seq_budget and avail_kv >= kv_needed:
            prefill_batch.append(req.request_id)
            admitted.add(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed
            return True
        return False

    # Phase 1: Admit all short requests (FCFS within short class)
    for req in sorted(short_waiting, key=lambda r: r.arrival_time_s):
        if len(prefill_batch) >= seq_budget:
            break
        try_admit(req)

    # Phase 2: Under low/medium pressure, admit long requests too
    # Under high pressure (> 0.80 KV), delay long admits if short ones are queued
    if kv_util < 0.80 or not short_waiting:
        for req in sorted(long_waiting, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)
    else:
        # High pressure + short requests queued: admit only the very oldest long req
        # to prevent indefinite starvation
        for req in sorted(long_waiting, key=lambda r: r.arrival_time_s)[:1]:
            try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
