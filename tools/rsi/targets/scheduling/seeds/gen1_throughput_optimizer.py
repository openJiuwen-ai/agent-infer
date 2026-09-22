"""
gen1_throughput_optimizer: Maximize decode throughput via greedy token-budget packing.

Strategy:
- Always decode all running requests (decode throughput = primary metric).
- For prefill admission, greedily pack the token budget using a two-pass approach:
    Pass 1 (large-first): sort waiting requests by descending remaining prompt tokens
            and admit any that still fit.  Large requests fill the budget efficiently
            and avoid the combinatorial overhead of true bin-packing.
    Pass 2 (small-fill):  after pass 1, sweep remaining requests smallest-first to
            use any leftover budget — squeezes out wasted capacity.
- Both passes respect KV block availability and seq_budget caps.
- In-flight prefill tokens are deducted first so the budget estimate is accurate.
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
    # --- Decode batch: all running non-prefill requests ---
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    # --- Token budget after reserving for in-flight prefills ---
    inflight_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - inflight_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    if seq_budget <= 0 or token_budget <= 0:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    admitted: set[str] = set()
    prefill_batch: list[str] = []

    def try_admit(req: RequestInfo) -> bool:
        nonlocal token_budget, available_kv_blocks
        if req.request_id in admitted:
            return False
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if (tokens <= token_budget
                and len(prefill_batch) < seq_budget
                and available_kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            admitted.add(req.request_id)
            token_budget -= tokens
            available_kv_blocks -= kv_needed
            return True
        return False

    # Pass 1: large-first — maximise token utilisation per admitted request
    for req in sorted(waiting_requests, key=lambda r: -r.remaining_prompt_tokens):
        if len(prefill_batch) >= seq_budget:
            break
        try_admit(req)

    # Pass 2: small-fill — use any residual budget with the smallest remaining requests
    for req in sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens):
        if len(prefill_batch) >= seq_budget or token_budget <= 0:
            break
        try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
