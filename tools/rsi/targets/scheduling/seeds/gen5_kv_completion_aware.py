"""
gen5_kv_completion_aware: Minimize KV block hold-time to maximize throughput.

Core insight from queuing theory (SRPT): minimizing service time for admitted
requests maximizes system throughput. For LLM PD-serving, "service time" for
a prefill request = prefill_time + expected_decode_time. The key resource is
KV blocks: a request holds KV blocks for its ENTIRE lifecycle (prefill + all
decode steps). Minimizing expected block-hold-time = preferring:
  1. Short prompts (small KV footprint)
  2. Short expected outputs (fewer decode steps while holding KV)

Under KV pressure, estimated block-hold-time = kv_blocks_needed * total_steps,
where total_steps ≈ prompt_tokens + expected_output_tokens.

Anti-starvation: requests waiting > 3x mean_wait get a priority boost.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_EXPECTED_OUTPUT_TOKENS = 150   # proxy when actual output length unknown
_STALE_MULTIPLIER = 3.0         # waiting > N * mean_wait → priority boost


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

    now = max(r.arrival_time_s for r in waiting_requests)
    wait_times = [now - r.arrival_time_s for r in waiting_requests]
    mean_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
    stale_threshold = max(0.5, mean_wait * _STALE_MULTIPLIER)

    # KV pressure detection
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)

    def kv_hold_time(req: RequestInfo) -> float:
        """Estimated KV-block-steps: blocks * total_tokens_to_process."""
        kv_blocks = max(1, req.remaining_prompt_tokens // 16)
        total_service = req.remaining_prompt_tokens + _EXPECTED_OUTPUT_TOKENS
        return kv_blocks * total_service

    def priority(req: RequestInfo, wait_t: float) -> float:
        is_stale = wait_t >= stale_threshold
        if is_stale:
            # Stale requests get highest priority (prevent p99 blowup)
            return 1e9 - req.arrival_time_s  # FCFS among stale
        # Otherwise: lower KV hold time = higher priority
        hold = kv_hold_time(req)
        # Normalize: reward shorter hold time
        return -hold  # sort ascending by hold time = descending by -hold

    scored = sorted(
        zip(waiting_requests, wait_times),
        key=lambda rw: priority(rw[0], rw[1]),
        reverse=True,  # highest priority first
    )

    prefill_batch: list[str] = []
    avail_kv = available_kv_blocks

    for req, _ in scored:
        if len(prefill_batch) >= seq_budget:
            break
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if tokens <= token_budget and avail_kv >= kv_needed:
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
