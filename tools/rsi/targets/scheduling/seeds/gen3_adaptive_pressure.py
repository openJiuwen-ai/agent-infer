"""
gen3_adaptive_pressure: Adaptive scheduling based on real-time KV pressure.

Strategy:
- Compute actual KV utilization from available_kv_blocks and a dynamic
  estimate of total capacity (sum of all kv_blocks_used + available).
- Under low pressure (util < 0.60): use large-first two-pass packing
  to maximise token throughput.
- Under medium pressure (0.60-0.80): FCFS admission but capped.
- Under high pressure (> 0.80): strict smallest-first admission to
  minimise per-request KV footprint and avoid evictions.
- Always decode all running decode requests first.
- Never admit if remaining KV blocks can't fit the request.
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
    # Always decode running decode requests
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    seq_budget = max_num_seqs - len(running_requests)
    inflight_tokens = sum(r.remaining_prompt_tokens for r in running_requests if r.is_prefill)
    token_budget = max_num_batched_tokens - inflight_tokens

    if seq_budget <= 0 or token_budget <= 0 or not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    # Estimate KV pressure from available vs used blocks
    used_blocks = sum(r.kv_blocks_used for r in running_requests) + sum(r.kv_blocks_used for r in waiting_requests)
    total_est = max(used_blocks + available_kv_blocks, available_kv_blocks + 1)
    kv_util = 1.0 - available_kv_blocks / total_est

    prefill_batch: list[str] = []
    admitted: set[str] = set()

    def try_admit(req: RequestInfo) -> bool:
        nonlocal token_budget, available_kv_blocks
        if req.request_id in admitted:
            return False
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if (tokens <= token_budget
                and len(prefill_batch) < seq_budget
                and available_kv_blocks >= kv_needed + 4):  # safety margin
            prefill_batch.append(req.request_id)
            admitted.add(req.request_id)
            token_budget -= tokens
            available_kv_blocks -= kv_needed
            return True
        return False

    if kv_util < 0.60:
        # Low pressure: large-first to maximize token throughput
        for req in sorted(waiting_requests, key=lambda r: -r.remaining_prompt_tokens):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)
        # Small-fill pass
        for req in sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens):
            if len(prefill_batch) >= seq_budget or token_budget <= 0:
                break
            try_admit(req)
    elif kv_util < 0.80:
        # Medium pressure: FCFS (stable, avoid starvation)
        for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)
    else:
        # High pressure: smallest-first to minimize KV footprint
        for req in sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
