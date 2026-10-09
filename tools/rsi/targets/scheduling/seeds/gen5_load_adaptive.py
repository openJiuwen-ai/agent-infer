"""
gen5_load_adaptive: Adapts scheduling policy based on measured system load.

Load regime detection:
  - LIGHT (util < 0.3): FCFS — no optimization needed, any order works
  - MODERATE (0.3-0.7): Large-first packing — maximize token throughput
  - HEAVY (> 0.7): Shortest-first with starvation protection — minimize
    KV block occupancy, prioritize fast-completing requests

The load metric is composite: KV utilization + queue depth.

Key difference from gen4 strategies: uses a three-regime approach
calibrated to the actual load level, with smooth transitions.
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

    # Load metrics
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)
    queue_pressure = min(1.0, len(waiting_requests) / max(max_num_seqs, 1))
    load = 0.6 * kv_util + 0.4 * queue_pressure

    # Anti-starvation: identify requests waiting too long
    now = max(r.arrival_time_s for r in waiting_requests)
    max_wait = max(now - r.arrival_time_s for r in waiting_requests)
    stale_threshold = max(1.0, max_wait * 0.5)

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

    # Phase 0 (always): starvation protection
    stale = sorted(
        [r for r in waiting_requests if (now - r.arrival_time_s) >= stale_threshold],
        key=lambda r: r.arrival_time_s
    )
    for req in stale:
        if len(prefill_batch) >= seq_budget:
            break
        try_admit(req)

    if load < 0.35:
        # LIGHT load: FCFS
        for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

    elif load < 0.70:
        # MODERATE load: large-first packing + small-fill
        for req in sorted(waiting_requests, key=lambda r: -r.remaining_prompt_tokens):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)
        for req in sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens):
            if len(prefill_batch) >= seq_budget or token_budget <= 0:
                break
            try_admit(req)

    else:
        # HEAVY load: shortest-first (minimize KV block occupancy)
        for req in sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
