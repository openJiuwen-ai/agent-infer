"""
gen13_cache_prio: Cache-hit-aware admission priority.

Built on gen5_bimodal_kv_guard. Key changes:
1. In short-request admission: sort by cache_hit_ratio descending, then FCFS.
   Requests with high prefix_cached_tokens have cheaper prefill (fewer new KV
   blocks needed), so servicing them first reduces KV pressure and improves
   throughput and P50.
2. More accurate KV estimate: use (prompt_tokens - prefix_cached_tokens) for
   actual new KV block needs, not the full prompt length.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_KV_PRESSURE_THRESHOLD = 0.55
_HETEROGENEITY_RATIO = 2.5
_STALE_SECONDS = 2.5


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

    # -- KV pressure measurement (use net new blocks for accuracy) --
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)
    high_kv_pressure = kv_util > _KV_PRESSURE_THRESHOLD

    # -- Workload heterogeneity measurement --
    sorted_lens = sorted(r.remaining_prompt_tokens for r in waiting_requests)
    median_len = sorted_lens[len(sorted_lens) // 2]
    max_len = sorted_lens[-1]
    is_heterogeneous = median_len > 0 and (max_len / median_len) > _HETEROGENEITY_RATIO

    # -- Starvation protection --
    now = max(r.arrival_time_s for r in waiting_requests)
    stale_ids = {r.request_id for r in waiting_requests
                 if (now - r.arrival_time_s) >= _STALE_SECONDS}

    prefill_batch: list[str] = []
    admitted: set[str] = set()
    avail_kv = available_kv_blocks

    def net_kv_needed(req: RequestInfo) -> int:
        """New KV blocks needed = (prompt - already_cached) / block_size."""
        new_tokens = max(0, req.remaining_prompt_tokens - req.prefix_cached_tokens)
        return max(1, new_tokens // 16)

    def cache_hit_ratio(req: RequestInfo) -> float:
        if req.num_prompt_tokens == 0:
            return 0.0
        return req.prefix_cached_tokens / req.num_prompt_tokens

    def try_admit(req: RequestInfo) -> bool:
        nonlocal token_budget, avail_kv
        if req.request_id in admitted:
            return False
        tokens = req.remaining_prompt_tokens
        kv_needed = net_kv_needed(req)
        if tokens <= token_budget and len(prefill_batch) < seq_budget and avail_kv >= kv_needed:
            prefill_batch.append(req.request_id)
            admitted.add(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed
            return True
        return False

    # Phase 0: Always admit stale requests first (anti-starvation)
    for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
        if req.request_id in stale_ids:
            try_admit(req)
        if len(prefill_batch) >= seq_budget:
            break

    if high_kv_pressure and is_heterogeneous:
        short_reqs = [r for r in waiting_requests if r.remaining_prompt_tokens <= median_len]
        long_reqs  = [r for r in waiting_requests if r.remaining_prompt_tokens > median_len]

        # Short requests: sort by cache hit ratio descending (cached-first),
        # then by arrival time (FCFS tie-break).
        for req in sorted(
            short_reqs,
            key=lambda r: (-cache_hit_ratio(r), r.arrival_time_s),
        ):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

        # Long requests: same cache-aware priority + KV gate
        for req in sorted(
            long_reqs,
            key=lambda r: (-cache_hit_ratio(r), r.arrival_time_s),
        ):
            if len(prefill_batch) >= seq_budget:
                break
            kv_needed = net_kv_needed(req)
            if avail_kv >= kv_needed * 1.3:
                try_admit(req)
    else:
        # Default: cache-aware FCFS (cached requests first, FCFS tie-break)
        for req in sorted(
            waiting_requests,
            key=lambda r: (-cache_hit_ratio(r), r.arrival_time_s),
        ):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
