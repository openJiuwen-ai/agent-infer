"""
gen10_wider_short: Widen the 'short' bucket to 75th percentile.

Built on gen5_bimodal_kv_guard. Key changes:
1. Heterogeneity threshold lowered (1.5) → fires on more workloads.
2. Short/long cutoff at p75 instead of p50 (median).
   → 75% of requests treated as short (preferred), 25% are gated.
   → At high load, more requests complete quickly (P50 win),
     and the 25% heaviest are still KV-gated (P99 preserved).
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_KV_PRESSURE_THRESHOLD = 0.55
_HETEROGENEITY_RATIO = 1.5   # triggers more broadly (max/p75 > 1.5)
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

    # -- KV pressure measurement --
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)
    high_kv_pressure = kv_util > _KV_PRESSURE_THRESHOLD

    # -- Workload heterogeneity measurement using p75 as cutoff --
    sorted_lens = sorted(r.remaining_prompt_tokens for r in waiting_requests)
    n = len(sorted_lens)
    p75_len = sorted_lens[int(n * 0.75)]   # 75th percentile
    max_len = sorted_lens[-1]
    # Trigger if top-25% are significantly longer than median
    is_heterogeneous = p75_len > 0 and (max_len / p75_len) > _HETEROGENEITY_RATIO

    # -- Starvation protection --
    now = max(r.arrival_time_s for r in waiting_requests)
    stale_ids = {r.request_id for r in waiting_requests
                 if (now - r.arrival_time_s) >= _STALE_SECONDS}

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

    # Phase 0: Always admit stale requests first (anti-starvation)
    for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
        if req.request_id in stale_ids:
            try_admit(req)
        if len(prefill_batch) >= seq_budget:
            break

    if high_kv_pressure and is_heterogeneous:
        # Short = bottom 75%, long = top 25% (heaviest)
        short_reqs = [r for r in waiting_requests if r.remaining_prompt_tokens <= p75_len]
        long_reqs  = [r for r in waiting_requests if r.remaining_prompt_tokens > p75_len]

        # Admit short requests in FCFS order
        for req in sorted(short_reqs, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

        # Admit long (heavy) requests only with extra KV margin
        for req in sorted(long_reqs, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            tokens = req.remaining_prompt_tokens
            kv_needed = max(1, tokens // 16)
            if avail_kv >= kv_needed * 1.3:
                try_admit(req)
    else:
        # Default: pure FCFS
        for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
