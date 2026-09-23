"""
gen9_age_sjf: Age-discounted SJF among short requests under KV pressure.

Built on gen5_bimodal_kv_guard. Key change:
Within the short-request bucket (when high KV pressure + heterogeneous),
sort by age-discounted length instead of FCFS.

Sort key = remaining_tokens - age_bonus
         = remaining_tokens - (age/max_age) * median_len

Oldest short requests get up to median_len discount on their effective length,
so the true shortest requests go first (P50 win) while older medium-short
requests aren't starved. Long requests are still KV-gated (P99 preserved).
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_KV_PRESSURE_THRESHOLD = 0.55   # tighter: trigger more aggressively (gen7 improvement)
_HETEROGENEITY_RATIO = 2.5      # lower: most real workloads are bimodal (gen7 improvement)
_STALE_SECONDS = 2.5            # faster anti-starvation


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

    # -- Workload heterogeneity measurement --
    sorted_lens = sorted(r.remaining_prompt_tokens for r in waiting_requests)
    median_len = sorted_lens[len(sorted_lens) // 2]
    max_len = sorted_lens[-1]
    is_heterogeneous = median_len > 0 and (max_len / median_len) > _HETEROGENEITY_RATIO

    # -- Starvation protection: identify stale requests --
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
        # Special regime: KV pressure + bimodal workload
        short_reqs = [r for r in waiting_requests if r.remaining_prompt_tokens <= median_len]
        long_reqs  = [r for r in waiting_requests if r.remaining_prompt_tokens > median_len]

        # Age-discounted SJF among short requests:
        # Sort by effective length = remaining_tokens - age_bonus.
        # Shorter requests go first (P50 win), but older requests get a
        # discount so medium-short requests don't starve.
        max_age_short = max((now - r.arrival_time_s) for r in short_reqs) if short_reqs else 1.0
        max_age_short = max_age_short or 1.0
        for req in sorted(
            short_reqs,
            key=lambda r: (
                r.remaining_prompt_tokens
                - (now - r.arrival_time_s) / max_age_short * median_len
            ),
        ):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

        # Admit long requests only if KV budget allows (gate on per-request KV)
        for req in sorted(long_reqs, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            tokens = req.remaining_prompt_tokens
            kv_needed = max(1, tokens // 16)
            # Extra gate: long requests need at least 30% of remaining KV blocks
            if avail_kv >= kv_needed * 1.3:
                try_admit(req)
    else:
        # Default: pure FCFS (no regression vs baseline)
        for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
