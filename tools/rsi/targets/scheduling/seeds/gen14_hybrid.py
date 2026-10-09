"""
gen14_hybrid: KV-pressure-adaptive scheduling with SRPT at low pressure.

Combines:
- gen5_bimodal_kv_guard's short-first+KV-gate when under pressure (P99 protection)
- gen7_age_srpt's SRPT ordering when NOT under pressure (P50 improvement at low load)
- Longer stale threshold (5.0s) to prevent SRPT starvation

Hypothesis: SRPT is only risky when KV is tight (causes starvation via KV blocking).
Under low-to-medium KV pressure, SRPT safely improves P50 without P99 harm.
Under high pressure, fall back to gen5's proven short-first+gate strategy.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_KV_PRESSURE_THRESHOLD = 0.55
_HETEROGENEITY_RATIO = 2.5
_STALE_SECONDS = 5.0   # longer window: SRPT starvation risk is lower


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

    # -- Starvation protection (longer window since SRPT has its own age-bonus) --
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
        # Under KV crisis: protect short requests, gate long ones
        short_reqs = [r for r in waiting_requests if r.remaining_prompt_tokens <= median_len]
        long_reqs  = [r for r in waiting_requests if r.remaining_prompt_tokens > median_len]

        for req in sorted(short_reqs, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

        for req in sorted(long_reqs, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            tokens = req.remaining_prompt_tokens
            kv_needed = max(1, tokens // 16)
            if avail_kv >= kv_needed * 1.3:
                try_admit(req)
    else:
        # Low/medium pressure: use age-weighted SRPT for P50 improvement.
        # Age bonus prevents starvation (complemented by 5s stale threshold above).
        if waiting_requests:
            min_len = min(r.remaining_prompt_tokens for r in waiting_requests)
            max_age = max(now - r.arrival_time_s for r in waiting_requests) or 1.0
            for req in sorted(
                waiting_requests,
                key=lambda r: -(
                    (now - r.arrival_time_s) / max_age          # age bonus [0, 1]
                    * min_len / max(r.remaining_prompt_tokens, 1)  # size penalty
                ),
            ):
                if len(prefill_batch) >= seq_budget:
                    break
                try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
