"""
gen5_bimodal_kv_guard: FCFS with selective KV-pressure gating.

Hypothesis: FCFS is near-optimal for uniform workloads. The only case where
a different policy beats FCFS is when BOTH conditions hold:
  A. KV cache is under significant pressure (util > 0.60)
  B. Workload is heterogeneous (large length variance)

Under those conditions, admitting large requests monopolizes the KV cache
and starves short ones. Gating long requests under pressure + giving
priority to short ones fixes this without hurting uniform workloads.

Algorithm:
  1. Measure KV utilization and workload heterogeneity.
  2. If both A and B: use length-weighted admission (short-first with long-gate)
  3. Otherwise: pure FCFS (guaranteed not to regress vs FCFS baseline)
  4. Always protect against starvation (any request waiting > 4s gets FCFS priority)
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_KV_PRESSURE_THRESHOLD = 0.60
_HETEROGENEITY_RATIO = 4.0    # max_len / median_len > this → "bimodal"
_STALE_SECONDS = 4.0          # time before a request gets FCFS protection


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
        # Short requests first (up to median_len), then FCFS for the rest
        short_reqs = [r for r in waiting_requests if r.remaining_prompt_tokens <= median_len]
        long_reqs  = [r for r in waiting_requests if r.remaining_prompt_tokens > median_len]

        # Admit all short requests (FCFS order among them)
        for req in sorted(short_reqs, key=lambda r: r.arrival_time_s):
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
