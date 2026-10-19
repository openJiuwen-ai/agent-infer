"""
gen5_score_v2: Improved multi-objective score — fixes heavy_mix regression.

Root cause of gen4_score's heavy_mix regression:
  pack_score prefers requests near 60% of token_budget (≈2458 tokens),
  strongly biasing toward large requests in uniform workloads → p99 blowup.

Fixes:
  1. Replace pack_score with a gentler "budget_fit" that just penalizes
     requests that exceed the REMAINING budget (feasibility) rather than
     pushing toward a specific size.
  2. Increase w_age floor (0.4 → 0.5) to ensure stronger anti-starvation.
  3. Remove the pack weight entirely; use the extra weight for age+kv.
  4. kv_score now directly scales with kv_util: 0 weight when KV is free,
     max weight when KV is almost full.

Score per request:
  S(r) = w_age * age_score(r)   [starvation prevention]
        + w_short * short_score(r) [p50 improvement]
        + w_kv * kv_score(r)    [KV efficiency under pressure]

where w_kv grows proportionally with kv_util (no pack bias for uniform workloads).
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_EPS = 1e-9


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
    wait_times = {r.request_id: now - r.arrival_time_s for r in waiting_requests}

    # KV pressure
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)

    # Age pressure: how urgently do we need to admit old requests?
    max_wait = max(wait_times.values()) if wait_times else 0.0
    age_pressure = min(1.0, max_wait / 3.0)  # saturates at 3s

    # --- Adaptive weights (no pack_score bias) ---
    # w_age: 0.5 base + up to 0.3 more under age pressure
    w_age   = 0.50 + 0.30 * age_pressure           # [0.50, 0.80]
    # w_kv: 0 when KV free, up to 0.30 at full utilization
    w_kv    = 0.30 * kv_util                        # [0.00, 0.30]
    # w_short: mild short-request preference (improves p50 slightly)
    w_short = 0.20

    total_w = w_age + w_kv + w_short + _EPS
    w_age /= total_w; w_kv /= total_w; w_short /= total_w

    prompt_lens = [r.remaining_prompt_tokens for r in waiting_requests]
    max_len = max(prompt_lens) + _EPS
    min_len = min(prompt_lens)
    len_range = max(max_len - min_len, _EPS)

    def score(req: RequestInfo) -> float:
        wait_t = wait_times[req.request_id]
        # age_score: linear [0, 1]
        age_s = wait_t / max(max_wait, _EPS)
        # short_score: 1 for shortest, 0 for longest
        short_s = 1.0 - (req.remaining_prompt_tokens - min_len) / len_range
        # kv_score: reward small KV footprint when under pressure
        kv_blocks = max(1, req.remaining_prompt_tokens // 16)
        kv_s = max(0.0, 1.0 - kv_blocks / max(available_kv_blocks, 1))
        return w_age * age_s + w_short * short_s + w_kv * kv_s

    scored = sorted(waiting_requests, key=score, reverse=True)

    prefill_batch: list[str] = []
    avail_kv = available_kv_blocks

    for req in scored:
        if len(prefill_batch) >= seq_budget:
            break
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if tokens <= token_budget and avail_kv >= kv_needed:
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
