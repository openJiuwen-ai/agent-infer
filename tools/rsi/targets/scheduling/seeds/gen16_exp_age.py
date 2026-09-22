"""
gen16_exp_age: Gen4 scoring with exponential age bonus from gen6.

The exponential age bonus (from gen6_kv_srpt) grows faster than linear:
  age_s = 2^(wait/threshold) - 1  → 0 at t=0, 1 at t=threshold, 7 at t=3x threshold

This more aggressively prevents starvation of long requests (P99 protection)
compared to gen4's linear age_s = wait/max_wait.

Combined with gen4's multi-objective scoring (short, pack, kv).
"""
from __future__ import annotations
import math
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_EPS = 1e-9
_AGE_THRESHOLD_S = 2.0   # time for age_s to reach 1.0


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

    # -- Dynamic weight estimation --
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)

    wait_times = [now - r.arrival_time_s for r in waiting_requests]
    max_wait = max(wait_times) if wait_times else 0.0
    age_pressure = min(1.0, max_wait / 2.0)

    # Adaptive weights (same as gen4)
    w_age   = 0.3 + 0.4 * age_pressure
    w_kv    = 0.2 + 0.2 * kv_util
    w_short = 0.2
    w_pack  = max(0.1, 0.3 - 0.2 * kv_util)

    total_w = w_age + w_kv + w_short + w_pack
    w_age /= total_w; w_kv /= total_w; w_short /= total_w; w_pack /= total_w

    prompt_lens = [r.remaining_prompt_tokens for r in waiting_requests]
    max_len = max(prompt_lens) + _EPS
    min_len = min(prompt_lens)
    len_range = max(max_len - min_len, _EPS)

    # Precompute exponential age bonuses; normalize by max
    raw_ages = []
    for wait_t in wait_times:
        ratio = wait_t / _AGE_THRESHOLD_S
        raw_ages.append(max(0.0, math.pow(2.0, ratio) - 1.0))  # 2^ratio - 1
    max_raw_age = max(raw_ages) + _EPS

    def score(req: RequestInfo, wait_t: float, raw_age: float) -> float:
        # Exponential age: grows fast, provides aggressive anti-starvation
        age_s = raw_age / max_raw_age
        # Short score: prefer smaller prompts
        short_s = 1.0 - (req.remaining_prompt_tokens - min_len) / len_range
        # Pack score (gen4's original: prefers "near ideal" size for packing)
        ideal = token_budget * 0.6
        pack_s = 1.0 - abs(req.remaining_prompt_tokens - ideal) / max(ideal, _EPS)
        pack_s = max(0.0, pack_s)
        # KV efficiency
        kv_blocks = max(1, req.remaining_prompt_tokens // 16)
        kv_s = 1.0 - (kv_blocks / max(available_kv_blocks, 1))
        kv_s = max(0.0, min(1.0, kv_s))
        return w_age * age_s + w_short * short_s + w_pack * pack_s + w_kv * kv_s

    scored = sorted(
        zip(waiting_requests, wait_times, raw_ages),
        key=lambda x: score(x[0], x[1], x[2]),
        reverse=True,
    )

    prefill_batch: list[str] = []
    avail_kv = available_kv_blocks

    for req, _, _ in scored:
        if len(prefill_batch) >= seq_budget:
            break
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if tokens <= token_budget and avail_kv >= kv_needed:
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
