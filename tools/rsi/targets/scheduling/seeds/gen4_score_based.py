"""
gen4_score_based: Multi-objective scoring that jointly optimizes TTFT and throughput.

Each waiting request gets a priority score that jointly accounts for:
  S(r) = w_age * age_score(r)          [prevent starvation, improve p99]
        + w_short * short_score(r)      [prefer short requests, improve p50]
        + w_pack * pack_score(r)        [prefer token-budget-efficient requests]
        + w_kv * kv_score(r)            [prefer KV-efficient requests under pressure]

Weights are dynamically adjusted based on queue state:
- If p99-proxy is high (old requests waiting), increase w_age
- Under KV pressure, increase w_kv
- Under token budget abundance, increase w_pack
"""
from __future__ import annotations
import math
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

    # -- Dynamic weight estimation --
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)

    wait_times = [now - r.arrival_time_s for r in waiting_requests]
    max_wait = max(wait_times) if wait_times else 0.0
    # If someone has been waiting > 2s, urgently increase age weight
    age_pressure = min(1.0, max_wait / 2.0)

    # Adaptive weights
    w_age   = 0.3 + 0.4 * age_pressure         # [0.3, 0.7]
    w_kv    = 0.2 + 0.2 * kv_util              # [0.2, 0.4]
    w_short = 0.2                               # constant: slight short preference
    w_pack  = max(0.1, 0.3 - 0.2 * kv_util)    # [0.1, 0.3] decreases under pressure

    # Normalize weights
    total_w = w_age + w_kv + w_short + w_pack
    w_age /= total_w; w_kv /= total_w; w_short /= total_w; w_pack /= total_w

    prompt_lens = [r.remaining_prompt_tokens for r in waiting_requests]
    max_len = max(prompt_lens) + _EPS
    min_len = min(prompt_lens)
    len_range = max(max_len - min_len, _EPS)

    def score(req: RequestInfo, wait_t: float) -> float:
        # age_score: linear ramp from 0 (just arrived) to 1 (waited max_wait)
        age_s = wait_t / max(max_wait, _EPS)
        # short_score: 1 for shortest, 0 for longest
        short_s = 1.0 - (req.remaining_prompt_tokens - min_len) / len_range
        # pack_score: prefer requests close to half token_budget (sweet spot for packing)
        ideal = token_budget * 0.6
        pack_s = 1.0 - abs(req.remaining_prompt_tokens - ideal) / max(ideal, _EPS)
        pack_s = max(0.0, pack_s)
        # kv_score: prefer smaller KV footprint under pressure
        kv_blocks = max(1, req.remaining_prompt_tokens // 16)
        kv_s = 1.0 - (kv_blocks / max(available_kv_blocks, 1))
        kv_s = max(0.0, min(1.0, kv_s))
        return w_age * age_s + w_short * short_s + w_pack * pack_s + w_kv * kv_s

    scored = sorted(
        zip(waiting_requests, wait_times),
        key=lambda rw: score(rw[0], rw[1]),
        reverse=True,
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
