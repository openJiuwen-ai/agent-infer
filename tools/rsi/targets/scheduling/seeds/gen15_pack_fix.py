"""
gen15_pack_fix: Gen4 scoring with fixed pack_score for bimodal workloads.

Built on gen4_score_based. Key change:
Pack score fixed from the biased "near ideal" formula to batch efficiency:
  OLD: pack_s = 1 - |tokens - 0.6*budget| / (0.6*budget)  → ≈0 for short requests
  NEW: pack_s = 1 / (1 + tokens/budget)                   → ≈0.977 for 96t, 0.865 for 640t

This correctly rewards small requests for their high batch efficiency (can fit many
per step) vs large requests that monopolize the token budget.

Also: w_short increased (0.2 → 0.25) for better bimodal performance.
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
    age_pressure = min(1.0, max_wait / 2.0)

    # Adaptive weights
    w_age   = 0.3 + 0.4 * age_pressure         # [0.3, 0.7]
    w_kv    = 0.2 + 0.2 * kv_util              # [0.2, 0.4]
    w_short = 0.25                              # slightly higher: better for bimodal
    w_pack  = max(0.05, 0.2 - 0.1 * kv_util)   # [0.05, 0.2] — smaller range

    # Normalize weights
    total_w = w_age + w_kv + w_short + w_pack
    w_age /= total_w; w_kv /= total_w; w_short /= total_w; w_pack /= total_w

    prompt_lens = [r.remaining_prompt_tokens for r in waiting_requests]
    max_len = max(prompt_lens) + _EPS
    min_len = min(prompt_lens)
    len_range = max(max_len - min_len, _EPS)

    def score(req: RequestInfo, wait_t: float) -> float:
        age_s = wait_t / max(max_wait, _EPS)
        short_s = 1.0 - (req.remaining_prompt_tokens - min_len) / len_range
        # Batch efficiency: smaller requests fit more per batch step
        tokens = req.remaining_prompt_tokens
        pack_s = 1.0 / (1.0 + tokens / max(token_budget, 1))
        # KV footprint under pressure
        kv_blocks = max(1, tokens // 16)
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
