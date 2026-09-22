"""
gen1_hybrid: Decode-first + chunk-aware prefill admission + adaptive KV pressure gate.

Strategy:
- Always drain the decode queue first (all running non-prefill requests).
- Account for in-flight prefill token consumption before admitting new prefills.
- Sort waiting requests so those that fit entirely in the remaining token budget
  come first (avoids partial-admit fragmentation); break ties by arrival time (FCFS).
- When KV utilization exceeds 80%, cap new prefill admissions to 1 per step to
  reduce cache eviction pressure and protect decode throughput.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

# Approximate total KV pool size used to estimate utilisation.
# In production this would be passed in; here we use a conservative constant.
_APPROX_TOTAL_KV_BLOCKS = 1024


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    # --- Decode batch: all running requests that are past prefill ---
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    # --- Reserve tokens already committed to in-flight prefills ---
    inflight_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - inflight_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    # --- Estimate KV utilisation to decide admission aggressiveness ---
    kv_used = _APPROX_TOTAL_KV_BLOCKS - available_kv_blocks
    kv_util = kv_used / _APPROX_TOTAL_KV_BLOCKS if _APPROX_TOTAL_KV_BLOCKS > 0 else 0.0

    # Under KV pressure: admit at most 1 new prefill this step.
    max_new_prefills = 1 if kv_util > 0.80 else max_num_seqs

    effective_seq_budget = min(seq_budget, max_new_prefills)

    if effective_seq_budget <= 0 or token_budget <= 0:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    # --- Sort: requests that fit entirely in the current budget first ---
    # Within each group, preserve FCFS order.
    def fit_key(r: RequestInfo):
        tokens = r.remaining_prompt_tokens
        fits = 0 if tokens <= token_budget else 1  # 0 = fits, 1 = overflow
        return (fits, r.arrival_time_s)

    prefill_batch: list[str] = []
    for req in sorted(waiting_requests, key=fit_key):
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if (tokens <= token_budget
                and len(prefill_batch) < effective_seq_budget
                and available_kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            available_kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
