"""
gen1_balanced: Multi-objective scoring balancing TTFT, throughput, and cache affinity.

Strategy:
- Score each waiting request on three normalised dimensions:
    w_wait  * wait_norm   — reward requests that have waited longer (anti-starvation)
    w_short * short_norm  — reward shorter prompts (lower TTFT)
    w_cache * cache_norm  — reward high prefix-cache hit ratio (lower KV pressure)
- Weights are dynamic: when the queue contains many long-waiting requests (TTFT
  pressure is high), w_wait increases and w_short decreases, shifting priority
  toward fairness; when the queue is fresh, weights shift toward short-prompt
  throughput.
- Decode batch always contains all running non-prefill requests (decode-first).
- In-flight prefill tokens deducted from budget before admitting new work.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    # --- Decode batch ---
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    if not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    # --- Token budget after in-flight prefills ---
    inflight_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - inflight_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    if seq_budget <= 0 or token_budget <= 0:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    # --- Compute normalisation ranges (O(n) single pass) ---
    earliest = min(r.arrival_time_s for r in waiting_requests)
    latest   = max(r.arrival_time_s for r in waiting_requests)
    wait_range = max(latest - earliest, 1e-9)

    max_prompt = max(r.num_prompt_tokens for r in waiting_requests)

    # --- Detect TTFT pressure: fraction of queue that has waited "long" ---
    # A request is considered a long-waiter if its arrival is in the oldest 20% of
    # the arrival-time span (i.e. it arrived early relative to the rest of the queue).
    long_waiter_threshold = earliest + 0.20 * wait_range
    long_waiters = sum(
        1 for r in waiting_requests if r.arrival_time_s <= long_waiter_threshold
    )
    ttft_pressure = long_waiters / len(waiting_requests)  # in [0, 1]

    # Dynamic weight schedule:
    #   high pressure  → w_wait↑, w_short↓  (protect long-waiters)
    #   low  pressure  → w_wait↓, w_short↑  (maximise short-prompt throughput)
    w_wait  = 0.40 + 0.30 * ttft_pressure   # 0.40 – 0.70
    w_short = 0.40 - 0.20 * ttft_pressure   # 0.20 – 0.40
    w_cache = 0.20                           # fixed: cache affinity bonus

    def score(r: RequestInfo) -> float:
        # wait_norm: 1.0 = waited the longest, 0.0 = just arrived
        wait_norm  = (latest - r.arrival_time_s) / wait_range
        # short_norm: 1.0 = shortest prompt, 0.0 = longest prompt
        short_norm = 1.0 - r.num_prompt_tokens / max(max_prompt, 1)
        # cache_norm: fraction of prompt tokens already in prefix cache
        cache_norm = r.prefix_cached_tokens / max(r.num_prompt_tokens, 1)
        return w_wait * wait_norm + w_short * short_norm + w_cache * cache_norm

    prefill_batch: list[str] = []
    for req in sorted(waiting_requests, key=score, reverse=True):
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if (tokens <= token_budget
                and len(prefill_batch) < seq_budget
                and available_kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            available_kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
