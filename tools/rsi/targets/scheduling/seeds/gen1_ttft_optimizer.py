"""
gen1_ttft_optimizer: Minimize Time-To-First-Token for all requests.

Strategy:
- Prioritize waiting requests with the fewest remaining prompt tokens — they are
  closest to completing prefill and will produce their first output token soonest.
- Cap per-request contribution to MAX_CHUNK_SIZE tokens per step so that a single
  large prompt cannot monopolise the entire token budget, keeping the pipeline fair.
- Ties broken by arrival time (FCFS) to prevent starvation of equally-short requests.
- In-flight prefill token cost is deducted from the budget before admitting new work,
  matching the baseline's accounting discipline.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

MAX_CHUNK_SIZE = 512  # max tokens consumed from a single request per step


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    # --- Decode batch: all non-prefill running requests ---
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    # --- Token budget after accounting for in-flight prefill work ---
    inflight_prefill_tokens = sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    token_budget = max_num_batched_tokens - inflight_prefill_tokens
    seq_budget = max_num_seqs - len(running_requests)

    if seq_budget <= 0 or token_budget <= 0:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    # --- Sort by effective chunk size ascending (shortest near-completion first),
    #     then by arrival time to break ties fairly ---
    def priority_key(r: RequestInfo):
        chunk = min(r.remaining_prompt_tokens, MAX_CHUNK_SIZE)
        return (chunk, r.arrival_time_s)

    prefill_batch: list[str] = []
    for req in sorted(waiting_requests, key=priority_key):
        chunk = min(req.remaining_prompt_tokens, MAX_CHUNK_SIZE)
        kv_needed = max(1, chunk // 16)
        if (chunk <= token_budget
                and len(prefill_batch) < seq_budget
                and available_kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            token_budget -= chunk
            available_kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
