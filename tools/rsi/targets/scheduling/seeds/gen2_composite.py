"""
Gen2 Strategy 4: Composite State-Machine Scheduler

Insight: No single heuristic wins everywhere. Adapt based on system state.
Evolution: State-machine scheduler that detects system state and applies the
most appropriate policy for the current conditions.

States (evaluated in priority order):
- KV_PRESSURE : available_kv_blocks < 50 → decode-priority + selective admit (1 request, cache-warm)
- HIGH_TTFT   : queue length > 2x seq capacity → SJF with chunking to drain backlog
- LOW_LOAD    : few waiters and running slots mostly free → FCFS (minimal overhead)
- NORMAL      : balanced multi-objective scoring (wait + short + cache weights)
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

CHUNK_SIZE = 384
KV_PRESSURE_THRESHOLD = 50       # free blocks below this = KV pressure
HIGH_TTFT_QUEUE_MULTIPLIER = 2   # queue > multiplier * seq_budget = HIGH_TTFT state
LOW_LOAD_QUEUE_MAX = 2           # waiters at or below this can be LOW_LOAD
LOW_LOAD_RUNNING_FRACTION = 0.5  # running < this fraction of max_seqs = LOW_LOAD


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    """
    State-machine composite scheduler.

    Evaluates system state (KV_PRESSURE > HIGH_TTFT > LOW_LOAD > NORMAL) and
    applies the optimal policy for each state to balance TTFT, throughput, and
    KV cache utilization.
    """
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    seq_budget = max_num_seqs - len(running_requests)
    kv_blocks = available_kv_blocks
    token_budget = max_num_batched_tokens
    prefill_batch: list[str] = []

    # --- Detect system state ---
    kv_pressure = kv_blocks < KV_PRESSURE_THRESHOLD
    high_ttft = len(waiting_requests) > seq_budget * HIGH_TTFT_QUEUE_MULTIPLIER
    low_load = (
        len(waiting_requests) <= LOW_LOAD_QUEUE_MAX
        and len(running_requests) < max_num_seqs * LOW_LOAD_RUNNING_FRACTION
    )

    if kv_pressure:
        # KV_PRESSURE: admit at most 1 request, strongly prefer cache-warm candidates
        candidates = sorted(
            waiting_requests,
            key=lambda r: (
                -r.prefix_cached_tokens / max(r.num_prompt_tokens, 1),
                r.arrival_time_s,
            ),
        )
        for req in candidates[:1]:
            tokens = req.num_prompt_tokens - req.num_computed_tokens
            kv_needed = max(1, tokens // 16)
            if tokens <= token_budget and kv_blocks >= kv_needed:
                prefill_batch.append(req.request_id)

    elif high_ttft:
        # HIGH_TTFT: SJF with chunking to drain the queue backlog quickly
        for req in sorted(
            waiting_requests,
            key=lambda r: r.num_prompt_tokens - r.num_computed_tokens,
        ):
            chunk = min(req.num_prompt_tokens - req.num_computed_tokens, CHUNK_SIZE)
            kv_needed = max(1, chunk // 16)
            if (chunk <= token_budget
                    and len(prefill_batch) < seq_budget
                    and kv_blocks >= kv_needed):
                prefill_batch.append(req.request_id)
                token_budget -= chunk
                kv_blocks -= kv_needed

    elif low_load:
        # LOW_LOAD: simple FCFS — minimal scheduling overhead
        for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
            tokens = req.num_prompt_tokens - req.num_computed_tokens
            kv_needed = max(1, tokens // 16)
            if (tokens <= token_budget
                    and len(prefill_batch) < seq_budget
                    and kv_blocks >= kv_needed):
                prefill_batch.append(req.request_id)
                token_budget -= tokens
                kv_blocks -= kv_needed

    else:
        # NORMAL: balanced multi-objective scoring (Gen1-Balanced inspired)
        if waiting_requests:
            max_wait = max(r.arrival_time_s for r in waiting_requests)
            min_wait = min(r.arrival_time_s for r in waiting_requests)
            wait_range = max(max_wait - min_wait, 1e-6)
            max_len = max(r.num_prompt_tokens for r in waiting_requests)

            def score(r: RequestInfo) -> float:
                wait_n  = (max_wait - r.arrival_time_s) / wait_range
                short_n = 1.0 - (r.num_prompt_tokens / max(max_len, 1))
                cache_n = r.prefix_cached_tokens / max(r.num_prompt_tokens, 1)
                return 0.5 * wait_n + 0.3 * short_n + 0.2 * cache_n

            for req in sorted(waiting_requests, key=score, reverse=True):
                tokens = req.num_prompt_tokens - req.num_computed_tokens
                kv_needed = max(1, tokens // 16)
                if (tokens <= token_budget
                        and len(prefill_batch) < seq_budget
                        and kv_blocks >= kv_needed):
                    prefill_batch.append(req.request_id)
                    token_budget -= tokens
                    kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
