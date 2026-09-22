"""
Gen2 Strategy 2: Preemption-Aware Scheduler

Insight: When KV cache is critically low, preempting decode requests frees KV
blocks for new prefills (which may have cache hits).
Evolution: Actively preempt the LEAST VALUABLE decode request when KV pressure
is critical.

Preemption value = tokens already decoded. Preempt the request that has generated
the fewest output tokens (farthest from completion = least valuable to keep).
After freeing space, admit new prefills preferring cache-warm requests.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

KV_CRITICAL_THRESHOLD = 0.05  # < 5% free blocks = critical
KV_TOTAL_ESTIMATE_MULTIPLIER = 20  # rough: total blocks ~ free_blocks * multiplier at start


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    """
    Preemption-aware scheduler that evicts the least valuable decode request
    under critical KV cache pressure to admit cache-warm prefills.
    """
    # Estimate total KV capacity to determine free fraction
    total_kv_estimate = max(available_kv_blocks * KV_TOTAL_ESTIMATE_MULTIPLIER, 1024)
    kv_free_fraction = available_kv_blocks / total_kv_estimate

    preempt_ids: list[str] = []
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    kv_blocks = available_kv_blocks

    # Under critical KV pressure with waiting requests, preempt least valuable decode
    if kv_free_fraction < KV_CRITICAL_THRESHOLD and waiting_requests and decode_batch:
        decode_running = [r for r in running_requests if not r.is_prefill]
        if decode_running:
            # Preempt the decode request with the fewest output tokens generated
            victim = min(decode_running, key=lambda r: r.num_output_tokens)
            preempt_ids.append(victim.request_id)
            decode_batch = [rid for rid in decode_batch if rid != victim.request_id]
            # Reclaim the victim's KV blocks
            kv_blocks += victim.kv_blocks_used

    token_budget = max_num_batched_tokens
    seq_budget = max_num_seqs - len(running_requests) + len(preempt_ids)
    prefill_batch: list[str] = []

    # Admit new prefills, preferring cache-warm requests (more prefix cached = higher priority)
    def admit_score(r: RequestInfo):
        cache_hit_boost = r.prefix_cached_tokens / max(r.num_prompt_tokens, 1)
        return (-cache_hit_boost, r.arrival_time_s)

    for req in sorted(waiting_requests, key=admit_score):
        tokens = req.num_prompt_tokens - req.num_computed_tokens
        kv_needed = max(1, tokens // 16)
        if (tokens <= token_budget
                and len(prefill_batch) < seq_budget
                and kv_blocks >= kv_needed):
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            kv_blocks -= kv_needed

    return ScheduleDecision(
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        preempt_ids=preempt_ids,
    )
