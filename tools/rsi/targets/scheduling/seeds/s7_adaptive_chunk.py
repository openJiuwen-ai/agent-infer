"""Adaptive chunked prefill: dynamically adjust chunk size based on queue depth."""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


def schedule_batch(waiting_requests, running_requests, max_num_batched_tokens,
                   max_num_seqs, available_kv_blocks, prefix_cache_hit_rate) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    # Dynamic chunk size: smaller chunks when queue is deep
    queue_depth = len(waiting_requests)
    if queue_depth > 50:
        chunk_size = 128
    elif queue_depth > 20:
        chunk_size = 256
    else:
        chunk_size = 512

    token_budget = min(max_num_batched_tokens, chunk_size * max(1, len(waiting_requests) // 4))
    seq_budget = max_num_seqs - len(running_requests)
    prefill_batch = []

    for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
        tokens = min(req.remaining_prompt_tokens, chunk_size)
        if tokens <= token_budget and len(prefill_batch) < seq_budget and available_kv_blocks >= max(1, tokens // 16):
            prefill_batch.append(req.request_id)
            token_budget -= tokens

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
