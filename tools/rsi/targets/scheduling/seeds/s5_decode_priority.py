"""Decode-priority: admit 1 prefill per 4 decode steps to maximize decode throughput."""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_step_counter = 0

def schedule_batch(waiting_requests, running_requests, max_num_batched_tokens,
                   max_num_seqs, available_kv_blocks, prefix_cache_hit_rate) -> ScheduleDecision:
    global _step_counter
    _step_counter += 1
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    prefill_batch = []
    if _step_counter % 4 == 0 and waiting_requests:
        req = min(waiting_requests, key=lambda r: r.arrival_time_s)
        tokens = req.remaining_prompt_tokens
        if tokens <= max_num_batched_tokens and available_kv_blocks >= max(1, tokens // 16):
            prefill_batch.append(req.request_id)
    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
