"""KV-gated FCFS: only admit prefill when KV utilization is below threshold."""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

KV_PRESSURE_THRESHOLD = 0.85
_total_blocks_seen: int = 0

def schedule_batch(waiting_requests, running_requests, max_num_batched_tokens,
                   max_num_seqs, available_kv_blocks, prefix_cache_hit_rate) -> ScheduleDecision:
    global _total_blocks_seen
    # Infer total blocks from max seen (increases monotonically until cache fills)
    # At startup all blocks are free, so the first call gives us the total.
    if available_kv_blocks > _total_blocks_seen:
        _total_blocks_seen = available_kv_blocks
    total_blocks = max(_total_blocks_seen, 1)
    kv_util = 1.0 - available_kv_blocks / total_blocks

    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    token_budget = max_num_batched_tokens
    seq_budget = max_num_seqs - len(running_requests)
    prefill_batch = []
    if kv_util < KV_PRESSURE_THRESHOLD:
        for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
            tokens = req.remaining_prompt_tokens
            if tokens <= token_budget and len(prefill_batch) < seq_budget and available_kv_blocks >= max(1, tokens // 16):
                prefill_batch.append(req.request_id)
                token_budget -= tokens
                available_kv_blocks -= max(1, tokens // 16)
    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
