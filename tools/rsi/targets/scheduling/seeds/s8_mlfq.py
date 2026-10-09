"""MLFQ: Multi-Level Feedback Queue - new requests get high priority."""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

# Priority levels: 0=high (new), 1=medium, 2=low (long-running)
LEVEL_THRESHOLDS = [0, 1, 3]  # num_preemptions thresholds

def schedule_batch(waiting_requests, running_requests, max_num_batched_tokens,
                   max_num_seqs, available_kv_blocks, prefix_cache_hit_rate) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    token_budget = max_num_batched_tokens
    seq_budget = max_num_seqs - len(running_requests)
    prefill_batch = []

    def priority(req: RequestInfo) -> int:
        p = req.num_preemptions
        if p == 0: return 0   # high priority
        if p < 3: return 1    # medium
        return 2              # low

    # Sort by priority level first, then arrival time within level
    sorted_waiting = sorted(waiting_requests, key=lambda r: (priority(r), r.arrival_time_s))

    for req in sorted_waiting:
        tokens = req.remaining_prompt_tokens
        if tokens <= token_budget and len(prefill_batch) < seq_budget and available_kv_blocks >= max(1, tokens // 16):
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            available_kv_blocks -= max(1, tokens // 16)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
