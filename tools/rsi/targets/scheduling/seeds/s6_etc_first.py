"""ETC (Estimated Time to Completion) first: prioritize requests closest to finishing."""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


def schedule_batch(waiting_requests, running_requests, max_num_batched_tokens,
                   max_num_seqs, available_kv_blocks, prefix_cache_hit_rate) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    token_budget = max_num_batched_tokens
    seq_budget = max_num_seqs - len(running_requests)
    prefill_batch = []
    # Short prompts finish sooner - prioritize by prompt length ascending
    for req in sorted(waiting_requests, key=lambda r: r.num_prompt_tokens):
        tokens = req.remaining_prompt_tokens
        if tokens <= token_budget and len(prefill_batch) < seq_budget and available_kv_blocks >= max(1, tokens // 16):
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            available_kv_blocks -= max(1, tokens // 16)
    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
