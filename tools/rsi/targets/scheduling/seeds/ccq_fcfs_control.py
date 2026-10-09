"""CCQ mechanism control: explicit stock-order pass-through."""
from __future__ import annotations

from integrations.frontier.ve_policy_api import ScheduleDecision
from targets.scheduling.skeleton import RequestInfo


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    token_budget = max_num_batched_tokens - sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    seq_budget = max_num_seqs - len(running_requests)
    prefill_batch = []
    for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
        need = req.remaining_prompt_tokens
        if need <= token_budget and len(prefill_batch) < seq_budget:
            prefill_batch.append(req.request_id)
            token_budget -= need
    return ScheduleDecision(
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        defer_ids=[],
        mechanism_applicable=False,
    )
