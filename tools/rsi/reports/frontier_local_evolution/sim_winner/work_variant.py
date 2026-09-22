"""D3Q schedule_batch variant (sim-search candidate; generated). VE_STRUCT:order=fcfs;gate=none;switch=1;dispersion=1"""
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
    queue_pressure = len(waiting_requests) + len(running_requests)
    outputs = sorted(r.num_output_tokens for r in waiting_requests)
    median_output = outputs[len(outputs) // 2] if outputs else 0
    has_prefix_signal = any(
        bool(getattr(r, 'has_prefix_hint', False))
        for r in waiting_requests
    )
    mixed_decode_modes = bool(
        outputs and 4 * outputs[0] <= max(1, median_output)
    )
    if queue_pressure <= max(1, max_num_seqs):
        _order = lambda r: r.arrival_time_s
    elif mixed_decode_modes and not has_prefix_signal:
        _order = lambda r: (-r.remaining_prompt_tokens, r.arrival_time_s)
    else:
        _order = lambda r: (r.remaining_prompt_tokens + r.num_output_tokens, r.arrival_time_s)
    defer_ids = []
    deferred = set(defer_ids)
    prefill_batch = []
    for req in sorted(waiting_requests, key=_order):
        if req.request_id in deferred:
            continue
        need = req.remaining_prompt_tokens
        if need <= token_budget and len(prefill_batch) < seq_budget:
            prefill_batch.append(req.request_id)
            token_budget -= need
    return ScheduleDecision(
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        defer_ids=defer_ids,
    )
