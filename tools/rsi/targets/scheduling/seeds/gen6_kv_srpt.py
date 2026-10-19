"""
gen6_kv_srpt: KV-Aware Shortest Remaining Processing Time (KV-SRPT).

Theoretical grounding: SRPT is optimal for minimizing mean sojourn time in
M/G/1 queues. For LLM PD-serving with KV-constrained memory, the "service
time" is not just prefill time but total KV-block-holding time:
    service_time(r) = (prompt_tokens + expected_output_tokens) * kv_blocks

KV-SRPT schedules based on this KV-weighted service time, rather than just
prompt length (SJF) or arrival order (FCFS).

Key improvements over gen4_score:
1. Explicit SRPT-derived priority: minimize kv-hold cost directly
2. Anti-starvation: exponential age bonus (2^(wait/threshold) - 1) grows
   faster than linear → more aggressive protection against p99 blowup
3. No pack_score bias: no hardcoded "ideal" size preference
4. Soft long-request gate: under extreme KV pressure (util > 0.85), long
   requests need 2x the normal KV available before being admitted

This is a principled generalization of KV-completion-aware scheduling.
"""
from __future__ import annotations
import math
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_EXPECTED_OUTPUT_TOKENS = 150  # proxy for unknown output length
_KV_BLOCK_SIZE = 16
_EXTREME_KV_THRESHOLD = 0.85
_STALE_THRESHOLD_S = 3.0      # base stale threshold


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    seq_budget = max_num_seqs - len(running_requests)
    inflight_tokens = sum(r.remaining_prompt_tokens for r in running_requests if r.is_prefill)
    token_budget = max_num_batched_tokens - inflight_tokens

    if seq_budget <= 0 or token_budget <= 0 or not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    now = max(r.arrival_time_s for r in waiting_requests)

    # KV state
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    kv_util = used_kv / max(used_kv + available_kv_blocks, 1)

    def kv_service_cost(req: RequestInfo) -> float:
        """KV-weighted service cost = blocks * total_tokens."""
        blocks = max(1, req.remaining_prompt_tokens // _KV_BLOCK_SIZE)
        total_toks = req.remaining_prompt_tokens + _EXPECTED_OUTPUT_TOKENS
        return blocks * total_toks

    def age_bonus(req: RequestInfo) -> float:
        """Exponential age bonus to prevent starvation."""
        wait = now - req.arrival_time_s
        # Grows exponentially: 0 at t=0, 1 at t=stale_threshold, 7 at t=3x threshold
        ratio = wait / _STALE_THRESHOLD_S
        return max(0.0, math.exp(ratio * math.log(2)) - 1)  # 2^(ratio) - 1

    # Dynamic weight between SRPT cost and age bonus based on KV pressure
    # High KV pressure → weight service-time heavily (SRPT dominates)
    # Low KV pressure → weight age heavily (fairness dominates)
    srpt_weight = 0.4 + 0.5 * kv_util   # [0.4, 0.9]
    age_weight = 1.0 - srpt_weight       # [0.1, 0.6]

    # Normalize service costs for comparability
    costs = [kv_service_cost(r) for r in waiting_requests]
    max_cost = max(costs) + 1e-9
    ages = [age_bonus(r) for r in waiting_requests]
    max_age = max(ages) + 1e-9 if max(ages) > 0 else 1.0

    def priority(req: RequestInfo, cost: float, age: float) -> float:
        cost_score = 1.0 - cost / max_cost   # higher priority = lower cost
        age_score  = age / max_age            # higher priority = older request
        return srpt_weight * cost_score + age_weight * age_score

    scored = sorted(
        zip(waiting_requests, costs, ages),
        key=lambda x: priority(x[0], x[1], x[2]),
        reverse=True,
    )

    prefill_batch: list[str] = []
    avail_kv = available_kv_blocks

    for req, cost, age in scored:
        if len(prefill_batch) >= seq_budget:
            break
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // _KV_BLOCK_SIZE)

        # Soft gate for very large requests under extreme KV pressure
        kv_gate_mult = 2.0 if kv_util > _EXTREME_KV_THRESHOLD else 1.0
        if tokens <= token_budget and avail_kv >= kv_needed * kv_gate_mult:
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed
        elif kv_gate_mult > 1.0 and avail_kv >= kv_needed and age > 2.0:
            # Allow stale requests through even under extreme pressure
            prefill_batch.append(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
