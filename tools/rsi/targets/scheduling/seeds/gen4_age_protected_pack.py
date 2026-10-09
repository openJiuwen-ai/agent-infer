"""
gen4_age_protected_pack: Two-phase packing with starvation protection.

Key insight: gen1_throughput_optimizer improves p50_ttft and throughput but
WORSENS p99_ttft because large-first admission can starve short requests.
This scheduler adds an age-based priority escalation:

Phase 1 (starvation check): Any request waiting > stale_threshold gets
  immediately admitted via FCFS (prevents p99 blowup).
Phase 2 (packing): For remaining budget, use large-first packing to
  maximize token utilisation.
Phase 3 (fill): Small-fill to use leftover budget.

The stale_threshold adapts: high KV pressure → lower threshold (faster
escalation to prevent deadlock); low pressure → higher threshold (more time
for packing to kick in).
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


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

    # Estimate KV utilization
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    total_est = max(used_kv + available_kv_blocks, 1)
    kv_util = used_kv / total_est

    # Stale threshold: under pressure, escalate quickly; else give more time for packing
    stale_threshold_s = 2.0 if kv_util < 0.7 else 1.0

    now = max((r.arrival_time_s for r in waiting_requests), default=0.0)

    prefill_batch: list[str] = []
    admitted: set[str] = set()
    avail_kv = available_kv_blocks

    def try_admit(req: RequestInfo) -> bool:
        nonlocal token_budget, avail_kv
        if req.request_id in admitted:
            return False
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, tokens // 16)
        if tokens <= token_budget and len(prefill_batch) < seq_budget and avail_kv >= kv_needed:
            prefill_batch.append(req.request_id)
            admitted.add(req.request_id)
            token_budget -= tokens
            avail_kv -= kv_needed
            return True
        return False

    # Phase 1: Starvation protection — admit oldest-waiting requests first
    stale = [r for r in waiting_requests if (now - r.arrival_time_s) >= stale_threshold_s]
    for req in sorted(stale, key=lambda r: r.arrival_time_s):
        if len(prefill_batch) >= seq_budget:
            break
        try_admit(req)

    # Phase 2: Large-first packing for remaining budget
    for req in sorted(waiting_requests, key=lambda r: -r.remaining_prompt_tokens):
        if len(prefill_batch) >= seq_budget:
            break
        try_admit(req)

    # Phase 3: Small-fill with remaining budget
    for req in sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens):
        if len(prefill_batch) >= seq_budget or token_budget <= 0:
            break
        try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
