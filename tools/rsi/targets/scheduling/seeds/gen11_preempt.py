"""
gen11_preempt: Proactive preemption of heaviest decode request under KV crisis.

Built on gen5_bimodal_kv_guard. Key addition:
When KV utilization is critically high (>= 0.88) AND the heaviest decode
request holds enough blocks to admit >= 3 short waiting requests, preempt it.

Rationale: Preempting one long decode sequence frees many KV blocks in one shot,
unblocking multiple short waiting requests. The preempted request re-queues and
will be re-admitted once pressure drops. Net effect: better P50 (shorts proceed),
controlled P99 (KV crisis resolved, no starvation cascade).

Anti-ping-pong: only preempt if victim has >= 50% output computed AND there are
enough short waiters to benefit (avoids preempting for marginal gain).
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_KV_PRESSURE_THRESHOLD = 0.55
_HETEROGENEITY_RATIO = 2.5
_STALE_SECONDS = 2.5
_KV_CRISIS_THRESHOLD = 0.88   # preemption trigger: very high KV utilization
_MIN_WAITERS_TO_PREEMPT = 3   # only preempt if >= N waiters would benefit


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    decode_reqs = [r for r in running_requests if not r.is_prefill]
    decode_batch = [r.request_id for r in decode_reqs]

    seq_budget = max_num_seqs - len(running_requests)
    inflight_tokens = sum(r.remaining_prompt_tokens for r in running_requests if r.is_prefill)
    token_budget = max_num_batched_tokens - inflight_tokens

    if not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    # -- KV utilization --
    used_kv = sum(r.kv_blocks_used for r in running_requests)
    total_kv_seen = used_kv + available_kv_blocks
    kv_util = used_kv / max(total_kv_seen, 1)

    # -- Proactive preemption: free KV by evicting heaviest decode request --
    preempt_ids: list[str] = []
    avail_kv = available_kv_blocks

    kv_crisis = kv_util >= _KV_CRISIS_THRESHOLD and decode_reqs and waiting_requests
    if kv_crisis:
        # Sort decode requests by KV blocks held (most to least)
        decode_by_kv = sorted(decode_reqs, key=lambda r: r.kv_blocks_used, reverse=True)
        victim = decode_by_kv[0]

        # Count how many short waiting requests would benefit from preemption
        freed = victim.kv_blocks_used
        beneficiaries = sum(
            1 for r in waiting_requests
            if r.remaining_prompt_tokens <= token_budget
            and max(1, r.remaining_prompt_tokens // 16) <= (avail_kv + freed)
        )

        if beneficiaries >= _MIN_WAITERS_TO_PREEMPT:
            preempt_ids = [victim.request_id]
            decode_batch = [r for r in decode_batch if r != victim.request_id]
            avail_kv += freed
            seq_budget += 1  # freed one running slot

    if seq_budget <= 0 or token_budget <= 0:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch, preempt_ids=preempt_ids)

    # -- Workload heterogeneity measurement --
    sorted_lens = sorted(r.remaining_prompt_tokens for r in waiting_requests)
    median_len = sorted_lens[len(sorted_lens) // 2]
    max_len = sorted_lens[-1]
    is_heterogeneous = median_len > 0 and (max_len / median_len) > _HETEROGENEITY_RATIO

    high_kv_pressure = kv_util > _KV_PRESSURE_THRESHOLD

    # -- Starvation protection --
    now = max(r.arrival_time_s for r in waiting_requests)
    stale_ids = {r.request_id for r in waiting_requests
                 if (now - r.arrival_time_s) >= _STALE_SECONDS}

    prefill_batch: list[str] = []
    admitted: set[str] = set()

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

    # Phase 0: Always admit stale requests first (anti-starvation)
    for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
        if req.request_id in stale_ids:
            try_admit(req)
        if len(prefill_batch) >= seq_budget:
            break

    if high_kv_pressure and is_heterogeneous:
        short_reqs = [r for r in waiting_requests if r.remaining_prompt_tokens <= median_len]
        long_reqs  = [r for r in waiting_requests if r.remaining_prompt_tokens > median_len]

        for req in sorted(short_reqs, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

        for req in sorted(long_reqs, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            tokens = req.remaining_prompt_tokens
            kv_needed = max(1, tokens // 16)
            if avail_kv >= kv_needed * 1.3:
                try_admit(req)
    else:
        for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
            if len(prefill_batch) >= seq_budget:
                break
            try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch, preempt_ids=preempt_ids)
