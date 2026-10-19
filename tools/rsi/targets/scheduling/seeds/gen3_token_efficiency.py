"""
gen3_token_efficiency: Maximize token-per-KV-block efficiency under pressure.

Key insight: In KV-constrained environments, the bottleneck is not compute
(token throughput) but memory (KV blocks). Each admitted prefill request
consumes ~tokens//16 blocks for the duration of its lifecycle. Maximizing
tokens_processed / blocks_consumed ratio is the right objective.

Strategy:
- Compute efficiency = output_potential / kv_cost for each request
  output_potential = min(max_output_tokens_estimate, 200) ← use 200 as proxy
  kv_cost = remaining_prompt_tokens // 16 + expected_decode_blocks
- Sort waiting by efficiency descending.
- Combine with a token-budget packing pass (small-fill after large-fill).
- Impose a hard KV safety margin: keep 10% of available blocks in reserve.
"""
from __future__ import annotations
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision

_KV_TOKENS_PER_BLOCK = 16
_DECODE_TOKENS_ESTIMATE = 150  # average output length estimate
_KV_RESERVE_FRACTION = 0.10


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

    # Reserve a fraction of KV blocks as safety buffer
    kv_reserve = max(4, int(available_kv_blocks * _KV_RESERVE_FRACTION))
    usable_kv = available_kv_blocks - kv_reserve

    if seq_budget <= 0 or token_budget <= 0 or usable_kv <= 0 or not waiting_requests:
        return ScheduleDecision(prefill_batch=[], decode_batch=decode_batch)

    def kv_cost(req: RequestInfo) -> int:
        prompt_blocks = max(1, req.remaining_prompt_tokens // _KV_TOKENS_PER_BLOCK)
        decode_blocks = max(1, _DECODE_TOKENS_ESTIMATE // _KV_TOKENS_PER_BLOCK)
        return prompt_blocks + decode_blocks

    def efficiency(req: RequestInfo) -> float:
        cost = kv_cost(req)
        # Output value: longer outputs amortize KV cost better
        value = _DECODE_TOKENS_ESTIMATE
        return value / cost

    prefill_batch: list[str] = []
    admitted: set[str] = set()
    avail = usable_kv

    def try_admit(req: RequestInfo) -> bool:
        nonlocal token_budget, avail
        if req.request_id in admitted:
            return False
        tokens = req.remaining_prompt_tokens
        cost = kv_cost(req)
        if tokens <= token_budget and len(prefill_batch) < seq_budget and avail >= cost:
            prefill_batch.append(req.request_id)
            admitted.add(req.request_id)
            token_budget -= tokens
            avail -= cost
            return True
        return False

    # Primary sort: highest efficiency first
    by_efficiency = sorted(waiting_requests, key=efficiency, reverse=True)
    for req in by_efficiency:
        if len(prefill_batch) >= seq_budget:
            break
        try_admit(req)

    # Secondary fill: use leftover budget with smallest remaining tokens
    by_size = sorted(waiting_requests, key=lambda r: r.remaining_prompt_tokens)
    for req in by_size:
        if len(prefill_batch) >= seq_budget or token_budget <= 0:
            break
        try_admit(req)

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
