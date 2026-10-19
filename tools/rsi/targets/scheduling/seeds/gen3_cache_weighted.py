"""
gen3_cache_weighted: Priority scoring that rewards prefix-cache affinity.

Strategy:
- Compute a priority score for each waiting request:
    score = cache_bonus - length_penalty + age_bonus
  where:
    cache_bonus = prefix_cached_tokens / num_prompt_tokens * 2.0
    length_penalty = remaining_prompt_tokens / max_num_batched_tokens
    age_bonus = min(wait_time / 5.0, 1.0)  (normalize to 5s saturation)
- High cache_bonus rewards requests whose prompts are already in the
  prefix KV cache, reducing actual block consumption.
- length_penalty prevents token-budget exhaustion by large requests.
- age_bonus prevents starvation of long-waiting requests.
- When prefix_cache_hit_rate is high (> 0.4), boost cache_bonus weight.
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

    # Estimate current time as max arrival in waiting (proxy for now)
    now = max(r.arrival_time_s for r in waiting_requests)

    # Cache weight scales with observed hit rate
    cache_weight = 1.0 + 2.0 * prefix_cache_hit_rate

    def priority(req: RequestInfo) -> float:
        cache_bonus = cache_weight * (req.prefix_cached_tokens / max(req.num_prompt_tokens, 1))
        length_penalty = req.remaining_prompt_tokens / max(max_num_batched_tokens, 1)
        age_bonus = min((now - req.arrival_time_s) / 5.0, 1.0)
        return cache_bonus - 0.5 * length_penalty + 0.3 * age_bonus

    prefill_batch: list[str] = []
    admitted: set[str] = set()

    for req in sorted(waiting_requests, key=priority, reverse=True):
        if len(prefill_batch) >= seq_budget:
            break
        tokens = req.remaining_prompt_tokens
        kv_needed = max(1, (tokens - req.prefix_cached_tokens) // 16)  # cached tokens need no new blocks
        if (tokens <= token_budget
                and available_kv_blocks >= kv_needed
                and req.request_id not in admitted):
            prefill_batch.append(req.request_id)
            admitted.add(req.request_id)
            token_budget -= tokens
            available_kv_blocks -= kv_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
