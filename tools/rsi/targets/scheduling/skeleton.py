"""
Scheduling target skeleton for vllm-evolve.

The function `schedule_batch` is the ONLY part that gets evolved.
Everything else is fixed infrastructure.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestInfo:
    """Public view of a vLLM request for scheduling decisions."""
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    arrival_time_s: float
    is_prefill: bool          # True = in waiting queue, False = running
    kv_blocks_used: int
    prefix_cached_tokens: int
    num_preemptions: int = 0

    @property
    def remaining_prompt_tokens(self) -> int:
        return self.num_prompt_tokens - self.num_computed_tokens

    @property
    def wait_time_s(self) -> float:
        """Approximate wait time (needs current_time from caller)."""
        return 0.0  # caller should compute: current_time - arrival_time_s


@dataclass
class ScheduleDecision:
    """Scheduling decision returned by schedule_batch."""
    prefill_batch: list[str]   # request_ids to prefill this step
    decode_batch: list[str]    # request_ids to decode this step
    preempt_ids: list[str] = field(default_factory=list)  # to preempt


# ============================================================
# EVOLVABLE REGION — only this function gets mutated
# ============================================================

def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    """
    Select which requests to prefill and decode in this step.

    Args:
        waiting_requests: Requests waiting to start prefill
        running_requests: Requests currently being processed
        max_num_batched_tokens: Token budget for this step
        max_num_seqs: Max concurrent sequences
        available_kv_blocks: Free KV cache blocks (each = 16 tokens)
        prefix_cache_hit_rate: Current cache hit rate [0, 1]

    Returns:
        ScheduleDecision with request_ids to process

    Constraints:
        - Only use request_ids from waiting_requests or running_requests
        - Do not exceed max_num_batched_tokens or max_num_seqs
        - No O(n^2) loops
    """
    # SEED IMPLEMENTATION (FCFS baseline)
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]

    token_budget = max_num_batched_tokens - sum(
        r.num_prompt_tokens - r.num_computed_tokens
        for r in running_requests if r.is_prefill
    )
    seq_budget = max_num_seqs - len(running_requests)

    prefill_batch = []
    for req in sorted(waiting_requests, key=lambda r: r.arrival_time_s):
        tokens_needed = req.num_prompt_tokens - req.num_computed_tokens
        if tokens_needed <= token_budget and len(prefill_batch) < seq_budget:
            if available_kv_blocks >= tokens_needed // 16 + 1:
                prefill_batch.append(req.request_id)
                token_budget -= tokens_needed

    return ScheduleDecision(prefill_batch=prefill_batch, decode_batch=decode_batch)
