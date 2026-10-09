"""CCQ: Cohort-Conserving Queue.

CCQ preserves vLLM's FCFS lifecycle mix and changes only prefix locality:

1. Inspect the oldest bounded admission window.
2. If the oldest schedulable prefix cohort has at least two members, make
   those members contiguous while preserving FCFS order inside the cohort and
   among every other request.
3. If no shared-prefix cohort exists, declare the mechanism inapplicable and
   return exact FCFS order.

Unlike lifecycle SJF, CCQ never globally sorts by prompt/decode size, so it
does not collapse continuous-batching width or strand long decodes at the tail.
"""
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
    fcfs = sorted(waiting_requests, key=lambda r: r.arrival_time_s)
    if seq_budget <= 0 or token_budget <= 0 or not fcfs:
        return ScheduleDecision(
            prefill_batch=[],
            decode_batch=decode_batch,
            mechanism_applicable=False,
        )

    window = fcfs[:max_num_seqs * 2]
    members = {}
    first_position = {}
    for position, req in enumerate(window):
        cohort = getattr(req, "session_id", None)
        if cohort is None:
            continue
        members.setdefault(cohort, []).append(req)
        first_position.setdefault(cohort, position)
    eligible_cohorts = [
        cohort
        for cohort, requests in members.items()
        if len(requests) >= 2 and first_position[cohort] < max_num_seqs
    ]
    if not eligible_cohorts:
        ordered = fcfs
        applicable = False
    else:
        cohort = min(
            eligible_cohorts,
            key=lambda item: (first_position[item], -len(members[item])),
        )
        cohort_ids = {req.request_id for req in members[cohort]}
        ordered_window = members[cohort] + [
            req for req in window if req.request_id not in cohort_ids
        ]
        window_ids = {req.request_id for req in window}
        ordered = ordered_window + [
            req for req in fcfs if req.request_id not in window_ids
        ]
        applicable = True

    prefill_batch = []
    for req in ordered:
        need = req.remaining_prompt_tokens
        if need <= token_budget and len(prefill_batch) < seq_budget:
            prefill_batch.append(req.request_id)
            token_budget -= need
    return ScheduleDecision(
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        defer_ids=[],
        mechanism_applicable=applicable,
    )
