# Fixture category: forbidden_pattern (L1 rejection).
# Uses ``eval`` so safety.FORBIDDEN_PATTERNS triggers.
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
    priority = eval("len(waiting_requests)")  # forbidden
    chosen = [r.request_id for r in waiting_requests[:priority]]
    return ScheduleDecision(
        prefill_batch=chosen,
        decode_batch=[r.request_id for r in running_requests],
    )
