# Fixture category: quadratic_loop (L2 rejection).
# Nested for-loops over the same iterable triggers the verify_tool heuristic.
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
    chosen: list[str] = []
    for outer in waiting_requests:
        for inner in waiting_requests:
            if outer.request_id != inner.request_id:
                chosen.append(outer.request_id)
                break
    return ScheduleDecision(
        prefill_batch=chosen[: max_num_seqs - len(running_requests)],
        decode_batch=[r.request_id for r in running_requests],
    )
