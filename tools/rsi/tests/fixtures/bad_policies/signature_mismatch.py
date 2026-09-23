# Fixture category: signature_mismatch (L2 rejection).
# Defines ``schedule_back`` instead of the target-required ``schedule_batch``
# so check_signatures fails with "Missing function: schedule_batch".
from __future__ import annotations

from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


def schedule_back(  # typo on purpose
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    return ScheduleDecision(
        prefill_batch=[r.request_id for r in waiting_requests[:1]],
        decode_batch=[r.request_id for r in running_requests],
    )
