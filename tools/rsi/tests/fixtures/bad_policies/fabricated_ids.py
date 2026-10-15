# Fixture category: fabricated_ids (L2 runtime/heuristic rejection).
# Returns ScheduleDecision containing request_ids that did not come from
# the input lists. Authoritative detection is a runtime contract enforced
# by ve bench; ve verify applies a static heuristic that flags suspect
# string literals like "FAKE_REQ_001".
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
    fabricated = ["FAKE_REQ_001", "FAKE_REQ_002"]
    return ScheduleDecision(
        prefill_batch=fabricated,
        decode_batch=[r.request_id for r in running_requests],
    )
