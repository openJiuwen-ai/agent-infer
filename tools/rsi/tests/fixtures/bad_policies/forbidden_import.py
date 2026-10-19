# Fixture category: forbidden_import (L1 rejection).
# Imports a module on safety.FORBIDDEN_IMPORTS so check_safety raises.
from __future__ import annotations

import os  # forbidden
from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    _ = os.getcwd()
    return ScheduleDecision(
        prefill_batch=[r.request_id for r in waiting_requests[:1]],
        decode_batch=[r.request_id for r in running_requests],
    )
