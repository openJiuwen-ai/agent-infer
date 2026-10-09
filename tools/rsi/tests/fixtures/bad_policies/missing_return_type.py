# Fixture category: missing_return_type (L1 rejection).
# Defines schedule_batch but without the -> ScheduleDecision annotation.
# The AST walk in ar_cli._verify_code detects FunctionDef.returns is None.
from __future__ import annotations

from targets.scheduling.skeleton import RequestInfo, ScheduleDecision


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
):
    return ScheduleDecision(
        prefill_batch=[r.request_id for r in waiting_requests[:1]],
        decode_batch=[r.request_id for r in running_requests],
    )
