from __future__ import annotations

from targets.scheduling.seeds.ftc import schedule_batch
from targets.scheduling.skeleton import RequestInfo


def _request(
    request_id: str,
    *,
    output_tokens: int,
    arrival: float,
    is_prefill: bool,
) -> RequestInfo:
    request = RequestInfo(
        request_id=request_id,
        num_prompt_tokens=64,
        num_computed_tokens=0 if is_prefill else 64,
        num_output_tokens=output_tokens,
        arrival_time_s=arrival,
        is_prefill=is_prefill,
        kv_blocks_used=0,
        prefix_cached_tokens=0,
    )
    request.generated_output_tokens = 0 if is_prefill else 1
    request.waiting_age_s = 0.05 if is_prefill else 0.0
    request.remaining_output_tokens = (
        output_tokens - request.generated_output_tokens
    )
    return request


def test_ftc_preempts_elephant_for_urgent_mouse_not_another_elephant():
    running = _request(
        "running-elephant",
        output_tokens=1792,
        arrival=0.0,
        is_prefill=False,
    )
    waiting_elephant = _request(
        "waiting-elephant",
        output_tokens=1792,
        arrival=0.01,
        is_prefill=True,
    )
    waiting_mouse = _request(
        "waiting-mouse",
        output_tokens=128,
        arrival=0.02,
        is_prefill=True,
    )

    decision = schedule_batch(
        [waiting_elephant, waiting_mouse],
        [running],
        8192,
        1,
        1 << 30,
        0.0,
    )

    assert decision.preempt_ids == ["running-elephant"]
    assert decision.prefill_batch == ["waiting-mouse"]
    assert decision.mechanism_applicable is True


def test_ftc_preserves_fcfs_when_only_elephants_are_waiting():
    running = _request(
        "running-elephant",
        output_tokens=1792,
        arrival=0.0,
        is_prefill=False,
    )
    waiting = _request(
        "waiting-elephant",
        output_tokens=1792,
        arrival=0.01,
        is_prefill=True,
    )

    decision = schedule_batch(
        [waiting],
        [running],
        8192,
        1,
        1 << 30,
        0.0,
    )

    assert decision.preempt_ids == []
    assert decision.prefill_batch == []
    assert decision.mechanism_applicable is False
