# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Tests for Scheduler-owned RequestPool and Program lifecycle execution."""

from unittest.mock import patch

import pytest

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.progress_ttl import ProgressTTLConfig, build_progress_ttl_strategy
from agentinfer.scheduling.runtime import ProgramScheduler

pytestmark = pytest.mark.cpu_test


def backend(capacity: int) -> BackendPoolInfo:
    """Build one fixed embedded backend view."""
    return BackendPoolInfo(
        (
            BackendInfo(
                backend_id="vllm-local",
                backend_url="embedded://vllm",
                healthy=True,
                dp_ranks=(DpRankInfo(0, True, True, total_hbm_kv_tokens=capacity),),
            ),
        )
    )


def test_scheduler_retains_queued_request_and_releases_it_after_capacity_handoff() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            target_min_segment_rounds=1,
            target_max_segment_rounds=2,
            decode_buffer_tokens=0,
            ttl_min_seconds=1000,
            ttl_max_seconds=1000,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
    )

    assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 600, backend(1000), "native-r1") is True
    scheduler.on_request_completion("r1", 600)
    p1_ref = scheduler.registry.current_ref("p1")
    assert p1_ref is not None
    assert scheduler.registry.require(p1_ref).status is ProgramStatus.ACTING

    assert scheduler.on_request_arrival("r2", AgentIdentity("p2"), 600, backend(1000), "native-r2") is False
    assert scheduler.retained_request_ids == ("r2",)

    scheduler.schedule_cycle(backend(1000))
    admitted = scheduler.consume_admitted_requests()

    assert [(entry.request_id, entry.retained_request, target.backend_id) for entry, target in admitted] == [
        ("r2", "native-r2", "vllm-local")
    ]
    assert scheduler.registry.require(p1_ref).state is ProgramState.PAUSED
    p2_ref = scheduler.registry.current_ref("p2")
    assert p2_ref is not None
    assert scheduler.registry.require(p2_ref).state is ProgramState.ACTIVE


def test_scheduler_privilege_handoff_pauses_parent_and_immediately_releases_child() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            privileged_lookahead_rounds=14,
            ttl_min_seconds=1000,
            ttl_max_seconds=1000,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(20_000),
    )
    parent_metadata = AgentIdentity(
        "temp:session:lead",
        task_id="task",
        session_id="session",
        agent_id="lead",
        agent_role="lead",
    )

    assert scheduler.on_request_arrival("parent-r1", parent_metadata, 700, backend(1000), "parent") is True
    scheduler.on_request_completion("parent-r1", 700)
    parent_ref = scheduler.registry.current_ref(parent_metadata.program_id)
    assert parent_ref is not None
    assert components.initial_factors.for_program(parent_ref).is_privileged is True

    child_metadata = AgentIdentity(
        "temp:session:child",
        task_id="task",
        session_id="session",
        agent_id="child",
        parent_program_id=parent_metadata.program_id,
        blocks_parent=True,
        expected_resume=False,
        agent_role="subagent",
    )
    assert scheduler.on_request_arrival("child-r1", child_metadata, 400, backend(1000), "child") is True

    child_ref = scheduler.registry.current_ref(child_metadata.program_id)
    assert child_ref is not None
    assert scheduler.registry.require(parent_ref).state is ProgramState.PAUSED
    assert scheduler.registry.require(child_ref).state is ProgramState.ACTIVE
    assert components.initial_factors.for_program(parent_ref).is_privileged is False
    assert components.initial_factors.for_program(child_ref).is_privileged is True
    assert scheduler.retained_request_count == 0


def test_cancel_retained_request_clears_request_binding_without_counting_completion() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(100),
    )

    assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 200, backend(100), "native-r1") is False
    entry = scheduler.cancel_request("r1")
    assert entry is not None and entry.retained_request == "native-r1"
    assert scheduler.retained_request_count == 0
    assert scheduler.registry.current_ref("p1") is None


def test_waiting_snapshot_contains_every_paused_program_without_a_pending_request() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(decode_buffer_tokens=0, ttl_min_seconds=0, ttl_max_seconds=0)
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
    )

    assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 100, backend(1000), "native-r1") is True
    scheduler.on_request_completion("r1", 120)
    scheduler.schedule_cycle(backend(1000))
    ref = scheduler.registry.current_ref("p1")

    assert ref is not None
    assert scheduler.registry.require(ref).state is ProgramState.PAUSED
    assert scheduler._snapshot().waiting_programs == (ref,)


def test_consecutive_request_for_active_program_is_immediately_admitted() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
    )
    metadata = AgentIdentity("p1")

    assert scheduler.on_request_arrival("r1", metadata, 100, backend(1000), "native-r1") is True
    scheduler.on_request_completion("r1", 120)
    assert scheduler.on_request_arrival("r2", metadata, 120, backend(1000), "native-r2") is True

    assert scheduler.retained_request_count == 0
    ref = scheduler.registry.current_ref("p1")
    assert ref is not None
    assert scheduler.registry.require(ref).status is ProgramStatus.REASONING


def test_api_terminal_completion_releases_only_an_idle_program() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
    )
    metadata = AgentIdentity("p1")

    assert scheduler.on_request_arrival("r1", metadata, 100, backend(1000), "native-r1") is True
    scheduler.on_request_completion("r1", 120)
    assert scheduler.on_response_completion("p1", ProgramLifecycle.CONTINUE) is False
    assert scheduler.registry.current_ref("p1") is not None
    assert scheduler.on_response_completion("p1", ProgramLifecycle.TERMINAL) is True
    assert scheduler.registry.current_ref("p1") is None


def test_late_api_terminal_completion_does_not_interrupt_new_reasoning() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
    )
    metadata = AgentIdentity("p1")

    assert scheduler.on_request_arrival("r1", metadata, 100, backend(1000), "native-r1") is True
    scheduler.on_request_completion("r1", 120)
    assert scheduler.on_request_arrival("r2", metadata, 120, backend(1000), "native-r2") is True

    assert scheduler.on_response_completion("p1", ProgramLifecycle.TERMINAL) is False
    ref = scheduler.registry.current_ref("p1")
    assert ref is not None
    assert scheduler.registry.require(ref).status is ProgramStatus.REASONING


def test_scheduler_throttles_periodic_work_but_preserves_due_deadlines() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            decode_buffer_tokens=0,
            ttl_min_seconds=1000,
            ttl_max_seconds=1000,
            ttl_prefill_seconds_per_1k_uncached_tokens=10_000,
            ttl_decode_throughput_alpha=1,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
        schedule_interval_seconds=5,
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic") as monotonic:
        monotonic.return_value = 10
        assert scheduler.needs_schedule_cycle(10) is False
        assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 100, backend(1000), "native-r1") is True
        assert scheduler.needs_schedule_cycle(10) is True

        scheduler.schedule_cycle(backend(1000))
        assert scheduler.needs_schedule_cycle(10) is False

        monotonic.return_value = 11
        scheduler.on_request_completion("r1", 120)
        assert scheduler.needs_schedule_cycle(11) is False
        assert scheduler.needs_schedule_cycle(14.9) is False
        assert scheduler.needs_schedule_cycle(15) is True
        with pytest.raises(ValueError, match="finite and non-negative"):
            scheduler.needs_schedule_cycle(float("inf"))


def test_scheduler_supplies_latency_gap_and_previous_context_to_rolling_stats() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
    )
    metadata = AgentIdentity("p1")

    with patch("agentinfer.scheduling.runtime.time.monotonic") as monotonic:
        monotonic.return_value = 10
        assert scheduler.on_request_arrival("r1", metadata, 100, backend(1000), "native-r1") is True
        monotonic.return_value = 14
        scheduler.on_request_completion("r1", 120)
        monotonic.return_value = 20
        assert scheduler.on_request_arrival("r2", metadata, 150, backend(1000), "native-r2") is True
        monotonic.return_value = 23
        scheduler.on_request_completion("r2", 170)

    samples = tuple(components.initial_factors.global_factors._request_samples)
    assert len(samples) == 2
    assert samples[0].request_latency_seconds == 4
    assert samples[0].input_token_growth is None
    assert samples[1].request_latency_seconds == 3
    assert samples[1].inter_request_gap_seconds == 6
    assert samples[1].input_token_growth == 50


def test_scheduler_retains_first_native_prefix_hit_during_warmup_freshness() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(20_000),
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=10):
        assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 300, backend(20_000), "native-r1") is True

    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=10):
        scheduler.on_prefix_cache_observation("r1", 250)
        scheduler.on_prefix_cache_observation("r1", 280)
    program = scheduler.registry.get(ProgramRef("p1", 0))

    assert program is not None
    assert program.tokens.shared_prefix_tokens == 250
    assert program.tokens.shared_prefix_fresh_until_monotonic_s is not None


def test_scheduler_refreshes_shared_prefix_only_after_its_freshness_deadline() -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(decode_buffer_tokens=0))
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(20_000),
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic") as monotonic:
        monotonic.return_value = 10
        assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 300, backend(20_000), "native-r1") is True
        scheduler.on_prefix_cache_observation("r1", 250)
        scheduler.on_request_completion("r1", 300)
        program = scheduler.registry.require(ProgramRef("p1", 0))
        program.state = ProgramState.PAUSED
        program.status = ProgramStatus.REASONING

        monotonic.return_value = 20
        assert scheduler.on_request_arrival("r2", AgentIdentity("p1"), 320, backend(20_000), "native-r2") is False
        scheduler.on_prefix_cache_observation("r2", 100)
        assert program.tokens.shared_prefix_tokens == 250

        monotonic.return_value = 111
        scheduler.on_prefix_cache_observation("r2", 100)
        assert program.tokens.shared_prefix_tokens == 100


def test_rolling_request_latency_includes_request_pool_wait() -> None:
    components = build_progress_ttl_strategy(
        ProgressTTLConfig(
            target_min_segment_rounds=1,
            decode_buffer_tokens=0,
            ttl_min_seconds=1000,
            ttl_max_seconds=1000,
        )
    )
    components.initial_factors.global_factors.avg_input_token_growth_per_round = 0
    scheduler = ProgramScheduler[str, object, object](
        components.strategy,
        components.initial_factors,
        backend(1000),
        schedule_interval_seconds=5,
    )

    with patch("agentinfer.scheduling.runtime.time.monotonic") as monotonic:
        monotonic.return_value = 0
        assert scheduler.on_request_arrival("r1", AgentIdentity("p1"), 600, backend(1000), "native-r1") is True
        monotonic.return_value = 5
        scheduler.on_request_completion("r1", 600)
        monotonic.return_value = 10
        assert scheduler.on_request_arrival("r2", AgentIdentity("p2"), 600, backend(1000), "native-r2") is False
        monotonic.return_value = 100
        scheduler.schedule_cycle(backend(1000))
        scheduler.consume_admitted_requests()
        monotonic.return_value = 104
        scheduler.on_request_completion("r2", 600)

    samples = tuple(components.initial_factors.global_factors._request_samples)
    assert samples[-1].request_latency_seconds == 94
