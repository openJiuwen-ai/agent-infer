# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Runtime-level MORI identity, token, and overlap contracts."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramStatus, TokenObservationSource
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.mori import MORIConfig, MORIMode, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler

pytestmark = pytest.mark.cpu_test


def backend(capacity: int = 10_000) -> BackendPoolInfo:
    return BackendPoolInfo(
        (
            BackendInfo(
                "vllm-local",
                "embedded://vllm",
                True,
                (
                    DpRankInfo(
                        0,
                        True,
                        True,
                        total_hbm_kv_tokens=capacity,
                        total_dram_kv_tokens=capacity,
                        hbm_block_size_tokens=16,
                        dram_chunk_size_tokens=16,
                    ),
                ),
            ),
        )
    )


def scheduler() -> ProgramScheduler[str, object, object]:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    return ProgramScheduler(
        components.strategy,
        components.initial_factors,
        backend(),
        schedule_interval_seconds=5,
    )


def test_server_observed_stream_and_final_usage_replace_prompt_estimate() -> None:
    runtime = scheduler()
    assert runtime.on_request_arrival("request", AgentIdentity("program"), 100, backend(), "native") is True
    ref = runtime.program_ref_for_request("request")
    assert ref is not None

    runtime.on_stream_output("request", 30)
    streamed = runtime.registry.require(ref).tokens
    assert streamed.estimated_context_tokens == 130
    assert streamed.source is TokenObservationSource.MIXED

    runtime.on_request_completion("request", 155)
    completed = runtime.registry.require(ref).tokens
    assert completed.estimated_context_tokens == 155
    assert completed.source is TokenObservationSource.BACKEND_USAGE


def test_same_external_program_overlaps_have_independent_owners() -> None:
    runtime = scheduler()
    lead = AgentIdentity("task:lead", task_id="task", agent_id="lead", agent_role="lead")
    assert runtime.on_request_arrival("lead-1", lead, 100, backend(), "lead")
    assert runtime.on_request_arrival("lead-2", lead, 120, backend(), "overlap")
    lead_ref = runtime.program_ref_for_request("lead-1")
    overlap_ref = runtime.program_ref_for_request("lead-2")
    assert lead_ref != overlap_ref
    runtime.on_request_completion("lead-2", 140)
    assert runtime.registry.get(overlap_ref) is None
    assert runtime.registry.require(lead_ref).status is ProgramStatus.REASONING
    assert not runtime.on_response_completion("task:lead", ProgramLifecycle.TERMINAL)
    runtime.on_request_completion("lead-1", 130)
    assert runtime.registry.require(lead_ref).status is ProgramStatus.ACTING


def test_lazy_demotion_completion_wakes_waiter_before_the_next_periodic_tick() -> None:
    runtime = scheduler()
    capacity = backend(416)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("resident", AgentIdentity("resident"), 100, capacity, "resident")
        assert runtime.on_request_arrival("peer", AgentIdentity("z-peer"), 100, capacity, "peer")
        runtime.schedule_cycle(capacity)
        clock.return_value = 1.0
        assert not runtime.on_request_arrival("old", AgentIdentity("old"), 100, capacity, "old")
        old = runtime.program_ref_for_request("old")
        assert runtime.needs_schedule_cycle(1.0)
        runtime.schedule_cycle(capacity)
        assert runtime.consume_admitted_requests() == ()
        assert runtime.registry.require(old).wait_started_at_monotonic_s == 1.0
        resident = runtime.program_ref_for_request("resident")
        assert not runtime.registry.require(resident).marked_for_pause
        clock.return_value = 31.0
        runtime.on_stream_output("resident", 20)
        runtime.schedule_cycle(capacity)
        assert runtime.registry.require(resident).marked_for_pause
        assert not runtime.needs_schedule_cycle(31.01)
        clock.return_value = 31.1
        runtime.on_request_completion("resident", 100)
        assert runtime.needs_schedule_cycle(31.1)
        runtime.schedule_cycle(capacity)
        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["old"]
        assert runtime.registry.require(old).wait_started_at_monotonic_s is None
        assert not runtime.needs_schedule_cycle(31.2)


def test_arrival_can_use_free_capacity_behind_a_non_fitting_waiter() -> None:
    runtime = scheduler()
    capacity = backend(1008)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("resident", AgentIdentity("resident"), 700, capacity, "resident")
        runtime.schedule_cycle(capacity)
        clock.return_value = 1.0
        assert not runtime.on_request_arrival("large", AgentIdentity("large"), 400, capacity, "large")
        clock.return_value = 1.1
        assert not runtime.on_request_arrival("small", AgentIdentity("small"), 50, capacity, "small")
        runtime.schedule_cycle(capacity)
        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["small"]
        assert runtime.request_pool.waiting_request_ids == ("large",)


def test_elapsed_acting_rebalances_without_explicit_terminal_or_capacity_debt() -> None:
    runtime = scheduler()
    capacity = backend(1008)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("resident", AgentIdentity("resident"), 700, capacity, "resident")
        runtime.schedule_cycle(capacity)
        clock.return_value = 1.0
        assert not runtime.on_request_arrival("large", AgentIdentity("large"), 400, capacity, "large")
        clock.return_value = 1.1
        runtime.on_request_completion("resident", 700)
        assert not runtime.on_request_arrival("small", AgentIdentity("small"), 50, capacity, "small")
        assert runtime.needs_schedule_cycle(1.1)
        runtime.schedule_cycle(capacity)
        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["small"]
        assert runtime.request_pool.waiting_request_ids == ("large",)
        # A normal periodic tick must turn elapsed tool time into placement
        # changes even though the existing GPU commitments remain under quota.
        clock.return_value = 6.2
        assert runtime.needs_schedule_cycle(6.2)
        runtime.schedule_cycle(capacity)
        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["large"]
        assert runtime.request_pool.waiting_request_ids == ()
