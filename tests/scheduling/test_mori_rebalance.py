# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Periodic MORI progress through real admission, execution, and owner lifecycles."""

from unittest.mock import patch

import pytest

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, MORITier, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler

pytestmark = pytest.mark.cpu_test


def pool(hbm=1000, dram=2000):
    return BackendPoolInfo(
        (
            BackendInfo(
                "local",
                "embedded://local",
                True,
                (
                    DpRankInfo(
                        0,
                        True,
                        True,
                        total_hbm_kv_tokens=hbm,
                        total_dram_kv_tokens=dram,
                        hbm_block_size_tokens=1,
                        dram_chunk_size_tokens=1,
                    ),
                ),
            ),
        )
    )


def scheduler(window_size=3):
    components = build_mori_strategy(MORIConfig(window_size=window_size, decode_buffer_tokens=0))
    return ProgramScheduler(components.strategy, components.initial_factors, pool(), schedule_interval_seconds=1)


@pytest.mark.parametrize("dram", [2000, 0])
def test_periodic_tick_admits_child_blocked_by_acting_parent_without_capacity_debt(dram):
    runtime = scheduler()
    capacity = pool(dram=dram)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("parent-1", AgentIdentity("parent", task_id="task"), 700, capacity, "parent")
        parent = runtime.program_ref_for_request("parent-1")
        runtime.on_native_execution("parent-1", running=True)
        clock.return_value = 1.0
        runtime.on_request_completion("parent-1", 700)
        clock.return_value = 2.0
        assert not runtime.on_request_arrival(
            "child-1",
            AgentIdentity("child", task_id="task", parent_program_id="parent", blocks_parent=True),
            400,
            capacity,
            "child",
        )
        child = runtime.program_ref_for_request("child-1")
        assert runtime.registry.require(parent).blocked_on_child
        before, _ = runtime.strategy._ledgers(runtime._snapshot(), runtime.strategy_factors)
        assert before.used == 700 < 1000

        runtime.schedule_cycle(capacity)

        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["child-1"]
        assert runtime.strategy_factors.for_program(child).tier is MORITier.GPU
        assert runtime.registry.require(child).state is ProgramState.ACTIVE
        assert runtime.registry.require(child).parent_program_id == "parent"
        if dram:
            assert runtime.registry.children(parent) == (child,)
            assert runtime.strategy_factors.for_program(parent).tier is MORITier.CPU
            assert runtime.registry.require(parent).blocked_on_child
        else:
            # Retirement must see the PAUSED state produced in this same tick.
            assert runtime.registry.get(parent) is None
            assert runtime.strategy_factors.for_program(parent) is None
            assert "parent" in runtime.registry._retired_metadata
            assert runtime.registry._children_by_parent_id["parent"] == {child}
        runtime.on_native_execution("child-1", running=True)
        clock.return_value = 3.0
        runtime.on_request_completion("child-1", 400)
        assert runtime.registry.require(child).step_count == 1


@pytest.mark.parametrize("window_size", [1, 3])
def test_idle_gpu_reclaims_acting_even_for_tied_or_higher_idleness_waiter(window_size):
    runtime = scheduler(window_size)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("waiter-1", AgentIdentity("waiter"), 400, pool(), "first")
        waiter = runtime.program_ref_for_request("waiter-1")
        runtime.on_native_execution("waiter-1", running=True)
        clock.return_value = 1.0
        runtime.on_request_completion("waiter-1", 400)
        clock.return_value = 100.0
        runtime.schedule_cycle(pool(hbm=0))
        assert runtime.strategy_factors.for_program(waiter).tier is MORITier.CPU
        assert runtime.on_request_arrival("resident", AgentIdentity("resident"), 700, pool(), "resident")
        resident = runtime.program_ref_for_request("resident")
        runtime.on_native_execution("resident", running=True)
        clock.return_value = 110.0
        runtime.on_request_completion("resident", 700)
        assert not runtime.on_request_arrival("waiter-2", AgentIdentity("waiter"), 400, pool(), "second")
        waiter_score = runtime.strategy.idleness_for_program(waiter, 110.0)
        resident_score = runtime.strategy.idleness_for_program(resident, 110.0)
        assert resident_score == 0
        assert waiter_score > resident_score

        runtime.schedule_cycle(pool())

        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["waiter-2"]
        assert runtime.strategy_factors.for_program(waiter).tier is MORITier.GPU
        assert runtime.strategy_factors.for_program(resident).tier is MORITier.CPU


def test_elapsed_acting_crosses_relative_score_while_other_gpu_reasoning_continues():
    runtime = scheduler()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("waiter-1", AgentIdentity("waiter"), 400, pool(), "first")
        waiter = runtime.program_ref_for_request("waiter-1")
        runtime.on_native_execution("waiter-1", running=True)
        clock.return_value = 1.0
        runtime.on_request_completion("waiter-1", 400)
        clock.return_value = 100.0
        runtime.schedule_cycle(pool(hbm=0))
        clock.return_value = 101.0
        assert runtime.on_request_arrival("resident", AgentIdentity("resident"), 600, pool(), "resident")
        resident = runtime.program_ref_for_request("resident")
        runtime.on_native_execution("resident", running=True)
        assert runtime.on_request_arrival("running", AgentIdentity("running"), 100, pool(), "running")
        running = runtime.program_ref_for_request("running")
        runtime.on_native_execution("running", running=True)
        clock.return_value = 111.0
        runtime.on_request_completion("resident", 600)
        clock.return_value = 112.0
        assert not runtime.on_request_arrival("waiter-2", AgentIdentity("waiter"), 400, pool(), "second")
        assert runtime.strategy.idleness_for_program(waiter, 112) > runtime.strategy.idleness_for_program(resident, 112)
        runtime.schedule_cycle(pool())
        assert runtime.consume_admitted_requests() == ()
        assert runtime.strategy_factors.for_program(resident).tier is MORITier.GPU

        clock.return_value = 1000.0
        assert runtime.needs_schedule_cycle(1000.0)
        assert runtime.strategy.idleness_for_program(waiter, 1000) < runtime.strategy.idleness_for_program(
            resident, 1000
        )
        runtime.schedule_cycle(pool())

        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["waiter-2"]
        assert runtime.strategy_factors.for_program(resident).tier is MORITier.CPU
        assert runtime.strategy_factors.for_program(running).tier is MORITier.GPU
        assert not runtime.registry.require(running).marked_for_pause


def test_newly_admitted_reasoning_cannot_be_reclaimed_for_later_candidate_in_same_tick():
    runtime = scheduler(window_size=1)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0):
        assert runtime.on_request_arrival("resident", AgentIdentity("resident"), 700, pool(), "resident")
        runtime.on_request_completion("resident", 700)
        assert not runtime.on_request_arrival("first", AgentIdentity("a-first"), 700, pool(), "first")
        assert not runtime.on_request_arrival("second", AgentIdentity("b-second"), 700, pool(), "second")

        runtime.schedule_cycle(pool())

        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["first"]
        first = runtime.program_ref_for_request("first")
        assert runtime.registry.require(first).status is ProgramStatus.REASONING
        assert runtime.registry.require(first).state is ProgramState.ACTIVE
        assert runtime.strategy_factors.for_program(first).tier is MORITier.GPU
        assert not runtime.registry.require(first).marked_for_pause
        assert runtime.request_pool.waiting_request_ids == ("second",)
