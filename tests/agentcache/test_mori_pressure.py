# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""HBM plan pressure wakes ordinary placement once, without forcing a transition."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import agentinfer.agentcache.core.scheduler as bridge_module
from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks
from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, MORIProgramFactors, MORITier, build_mori_strategy
from tests.scheduling.test_mori_runtime import backend, scheduler
from tests.scheduling.test_mori_waiting import pool, runtime

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def hbm():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    policy = MORIHBMPolicy(state)
    for block, tier in ((1, MORITier.GPU), (2, MORITier.CPU)):
        owner = ProgramRef(str(block), 0)
        state.bind_request(str(block), owner)
        components.strategy._set_factor(components.initial_factors, owner, MORIProgramFactors(tier=tier))
        policy.on_block_owner_added(block, str(block))
    for block in (1, 2, 3):
        policy.on_block_ready(block, block)
    yield policy, state, components
    policy.close()
    state.unbind_provider(components.strategy)


@pytest.mark.parametrize(
    ("count", "expected", "pressure"),
    [(0, (), False), (1, (3,), False), (2, (3, 2), False), (3, (3, 2, 1), True), (4, None, False)],
)
def test_only_complete_hbm_plan_including_busy_reports_pressure(hbm, count, expected, pressure):
    policy, state, _ = hbm
    assert policy.propose(count) == expected
    # An incomplete plan may scan BUSY entries but must not publish pressure.
    assert state.consume_hbm_pressure() is pressure
    assert not state.consume_hbm_pressure()


def test_proposals_coalesce_without_mutating_victims_and_generic_queries_stay_pure(hbm):
    policy, state, _ = hbm
    owner = ProgramRef("1", 0)
    assert policy.index.propose_exact(3) == (3, 2, 1)
    assert not state.consume_hbm_pressure()
    assert policy.propose(3) == policy.propose(3) == (3, 2, 1)
    assert policy.index.owners(1) == frozenset({owner})
    assert policy.index.entries_for_program(owner) == (1,)
    assert state.consume_hbm_pressure()
    assert not state.consume_hbm_pressure()


def test_cpu_busy_backup_advice_does_not_report_hbm_pressure(hbm):
    _, state, _ = hbm
    cpu = MORISimpleCPUOffloadPolicy(state)
    try:
        cpu.on_block_owner_added(1, "1")
        cpu.on_block_ready(1, 1)
        assert cpu.propose(1) == (1,)
        assert not state.consume_hbm_pressure()
    finally:
        cpu.close()


@pytest.mark.parametrize("cleanup", ["cache_reset", "unbind", "runtime_reset"])
def test_reset_and_unbind_discard_pending_pressure(hbm, cleanup):
    policy, state, components = hbm
    assert policy.propose(3) == (3, 2, 1)
    if cleanup == "cache_reset":
        policy.on_cache_reset()
    elif cleanup == "unbind":
        state.unbind_provider(components.strategy)
    else:
        state.reset()
    assert not state.consume_hbm_pressure()


def make_hooks(controller, capacity):
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.controller = controller
    hooks._drain_lifecycle_signals = Mock()
    hooks._prefix_refresh_pending = False
    hooks.backend_pool_info = lambda owner: capacity
    hooks.backend_id = capacity.backends[0].backend_id
    hooks.dp_rank = 0
    hooks.tracked_native = set()
    return hooks


def test_bridge_consumes_pressure_once_and_early_cycle_may_leave_placement_unchanged(monkeypatch):
    controller = runtime()
    state = MORIRuntimeState()
    state.bind_provider(controller.strategy)
    policy = MORIHBMPolicy(state)
    monkeypatch.setattr(bridge_module, "MORI_RUNTIME_STATE", state)
    hooks = make_hooks(controller, pool())
    native_add = Mock()
    try:
        with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
            assert controller.on_request_arrival("r", AgentIdentity("p"), 32, pool(), "r")
            ref = controller.program_ref_for_request("r")
            controller.consume_admitted_requests()
            controller.schedule_cycle(pool())
            clock.return_value = 0.01
            controller.on_request_completion("r", 32)
            controller.schedule_cycle(pool())
            before = controller.strategy_factors.global_factors.schedule_cycles
            state.bind_request("cache-owner", ref)
            policy.on_block_owner_added(1, "cache-owner")
            policy.on_block_ready(1, 1)

            clock.return_value = 0.1
            assert not controller.needs_schedule_cycle(0.1)
            for _ in range(3):
                assert policy.propose(1) == (1,)
            # The allocation callback only records a flag; it does not enter scheduling.
            assert controller.strategy_factors.global_factors.schedule_cycles == before
            assert not controller.needs_schedule_cycle(0.1)

            hooks.before_schedule(SimpleNamespace(), native_add)
            assert controller.strategy_factors.global_factors.schedule_cycles == before + 1
            assert controller.strategy_factors.for_program(ref).tier is MORITier.GPU
            current = controller.registry.require(ref)
            assert (current.state, current.status) == (ProgramState.ACTIVE, ProgramStatus.ACTING)
            assert not current.marked_for_pause
            assert not state.consume_hbm_pressure()
            assert not controller.needs_schedule_cycle(0.1)

            hooks.before_schedule(SimpleNamespace(), native_add)
            assert controller.strategy_factors.global_factors.schedule_cycles == before + 1
            native_add.assert_not_called()
    finally:
        policy.close()
        state.unbind_provider(controller.strategy)


def test_other_scheduler_does_not_consume_mori_pressure(hbm, monkeypatch):
    policy, state, _ = hbm
    controller = SimpleNamespace(
        strategy=object(),
        on_cache_pressure=Mock(),
        run_due_lightweight_checks=Mock(),
        needs_schedule_cycle=lambda now: False,
    )
    hooks = make_hooks(controller, pool())
    monkeypatch.setattr(bridge_module, "MORI_RUNTIME_STATE", state)
    policy.propose(3)
    hooks.before_schedule(SimpleNamespace(), Mock())
    controller.on_cache_pressure.assert_not_called()
    assert state.consume_hbm_pressure()


def test_pressure_wake_reuses_relative_rebalance_before_periodic_tick():
    controller = scheduler()
    capacity = backend(1008)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert controller.on_request_arrival("resident", AgentIdentity("resident"), 700, capacity, "resident")
        controller.schedule_cycle(capacity)
        clock.return_value = 1.0
        assert not controller.on_request_arrival("large", AgentIdentity("large"), 400, capacity, "large")
        clock.return_value = 1.1
        controller.on_request_completion("resident", 700)
        assert not controller.on_request_arrival("small", AgentIdentity("small"), 50, capacity, "small")
        controller.schedule_cycle(capacity)
        assert [entry.request_id for entry, _ in controller.consume_admitted_requests()] == ["small"]
        assert controller.request_pool.waiting_request_ids == ("large",)

        clock.return_value = 2.0
        assert not controller.needs_schedule_cycle(2.0)
        controller.on_cache_pressure(2.0)
        assert controller.needs_schedule_cycle(2.0)
        controller.schedule_cycle(capacity)
        assert [entry.request_id for entry, _ in controller.consume_admitted_requests()] == ["large"]


@pytest.mark.parametrize("now", [-1.0, float("inf"), float("nan")])
def test_pressure_wake_rejects_invalid_time(now):
    controller = runtime()
    with pytest.raises(ValueError, match="cache-pressure time"):
        controller.on_cache_pressure(now)
