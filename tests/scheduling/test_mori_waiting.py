# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Current-cycle waiting credit through real scheduler and native execution events."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, MORIProgramFactors, MORITier, build_mori_strategy
from agentinfer.scheduling.mori.diagnostics import score_components
from agentinfer.scheduling.runtime import ProgramScheduler
from tests.scheduling.test_mori_strategy import accepting_controller, snapshot, view

pytestmark = pytest.mark.cpu_test


def pool(hbm=96, dram=96):
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
                        hbm_block_size_tokens=16,
                        dram_chunk_size_tokens=16,
                    ),
                ),
            ),
        )
    )


def runtime(window_size=3):
    components = build_mori_strategy(MORIConfig(window_size=window_size, decode_buffer_tokens=0))
    return ProgramScheduler(components.strategy, components.initial_factors, pool(), schedule_interval_seconds=1)


def test_waiting_survives_cpu_to_waiting_but_admission_clears_it_without_changing_history():
    scheduler = runtime()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert scheduler.on_request_arrival("first", AgentIdentity("p"), 96, pool(), "first")
        ref = scheduler.program_ref_for_request("first")
        scheduler.on_native_execution("first", running=True)
        clock.return_value = 2.0
        scheduler.on_request_completion("first", 96)
        clock.return_value = 100.0
        scheduler.schedule_cycle(pool(hbm=0))
        assert scheduler.strategy_factors.for_program(ref).tier is MORITier.CPU
        # Long tool time does not itself become waiting compensation.
        assert scheduler.strategy.idleness_for_program(ref, 100.0) == pytest.approx(98 / 100)

        clock.return_value = 101.0
        assert not scheduler.on_request_arrival("second", AgentIdentity("p"), 96, pool(hbm=0), "second")
        assert scheduler.strategy_factors.for_program(ref).cycles == ((2.0, 99.0),)
        assert scheduler.strategy.idleness_for_program(ref, 111.0) == pytest.approx(99 / 111)
        # Repeated reads/ticks do not accumulate the same elapsed wait again.
        for _ in range(2):
            clock.return_value = 111.0
            scheduler.schedule_cycle(pool(hbm=0, dram=0))
            assert scheduler.strategy.idleness_for_program(ref, 111.0) == pytest.approx(99 / 111)
        assert scheduler.strategy_factors.for_program(ref).tier is MORITier.WAITING
        assert scheduler.request_pool.waiting_request_ids == ("second",)

        clock.return_value = 121.0
        scheduler.schedule_cycle(pool())
        assert [entry.request_id for entry, _ in scheduler.consume_admitted_requests()] == ["second"]
        admitted = scheduler.strategy_factors.for_program(ref)
        assert admitted.reasoning_wait_seconds == 0.0
        assert admitted.reasoning_wait_started_at_monotonic_s is None
        assert scheduler.registry.require(ref).wait_started_at_monotonic_s is None
        assert scheduler.strategy.idleness_for_program(ref, 121.0) == pytest.approx(99 / 101)

        clock.return_value = 124.0
        # Native waiting receives no fairness credit and occupies no window slot.
        assert scheduler.strategy.idleness_for_program(ref, 124.0) == pytest.approx(99 / 101)
        scheduler.on_native_execution("second", running=True)
        assert scheduler.strategy.idleness_for_program(ref, 124.0) == pytest.approx(99 / 101)
        clock.return_value = 126.0
        scheduler.on_native_execution("second", running=False)
        # Native preemption is neither real execution nor CPU/WAITING credit.
        assert scheduler.strategy.idleness_for_program(ref, 136.0) == pytest.approx(99 / 103)
        clock.return_value = 136.0
        scheduler.on_native_execution("second", running=True)
        clock.return_value = 139.0
        scheduler.on_request_completion("second", 96)
        completed = scheduler.strategy_factors.for_program(ref)
        assert completed.pending_reasoning_seconds == 5.0
        assert completed.reasoning_wait_seconds == 0.0
        assert completed.reasoning_wait_started_at_monotonic_s is None
        assert scheduler.strategy.idleness_for_program(ref, 139.0) == pytest.approx(99 / 106)

        clock.return_value = 149.0
        assert scheduler.on_request_arrival("third", AgentIdentity("p"), 96, pool(), "third")
        assert scheduler.strategy_factors.for_program(ref).cycles == ((2.0, 99.0), (5.0, 10.0))


@pytest.mark.parametrize("window,expected", [(1, 9 / 15), (2, 109 / 116), (3, 109 / 116)])
def test_waiting_credit_preserves_completed_window(window, expected):
    components = build_mori_strategy(MORIConfig(window_size=window))
    owner = ProgramRef("waiting", 0)
    sidecar = MORIProgramFactors(
        tier=MORITier.CPU,
        cycles=((1.0, 100.0), (1.0, 9.0)),
        reasoning_wait_started_at_monotonic_s=10.0,
    )
    components.strategy._set_factor(components.initial_factors, owner, sidecar)
    assert components.strategy.idleness_for_program(owner, 15.0) == pytest.approx(expected)
    assert components.initial_factors.for_program(owner).cycles == ((1.0, 100.0), (1.0, 9.0))


def test_disabled_wait_credit_preserves_score_while_waiting():
    components = build_mori_strategy(MORIConfig(window_size=1, enable_wait_credit=False))
    owner = ProgramRef("waiting", 0)
    sidecar = MORIProgramFactors(
        tier=MORITier.CPU,
        cycles=((10.0, 90.0),),
        reasoning_wait_started_at_monotonic_s=10.0,
    )
    components.strategy._set_factor(components.initial_factors, owner, sidecar)

    assert components.strategy.idleness_for_program(owner, 10.0) == pytest.approx(0.9)
    assert components.strategy.idleness_for_program(owner, 110.0) == pytest.approx(0.9)
    fields = score_components(sidecar, 110.0, 1, wait_credit_enabled=False)
    assert fields["current_wait_credit_seconds"] == 0.0
    assert fields["wait_credit_enabled"] is False
    assert fields["wait_affects_idleness"] is False
    assert fields["idleness"] == pytest.approx(0.9)


@pytest.mark.parametrize("window", [1, 5])
@pytest.mark.parametrize("tier", [MORITier.CPU, MORITier.WAITING, MORITier.GPU])
@pytest.mark.parametrize("oldest_cycle", [(1.0, 99.0), (99.0, 1.0)])
def test_entering_wait_does_not_drop_history_or_jump_score(window, tier, oldest_cycle):
    components = build_mori_strategy(MORIConfig(window_size=window))
    cycles = (oldest_cycle,) + ((50.0, 50.0),) * (window - 1)
    before = MORIProgramFactors(tier=tier, cycles=cycles)
    native_queue = tier is MORITier.GPU
    queued = replace(
        before,
        reasoning_started_at_monotonic_s=10.0 if native_queue else None,
        reasoning_wait_started_at_monotonic_s=None if native_queue else 10.0,
    )
    score = components.strategy._idleness
    assert score(queued, 10.0) == score(before, 10.0)
    reasoning = sum(cycle[0] for cycle in cycles)
    acting = sum(cycle[1] for cycle in cycles)
    credit = 0.0 if native_queue else 10.0
    assert score(queued, 20.0) == pytest.approx(acting / (reasoning + acting + credit))
    fields = score_components(queued, 20.0, window)
    assert fields["idleness"] == score(queued, 20.0)
    assert fields["retained_cycles"] == window
    assert fields["dropped_cycles"] == 0
    assert fields["partial_reasoning_seconds"] == 0
    assert fields["current_wait_credit_seconds"] == credit
    assert fields["wait_affects_idleness"] is not native_queue


@pytest.mark.parametrize("window", [1, 5])
def test_first_execution_uses_partial_slot_without_wait_credit(window):
    components = build_mori_strategy(MORIConfig(window_size=window))
    owner = ProgramRef("queued", 0)
    cycles = ((1.0, 99.0),) + ((50.0, 50.0),) * (window - 1)
    queued = MORIProgramFactors(
        tier=MORITier.GPU,
        cycles=cycles,
        reasoning_started_at_monotonic_s=10.0,
    )
    components.strategy._set_factor(components.initial_factors, owner, queued)
    components.strategy.on_execution_changed(components.initial_factors, owner, 20.0, running=True)
    running = components.initial_factors.for_program(owner)
    fields = score_components(running, 23.0, window)
    assert fields["retained_cycles"] == window - 1
    assert fields["dropped_cycles"] == 1
    assert fields["partial_reasoning_seconds"] == 3.0
    assert fields["current_wait_credit_seconds"] == 0.0
    assert fields["wait_affects_idleness"] is False
    assert fields["idleness"] == pytest.approx(50 * (window - 1) / (100 * (window - 1) + 3))
    assert fields["idleness"] == components.strategy._idleness(running, 23.0)
    assert running.cycles == cycles


def test_first_admission_clears_fairness_credit_before_native_execution():
    scheduler = runtime()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert not scheduler.on_request_arrival("first", AgentIdentity("p"), 96, pool(hbm=0), "first")
        ref = scheduler.program_ref_for_request("first")
        waiting = scheduler.strategy_factors.for_program(ref)
        assert score_components(waiting, 100.0, 3)["current_wait_credit_seconds"] == 100.0

        clock.return_value = 100.0
        scheduler.schedule_cycle(pool())
        assert [entry.request_id for entry, _ in scheduler.consume_admitted_requests()] == ["first"]
        admitted = scheduler.strategy_factors.for_program(ref)
        assert admitted.tier is MORITier.GPU
        assert not admitted.execution_observed
        assert admitted.reasoning_wait_seconds == 0.0
        assert admitted.reasoning_wait_started_at_monotonic_s is None
        fields = score_components(admitted, 110.0, 3)
        assert fields["current_wait_credit_seconds"] == 0.0
        assert fields["partial_reasoning_seconds"] == 0.0
        assert fields["wait_affects_idleness"] is False


@pytest.mark.parametrize(
    "state",
    [
        {"tier": MORITier.GPU, "reasoning_started_at_monotonic_s": 10.0},
        {"tier": MORITier.CPU, "acting_started_at_monotonic_s": 10.0},
        {"tier": MORITier.WAITING, "acting_started_at_monotonic_s": 10.0},
    ],
)
def test_gpu_or_acting_state_cannot_receive_stale_wait_credit(state):
    components = build_mori_strategy()
    clean = MORIProgramFactors(cycles=((10.0, 90.0),), **state)
    stale = replace(clean, reasoning_wait_seconds=100.0, reasoning_wait_started_at_monotonic_s=0.0)
    assert components.strategy._idleness(stale, 100.0) == components.strategy._idleness(clean, 100.0)
    fields = score_components(stale, 100.0, 5)
    assert fields["current_wait_credit_seconds"] == 0.0
    assert fields["wait_affects_idleness"] is False
    assert fields["idleness"] == components.strategy._idleness(stale, 100.0)


@pytest.mark.parametrize("tier", [MORITier.CPU, MORITier.WAITING])
def test_uniform_waiting_score_selects_older_peer_and_matches_diagnostics(tier):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    young = replace(
        view("a-young", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=96, step_count=1),
        wait_started_at_monotonic_s=99.0,
    )
    old = replace(young, ref=ProgramRef("z-old", 0), wait_started_at_monotonic_s=0.0)
    for program in (young, old):
        components.strategy._set_factor(
            components.initial_factors,
            program.ref,
            MORIProgramFactors(
                tier=tier,
                cycles=((10.0, 90.0),),
                reasoning_wait_started_at_monotonic_s=program.wait_started_at_monotonic_s,
            ),
        )
    fixed = snapshot(young, old, hbm=96)
    scores = [components.strategy.idleness_for_program(p.ref, 100.0) for p in (young, old)]
    assert scores[1] < scores[0]
    report = dict(components.strategy.diagnostics(fixed, components.initial_factors)[0].fields)
    assert report["mean_idleness"] == pytest.approx(sum(scores) / 2)
    calls = []
    components.strategy.schedule_resume(
        fixed, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
    )
    assert [call.program for call in calls] == [old.ref]


def test_cancelled_cpu_wait_discards_credit_without_counting_wait_as_reasoning():
    scheduler = runtime()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert scheduler.on_request_arrival("first", AgentIdentity("p"), 96, pool(), "first")
        ref = scheduler.program_ref_for_request("first")
        clock.return_value = 2.0
        scheduler.on_request_completion("first", 96)
        clock.return_value = 3.0
        scheduler.schedule_cycle(pool(hbm=0))
        clock.return_value = 5.0
        assert not scheduler.on_request_arrival("second", AgentIdentity("p"), 96, pool(hbm=0), "second")
        clock.return_value = 105.0
        scheduler.cancel_request("second")
        factor = scheduler.strategy_factors.for_program(ref)
        assert scheduler.registry.require(ref).status is ProgramStatus.ACTING
        assert factor.tier is MORITier.CPU
        assert factor.reasoning_wait_seconds == 0.0
        assert factor.reasoning_wait_started_at_monotonic_s is None
        assert factor.pending_reasoning_seconds == 0.0
        clock.return_value = 110.0
        scheduler.on_request_arrival("third", AgentIdentity("p"), 96, pool(hbm=0), "third")
        assert scheduler.strategy_factors.for_program(ref).cycles == ((2.0, 3.0), (0.0, 5.0))


def test_reading_retired_owner_does_not_recreate_factors_or_classes():
    components = build_mori_strategy()
    stale = view("retired", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=96)
    components.strategy.diagnostics(snapshot(stale), components.initial_factors)
    components.strategy.on_execution_changed(components.initial_factors, stale.ref, 100.0, running=True)
    assert components.initial_factors.for_program(stale.ref) is None
    assert stale.ref not in components.strategy._classes
