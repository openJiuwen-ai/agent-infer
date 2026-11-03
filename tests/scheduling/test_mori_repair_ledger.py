# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Repair-scoped MORI ledger reuse preserves the former two-build policy."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling import (
    CacheClass,
    ProgramRef,
    ProgramState,
    ProgramStatus,
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)
from agentinfer.scheduling.mori import MORIConfig, MORIMode, MORIProgramFactors, MORITier, build_mori_strategy
from tests.scheduling.test_mori_strategy import accepting_controller, snapshot, view

pytestmark = pytest.mark.cpu_test


class _FootprintSpy:
    def __init__(self, blocks: dict[tuple[ProgramRef, str], frozenset[int]]) -> None:
        self.blocks = blocks
        self.calls: list[tuple[ProgramRef, str]] = []

    def footprint(self, program: ProgramRef, tier: str) -> frozenset[int]:
        self.calls.append((program, tier))
        return self.blocks.get((program, tier), frozenset())

    def invalidate_footprint(self, program: ProgramRef) -> None:
        del program

    def replace_context(self, program: ProgramRef) -> None:
        del program


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[ProgramRef, CacheClass, CacheClass] | tuple[ProgramRef]] = []

    def on_cache_class_changed(self, program: ProgramRef, previous: CacheClass, current: CacheClass) -> None:
        self.events.append((program, previous, current))

    def on_program_released(self, program: ProgramRef) -> None:
        self.events.append((program,))


def _factor(tier: MORITier, *, idle: float, pending: MORITier | None = None) -> MORIProgramFactors:
    return MORIProgramFactors(tier=tier, pending_tier=pending, cycles=((1.0 - idle, idle),))


def _scenario(name: str):
    blocks: dict[tuple[ProgramRef, str], frozenset[int]] = {}
    if name == "under_quota":
        gpu = view("gpu", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=300)
        cpu = view("cpu", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=200, step_count=1)
        programs = (gpu, cpu)
        factors = {gpu.ref: _factor(MORITier.GPU, idle=0.4), cpu.ref: _factor(MORITier.CPU, idle=0.6)}
        return programs, factors, 1000, 1000, blocks, 2
    if name == "cpu_shared_debt":
        first = view("cpu-a", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=600, step_count=1)
        second = view("cpu-b", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=500, step_count=1)
        programs = (first, second)
        factors = {first.ref: _factor(MORITier.CPU, idle=0.9), second.ref: _factor(MORITier.CPU, idle=0.1)}
        blocks[(first.ref, "cpu")] = frozenset(range(1, 201))
        blocks[(second.ref, "cpu")] = frozenset(range(1, 201))
        return programs, factors, 1000, 600, blocks, 1
    if name == "gpu_acting_debt":
        acting = view("acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
        reasoning = view("reasoning", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=300)
        programs = (acting, reasoning)
        factors = {
            acting.ref: _factor(MORITier.GPU, idle=0.9),
            reasoning.ref: _factor(MORITier.GPU, idle=0.1),
        }
        return programs, factors, 800, 2000, blocks, 2
    if name == "gpu_reasoning_debt":
        first = view("reasoning-a", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
        second = view("reasoning-b", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=200)
        programs = (first, second)
        factors = {first.ref: _factor(MORITier.GPU, idle=0.9), second.ref: _factor(MORITier.GPU, idle=0.1)}
        return programs, factors, 600, 2000, blocks, 2
    if name == "simultaneous_shared_debt":
        cpu = view("cpu", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=500, step_count=1)
        acting = view("acting", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=600)
        reasoning = view("reasoning", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
        programs = (cpu, acting, reasoning)
        factors = {
            cpu.ref: _factor(MORITier.CPU, idle=0.95),
            acting.ref: _factor(MORITier.GPU, idle=0.8),
            reasoning.ref: _factor(MORITier.GPU, idle=0.2),
        }
        blocks[(cpu.ref, "cpu")] = frozenset(range(101, 105))
        blocks[(acting.ref, "gpu")] = frozenset(range(1, 25))
        blocks[(acting.ref, "cpu")] = frozenset(range(101, 105))
        blocks[(reasoning.ref, "gpu")] = frozenset(range(1, 25))
        return programs, factors, 500, 400, blocks, 2
    if name == "pending_cpu_lazy_relief":
        reasoning = view("reasoning", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
        programs = (reasoning,)
        factors = {reasoning.ref: _factor(MORITier.GPU, idle=0.9, pending=MORITier.CPU)}
        return programs, factors, 600, 800, blocks, 2
    raise AssertionError(f"unknown scenario: {name}")


def _install(components, factors: dict[ProgramRef, MORIProgramFactors]) -> None:
    for ref, sidecar in factors.items():
        components.strategy._set_factor(components.initial_factors, ref, sidecar)


def _fixed_snapshot(scenario: str, programs, hbm: int, dram: int):
    fixed = snapshot(*programs, hbm=hbm, dram=dram)
    if scenario == "simultaneous_shared_debt":
        return replace(fixed, hbm_block_size_tokens=16, dram_chunk_size_tokens=64)
    return fixed


def _controller(components, calls: list[TransitionRequest], behavior: str) -> TransitionController:
    accepted = accepting_controller(components.strategy, components.initial_factors, calls)

    def apply(request: TransitionRequest) -> TransitionResult:
        if request.kind in (TransitionKind.PAUSE, TransitionKind.MARK_FOR_PAUSE):
            if behavior == "reject":
                calls.append(request)
                return TransitionResult(request.kind, request.program, False, "test_rejected")
            if behavior == "raise":
                calls.append(request)
                raise RuntimeError("test transition failure")
        return accepted.apply(request)

    return TransitionController(apply)


def _legacy_repair(components, fixed, controller: TransitionController, ledgers: list) -> None:
    """Execute the pre-iteration-06 build/CPU/build/GPU helper sequence."""
    strategy = components.strategy
    factors = components.initial_factors
    if strategy.config.mode is MORIMode.FULL:
        _gpu, cpu = strategy._ledgers(fixed, factors)
        strategy._repair_cpu_capacity(fixed, factors, cpu)
    gpu, cpu = strategy._ledgers(fixed, factors)
    ledgers.append((gpu, cpu))
    strategy._repair_gpu_capacity(fixed, factors, controller, gpu, cpu)


def _ledger_pair_signature(pair):
    gpu, cpu = pair
    return (
        gpu.used,
        gpu.claims.copy(),
        gpu.owners.copy(),
        cpu.used,
        cpu.claims.copy(),
        cpu.owners.copy(),
    )


def _policy_counters(strategy) -> Counter[str]:
    excluded = {
        "planning_program_reads",
        "repair_cycles",
        "repair_ledger_builds",
        "repair_programs_observed",
        "repair_program_reads_avoided",
    }
    return Counter({name: value for name, value in strategy.counters.items() if name not in excluded})


def test_repair_builds_one_hook_local_ledger_pair_per_cycle():
    expected_avoided = 2
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL, decode_buffer_tokens=0))
    programs, factors, hbm, dram, _blocks, _second_build_reads = _scenario("gpu_acting_debt")
    _install(components, factors)
    fixed = snapshot(*programs, hbm=hbm, dram=dram)
    calls: list[TransitionRequest] = []
    controller = _controller(components, calls, "accept")

    with patch.object(components.strategy, "_ledgers", wraps=components.strategy._ledgers) as builds:
        components.strategy.repair_capacity(fixed, components.initial_factors, controller)

    assert builds.call_count == 1
    assert components.strategy.counters["planning_program_reads"] == len(programs)
    assert components.strategy.counters["repair_cycles"] == 1
    assert components.strategy.counters["repair_ledger_builds"] == 1
    assert components.strategy.counters["repair_programs_observed"] == len(programs)
    assert components.strategy.counters["repair_program_reads_avoided"] == expected_avoided
    assert components.strategy.repair_ledger_cpu_seconds >= 0.0
    fields = dict(components.strategy.diagnostics(fixed, components.initial_factors)[0].fields)
    assert fields["repair_cycles"] == 1
    assert fields["repair_ledger_builds"] == 1
    assert fields["repair_programs_observed"] == len(programs)
    assert fields["repair_program_reads_avoided"] == expected_avoided
    assert fields["repair_ledger_cpu_seconds"] >= 0.0


def test_repair_ledger_pair_is_not_reused_across_cycles():
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL, decode_buffer_tokens=0))
    programs, factors, hbm, dram, _blocks, _second_build_reads = _scenario("under_quota")
    _install(components, factors)
    fixed = snapshot(*programs, hbm=hbm, dram=dram)
    calls: list[TransitionRequest] = []
    controller = _controller(components, calls, "accept")

    with patch.object(components.strategy, "_ledgers", wraps=components.strategy._ledgers) as builds:
        components.strategy.repair_capacity(fixed, components.initial_factors, controller)
        components.strategy.repair_capacity(fixed, components.initial_factors, controller)

    assert builds.call_count == 2
    assert components.strategy.counters["repair_cycles"] == 2
    assert components.strategy.counters["repair_ledger_builds"] == 2
    assert components.strategy.counters["repair_programs_observed"] == 2 * len(programs)
    assert components.strategy.counters["repair_program_reads_avoided"] == 2 * len(programs)


@pytest.mark.parametrize(
    ("scenario", "behavior"),
    [
        ("under_quota", "accept"),
        ("cpu_shared_debt", "accept"),
        ("gpu_acting_debt", "accept"),
        ("gpu_reasoning_debt", "accept"),
        ("simultaneous_shared_debt", "accept"),
        ("pending_cpu_lazy_relief", "accept"),
        ("gpu_acting_debt", "reject"),
        ("gpu_acting_debt", "raise"),
    ],
)
def test_shared_repair_ledgers_match_the_former_rebuild_path(scenario, behavior):
    programs, factors, hbm, dram, blocks, second_build_reads = _scenario(scenario)
    candidate = build_mori_strategy(MORIConfig(mode=MORIMode.FULL, decode_buffer_tokens=0))
    reference = build_mori_strategy(MORIConfig(mode=MORIMode.FULL, decode_buffer_tokens=0))
    _install(candidate, factors)
    _install(reference, factors)
    candidate_observer = _Observer()
    reference_observer = _Observer()
    candidate.strategy.subscribe_cache_class_observer(candidate_observer)
    reference.strategy.subscribe_cache_class_observer(reference_observer)
    candidate_footprints = _FootprintSpy(blocks)
    reference_footprints = _FootprintSpy(blocks)
    candidate.strategy.bind_footprints(candidate_footprints)
    reference.strategy.bind_footprints(reference_footprints)
    fixed = _fixed_snapshot(scenario, programs, hbm, dram)
    candidate_calls: list[TransitionRequest] = []
    reference_calls: list[TransitionRequest] = []
    candidate_ledgers: list = []
    reference_ledgers: list = []
    candidate_gpu_repair = candidate.strategy._repair_gpu_capacity

    def observe_candidate_ledgers(fixed, factors, transitions, gpu, cpu):
        candidate_ledgers.append((gpu, cpu))
        return candidate_gpu_repair(fixed, factors, transitions, gpu, cpu)

    candidate_error = None
    reference_error = None
    with patch.object(candidate.strategy, "_repair_gpu_capacity", side_effect=observe_candidate_ledgers):
        try:
            candidate.strategy.repair_capacity(
                fixed, candidate.initial_factors, _controller(candidate, candidate_calls, behavior)
            )
        except RuntimeError as error:
            candidate_error = str(error)
    try:
        _legacy_repair(
            reference,
            fixed,
            _controller(reference, reference_calls, behavior),
            reference_ledgers,
        )
    except RuntimeError as error:
        reference_error = str(error)

    assert candidate_error == reference_error
    assert candidate_calls == reference_calls
    assert candidate.initial_factors.program_factors == reference.initial_factors.program_factors
    assert candidate.strategy.cache_class_epoch == reference.strategy.cache_class_epoch
    assert candidate_observer.events == reference_observer.events
    assert _policy_counters(candidate.strategy) == _policy_counters(reference.strategy)
    assert len(reference_footprints.calls) - len(candidate_footprints.calls) == second_build_reads
    if scenario == "gpu_reasoning_debt" and behavior == "accept":
        assert [call.kind for call in candidate_calls] == [TransitionKind.MARK_FOR_PAUSE]
    if scenario == "simultaneous_shared_debt" and behavior == "accept":
        assert fixed.hbm_block_size_tokens == 16
        assert fixed.dram_chunk_size_tokens == 64
        assert len(blocks[(programs[1].ref, "gpu")]) * 16 <= 608
        assert len(blocks[(programs[2].ref, "gpu")]) * 16 <= 400
        assert len(blocks[(programs[0].ref, "cpu")]) * 64 <= 512
        assert len(blocks[(programs[1].ref, "cpu")]) * 64 <= 640
        assert [(call.kind, call.program) for call in candidate_calls] == [(TransitionKind.PAUSE, programs[1].ref)]
    assert len(candidate_ledgers) == len(reference_ledgers) == 1
    assert _ledger_pair_signature(candidate_ledgers[0]) == _ledger_pair_signature(reference_ledgers[0])
