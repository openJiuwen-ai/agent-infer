# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Independent union arithmetic and planner scaling checks."""

from random import Random
from unittest.mock import patch

import pytest

from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.mori import MORITier, build_mori_strategy
from agentinfer.scheduling.mori.capacity import CapacityClaim, CapacityLedger, ReliefProjection
from agentinfer.scheduling.transitions import TransitionKind
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


def test_union_and_marginal_reclaim_are_not_max_or_full_context_sum():
    a, b = ProgramRef("a", 0), ProgramRef("b", 0)
    ledger = CapacityLedger(16)
    ledger.add(a, CapacityClaim(40 * 16, frozenset(range(1, 41))))
    ledger.add(b, CapacityClaim(30 * 16, frozenset(range(21, 51))))
    assert ledger.used == 50 * 16
    assert ledger.remove(a) == 20 * 16
    assert ledger.used == 30 * 16


def test_union_ledger_matches_brute_force_with_private_growth_promises():
    random = Random(178)
    ledger = CapacityLedger(16)
    active = {}
    for _ in range(500):
        ref = ProgramRef(str(random.randrange(20)), 0)
        if random.randrange(3) == 0:
            ledger.remove(ref)
            active.pop(ref, None)
        else:
            keys = frozenset(random.sample(range(1, 100), random.randrange(20)))
            private = random.randrange(10) * 16
            claim = CapacityClaim(len(keys) * 16 + private, keys)
            ledger.add(ref, claim)
            active[ref] = claim
        union = set().union(*(claim.blocks for claim in active.values()))
        expected = len(union) * 16 + sum(claim.tokens - len(claim.blocks) * 16 for claim in active.values())
        assert ledger.used == expected


def test_used_after_swap_matches_copy_mutation_for_random_overlapping_claims():
    random = Random(743)
    for iteration in range(300):
        ledger = CapacityLedger(16)
        refs = [ProgramRef(f"owner-{iteration}-{index}", 0) for index in range(8)]
        for ref in refs:
            blocks = frozenset(random.sample(range(32), random.randrange(12)))
            ledger.add(ref, CapacityClaim(16 * (len(blocks) + random.randrange(6)), blocks))
        incoming_ref = random.choice(refs)
        restore_blocks = frozenset(random.sample(range(32), random.randrange(12)))
        restore_claim = CapacityClaim(16 * (len(restore_blocks) + random.randrange(6)), restore_blocks)
        before_claims = ledger.claims.copy()
        before_owners = ledger.owners.copy()
        before_used = ledger.used

        expected = ledger.copy()
        expected.add(ProgramRef(f"restore-{iteration}", 0), restore_claim)
        expected.remove(incoming_ref)

        assert ledger.used_after_swap(restore_claim, incoming_ref) == expected.used
        assert ledger.claims == before_claims
        assert ledger.owners == before_owners
        assert ledger.used == before_used


def test_used_after_swap_accounts_for_candidate_and_outgoing_incoming_overlap():
    candidate, incoming = ProgramRef("candidate", 0), ProgramRef("incoming", 0)
    ledger = CapacityLedger(16)
    ledger.add(candidate, CapacityClaim(64, frozenset({1, 2, 5})))
    ledger.add(incoming, CapacityClaim(64, frozenset({2, 3, 4})))
    outgoing_claim = CapacityClaim(80, frozenset({1, 3, 6}))

    expected = ledger.copy()
    expected.add(ProgramRef("outgoing", 0), outgoing_claim)
    expected.remove(incoming)

    assert ledger.used_after_swap(outgoing_claim, incoming) == expected.used
    assert ledger.used == 112


def test_used_after_swap_keeps_unproven_private_and_decode_tokens():
    incoming = ProgramRef("incoming", 0)
    ledger = CapacityLedger(16)
    ledger.add(incoming, CapacityClaim(160))
    ledger.add(ProgramRef("resident", 0), CapacityClaim(96, frozenset({1, 2})))
    restore_claim = CapacityClaim(192, frozenset({2, 3, 4}))

    expected = ledger.copy()
    expected.add(ProgramRef("outgoing", 0), restore_claim)
    expected.remove(incoming)

    assert ledger.used_after_swap(restore_claim, incoming) == expected.used


def test_legacy_plan_keeps_zero_relief_owner_until_shared_last_owner_releases():
    components = build_mori_strategy()
    first = view("first", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=32)
    second = view("second", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=32)
    candidate = ProgramRef("candidate", 0)
    ledger = CapacityLedger(16)
    shared = CapacityClaim(32, frozenset({1, 2}))
    ledger.add(first.ref, shared)
    ledger.add(second.ref, shared)
    ledger.add(candidate, CapacityClaim(32))

    plan = components.strategy._legacy_victim_plan(ledger, [first, second], quota=32)

    assert plan is not None
    assert [victim.ref for victim in plan.victims] == [first.ref, second.ref]
    assert plan.final_used == 32


def test_relief_projection_matches_independent_union_after_admissions_at_every_boundary():
    random = Random(539)
    initial = CapacityLedger(16)
    victims = tuple(ProgramRef(str(i), 0) for i in range(20))

    def claim():
        blocks = frozenset(random.sample(range(40), random.randrange(15)))
        return CapacityClaim(16 * (len(blocks) + random.randrange(6)), blocks)

    for ref in (*victims, ProgramRef("running", 0)):
        initial.add(ref, claim())
    projection = ReliefProjection(initial, victims)
    admitted = []
    for _ in range(30):
        candidate = claim()
        for boundary in range(len(victims) + 1):
            remaining = [c for ref, c in initial.claims.items() if ref not in victims[:boundary]]
            claims = [*remaining, *admitted, candidate]
            union = set().union(*(item.blocks for item in claims))
            expected = 16 * len(union) + sum(item.tokens - 16 * len(item.blocks) for item in claims)
            assert projection.required(boundary, candidate) == expected
        admitted.append(candidate)
        projection.admit(candidate)


def test_capacity_repair_uses_marginal_shared_relief_and_keeps_lazy_debt():
    components = build_mori_strategy()
    a = view("a", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    b = view("b", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    waiting = view("wait", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)

    class Shared:
        def footprint(self, ref, tier):
            return frozenset(range(1, 301)) if tier == "gpu" and ref in (a.ref, b.ref) else frozenset()

    components.strategy.bind_footprints(Shared())
    set_tier(components, a, MORITier.GPU, idleness=0.9)
    set_tier(components, b, MORITier.GPU, idleness=0.8)
    calls = []
    fixed = snapshot(a, b, waiting, hbm=350, dram=1000)
    controller = accepting_controller(components.strategy, components.initial_factors, calls)
    components.strategy.repair_capacity(fixed, components.initial_factors, controller)
    components.strategy.schedule_resume(fixed, components.initial_factors, controller)
    # Actual 400-token contexts cannot be capped below their footprint. A
    # releases only 100 tokens; the shared 300 remain owned by executing B.
    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, a.ref),
        (TransitionKind.MARK_FOR_PAUSE, b.ref),
    ]
    assert components.initial_factors.for_program(b.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(b.ref).pending_tier is MORITier.CPU
    assert components.strategy._ledgers(fixed, components.initial_factors)[0].used == 400


@pytest.mark.parametrize("quota", [1200, 1600])
def test_shared_candidate_rebalances_only_necessary_owners_when_slack_is_insufficient(quota):
    components = build_mori_strategy()
    a = view("a", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    b = view("b", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    spare = view("spare", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=100)
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    waiting = view("wait", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=700, step_count=1)

    class Shared:
        def footprint(self, ref, tier):
            return frozenset(range(1, 301)) if tier == "gpu" and ref in (a.ref, b.ref, waiting.ref) else frozenset()

    components.strategy.bind_footprints(Shared())
    for program in (a, b, spare, running):
        set_tier(components, program, MORITier.GPU, idleness=0.9)
    set_tier(components, waiting, MORITier.CPU, idleness=0.1)
    calls = []
    fixed = snapshot(a, b, spare, running, waiting, hbm=quota, dram=2000)
    components.strategy.schedule_resume(
        fixed, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
    )
    expected = [(TransitionKind.PAUSE, a.ref), (TransitionKind.PAUSE, b.ref)] if quota == 1200 else []
    expected.append((TransitionKind.RESUME, waiting.ref))
    assert [(call.kind, call.program) for call in calls] == expected
    for program in (a, b, spare, running):
        tier = MORITier.CPU if quota == 1200 and program in (a, b) else MORITier.GPU
        assert components.initial_factors.for_program(program.ref).tier is tier
        assert components.initial_factors.for_program(program.ref).pending_tier is None
    assert components.strategy._ledgers(fixed, components.initial_factors)[0].used == quota


def test_non_fitting_candidates_do_not_rescan_or_demote_acting_residents():
    counts = []
    for size in (64, 128, 256, 512):
        components = build_mori_strategy()
        victims = [
            view(f"victim-{i}", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=1) for i in range(size)
        ]
        running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100 * size)
        waiters = [
            view(f"wait-{i}", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=101 * size)
            for i in range(size)
        ]
        for program in (*victims, running):
            set_tier(components, program, MORITier.GPU, idleness=0.9)
        calls = []
        with patch.object(components.strategy, "_factor", wraps=components.strategy._factor) as factor_reads:
            components.strategy.schedule_resume(
                snapshot(*victims, running, *waiters, hbm=201 * size + 100, dram=1000 * size),
                components.initial_factors,
                accepting_controller(components.strategy, components.initial_factors, calls),
            )
        counts.append(factor_reads.call_count)
        assert components.strategy.counters["planning_candidates"] == size
        assert not calls  # Even removing all Acting owners cannot fit a candidate.
    assert all(after < before * 2.1 for before, after in zip(counts, counts[1:], strict=False))


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("has_running", [False, True])
def test_rebalance_matches_brute_force_union_planner(seed, has_running):
    random = Random(912 + seed)
    components = build_mori_strategy()
    programs, claims, scores, tiers = {}, {}, {}, {}
    for index in range(20):
        gpu = index < 10
        program = view(
            str(index),
            state=ProgramState.ACTIVE if gpu else ProgramState.PAUSED,
            status=ProgramStatus.ACTING if gpu and (index < 8 or not has_running) else ProgramStatus.REASONING,
            tokens=200 + random.randrange(400),
            step_count=random.randrange(2),
        )
        ref = program.ref
        programs[ref] = program
        blocks = frozenset(random.sample(range(1, 301), random.randrange(200)))
        claims[ref] = CapacityClaim(program.tokens.estimated_context_tokens + 100, blocks)
        scores[ref] = random.choice((0.0, 0.1, 0.5, 0.9, 1.0))
        tiers[ref] = MORITier.GPU if gpu else random.choice((MORITier.CPU, MORITier.WAITING))
        set_tier(components, program, tiers[ref], idleness=scores[ref])

    class Shared:
        def footprint(self, ref, tier):
            return claims[ref].blocks if tier == "gpu" else frozenset()

    components.strategy.bind_footprints(Shared())

    def union_tokens(refs):
        blocks = set().union(*(claims[ref].blocks for ref in refs))
        return len(blocks) + sum(claims[ref].tokens - len(claims[ref].blocks) for ref in refs)

    active = {ref for ref, tier in tiers.items() if tier is MORITier.GPU}
    quota = union_tokens(active) + random.randrange(400)

    def promotion_key(ref):
        group = 0 if tiers[ref] is MORITier.CPU else 1 if programs[ref].step_count else 2
        return group, claims[ref].tokens if group == 2 else 0, scores[ref], ref

    candidates = sorted(set(programs) - active, key=promotion_key)
    victims = sorted(
        (ref for ref in active if programs[ref].status is ProgramStatus.ACTING), key=lambda ref: (-scores[ref], ref)
    )
    expected = []
    for candidate in candidates:
        eligible = [ref for ref in victims if ref in active and scores[ref] > scores[candidate]]
        if (
            not any(programs[ref].status is ProgramStatus.REASONING for ref in active)
            and union_tokens((active - set(eligible)) | {candidate}) > quota
        ):
            eligible = [ref for ref in victims if ref in active]
        for count in range(len(eligible) + 1):
            projected = (active - set(eligible[:count])) | {candidate}
            if union_tokens(projected) <= quota:
                expected.extend((TransitionKind.PAUSE, ref) for ref in eligible[:count])
                kind = TransitionKind.RESUME if programs[candidate].step_count else TransitionKind.ADMIT
                expected.append((kind, candidate))
                active = projected
                break
    calls = []
    fixed = snapshot(*programs.values(), hbm=quota, dram=100_000)
    components.strategy.schedule_resume(
        fixed,
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [(call.kind, call.program) for call in calls] == expected
    assert components.strategy._ledgers(fixed, components.initial_factors)[0].used == union_tokens(active)


def test_non_fitting_waiter_scan_grows_linearly_in_program_reads():
    counts = []
    for size in (64, 128, 256, 512):
        components = build_mori_strategy()
        resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=900)
        waiters = [
            view(str(i), state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100) for i in range(size)
        ]
        set_tier(components, resident, MORITier.GPU)
        calls = []
        with patch.object(components.strategy, "_factor", wraps=components.strategy._factor) as factor_reads:
            components.strategy.schedule_resume(
                snapshot(resident, *waiters),
                components.initial_factors,
                accepting_controller(components.strategy, components.initial_factors, calls),
            )
        counts.append(factor_reads.call_count)
        assert components.strategy.counters["planning_candidates"] == size
        assert not calls
    assert all(after < before * 2.1 for before, after in zip(counts, counts[1:], strict=False))
