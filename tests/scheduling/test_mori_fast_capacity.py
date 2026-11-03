# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exact accounting differential checks against the retained slow implementation."""

import random

import pytest

from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.mori.capacity import CapacityClaim, CapacityLedger

pytestmark = pytest.mark.cpu_test


def union_usage(ledger):
    blocks = set().union(*(claim.blocks for claim in ledger.claims.values()))
    private = sum(claim.tokens - ledger.quantum * len(claim.blocks) for claim in ledger.claims.values())
    return private + len(blocks) * ledger.quantum


def same(slow, fast):
    assert fast.claims == slow.claims
    assert fast.owners == slow.owners
    assert fast.used == slow.used == union_usage(fast)
    assert fast._shared == {block for block, owners in fast.owners.items() if owners > 1}
    for owner in slow.claims:
        assert fast.reclaimable(owner) == slow.reclaimable(owner)


@pytest.mark.parametrize("quantum", [1, 16, 64])
@pytest.mark.parametrize("seed", range(12))
def test_random_add_replace_remove_copy_and_swap_preserve_exact_union_accounting(seed, quantum):
    rng = random.Random(seed)
    slow, fast = CapacityLedger(quantum), CapacityLedger(quantum, fast=True)
    owners = [ProgramRef(str(i), generation) for i in range(5) for generation in [0, 1]]
    for _ in range(250):
        owner = rng.choice(owners)
        blocks = frozenset(rng.sample(range(48), rng.randrange(25)))
        claim = CapacityClaim(len(blocks) * quantum + rng.randrange(100), blocks)
        assert fast.incremental(claim) == slow.incremental(claim)
        if rng.random() < 0.65:
            slow.add(owner, claim)
            fast.add(owner, claim)
        else:
            assert fast.remove(owner) == slow.remove(owner)
        same(slow, fast)
        if slow.claims:
            removed = rng.choice(list(slow.claims))
            projected = slow.copy()
            projected.add(ProgramRef("absent-restore", 99), claim)
            projected.remove(removed)
            assert fast.used_after_swap(claim, removed) == slow.used_after_swap(claim, removed) == projected.used
        # A speculative copy can be changed without mutating either parent.
        copied_slow, copied_fast = slow.copy(), fast.copy()
        assert copied_fast.fast
        assert copied_fast._shared is not fast._shared
        copied_slow.remove(owner)
        copied_fast.remove(owner)
        same(copied_slow, copied_fast)
        same(slow, fast)


def test_three_to_two_to_one_owner_transitions_and_replacement():
    ledger = CapacityLedger(16, fast=True)
    owners = [ProgramRef(str(i), 0) for i in range(3)]
    for owner in owners:
        ledger.add(owner, CapacityClaim(32, frozenset({1, 2})))
    assert ledger.used == 32
    ledger.remove(owners[0])
    assert ledger._shared == {1, 2}
    ledger.remove(owners[1])
    assert not ledger._shared
    ledger.add(owners[2], CapacityClaim(16, frozenset({7})))
    assert ledger.used == 16 and set(ledger.owners) == {7}
    assert ledger.remove(owners[2]) == 16
    assert ledger.used == 0 and not ledger.owners and not ledger._shared
