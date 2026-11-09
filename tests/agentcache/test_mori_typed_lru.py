# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Tests for the shared incremental typed-LRU index."""

from __future__ import annotations

import pytest

from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.typed_lru import TypedLRUIndex
from agentinfer.scheduling import CacheClass, ProgramRef
from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy

pytestmark = pytest.mark.cpu_test


def test_hbm_order_preserves_native_lru_inside_each_class() -> None:
    classes = {
        ProgramRef("busy", 0): CacheClass.BUSY,
        ProgramRef("idle", 0): CacheClass.IDLE,
        ProgramRef("inactive", 0): CacheClass.INACTIVE,
    }
    index = TypedLRUIndex[int](
        eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2},
        class_for_program=classes.__getitem__,
    )
    index.upsert(1, owners=(ProgramRef("busy", 0),), native_lru_seq=1, ready=True)
    index.upsert(2, owners=(ProgramRef("idle", 0),), native_lru_seq=4, ready=True)
    index.upsert(3, owners=(ProgramRef("inactive", 0),), native_lru_seq=8, ready=True)
    index.upsert(4, owners=(ProgramRef("inactive", 0),), native_lru_seq=2, ready=True)

    assert index.propose_exact(4) == (4, 3, 2, 1)
    assert index.propose_exact(4) == (4, 3, 2, 1)


def test_reservation_commits_removed_entries_without_restoring_them() -> None:
    owner = ProgramRef("owner", 0)
    index = TypedLRUIndex[str](
        eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2},
        class_for_program=lambda program: CacheClass.INACTIVE,
    )
    index.upsert("first", owners=(owner,), native_lru_seq=1, ready=True)
    index.upsert("second", owners=(owner,), native_lru_seq=2, ready=True)

    with index.reserve_exact(1) as proposed:
        assert proposed == ("first",)
        index.remove("first")

    assert index.propose_exact(1) == ("second",)


def test_reservation_restores_live_entries_after_rejected_commit() -> None:
    owner = ProgramRef("owner", 0)
    index = TypedLRUIndex[str](
        eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2},
        class_for_program=lambda program: CacheClass.INACTIVE,
    )
    index.upsert("first", owners=(owner,), native_lru_seq=1, ready=True)

    with index.reserve_exact(1) as proposed:
        assert proposed == ("first",)

    assert index.propose_exact(1) == ("first",)


def test_shared_entry_uses_most_protected_owner_and_reclassifies_eagerly() -> None:
    idle = ProgramRef("idle", 0)
    busy = ProgramRef("busy", 0)
    classes = {idle: CacheClass.IDLE, busy: CacheClass.BUSY}
    index = TypedLRUIndex[int](
        eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2},
        class_for_program=classes.__getitem__,
    )
    index.upsert(10, owners=(idle, busy), native_lru_seq=3, ready=True)

    assert index.cache_class(10) is CacheClass.BUSY
    classes[busy] = CacheClass.INACTIVE
    assert index.reclassify_program(busy) == 1
    assert index.cache_class(10) is CacheClass.IDLE
    assert index.stats["entries_reclassified"] == 1


def test_cpu_order_is_independent_and_protected_entries_are_skipped() -> None:
    owners = {
        ProgramRef("inactive", 0): CacheClass.INACTIVE,
        ProgramRef("busy", 0): CacheClass.BUSY,
        ProgramRef("idle", 0): CacheClass.IDLE,
    }
    index = TypedLRUIndex[str](
        eviction_order=(CacheClass.INACTIVE, CacheClass.BUSY, CacheClass.IDLE),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.BUSY: 1, CacheClass.IDLE: 2},
        class_for_program=owners.__getitem__,
    )
    for sequence, name in enumerate(("inactive", "busy", "idle"), 1):
        index.upsert(name, owners=(ProgramRef(name, 0),), native_lru_seq=sequence, ready=True)

    assert index.propose_exact(2, protected=("inactive",)) == ("busy", "idle")
    assert index.propose_exact(3, protected=("inactive",)) is None


def test_class_change_preserves_old_native_sequence() -> None:
    owner = ProgramRef("owner", 0)
    peer = ProgramRef("peer", 0)
    classes = {owner: CacheClass.BUSY, peer: CacheClass.INACTIVE}
    index = TypedLRUIndex[int](
        eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2},
        class_for_program=classes.__getitem__,
    )
    index.upsert(1, owners=(owner,), native_lru_seq=1, ready=True)
    index.upsert(2, owners=(peer,), native_lru_seq=2, ready=True)
    classes[owner] = CacheClass.INACTIVE
    index.reclassify_program(owner)

    assert index.propose_exact(2) == (1, 2)


def test_repeated_touches_compact_stale_records_without_changing_order() -> None:
    owner = ProgramRef("owner", 0)
    index = TypedLRUIndex[int](
        eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2},
        class_for_program=lambda program: CacheClass.BUSY,
    )
    index.upsert(1, owners=(owner,), native_lru_seq=1, ready=True)
    index.upsert(2, owners=(owner,), native_lru_seq=2, ready=True)

    for sequence in range(3, 260):
        index.touch(1, sequence)

    assert index.stats["heap_compactions"] > 0
    assert index.propose_exact(2) == (2, 1)


def test_hbm_owner_callback_resolves_request_only_inside_server_runtime() -> None:
    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    policy = MORIHBMPolicy(state)
    program = ProgramRef("program", 0)
    state.bind_request("request", program)

    policy.on_block_owner_added(7, "request")
    policy.on_block_ready(7, 12)
    assert policy.index.owners(7) == frozenset((program,))
    assert policy.index.cache_class(7) is CacheClass.INACTIVE

    components.strategy._set_factor(
        components.initial_factors,
        program,
        MORIProgramFactors(tier=MORITier.GPU),
    )
    assert policy.index.cache_class(7) is CacheClass.BUSY

    policy.close()
    state.unbind_provider(components.strategy)


def test_large_stale_heap_recovers_without_losing_lru_order() -> None:
    policy = MORIHBMPolicy(MORIRuntimeState())
    for key in range(5000):
        policy.on_block_ready(key, key)
    for key in range(4999):
        policy.on_block_touched(key, 5000 + key)

    expected = (4999, *range(4999))
    assert policy.propose(5000) == expected
    assert policy.propose(5000) == expected
    assert policy.index.stats["stale_repair_limit"] == 1
    policy.close()


def test_stale_rebuild_preserves_reserved_and_protected_entries_exactly_once() -> None:
    index = TypedLRUIndex[int](
        eviction_order=(CacheClass.INACTIVE, CacheClass.IDLE, CacheClass.BUSY),
        protection_rank={CacheClass.INACTIVE: 0, CacheClass.IDLE: 1, CacheClass.BUSY: 2},
        class_for_program=lambda program: CacheClass.INACTIVE,
        max_stale_repairs_per_proposal=2,
    )
    for key in range(10):
        index.upsert(key, owners=(), native_lru_seq=key, ready=True)
    for key in range(2, 9):
        index.touch(key, 10 + key)

    with index.reserve_exact(3, protected=(1,)) as proposed:
        assert proposed == (0, 9, 2)
        index.remove(0)
    assert index.propose_exact(9) == (1, 9, *range(2, 9))
    assert index.propose_exact(10) is None
