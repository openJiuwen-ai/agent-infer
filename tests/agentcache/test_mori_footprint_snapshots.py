# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Cached physical proof must invalidate on every native ownership mutation."""

import random

import pytest
from vllm.v1.core.block_pool import BlockPool

from agentinfer.agentcache.mori.footprints import FootprintBook
from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
from agentinfer.scheduling import ProgramRef, ProgramState, ProgramStatus, TransitionController
from agentinfer.scheduling.mori import MORIConfig, MORIMode, MORIStrategy, MORITier, build_mori_strategy
from tests.scheduling.test_mori_strategy import set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("seed", range(10))
def test_cached_and_uncached_proof_match_through_random_native_deltas(seed):
    rng = random.Random(seed)
    plain, cached = FootprintBook(), FootprintBook()
    plain.enabled = cached.enabled = True
    cached.configure_snapshot_cache(True)
    owners = [ProgramRef(str(i), generation) for i in range(4) for generation in [0, 1]]
    for _ in range(300):
        owner, tier, block = rng.choice(owners), rng.choice(["gpu", "cpu"]), rng.randrange(20)
        op = rng.randrange(10)
        for book in [plain, cached]:
            if op < 6:
                book.add(tier, block, owner)
            elif op < 8:
                book.remove_block(tier, block)
            elif op == 8:
                book.invalidate(owner)
            else:
                book.clear(tier)
        for ref in owners:
            for pool in ["gpu", "cpu"]:
                expected = plain.get(ref, pool)
                actual = cached.get(ref, pool)
                assert actual == expected
                if actual:
                    assert cached.get(ref, pool) is actual
        assert len(cached._snapshots) <= len(cached._by_owner)


def test_prior_immutable_snapshot_does_not_change_after_native_reuse_or_context_release():
    book = FootprintBook()
    book.enabled = True
    book.configure_snapshot_cache(True)
    old, fresh = ProgramRef("same-program", 0), ProgramRef("same-program", 1)
    book.add("gpu", 1, old)
    first = book.get(old, "gpu")
    book.remove_block("gpu", 1)
    book.add("gpu", 1, fresh)
    assert first == frozenset({1})
    assert book.get(old, "gpu") == frozenset()
    assert book.get(fresh, "gpu") == frozenset({1})
    book.invalidate(fresh)
    assert not book.get(fresh, "gpu")
    assert not book._snapshots


def test_each_pool_reset_and_global_reset_discard_cached_proof():
    book = FootprintBook()
    book.enabled = True
    book.configure_snapshot_cache(True)
    owner = ProgramRef("p", 0)
    for tier in ["gpu", "cpu"]:
        book.add(tier, 1, owner)
        book.get(owner, tier)
    book.clear("gpu")
    assert not book.get(owner, "gpu")
    assert book.get(owner, "cpu") == frozenset({1})
    book.clear()
    assert not book._snapshots and not book.get(owner, "cpu")


def test_actual_native_block_id_reuse_invalidates_the_cached_footprint():
    state = MORIRuntimeState()
    strategy = MORIStrategy(MORIConfig(fast_capacity_accounting=True))
    state.bind_provider(strategy)
    state.footprints.enabled = True
    owner = ProgramRef("native-owner", 0)
    state.bind_request("native-request", owner)
    policy = MORIHBMPolicy(state)
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    pool.register_mori_block_selection_provider(policy)
    blocks = pool.get_new_blocks(1)
    pool.observe_mori_block_owner("native-request", blocks)
    proof = state.footprint(owner, "gpu")
    assert proof == frozenset({blocks[0].block_id})
    assert state.footprint(owner, "gpu") is proof
    assert state.footprint_snapshot_hits > 0
    pool.free_blocks(blocks)
    pool.get_new_blocks(7)
    assert not state.footprint(owner, "gpu")
    pool.unregister_mori_block_selection_provider(policy)
    policy.close()
    state.reset()
    assert not state.footprints._snapshots


def test_recycled_block_and_new_prompt_invalidate_only_relevant_proof():
    book = FootprintBook()
    book.enabled = True
    a, b = ProgramRef("a", 0), ProgramRef("b", 0)
    for owner in (a, b):
        book.add("gpu", 1, owner)
    book.add("cpu", 1, a)
    book.remove_block("gpu", 1)
    assert book.get(a, "gpu") == book.get(b, "gpu") == frozenset()
    assert book.get(a, "cpu") == frozenset({1})
    book.add("gpu", 1, b)  # new contents, old owner proof cannot survive reuse
    book.invalidate(a)
    assert book.get(b, "gpu") == frozenset({1})
    assert not book.get(a, "cpu")


@pytest.mark.parametrize(("policy_type", "tier"), [(MORIHBMPolicy, "gpu"), (MORISimpleCPUOffloadPolicy, "cpu")])
def test_native_ownership_reestablishes_proof_after_prompt_change_and_clears_on_reuse(policy_type, tier):
    """The native ownership delta supplies proof even if the LRU owner was already known."""
    state = MORIRuntimeState()
    state.footprints.enabled = True
    owner = ProgramRef("agent", 0)
    state.bind_request("request", owner)
    policy = policy_type(state)
    policy.on_block_owner_added(8, "request")
    assert state.footprint(owner, tier) == frozenset({8})
    state.invalidate_footprint(owner)
    policy.on_block_owner_added(8, "request")
    assert state.footprint(owner, tier) == frozenset({8})
    policy.on_block_removed(8)
    assert not state.footprint(owner, tier)
    policy.on_block_owner_added(8, "request")
    policy.on_cache_reset()
    assert not state.footprint(owner, tier)
    policy.close()


def test_runtime_state_idle_to_inactive_callback_preserves_footprint_book():
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL, decode_buffer_tokens=0))
    program = view("cpu", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=60, step_count=1)
    set_tier(components, program, MORITier.CPU, idleness=0.9)
    state = MORIRuntimeState()
    state.footprints.enabled = True

    class Index:
        def __init__(self) -> None:
            self.reclassified: list[ProgramRef] = []

        def reclassify_program(self, program: ProgramRef) -> int:
            self.reclassified.append(program)
            return 0

        def release_program(self, program: ProgramRef) -> int:
            del program
            return 0

    index = Index()
    state.register_index(index)
    state.bind_provider(components.strategy)
    state.footprints.add("cpu", 7, program.ref)
    before = state.footprints.get(program.ref, "cpu")
    try:
        components.strategy.repair_capacity(
            snapshot(program, hbm=1000, dram=0),
            components.initial_factors,
            TransitionController(lambda request: None),
        )

        assert components.initial_factors.for_program(program.ref).tier is MORITier.WAITING
        assert index.reclassified == [program.ref]
        assert before == frozenset({7})
        assert state.footprints.get(program.ref, "cpu") == before
    finally:
        state.unbind_provider(components.strategy)
