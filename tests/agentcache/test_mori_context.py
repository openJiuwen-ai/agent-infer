# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Prompt changes invalidate cache ownership without destroying native contents."""

from __future__ import annotations

import pytest

from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState, ProgramCacheContext
from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
from agentinfer.scheduling import CacheClass, ProgramRef
from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy

pytestmark = pytest.mark.cpu_test


def test_pending_prompt_drops_capacity_proof_but_preserves_typed_owner_until_dispatch():
    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    state.footprints.enabled = True
    policy = MORISimpleCPUOffloadPolicy(state)
    program = ProgramRef("agent", 0)
    components.strategy._set_factor(
        components.initial_factors,
        program,
        MORIProgramFactors(tier=MORITier.CPU),
    )
    state.bind_request("old", program)
    policy.on_block_owner_added(1, "old")
    policy.on_block_ready(1, 10)

    components.strategy.on_context_pending(program)

    assert not state.footprint(program, "cpu")
    assert state.program_for_request("old") == program
    assert policy.index.owners(1) == frozenset({program})
    assert policy.index.cache_class(1) is CacheClass.IDLE

    components.strategy.on_context_replaced(program)

    assert state.program_for_request("old") is None
    assert not policy.index.owners(1)
    assert policy.index.cache_class(1) is CacheClass.INACTIVE
    policy.close()
    state.unbind_provider(components.strategy)


@pytest.mark.parametrize(("policy_type", "tier"), [(MORIHBMPolicy, "gpu"), (MORISimpleCPUOffloadPolicy, "cpu")])
def test_prompt_replacement_removes_old_ownership_but_keeps_shared_cache_and_pins(policy_type, tier):
    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    state.footprints.enabled = True
    policy = policy_type(state)
    program, peer = ProgramRef("agent", 0), ProgramRef("peer", 0)
    factors = MORIProgramFactors(tier=MORITier.GPU)
    components.strategy._set_factor(components.initial_factors, program, factors)
    components.strategy._set_factor(components.initial_factors, peer, MORIProgramFactors(tier=MORITier.CPU))
    state.bind_request("old", program)
    state.bind_request("peer", peer)
    ticket = state.capture_context("old")
    for block in (1, 2, 3):
        policy.on_block_owner_added(block, "old")
    policy.on_block_owner_added(2, "peer")
    policy.on_block_ready(1, 10)
    policy.on_block_ready(2, 20)
    # Block 3 still carries a native transfer pin when the next prompt arrives.
    components.strategy.on_context_replaced(program)

    assert state.program_for_context(ticket) is None
    assert state.program_for_request("old") is None
    assert state.program_for_request("peer") == peer
    assert not state.footprint(program, tier)
    assert state.footprint(peer, tier) == frozenset({2})
    assert policy.index.owners(1) == policy.index.owners(3) == frozenset()
    assert policy.index.owners(2) == frozenset({peer})
    assert policy.index.cache_class(1) is CacheClass.INACTIVE
    assert policy.propose(3) is None  # Dropping ownership never drops a transfer pin.
    assert policy.propose(2) == (1, 2)
    assert components.strategy._factor(components.initial_factors, program) == factors

    # A late old callback may make native data reusable, but may not restore ownership.
    policy.on_block_owner_added(3, "old")
    policy.on_block_ready(3, 30)
    assert policy.index.owners(3) == frozenset()
    assert policy.propose(3) == (1, 3, 2)

    # Even an identical/shared prefix must be reobserved for the new request.
    state.bind_request("new", program)
    policy.on_block_owner_added(2, "new")
    policy.on_block_owner_added(4, "new")
    assert policy.index.owners(2) == frozenset({program, peer})
    assert state.footprint(program, tier) == frozenset({2, 4})
    assert set(policy.index.entries_for_program(program)) == {2, 4}
    assert state.capture_context("new") is not ticket
    policy.close()
    state.unbind_provider(components.strategy)


def test_context_receipt_survives_finish_but_not_replacement_release_or_shutdown():
    state = MORIRuntimeState()
    program = ProgramRef("agent", 0)
    state.bind_request("first", program)
    first = state.capture_context("first")
    state.unbind_request("first")
    assert state.program_for_context(first) == program
    assert state.program_for_context(ProgramCacheContext(program)) is None
    assert state.program_for_context(object()) is None

    state.replace_context(program)
    state.bind_request("second", program)
    second = state.capture_context("second")
    assert state.program_for_context(first) is None
    assert state.program_for_context(second) == program
    state.on_program_released(program)
    assert state.program_for_context(second) is None

    newer = ProgramRef("agent", 1)
    state.bind_request("third", newer)
    third = state.capture_context("third")
    state.on_program_released(program)  # An old generation cannot invalidate the new one.
    assert state.program_for_context(third) == newer
    state.clear_requests()
    assert state.program_for_context(third) is None
    assert not state._contexts


def test_context_replacement_clears_pending_generic_cpu_owners_before_late_insert():
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

    from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
    from agentinfer.agentcache.mori_cpu_policy import MoriCPUCachePolicy

    state = MORI_RUNTIME_STATE
    state.reset()
    policy = MoriCPUCachePolicy(4)
    owner, peer = ProgramRef("owner", 0), ProgramRef("peer", 0)
    state.bind_request("old", owner)
    state.bind_request("peer", peer)
    shared, private = [make_offload_key(key, 0) for key in (b"shared", b"private")]
    policy.on_store_miss((shared, private), ReqContext("old"))
    policy.on_store_miss((shared,), ReqContext("peer"))
    try:
        state.replace_context(owner)
        assert state.program_for_request("old") is None
        assert policy._pending_owners == {shared: {peer}}
        policy.on_request_finished(((shared, private),), set(), set(), ReqContext("old"))
        for key, block_id in ((shared, 1), (private, 2)):
            chunk = ChunkStatus(block_id)
            chunk.ref_cnt = 0
            policy.insert(key, chunk)
        assert policy.index.owners(shared) == frozenset({peer})
        assert not policy.index.owners(private)
        assert not policy.index.entries_for_program(owner)
        assert not policy._pending_owners
    finally:
        policy.close()
        state.reset()
