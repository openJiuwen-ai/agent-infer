# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Owner retirement keeps native KV usable without reviving an old generation."""

from __future__ import annotations

import pytest
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE, MORIRuntimeState
from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
from agentinfer.agentcache.mori_cpu_policy import MoriCPUCachePolicy
from agentinfer.scheduling import CacheClass, ProgramRef
from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(("policy_type", "tier"), [(MORIHBMPolicy, "gpu"), (MORISimpleCPUOffloadPolicy, "cpu")])
def test_retirement_preserves_shared_kv_native_order_and_pinned_transfers(policy_type, tier):
    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    state.footprints.enabled = True
    policy = policy_type(state)
    retired, peer = ProgramRef("retired", 0), ProgramRef("peer", 0)
    for owner in (retired, peer):
        state.bind_request(owner.program_id, owner)
        components.strategy._set_factor(components.initial_factors, owner, MORIProgramFactors(tier=MORITier.CPU))
    state.bind_request("retired-second-request", retired)
    for block in (1, 2, 3):
        policy.on_block_owner_added(block, retired.program_id)
    policy.on_block_owner_added(2, peer.program_id)
    policy.on_block_ready(1, 10)
    policy.on_block_ready(2, 20)
    # Block 3 is still pinned by a native transfer when the owner retires.
    state.on_program_released(retired)

    assert state.program_for_request("retired") is None
    assert state.program_for_request("retired-second-request") is None
    assert retired not in state._program_requests
    assert state.program_for_request("peer") == peer
    assert not state.footprint(retired, tier)
    assert state.footprint(peer, tier) == frozenset({2})
    assert state.footprints._by_block == {(tier, 2): {peer}}
    assert not policy.index.entries_for_program(retired)
    assert policy.index.owners(1) == policy.index.owners(3) == frozenset()
    assert policy.index.owners(2) == frozenset({peer})
    assert policy.index.cache_class(1) is CacheClass.INACTIVE
    assert policy.index.cache_class(2) is CacheClass.IDLE
    assert policy.propose(3) is None  # retirement never makes a pinned block evictable
    assert policy.propose(2) == (1, 2)

    # Delayed ownership and completion callbacks cannot resurrect the old owner.
    policy.on_block_owner_added(3, "retired")
    policy.on_block_ready(3, 30)
    assert policy.index.owners(3) == frozenset()
    assert not state.footprint(retired, tier)
    assert policy.propose(3) == (1, 3, 2)
    state.on_program_released(peer)
    assert policy.propose(3) == (1, 2, 3)  # original native recency survived both releases
    assert not state.footprints._by_owner
    assert not state.footprints._by_block
    policy.close()
    state.unbind_provider(components.strategy)


@pytest.mark.parametrize("policy_type", [MORIHBMPolicy, MORISimpleCPUOffloadPolicy])
def test_late_old_generation_release_does_not_remove_new_owner(policy_type):
    state = MORIRuntimeState()
    state.footprints.enabled = True
    policy = policy_type(state)
    old, new = ProgramRef("same-program", 0), ProgramRef("same-program", 1)
    state.bind_request("old-request", old)
    policy.on_block_owner_added(7, "old-request")
    state.on_program_released(old)
    state.bind_request("new-request", new)
    policy.on_block_owner_added(7, "new-request")
    state.on_program_released(old)
    policy.on_block_owner_added(7, "old-request")
    policy.on_block_ready(7, 12)

    assert policy.index.owners(7) == frozenset({new})
    assert policy.index.entries_for_program(new) == (7,)
    assert not policy.index.entries_for_program(old)
    assert state.program_for_request("new-request") == new
    assert policy.propose(1) == (7,)
    policy.close()


def test_generic_cpu_retirement_clears_live_and_pending_owners_before_late_insert():
    MORI_RUNTIME_STATE.reset()
    policy = MoriCPUCachePolicy(4)
    retired, peer = ProgramRef("retired", 0), ProgramRef("peer", 0)
    for owner in (retired, peer):
        MORI_RUNTIME_STATE.bind_request(owner.program_id, owner)
    live, shared_pending, private_pending = [make_offload_key(name, 0) for name in (b"live", b"shared", b"private")]
    policy.on_store_miss((live, shared_pending, private_pending), ReqContext("retired"))
    policy.on_store_miss((live, shared_pending), ReqContext("peer"))
    live_chunk = ChunkStatus(1)
    live_chunk.ref_cnt = 1
    policy.insert(live, live_chunk)

    try:
        MORI_RUNTIME_STATE.on_program_released(retired)
        assert policy.get(live) is live_chunk
        assert live_chunk.ref_cnt == 1
        assert policy._owners[live] == {peer}
        assert policy.index.owners(live) == frozenset({peer})
        assert policy._pending_owners == {shared_pending: {peer}}
        assert policy._pending_program_keys == {peer: {shared_pending}}
        assert policy.evict(1, set()) is None

        # Native completion can arrive after retirement and create the physical
        # chunk, but it must not restore a saved pending owner or a stale request.
        policy.on_request_finished(((live, shared_pending, private_pending),), set(), set(), ReqContext("retired"))
        shared_chunk, private_chunk = ChunkStatus(2), ChunkStatus(3)
        shared_chunk.ref_cnt = private_chunk.ref_cnt = 0
        policy.insert(shared_pending, shared_chunk)
        policy.insert(private_pending, private_chunk)
        assert policy.index.owners(shared_pending) == frozenset({peer})
        assert policy.index.owners(private_pending) == frozenset()
        assert policy._pending_owners == policy._pending_program_keys == {}
        assert not policy.index.entries_for_program(retired)
        assert MORI_RUNTIME_STATE.program_for_request("retired") is None
        assert policy.get(live) is live_chunk
        assert policy.evict(2, set()) == [(shared_pending, shared_chunk), (private_pending, private_chunk)]
        assert policy.get(live) is live_chunk  # native refcount still protects the shared live chunk
    finally:
        policy.close()
        MORI_RUNTIME_STATE.reset()
