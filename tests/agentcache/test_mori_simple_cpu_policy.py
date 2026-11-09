# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""SimpleCPUOffloadConnector typed victim-provider tests."""

from __future__ import annotations

import pytest

from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
from agentinfer.scheduling import CacheClass, ProgramRef
from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy

pytestmark = pytest.mark.cpu_test


def test_simple_cpu_provider_uses_cpu_tier_order_and_native_lru() -> None:
    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    policy = MORISimpleCPUOffloadPolicy(state)
    programs = {
        MORITier.WAITING: ProgramRef("waiting", 0),
        MORITier.GPU: ProgramRef("gpu", 0),
        MORITier.CPU: ProgramRef("cpu", 0),
    }

    for block_id, (tier, program) in enumerate(programs.items(), 1):
        request_id = f"request-{tier.value}"
        state.bind_request(request_id, program)
        components.strategy._set_factor(
            components.initial_factors,
            program,
            MORIProgramFactors(tier=tier),
        )
        policy.on_block_owner_added(block_id, request_id)
        policy.on_block_ready(block_id, block_id + 10)

    assert policy.propose(3) == (1, 2, 3)
    assert policy.index.cache_class(1) is CacheClass.INACTIVE
    assert policy.index.cache_class(2) is CacheClass.BUSY
    assert policy.index.cache_class(3) is CacheClass.IDLE

    policy.close()
    state.unbind_provider(components.strategy)


def test_simple_cpu_provider_tracks_shared_owners_touch_and_abandon() -> None:
    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    policy = MORISimpleCPUOffloadPolicy(state)
    busy = ProgramRef("busy", 0)
    idle = ProgramRef("idle", 0)
    state.bind_request("busy-request", busy)
    state.bind_request("idle-request", idle)
    components.strategy._set_factor(
        components.initial_factors,
        busy,
        MORIProgramFactors(tier=MORITier.GPU),
    )
    components.strategy._set_factor(
        components.initial_factors,
        idle,
        MORIProgramFactors(tier=MORITier.CPU),
    )

    policy.on_block_owner_added(8, "busy-request")
    policy.on_block_owner_added(8, "idle-request")
    policy.on_block_ready(8, 20)
    assert policy.index.owners(8) == frozenset((busy, idle))
    assert policy.index.cache_class(8) is CacheClass.IDLE

    policy.on_block_evictability_changed(8, False)
    assert policy.propose(1) is None
    policy.on_block_ready(8, 30)
    assert policy.propose(1) == (8,)

    policy.on_block_removed(8)
    assert policy.propose(1) is None
    policy.close()
    state.unbind_provider(components.strategy)


@pytest.mark.parametrize(
    "incoming,allowed", [(MORITier.GPU, (1, 2)), (MORITier.CPU, (1, 2, 3)), (MORITier.WAITING, (1,))]
)
def test_store_admission_cannot_replace_higher_cpu_protection(incoming, allowed):
    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    policy = MORISimpleCPUOffloadPolicy(state)
    for block_id, tier in enumerate((MORITier.WAITING, MORITier.GPU, MORITier.CPU), 1):
        owner = ProgramRef(tier.value, 0)
        state.bind_request(tier.value, owner)
        components.strategy._set_factor(components.initial_factors, owner, MORIProgramFactors(tier=tier))
        policy.on_block_owner_added(block_id, tier.value)
        policy.on_block_ready(block_id, block_id)
    assert policy.store_capacity(incoming.value) == len(allowed)
    assert policy.propose_for_store(len(allowed), incoming.value) == allowed
    assert policy.propose_for_store(len(allowed) + 1, incoming.value) is None
    assert policy.propose(3) == (1, 2, 3)  # rejection did not remove entries
    assert policy.store_capacity("unknown-request") == 1
    # The shared block must retain the stronger CPU protection.
    policy.on_block_owner_added(2, "cpu")
    assert policy.store_capacity("gpu") == 1
    policy.on_block_evictability_changed(1, False)
    assert policy.store_capacity("gpu") == 0
    policy.close()
    state.unbind_provider(components.strategy)
