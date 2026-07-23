# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Tests for host-neutral identity, backend, and Program fact contracts."""

import pytest

from agentinfer.scheduling import BackendInfo, BackendPoolInfo, DpRankInfo, ProgramRef, ProgramTokenObservation

pytestmark = pytest.mark.cpu_test


def test_program_ref_and_token_observation_validate_counts() -> None:
    with pytest.raises(ValueError, match="program_id"):
        ProgramRef("", 0)
    with pytest.raises(ValueError, match="generation"):
        ProgramRef("program-a", -1)
    with pytest.raises(ValueError, match="non-negative"):
        ProgramTokenObservation(estimated_context_tokens=-1)


def test_missing_physical_token_observation_remains_unknown() -> None:
    observation = ProgramTokenObservation(estimated_context_tokens=256)

    assert observation.actual_resident_tokens is None
    assert observation.actual_allocated_blocks is None


def test_token_observation_requires_positive_supplied_block_size() -> None:
    with pytest.raises(ValueError, match="block_size_tokens must be positive"):
        ProgramTokenObservation(estimated_context_tokens=256, actual_allocated_blocks=1, block_size_tokens=0)

    observation = ProgramTokenObservation(
        estimated_context_tokens=256,
        actual_allocated_blocks=0,
        block_size_tokens=16,
    )

    assert observation.actual_allocated_blocks == 0
    assert observation.block_size_tokens == 16


def test_capacity_is_rank_local_and_remote_capacity_is_pool_scoped() -> None:
    backend = BackendInfo(
        backend_id="backend-a",
        backend_url="embedded://vllm",
        healthy=True,
        dp_ranks=(
            DpRankInfo(0, True, True, total_hbm_kv_tokens=2048, expected_reasoning_agent_nums=4),
            DpRankInfo(1, True, True, total_hbm_kv_tokens=2048, expected_reasoning_agent_nums=4),
        ),
    )
    pool = BackendPoolInfo((backend,), total_remote_store_kv_tokens=8192)

    assert pool.backends[0].dp_ranks[1].total_hbm_kv_tokens == 2048
    assert pool.total_remote_store_kv_tokens == 8192


def test_backend_observations_detach_from_mutable_rank_and_backend_lists() -> None:
    ranks = [DpRankInfo(0, True, True, total_hbm_kv_tokens=2048)]
    backend = BackendInfo("backend-a", "embedded://vllm", True, dp_ranks=ranks)
    backends = [backend]
    pool = BackendPoolInfo(backends)

    ranks.append(DpRankInfo(1, True, True, total_hbm_kv_tokens=2048))
    backends.clear()

    assert backend.dp_ranks == (ranks[0],)
    assert pool.backends == (backend,)


def test_backend_rejects_duplicate_ranks() -> None:
    with pytest.raises(ValueError, match="DP ranks"):
        BackendInfo(
            backend_id="backend-a",
            backend_url="embedded://vllm",
            healthy=True,
            dp_ranks=(DpRankInfo(1, True, True), DpRankInfo(1, True, True)),
        )


def test_dp_rank_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError, match="DP-rank capacities"):
        DpRankInfo(0, True, True, total_hbm_kv_tokens=-1)
