# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Equivalence tests for the original Progress-TTL bounded rolling statistics."""

import pytest

from agentinfer.scheduling.progress_ttl import ProgressTTLConfig, ProgressTTLGlobalFactors

pytestmark = pytest.mark.cpu_test


def test_config_rejects_invalid_segment_and_capacity_bounds() -> None:
    with pytest.raises(ValueError, match="target_max_segment_rounds"):
        ProgressTTLConfig(target_min_segment_rounds=8, target_max_segment_rounds=7)
    with pytest.raises(ValueError, match="resume_capacity_ratio"):
        ProgressTTLConfig(resume_capacity_ratio=0)
    with pytest.raises(ValueError, match="uncached_ratio_default"):
        ProgressTTLConfig(uncached_ratio_default=1.1)


def test_config_defaults_match_the_two_l20_reference_policy() -> None:
    config = ProgressTTLConfig()

    assert config.target_min_segment_rounds == 9
    assert config.target_max_segment_rounds == 14
    assert config.resume_capacity_ratio == 0.9
    assert config.resume_reclaim_acting_programs is True
    assert config.pause_capacity_ratio == 0.95
    assert config.pause_capacity_lookahead_rounds == 2
    assert config.privileged_lookahead_rounds == 14


def test_config_rejects_non_boolean_resume_reclaim_switch() -> None:
    with pytest.raises(ValueError, match="resume_reclaim_acting_programs"):
        ProgressTTLConfig(resume_reclaim_acting_programs=1)


def update(
    stats: ProgressTTLGlobalFactors,
    *,
    prompt_tokens: int = 2000,
    completion_tokens: int = 100,
    total_tokens: int = 2100,
    latency: float = 2.0,
    active_programs: int = 1,
    waiting_programs: int = 0,
    growth: int = 100,
    gap: float = 0.5,
) -> None:
    """Append one deterministic request sample."""
    stats.update_request(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        request_latency_seconds=latency,
        active_programs=active_programs,
        waiting_programs=waiting_programs,
        input_token_growth=growth,
        inter_request_gap_seconds=gap,
    )


def test_global_factors_is_policy_owned_without_a_router_compatibility_base() -> None:
    assert ProgressTTLGlobalFactors.__bases__ == (object,)


def test_global_factors_reject_invalid_window_and_initial_numeric_values() -> None:
    with pytest.raises(ValueError, match="window sizes"):
        ProgressTTLGlobalFactors(request_window_size=32)
    with pytest.raises(ValueError, match="window growth and eviction"):
        ProgressTTLGlobalFactors(max_evictions_per_update=0)
    with pytest.raises(ValueError, match="min_update_window_fraction"):
        ProgressTTLGlobalFactors(min_update_window_fraction=0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ProgressTTLGlobalFactors(avg_request_latency_seconds=float("nan"))


def test_stats_wait_for_half_window_before_replacing_cold_start_values() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(31):
        update(stats)

    assert stats.request_window_size == 64
    assert stats.avg_prompt_tokens == 1024.0
    assert stats.avg_request_latency_seconds == 1.0
    assert stats.avg_input_token_growth_per_round == 1024.0

    update(stats)

    assert stats.avg_prompt_tokens == 2000
    assert stats.avg_completion_tokens == 100
    assert stats.avg_total_tokens == 2100
    assert stats.avg_request_latency_seconds == 2.0
    assert stats.avg_active_programs == 1
    assert stats.avg_waiting_programs == 0
    assert stats.avg_input_token_growth_per_round == 100
    assert stats.avg_inter_request_gap_seconds == 0.5


def test_stats_cap_large_growth_samples_to_previous_context() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(32):
        update(stats, prompt_tokens=11_000, completion_tokens=50, total_tokens=11_050, growth=10_000)

    assert stats.avg_input_token_growth_per_round == 1050


def test_stats_ignore_invalid_growth_until_a_valid_sample_arrives() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(32):
        update(stats, prompt_tokens=1000, completion_tokens=50, total_tokens=1050, growth=0)

    assert stats.avg_input_token_growth_per_round == 1024.0

    update(stats, prompt_tokens=1100, completion_tokens=50, total_tokens=1150, growth=100)

    assert stats.avg_input_token_growth_per_round == 100


@pytest.mark.parametrize(
    ("latency", "gap"),
    [
        (float("nan"), 0.5),
        (float("inf"), 0.5),
        (2.0, float("nan")),
        (2.0, float("inf")),
    ],
)
def test_stats_reject_non_finite_duration_samples_without_mutation(latency: float, gap: float) -> None:
    stats = ProgressTTLGlobalFactors()

    with pytest.raises(ValueError, match="must be finite"):
        update(stats, latency=latency, gap=gap)

    assert stats.request_count == 0
    assert len(stats._request_samples) == 0


def test_stats_evict_at_most_five_samples_when_window_shrinks() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(100):
        update(stats, active_programs=10)
    assert len(stats._request_samples) == 100

    update(stats, active_programs=1)

    assert stats.request_window_size == 64
    assert len(stats._request_samples) == 96


def test_stats_window_tracks_recent_workload_without_unbounded_sums() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(64):
        update(stats, prompt_tokens=1000, completion_tokens=100, total_tokens=1100, latency=1)
    for _ in range(64):
        update(stats, prompt_tokens=4000, completion_tokens=400, total_tokens=4400, latency=4)

    assert len(stats._request_samples) == 64
    assert stats._prompt_token_sum == 64 * 4000
    assert stats.avg_prompt_tokens == 4000
    assert stats.avg_completion_tokens == 400
    assert stats.avg_request_latency_seconds == 4


def test_pause_window_is_bounded_and_uses_the_same_update_threshold() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(31):
        stats.update_pause(served_rounds=9)
    assert stats.avg_segment_served_rounds_on_pause == 1.0

    stats.update_pause(served_rounds=9)

    assert stats.avg_segment_served_rounds_on_pause == 9
    for _ in range(400):
        stats.update_pause(served_rounds=14)
    assert len(stats._pause_served_rounds) == stats.max_request_window_size
