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


def test_config_defaults_match_the_two_l20_reference_policy() -> None:
    config = ProgressTTLConfig()

    assert config.target_min_segment_rounds == 9
    assert config.target_max_segment_rounds == 14
    assert config.resume_capacity_ratio == 0.95
    assert config.resume_reclaim_acting_programs is True
    assert config.pause_capacity_ratio == 1.0
    assert config.pause_capacity_lookahead_rounds == 2
    assert config.privileged_lookahead_rounds == 14
    assert config.use_fixed_input_token_growth is False
    assert config.fixed_input_token_growth_per_round == 1024


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_fixed_input_token_growth", 1, "use_fixed_input_token_growth"),
        ("fixed_input_token_growth_per_round", True, "fixed_input_token_growth_per_round"),
        ("fixed_input_token_growth_per_round", -1, "fixed_input_token_growth_per_round"),
    ],
)
def test_config_rejects_invalid_fixed_growth_settings(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProgressTTLConfig(**{field: value})


def test_config_rejects_non_boolean_resume_reclaim_switch() -> None:
    with pytest.raises(ValueError, match="resume_reclaim_acting_programs"):
        ProgressTTLConfig(resume_reclaim_acting_programs=1)


def update(
    stats: ProgressTTLGlobalFactors,
    *,
    prompt_tokens: int = 2000,
    cached_prefix_tokens: int = 0,
    completion_tokens: int = 100,
    total_tokens: int = 2100,
    latency: float = 2.0,
    active_programs: int = 1,
    waiting_programs: int = 0,
    growth: int = 100,
    gap: float = 0.5,
    rounds_since_ttl_pause: int = 0,
) -> None:
    """Append one deterministic request sample."""
    stats.update_request(
        prompt_tokens=prompt_tokens,
        cached_prefix_tokens=cached_prefix_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        request_latency_seconds=latency,
        active_programs=active_programs,
        waiting_programs=waiting_programs,
        input_token_growth=growth,
        inter_request_gap_seconds=gap,
        rounds_since_ttl_pause=rounds_since_ttl_pause,
    )


def test_global_factors_is_policy_owned_without_a_router_compatibility_base() -> None:
    assert ProgressTTLGlobalFactors.__bases__ == (object,)


def test_global_factors_reject_invalid_window_and_initial_numeric_values() -> None:
    with pytest.raises(ValueError, match="request_window_size"):
        ProgressTTLGlobalFactors(request_window_size=0)
    with pytest.raises(ValueError, match="min_update_window_fraction"):
        ProgressTTLGlobalFactors(min_update_window_fraction=0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ProgressTTLGlobalFactors(avg_request_latency_seconds=float("nan"))


def test_stats_wait_for_half_window_before_replacing_cold_start_values() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(49):
        update(stats)

    assert stats.request_window_size == 100
    assert stats.avg_prompt_tokens == 1024.0
    assert stats.avg_request_latency_seconds == 1.0
    assert stats.avg_input_token_growth_per_round == 1024.0

    update(stats)

    assert stats.avg_prompt_tokens == 2000
    assert stats.avg_cached_prefix_tokens == 0
    assert stats.avg_uncached_prompt_tokens == 2000
    assert stats.avg_completion_tokens == 100
    assert stats.avg_total_tokens == 2100
    assert stats.avg_request_latency_seconds == 2.0
    assert stats.avg_active_programs == 1
    assert stats.avg_waiting_programs == 0
    assert stats.avg_input_token_growth_per_round == 100
    assert stats.avg_inter_request_gap_seconds == 0.5


def test_stats_exclude_large_growth_samples_from_capacity_estimate() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(50):
        update(stats, prompt_tokens=11_000, completion_tokens=50, total_tokens=11_050, growth=10_000)

    assert stats.avg_input_token_growth_per_round == 1024.0


def test_stats_ignore_invalid_growth_until_a_valid_sample_arrives() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(50):
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


def test_stats_window_is_fixed_independent_of_active_program_count() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(100):
        update(stats, active_programs=10)
    assert len(stats._request_samples) == 100

    for _ in range(20):
        update(stats, active_programs=1)

    assert stats.request_window_size == 100
    assert len(stats._request_samples) == 100


def test_stats_window_tracks_recent_workload_without_unbounded_sums() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(64):
        update(stats, prompt_tokens=1000, completion_tokens=100, total_tokens=1100, latency=1)
    for _ in range(64):
        update(stats, prompt_tokens=4000, completion_tokens=400, total_tokens=4400, latency=4)

    assert len(stats._request_samples) == 100
    assert stats._prompt_token_sum == 36 * 1000 + 64 * 4000
    assert stats.avg_prompt_tokens == 2920
    assert stats.avg_completion_tokens == 292
    assert stats.avg_request_latency_seconds == 2.92


def test_stats_keep_bounded_cached_and_uncached_prompt_token_averages() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2, min_update_window_fraction=1.0)

    update(stats, prompt_tokens=1000, cached_prefix_tokens=800)
    update(stats, prompt_tokens=600, cached_prefix_tokens=1000)

    assert stats.avg_cached_prefix_tokens == 700
    assert stats.avg_uncached_prompt_tokens == 100
    assert stats._cached_prefix_token_sum == 1400
    assert stats._uncached_prompt_token_sum == 200


def test_pause_window_is_bounded_and_uses_the_same_update_threshold() -> None:
    stats = ProgressTTLGlobalFactors()
    for _ in range(49):
        stats.update_pause(served_rounds=9)
    assert stats.avg_segment_served_rounds_on_pause == 1.0

    stats.update_pause(served_rounds=9)

    assert stats.avg_segment_served_rounds_on_pause == 9
    for _ in range(400):
        stats.update_pause(served_rounds=14)
    assert len(stats._pause_served_rounds) == stats.request_window_size


def test_continuity_mode_keeps_theoretical_samples_while_disabled() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)

    for _ in range(2):
        stats.observe_continuity(interval_seconds=1, cache_miss_impact_seconds=0, assigned_ttl_seconds=1)

    assert stats.continuity_window_complete is True
    assert stats.continuity_enabled is False
    assert stats.continuity_utility_seconds == -2

    for _ in range(2):
        stats.observe_continuity(interval_seconds=0.5, cache_miss_impact_seconds=10, assigned_ttl_seconds=1)

    assert stats.continuity_utility_seconds == 19
    assert stats.update_continuity_mode(enable_threshold_seconds=20, disable_threshold_seconds=5) is False
    stats.observe_continuity(interval_seconds=0.5, cache_miss_impact_seconds=20, assigned_ttl_seconds=1)
    assert stats.update_continuity_mode(enable_threshold_seconds=20, disable_threshold_seconds=5) is True
    assert stats.continuity_enabled is True


def test_continuity_window_recalibration_replaces_warmup_ttls_only_in_accounting() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)

    stats.observe_continuity(interval_seconds=0.2, cache_miss_impact_seconds=0, assigned_ttl_seconds=1)
    stats.observe_continuity(interval_seconds=2.0, cache_miss_impact_seconds=0, assigned_ttl_seconds=1)
    recalibrated = stats.recalibrate_continuity_window(
        minimum_seconds=0.05,
        maximum_seconds=32,
        impact_ratio=1,
    )

    assert recalibrated is not None
    assert [sample.assigned_ttl_seconds for sample in stats._continuity_samples] == [0, 0]
    assert stats.continuity_utility_seconds == 0


def test_fitted_negative_ttl_utility_disables_future_ttl() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)

    stats.observe_continuity(interval_seconds=10, cache_miss_impact_seconds=1, assigned_ttl_seconds=1)
    stats.observe_continuity(interval_seconds=20, cache_miss_impact_seconds=1, assigned_ttl_seconds=1)
    estimate = stats.estimate_ttl(
        impact_seconds=1,
        minimum_seconds=0.05,
        maximum_seconds=32,
        impact_ratio=1,
    )

    assert estimate.uses_fitted_distribution is True
    assert estimate.candidate_utility_seconds is not None
    assert estimate.candidate_utility_seconds < 0
    assert estimate.ttl_seconds == 0


def test_continuity_mode_uses_asymmetric_positive_hysteresis_thresholds() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)
    stats.observe_continuity(interval_seconds=1, cache_miss_impact_seconds=1, assigned_ttl_seconds=1)
    stats.observe_continuity(interval_seconds=2, cache_miss_impact_seconds=1, assigned_ttl_seconds=1)

    stats.continuity_utility_seconds = 20
    assert stats.update_continuity_mode(enable_threshold_seconds=20, disable_threshold_seconds=5) is True
    stats.continuity_utility_seconds = 5
    assert stats.update_continuity_mode(enable_threshold_seconds=20, disable_threshold_seconds=5) is False
    assert stats.continuity_enabled is True
    stats.continuity_utility_seconds = 4.99
    assert stats.update_continuity_mode(enable_threshold_seconds=20, disable_threshold_seconds=5) is True
    assert stats.continuity_enabled is False


def test_continuity_utility_rewards_only_intervals_that_return_before_ttl() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)

    stats.observe_continuity(interval_seconds=1, cache_miss_impact_seconds=10, assigned_ttl_seconds=2)
    stats.observe_continuity(interval_seconds=3, cache_miss_impact_seconds=10, assigned_ttl_seconds=2)

    # interval <= TTL contributes C - interval; interval > TTL contributes -TTL.
    assert stats.continuity_utility_seconds == 7


def test_continuity_ttl_uses_impact_during_warmup_then_lognormal_window() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)

    assert stats.recommended_ttl_seconds(impact_seconds=7, minimum_seconds=1, maximum_seconds=10) == 7
    stats.observe_continuity(interval_seconds=2, cache_miss_impact_seconds=7, assigned_ttl_seconds=7)
    stats.observe_continuity(interval_seconds=4, cache_miss_impact_seconds=7, assigned_ttl_seconds=7)

    recommended = stats.recommended_ttl_seconds(impact_seconds=7, minimum_seconds=1, maximum_seconds=10)
    assert 1 <= recommended <= 10
    assert stats.continuity_log_mu is not None


def test_continuity_ttl_uses_a_non_minimum_value_when_the_lognormal_hit_curve_supports_it() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)

    stats.observe_continuity(interval_seconds=0.2, cache_miss_impact_seconds=6, assigned_ttl_seconds=1)
    stats.observe_continuity(interval_seconds=2.0, cache_miss_impact_seconds=6, assigned_ttl_seconds=1)

    recommended = stats.recommended_ttl_seconds(impact_seconds=6, minimum_seconds=0.05, maximum_seconds=32)

    assert recommended > 0.05
    assert recommended == pytest.approx(17.12317023578921)


def test_protected_min_rounds_uses_ttl_pause_history_only_after_a_full_window() -> None:
    stats = ProgressTTLGlobalFactors(request_window_size=2)

    update(stats, rounds_since_ttl_pause=3)
    assert stats.protected_min_segment_rounds(9) == 9
    update(stats, rounds_since_ttl_pause=3)

    assert stats.avg_rounds_since_ttl_pause == 3
    assert stats.protected_min_segment_rounds(9) == 6
