# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Tests for explicit vLLM controller factory policy settings."""

import pytest

from agentinfer.agentcache.core.factory import build_progress_ttl_controller
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.progress_ttl import ProgressTTLMode

pytestmark = pytest.mark.cpu_test


def backend() -> BackendPoolInfo:
    """Build the single embedded backend required by the current runtime."""
    return BackendPoolInfo(
        (
            BackendInfo(
                "vllm-local",
                "embedded://vllm",
                True,
                (DpRankInfo(0, True, True, total_hbm_kv_tokens=1000),),
            ),
        )
    )


def test_factory_applies_nested_progress_ttl_settings() -> None:
    controller = build_progress_ttl_controller(
        backend(),
        {
            "progress_ttl": {
                "target_min_segment_rounds": 9,
                "target_max_segment_rounds": 14,
                "ttl_min_seconds": 5,
                "ttl_max_cache_miss_impact_ratio": 0.75,
                "auto_enable_utility_seconds": 30,
                "auto_disable_utility_seconds": 8,
                "resume_capacity_ratio": 0.9,
                "resume_reclaim_acting_programs": False,
                "pause_capacity_ratio": 0.95,
                "pause_capacity_lookahead_rounds": 2,
                "privileged_lookahead_rounds": 14,
                "privileged_max_context_tokens": 262144,
                "use_fixed_input_token_growth": True,
                "fixed_input_token_growth_per_round": 768,
                "paused_program_ttl_seconds": 600,
            }
        },
    )

    assert controller.strategy.config.target_min_segment_rounds == 9
    assert controller.strategy.config.target_max_segment_rounds == 14
    assert controller.strategy.config.ttl_min_seconds == 5
    assert controller.strategy.config.ttl_max_cache_miss_impact_ratio == 0.75
    assert controller.strategy.config.auto_enable_utility_seconds == 30
    assert controller.strategy.config.auto_disable_utility_seconds == 8
    assert controller.strategy.config.resume_capacity_ratio == 0.9
    assert controller.strategy.config.resume_reclaim_acting_programs is False
    assert controller.strategy.config.pause_capacity_ratio == 0.95
    assert controller.strategy.config.pause_capacity_lookahead_rounds == 2
    assert controller.strategy.config.privileged_lookahead_rounds == 14
    assert controller.strategy.config.privileged_max_context_tokens == 262144
    assert controller.strategy.config.use_fixed_input_token_growth is True
    assert controller.strategy.config.fixed_input_token_growth_per_round == 768
    assert controller.strategy.config.paused_program_ttl_seconds == 600


def test_factory_uses_two_l20_reference_defaults_without_policy_overrides() -> None:
    controller = build_progress_ttl_controller(backend(), {})

    assert controller.strategy.config.target_min_segment_rounds == 9
    assert controller.strategy.config.target_max_segment_rounds == 14
    assert controller.strategy.config.resume_capacity_ratio == 0.95
    assert controller.strategy.config.pause_capacity_ratio == 1.0
    assert controller.strategy.config.pause_capacity_lookahead_rounds == 2
    assert controller.strategy.config.privileged_lookahead_rounds == 14
    assert controller.strategy.config.mode is ProgressTTLMode.ON
    assert controller.strategy.config.ttl_max_cache_miss_impact_ratio == 1.0
    assert controller.strategy.config.auto_enable_utility_seconds == 20.0
    assert controller.strategy.config.auto_disable_utility_seconds == 5.0
    assert controller._observability.enabled is False
    assert controller._observability.log_interval_seconds == 5


def test_factory_accepts_explicit_progress_ttl_mode() -> None:
    controller = build_progress_ttl_controller(backend(), {"progress_ttl": {"mode": "auto"}})

    assert controller.strategy.config.mode is ProgressTTLMode.AUTO


def test_factory_applies_observability_controls() -> None:
    controller = build_progress_ttl_controller(
        backend(),
        {"observability": {"enabled": True, "log_interval_seconds": 15}},
    )

    assert controller._observability.enabled is True
    assert controller._observability.log_interval_seconds == 15


def test_factory_rejects_non_mapping_policy_settings() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        build_progress_ttl_controller(backend(), {"progress_ttl": "invalid"})


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"observability": "invalid"}, "observability must be an object"),
        ({"observability": {"enabled": 1}}, "observability.enabled must be a boolean"),
        ({"observability": {"log_interval_seconds": 0}}, "log_interval_seconds must be finite and positive"),
    ],
)
def test_factory_rejects_invalid_observability_controls(settings: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_progress_ttl_controller(backend(), settings)


@pytest.mark.parametrize(
    "setting",
    (
        "resume_reclaim_acting_programs",
        "use_fixed_input_token_growth",
    ),
)
def test_factory_rejects_non_boolean_switches(setting: str) -> None:
    with pytest.raises(ValueError, match=f"{setting} must be a boolean"):
        build_progress_ttl_controller(backend(), {"progress_ttl": {setting: 1}})


@pytest.mark.parametrize(
    "setting",
    (
        "resume_fairness_weight",
        "resume_resource_penalty_weight",
        "capacity_safety_margin_tokens",
        "decode_buffer_tokens",
    ),
)
def test_factory_rejects_fixed_policy_settings(setting: str) -> None:
    with pytest.raises(ValueError, match="fixed implementation values"):
        build_progress_ttl_controller(backend(), {"progress_ttl": {setting: 1}})


@pytest.mark.parametrize("setting", ("ttl_impact_multiplier", "uncached_ratio_default"))
def test_factory_rejects_removed_policy_settings(setting: str) -> None:
    with pytest.raises(ValueError, match="were removed"):
        build_progress_ttl_controller(backend(), {"progress_ttl": {setting: 1}})


@pytest.mark.parametrize(
    ("settings", "message"),
    (
        ({"ttl_min_second": 1}, "ttl_min_second"),
        ({"zzz_unknown": 1, "aaa_unknown": 2}, "aaa_unknown, zzz_unknown"),
    ),
)
def test_factory_rejects_unknown_policy_settings(settings: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_progress_ttl_controller(backend(), {"progress_ttl": settings})


@pytest.mark.parametrize(
    "invalid_value",
    [pytest.param(float("inf"), id="infinity"), pytest.param(10**10000, id="overflowing-integer")],
)
def test_factory_rejects_non_finite_numeric_settings(invalid_value: int | float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_progress_ttl_controller(backend(), {"progress_ttl": {"ttl_min_seconds": invalid_value}})
