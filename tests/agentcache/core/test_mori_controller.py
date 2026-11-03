# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Controller selection and strict MORI configuration tests."""

from __future__ import annotations

import pytest

from agentinfer.agentcache.core.controller import build_controller, build_mori_controller
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.mori import MORIConfig, MORIMode, MORIStrategy

pytestmark = pytest.mark.cpu_test

BOOLEAN_SETTINGS = [
    ("enable_wait_credit", True),
    ("fast_capacity_accounting", False),
    ("audit_writeback_sources", False),
    ("repair_writeback_endpoints", False),
    ("history_gated_gpu_service", False),
]


def backend(*, dram: int | None = 2000, chunk: int | None = 16) -> BackendPoolInfo:
    return BackendPoolInfo(
        (
            BackendInfo(
                "vllm-local",
                "embedded://vllm",
                True,
                (
                    DpRankInfo(
                        0,
                        True,
                        True,
                        total_hbm_kv_tokens=1000,
                        total_dram_kv_tokens=dram,
                        hbm_block_size_tokens=16,
                        dram_chunk_size_tokens=chunk,
                    ),
                ),
            ),
        )
    )


def test_controller_selects_mori_without_a_dynamic_factory() -> None:
    controller = build_controller(backend(), {"strategy": "mori", "mori": {"window_size": 5}})

    assert isinstance(controller.strategy, MORIStrategy)
    assert controller.strategy.config.mode is MORIMode.FULL
    assert controller.strategy.config.enable_wait_credit is True
    assert controller.strategy.config.decode_buffer_tokens == 100
    assert controller._schedule_interval_seconds == 5.0


def test_full_mori_fails_closed_without_dram_capacity() -> None:
    with pytest.raises(ValueError, match="DRAM quota"):
        build_mori_controller(backend(dram=None, chunk=None), {"mori": {}})


def test_removed_placement_only_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mori.mode must be: full"):
        build_mori_controller(backend(), {"mori": {"mode": "placement_only"}})
    assert build_mori_controller(backend(), {"mori": {"mode": "full"}}).strategy.config.mode is MORIMode.FULL


def test_decode_buffer_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="fixed implementation"):
        build_mori_controller(backend(), {"mori": {"decode_buffer_tokens": 999}})


@pytest.mark.parametrize(("setting", "default"), BOOLEAN_SETTINGS)
def test_boolean_configuration_defaults_and_overrides(setting, default):
    options = {"gpu_service_requests": 4} if setting == "history_gated_gpu_service" else {}
    assert getattr(MORIConfig(**options), setting) is default
    assert getattr(build_mori_controller(backend(), {"mori": options}).strategy.config, setting) is default
    for value in (False, True):
        configured = {**options, setting: value}
        assert getattr(MORIConfig(**configured), setting) is value
        assert getattr(build_mori_controller(backend(), {"mori": configured}).strategy.config, setting) is value


@pytest.mark.parametrize("setting", [name for name, _ in BOOLEAN_SETTINGS])
@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_boolean_configuration_requires_exact_boolean(setting, value):
    options = {"gpu_service_requests": 4} if setting == "history_gated_gpu_service" else {}
    configured = {**options, setting: value}
    with pytest.raises(ValueError, match=f"{setting} must be a boolean"):
        MORIConfig(**configured)
    with pytest.raises(ValueError, match=f"{setting} must be a boolean"):
        build_mori_controller(backend(), {"mori": configured})


def test_ready_wait_reservation_is_explicit_and_defaults_off() -> None:
    assert build_mori_controller(backend(), {}).strategy.config.ready_wait_limit_seconds == 0
    controller = build_mori_controller(backend(), {"mori": {"ready_wait_limit_seconds": 30.0}})
    assert controller.strategy.config.ready_wait_limit_seconds == 30.0


def test_gpu_service_request_budget_defaults_off_and_is_explicit():
    assert build_mori_controller(backend(), {}).strategy.config.gpu_service_requests == 0
    assert (
        build_mori_controller(backend(), {"mori": {"gpu_service_requests": 4}}).strategy.config.gpu_service_requests
        == 4
    )


@pytest.mark.parametrize("value", [-1, 1.5, True, "4"])
def test_gpu_service_request_budget_requires_non_negative_integer(value):
    with pytest.raises(ValueError):
        MORIConfig(gpu_service_requests=value)
    with pytest.raises(ValueError):
        build_mori_controller(backend(), {"mori": {"gpu_service_requests": value}})


def test_history_gated_service_requires_service_budget():
    with pytest.raises(ValueError, match="requires full mode and a positive service budget"):
        MORIConfig(history_gated_gpu_service=True)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_invalid_ready_wait_limit_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        build_mori_controller(backend(), {"mori": {"ready_wait_limit_seconds": value}})
    with pytest.raises(ValueError):
        MORIConfig(ready_wait_limit_seconds=value)


def test_slow_request_diagnostics_default_and_explicit_settings() -> None:
    default = build_mori_controller(backend(), {})
    tracker = default.strategy.request_diagnostics()
    assert tracker is not None
    assert tracker.threshold_seconds == 30.0
    assert tracker.repeat_seconds == 60.0
    configured = build_mori_controller(
        backend(), {"mori": {"slow_request_threshold_seconds": 15.0, "slow_request_repeat_seconds": 90.0}}
    )
    tracker = configured.strategy.request_diagnostics()
    assert tracker is not None
    assert tracker.threshold_seconds == 15.0
    assert tracker.repeat_seconds == 90.0
    disabled = build_mori_controller(backend(), {"mori": {"slow_request_threshold_seconds": 0}})
    assert disabled.strategy.request_diagnostics() is None


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
@pytest.mark.parametrize("setting", ["slow_request_threshold_seconds", "slow_request_repeat_seconds"])
def test_invalid_slow_request_settings_fail_early(setting: str, value: float) -> None:
    with pytest.raises(ValueError):
        build_mori_controller(backend(), {"mori": {setting: value}})
    with pytest.raises(ValueError):
        MORIConfig(**{setting: value})


def test_zero_repeat_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="slow_request_repeat_seconds"):
        build_mori_controller(backend(), {"mori": {"slow_request_repeat_seconds": 0}})


@pytest.mark.parametrize(
    "setting", ["starvation_timeout_seconds", "min_residency_seconds", "idleness_margin", "max_wait_seconds"]
)
def test_controller_rejects_removed_policy_extensions(setting) -> None:
    with pytest.raises(ValueError, match=f"not supported: {setting}"):
        build_mori_controller(backend(), {"mori": {setting: 60.0}})
