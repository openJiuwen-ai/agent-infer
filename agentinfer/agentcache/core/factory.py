# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Explicit controller factories referenced by vLLM ``additional_config``."""

from __future__ import annotations

import math

from vllm.v1.request import Request

from agentinfer.scheduling.backend import BackendPoolInfo
from agentinfer.scheduling.identity import JsonMapping
from agentinfer.scheduling.observability import SchedulerObservabilityConfig
from agentinfer.scheduling.progress_ttl import (
    ProgressTTLConfig,
    ProgressTTLGlobalFactors,
    ProgressTTLMode,
    ProgressTTLProgramFactors,
    ProgressTTLResumeOrder,
    build_progress_ttl_strategy,
)
from agentinfer.scheduling.runtime import ProgramScheduler

_FIXED_PROGRESS_TTL_SETTINGS = frozenset({"decode_buffer_tokens"})

_CONFIGURABLE_PROGRESS_TTL_SETTINGS = frozenset(
    {
        "mode",
        "target_max_segment_rounds",
        "ttl_min_seconds",
        "ttl_max_seconds",
        "ttl_max_cache_miss_impact_ratio",
        "auto_enable_utility_seconds",
        "auto_disable_utility_seconds",
        "ttl_prefill_model_intercept_seconds",
        "ttl_prefill_model_linear_seconds_per_1k_tokens",
        "ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared",
        "ttl_decode_throughput_alpha",
        "shared_prefix_freshness_warmup_seconds",
        "shared_prefix_freshness_kv_turnovers",
        "resume_capacity_ratio",
        "resume_order",
        "resume_reclaim_acting_programs",
        "pause_capacity_ratio",
        "privileged_max_context_tokens",
        "privileged_ttl_seconds",
        "use_fixed_input_token_growth",
        "fixed_input_token_growth_per_round",
        "enable_batch_gain_admission",
        "decode_step_fixed_seconds",
        "decode_step_seconds_per_request",
        "decode_step_seconds_per_context_token",
        "force_resume_timeout_scale",
        "force_resume_timeout_min_seconds",
        "force_resume_timeout_max_seconds",
        "paused_program_ttl_seconds",
    }
)

_REMOVED_PROGRESS_TTL_SETTINGS = frozenset(
    {
        "ttl_impact_multiplier",
        "uncached_ratio_default",
        "target_min_segment_rounds",
        "privileged_lookahead_rounds",
        "ttl_prefill_seconds_per_1k_uncached_tokens",
        "resume_fairness_weight",
        "resume_resource_penalty_weight",
        "capacity_safety_margin_tokens",
        "pause_capacity_lookahead_rounds",
        "force_resume_timeout_seconds",
    }
)


def build_progress_ttl_controller(
    backend_pool_info: BackendPoolInfo,
    settings: JsonMapping,
) -> ProgramScheduler[Request, ProgressTTLGlobalFactors, ProgressTTLProgramFactors]:
    """Build an explicitly configured embedded Progress-TTL Scheduler.

    ``settings`` is the read-only ``additional_config.agentcache`` object supplied by the bridge. Only the nested
    ``progress_ttl`` mapping is consumed here, keeping policy fields out of the vLLM lifecycle Adapter.
    """
    raw = settings.get("progress_ttl", {})
    if not isinstance(raw, dict):
        raise ValueError("additional_config.agentcache.progress_ttl must be an object")
    fixed_settings = sorted(_FIXED_PROGRESS_TTL_SETTINGS.intersection(raw))
    if fixed_settings:
        joined = ", ".join(fixed_settings)
        raise ValueError(f"progress_ttl settings are fixed implementation values and cannot be configured: {joined}")
    removed_settings = sorted(_REMOVED_PROGRESS_TTL_SETTINGS.intersection(raw))
    if removed_settings:
        joined = ", ".join(removed_settings)
        raise ValueError(f"progress_ttl settings were removed and must not be configured: {joined}")
    unknown_settings = sorted(set(raw).difference(_CONFIGURABLE_PROGRESS_TTL_SETTINGS))
    if unknown_settings:
        joined = ", ".join(unknown_settings)
        raise ValueError(f"progress_ttl settings are not supported: {joined}")
    defaults = ProgressTTLConfig()
    config = ProgressTTLConfig(
        mode=_mode_setting(raw, "mode", defaults.mode),
        target_max_segment_rounds=_int_setting(raw, "target_max_segment_rounds", defaults.target_max_segment_rounds),
        ttl_min_seconds=_float_setting(raw, "ttl_min_seconds", defaults.ttl_min_seconds),
        ttl_max_seconds=_float_setting(raw, "ttl_max_seconds", defaults.ttl_max_seconds),
        ttl_max_cache_miss_impact_ratio=_float_setting(
            raw,
            "ttl_max_cache_miss_impact_ratio",
            defaults.ttl_max_cache_miss_impact_ratio,
        ),
        auto_enable_utility_seconds=_float_setting(
            raw,
            "auto_enable_utility_seconds",
            defaults.auto_enable_utility_seconds,
        ),
        auto_disable_utility_seconds=_float_setting(
            raw,
            "auto_disable_utility_seconds",
            defaults.auto_disable_utility_seconds,
        ),
        ttl_prefill_model_intercept_seconds=_float_setting(
            raw,
            "ttl_prefill_model_intercept_seconds",
            defaults.ttl_prefill_model_intercept_seconds,
        ),
        ttl_prefill_model_linear_seconds_per_1k_tokens=_float_setting(
            raw,
            "ttl_prefill_model_linear_seconds_per_1k_tokens",
            defaults.ttl_prefill_model_linear_seconds_per_1k_tokens,
        ),
        ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared=_float_setting(
            raw,
            "ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared",
            defaults.ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared,
        ),
        ttl_decode_throughput_alpha=_float_setting(
            raw,
            "ttl_decode_throughput_alpha",
            defaults.ttl_decode_throughput_alpha,
        ),
        shared_prefix_freshness_warmup_seconds=_float_setting(
            raw,
            "shared_prefix_freshness_warmup_seconds",
            defaults.shared_prefix_freshness_warmup_seconds,
        ),
        shared_prefix_freshness_kv_turnovers=_float_setting(
            raw,
            "shared_prefix_freshness_kv_turnovers",
            defaults.shared_prefix_freshness_kv_turnovers,
        ),
        resume_capacity_ratio=_float_setting(
            raw,
            "resume_capacity_ratio",
            defaults.resume_capacity_ratio,
        ),
        resume_order=_resume_order_setting(raw, "resume_order", defaults.resume_order),
        resume_reclaim_acting_programs=_bool_setting(
            raw,
            "resume_reclaim_acting_programs",
            defaults.resume_reclaim_acting_programs,
        ),
        pause_capacity_ratio=_float_setting(
            raw,
            "pause_capacity_ratio",
            defaults.pause_capacity_ratio,
        ),
        privileged_max_context_tokens=_int_setting(
            raw,
            "privileged_max_context_tokens",
            defaults.privileged_max_context_tokens,
        ),
        privileged_ttl_seconds=_float_setting(
            raw,
            "privileged_ttl_seconds",
            defaults.privileged_ttl_seconds,
        ),
        use_fixed_input_token_growth=_bool_setting(
            raw,
            "use_fixed_input_token_growth",
            defaults.use_fixed_input_token_growth,
        ),
        fixed_input_token_growth_per_round=_int_setting(
            raw,
            "fixed_input_token_growth_per_round",
            defaults.fixed_input_token_growth_per_round,
        ),
        enable_batch_gain_admission=_bool_setting(
            raw,
            "enable_batch_gain_admission",
            defaults.enable_batch_gain_admission,
        ),
        decode_step_fixed_seconds=_float_setting(
            raw,
            "decode_step_fixed_seconds",
            defaults.decode_step_fixed_seconds,
        ),
        decode_step_seconds_per_request=_float_setting(
            raw,
            "decode_step_seconds_per_request",
            defaults.decode_step_seconds_per_request,
        ),
        decode_step_seconds_per_context_token=_float_setting(
            raw,
            "decode_step_seconds_per_context_token",
            defaults.decode_step_seconds_per_context_token,
        ),
        force_resume_timeout_scale=_float_setting(
            raw,
            "force_resume_timeout_scale",
            defaults.force_resume_timeout_scale,
        ),
        force_resume_timeout_min_seconds=_float_setting(
            raw,
            "force_resume_timeout_min_seconds",
            defaults.force_resume_timeout_min_seconds,
        ),
        force_resume_timeout_max_seconds=_float_setting(
            raw,
            "force_resume_timeout_max_seconds",
            defaults.force_resume_timeout_max_seconds,
        ),
        paused_program_ttl_seconds=_float_setting(
            raw,
            "paused_program_ttl_seconds",
            defaults.paused_program_ttl_seconds,
        ),
    )
    components = build_progress_ttl_strategy(config)
    observability_raw = settings.get("observability", {})
    if not isinstance(observability_raw, dict):
        raise ValueError("additional_config.agentcache.observability must be an object")
    observability = SchedulerObservabilityConfig(
        enabled=_bool_setting(
            observability_raw,
            "enabled",
            False,
            scope="agentcache.observability",
        ),
        log_interval_seconds=_float_setting(
            observability_raw,
            "log_interval_seconds",
            5.0,
            scope="agentcache.observability",
        ),
    )
    schedule_interval_seconds = _float_setting(
        settings,
        "schedule_interval_seconds",
        1.0,
        scope="agentcache",
    )
    return ProgramScheduler(
        components.strategy,
        components.initial_factors,
        backend_pool_info,
        schedule_interval_seconds=schedule_interval_seconds,
        observability=observability,
    )


def _int_setting(settings: JsonMapping, name: str, default: int) -> int:
    """Read an exact integer policy value without accepting booleans."""
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"progress_ttl.{name} must be an integer")
    return value


def _bool_setting(
    settings: JsonMapping,
    name: str,
    default: bool,
    *,
    scope: str = "progress_ttl",
) -> bool:
    """Read an exact boolean value from one configuration scope."""
    value = settings.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{scope}.{name} must be a boolean")
    return value


def _mode_setting(settings: JsonMapping, name: str, default: ProgressTTLMode) -> ProgressTTLMode:
    """Read one explicit Progress-TTL control mode."""
    value = settings.get(name, default.value)
    if not isinstance(value, str):
        raise ValueError(f"progress_ttl.{name} must be one of: on, off, auto")
    try:
        return ProgressTTLMode(value)
    except ValueError as exc:
        raise ValueError(f"progress_ttl.{name} must be one of: on, off, auto") from exc


def _resume_order_setting(
    settings: JsonMapping,
    name: str,
    default: ProgressTTLResumeOrder,
) -> ProgressTTLResumeOrder:
    """Read the ordering used for ordinary resume candidates."""
    value = settings.get(name, default.value)
    if not isinstance(value, str):
        raise ValueError(f"progress_ttl.{name} must be one of: mru, fcfs")
    try:
        return ProgressTTLResumeOrder(value)
    except ValueError as exc:
        raise ValueError(f"progress_ttl.{name} must be one of: mru, fcfs") from exc


def _float_setting(
    settings: JsonMapping,
    name: str,
    default: float,
    *,
    scope: str = "progress_ttl",
) -> float:
    """Read a numeric policy value without accepting booleans."""
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{scope}.{name} must be numeric")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{scope}.{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{scope}.{name} must be finite")
    return normalized
