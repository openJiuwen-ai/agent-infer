# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Strict configuration for Trace Replay planning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReplayStrictModel(BaseModel):
    """Reject unknown fields and non-finite numeric values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReplayExperimentConfig(ReplayStrictModel):
    """Control sampled tasks, concurrency, and the result directory."""

    task_num: int | None = Field(None, ge=1, json_schema_extra={"cli": True})
    result_dir: Path = Field(Path("replay-results"), json_schema_extra={"cli": True})
    max_concurrency: int = Field(1, ge=1, json_schema_extra={"cli": True})
    task_timeout_seconds: int = Field(3600, ge=1)
    run_timeout_seconds: int | None = Field(None, ge=1)


class ReplayBackendConfig(ReplayStrictModel):
    """Describe the Replay request and tokenizer endpoints."""

    type: Literal["vllm"] = "vllm"
    base_url: str = Field("http://127.0.0.1:8000", json_schema_extra={"cli": True})
    metrics_url: str | None = Field(None, json_schema_extra={"cli": {"flag": "--metrics-url"}})
    tokenizer_base_url: str | None = None
    model: str = Field(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        json_schema_extra={"cli": True},
    )
    endpoint: Literal["/v1/chat/completions", "/v1/messages"] = Field(
        "/v1/chat/completions",
        json_schema_extra={"cli": True},
    )
    api_key_env: str | None = None

    @property
    def resolved_tokenizer_base_url(self) -> str:
        """Return the explicit tokenizer URL or the inference base URL."""

        return self.tokenizer_base_url or self.base_url

    @property
    def effective_metrics_url(self) -> str:
        """Return the complete Prometheus endpoint used for vLLM evidence."""

        return self.metrics_url or f"{self.base_url.rstrip('/')}/metrics"


class ReplayIntervalLognormalConfig(ReplayStrictModel):
    """User-selected quantile anchors for the Lognormal interval distribution."""

    p50_seconds: float = Field(gt=0)
    p95_seconds: float = Field(gt=0)
    p99_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> ReplayIntervalLognormalConfig:
        if not (self.p50_seconds < self.p95_seconds <= self.p99_seconds):
            raise ValueError("interval anchors must satisfy p50 < p95 <= p99")
        return self


class ReplayConfig(ReplayStrictModel):
    """Configure source analysis, sampling, timing, and synthetic prefixes."""

    trace_type: Literal["agentinfer", "inferact_codex_swebenchpro", "agentX", "tracelab"] = Field(
        "agentinfer",
        json_schema_extra={"cli": True},
    )
    trace_path: Path = Field(json_schema_extra={"cli": True})
    trace_same_agent_gap_scale: float = Field(
        1.0,
        ge=0,
        description="Scale historical completion-to-next-start gaps between requests from the same Agent.",
    )
    trace_same_agent_gap_offset_seconds: float = Field(
        0.0,
        description="Signed offset applied after scaling a same-Agent Trace request gap.",
    )
    interval_mode: Literal["trace", "lognormal"] = Field("trace", json_schema_extra={"cli": True})
    interval_lognormal: ReplayIntervalLognormalConfig | None = None
    sample_seed: int = 0
    prompt_shape: Literal["claude_code_minimal_v1", "trace_record"] = Field(
        "claude_code_minimal_v1",
        json_schema_extra={"cli": True},
    )
    lead_title_sys_shared_prefix: int = Field(0, ge=0)
    lead_name_sys_shared_prefix: int = Field(0, ge=0)
    lead_1st_sys_shared_prefix: int = Field(0, ge=0)
    lead_1st_sys_session_prefix: int = Field(0, ge=0)
    lead_1st_tool_shared_prefix: int = Field(0, ge=0)
    lead_1st_msg_shared_prefix: int = Field(0, ge=0)
    lead_1st_trailing_system_prefix: int = Field(0, ge=0)
    subagent_tool_1st_shared_prefix: int = Field(0, ge=0)
    subagent_1st_sys_shared_prefix: int = Field(0, ge=0)
    subagent_1st_sys_session_prefix: int = Field(0, ge=0)
    subagent_1st_msg_shared_prefix: int = Field(0, ge=0)
    lead_continuation_extra_system_ratio: float = Field(0, ge=0, le=1)
    subagent_continuation_extra_system_ratio: float = Field(0, ge=0, le=1)
    lead_continuation_system_tokens: int = Field(0, ge=0)
    subagent_continuation_system_tokens: int = Field(0, ge=0)
    max_input_tokens: int | None = Field(None, ge=1)
    max_output_tokens: int | None = Field(None, ge=1)
    context_adjustment_mode: Literal["strict", "adaptive"] = "strict"
    context_micro_trim_max_tokens: int = Field(64, ge=0)
    context_micro_trim_max_ratio: float = Field(0.005, ge=0, le=1)
    prompt_calibration_tolerance_tokens: int = Field(
        1,
        ge=0,
        le=8,
        description="Accept a small audited residual after exact repair; capped at 8 to expose larger drift.",
    )
    request_timeout_seconds: int = Field(3600, ge=1)

    @model_validator(mode="after")
    def validate_replay_modes(self) -> ReplayConfig:
        if self.interval_mode == "lognormal":
            if self.interval_lognormal is None:
                raise ValueError("interval_mode=lognormal requires interval_lognormal anchors")
        elif self.interval_lognormal is not None:
            raise ValueError("interval_lognormal anchors require interval_mode=lognormal")
        if self.trace_type == "inferact_codex_swebenchpro":
            if self.prompt_shape != "trace_record":
                raise ValueError("inferact_codex_swebenchpro requires prompt_shape=trace_record")
            if self.interval_mode != "lognormal":
                raise ValueError("inferact_codex_swebenchpro requires interval_mode=lognormal")
        if self.trace_type == "agentinfer" and self.prompt_shape == "trace_record":
            raise ValueError("agentinfer trace_type does not provide a unified trace_record IR")
        return self

    def context_micro_trim_limit(self, target: int) -> int:
        """Return the shared Planner and Prompt calibration trim threshold."""

        return max(
            self.context_micro_trim_max_tokens,
            math.ceil(target * self.context_micro_trim_max_ratio),
        )


class ReplayBenchConfig(ReplayStrictModel):
    """Root configuration for Replay planning and execution."""

    experiment: ReplayExperimentConfig = Field(default_factory=ReplayExperimentConfig)
    backend: ReplayBackendConfig = Field(default_factory=ReplayBackendConfig)
    replay: ReplayConfig

    @model_validator(mode="after")
    def validate_trace_backend(self) -> ReplayBenchConfig:
        """Reject trace adapters whose accounting does not match the wire endpoint."""

        if self.replay.trace_type == "inferact_codex_swebenchpro" and self.backend.endpoint != "/v1/chat/completions":
            raise ValueError("inferact_codex_swebenchpro requires backend.endpoint=/v1/chat/completions")
        return self


def resolve_replay_config_paths(
    config: ReplayBenchConfig,
    base_dir: Path,
) -> ReplayBenchConfig:
    """Resolve YAML paths relative to the configuration file directory."""

    for owner, name in (
        (config.experiment, "result_dir"),
        (config.replay, "trace_path"),
    ):
        value = getattr(owner, name)
        if not value.is_absolute():
            setattr(owner, name, (base_dir / value).resolve())
    return config


def load_replay_config(path: Path) -> ReplayBenchConfig:
    """Load and resolve one Replay YAML configuration."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = ReplayBenchConfig.model_validate(raw)
    return resolve_replay_config_paths(config, path.resolve().parent)
