# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Strict configuration for Trace Replay planning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ReplayTraceType = Literal["agentinfer", "inferact_codex_swebenchpro", "agentX", "tracelab"]

_BUILTIN_REPLAY_CONFIGS: dict[str, str] = {
    "agentinfer": "replay_agentinfer.yaml",
    "inferact_codex_swebenchpro": "replay_inferact.yaml",
    "tracelab": "replay_tracelab.yaml",
    "agentX": "replay_agentX.yaml",
}


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
    chat_template_kwargs: dict[str, object] = Field(default_factory=dict)
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

    trace_type: ReplayTraceType = Field(
        "agentinfer",
        json_schema_extra={"cli": True},
    )
    trace_path: Path | None = Field(
        None,
        description="Local trace path; TraceLab downloads a pinned dataset snapshot when omitted.",
        json_schema_extra={"cli": True},
    )
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
    context_adjustment_mode: Literal["strict", "adaptive"] = "strict"
    context_micro_trim_max_tokens: int = Field(64, ge=0)
    context_micro_trim_max_ratio: float = Field(0.005, ge=0, le=1)
    prompt_calibration_tolerance_tokens: int = Field(
        0,
        ge=0,
        le=8,
        description="Residual allowance for synthetic prompts; Inferact always uses exact input lengths.",
    )
    request_timeout_seconds: int = Field(3600, ge=1)

    @property
    def prompt_shape(self) -> str:
        """Derive the internal prompt shape from the selected trace adapter."""

        return {
            "inferact_codex_swebenchpro": "inferact_synthetic",
            "agentinfer": "agentinfer_synthetic",
            "tracelab": "tracelab_synthetic",
            "agentX": "agentX_synthetic",
        }[self.trace_type]

    @model_validator(mode="after")
    def validate_replay_modes(self) -> ReplayConfig:
        if self.trace_path is None and self.trace_type != "tracelab":
            raise ValueError("trace_path may be null only when trace_type=tracelab")
        if self.interval_mode == "lognormal":
            if self.interval_lognormal is None:
                raise ValueError("interval_mode=lognormal requires interval_lognormal anchors")
        elif self.interval_lognormal is not None:
            raise ValueError("interval_lognormal anchors require interval_mode=lognormal")
        if self.trace_type == "inferact_codex_swebenchpro":
            if self.interval_mode != "lognormal":
                raise ValueError("inferact_codex_swebenchpro requires interval_mode=lognormal")
            if self.prompt_calibration_tolerance_tokens != 0:
                raise ValueError("inferact_codex_swebenchpro requires prompt_calibration_tolerance_tokens=0")
        if self.trace_type == "tracelab":
            if self.prompt_calibration_tolerance_tokens != 0:
                raise ValueError("tracelab requires prompt_calibration_tolerance_tokens=0")
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

        if self.replay.trace_path is None and self.experiment.task_num is None:
            raise ValueError("TraceLab automatic dataset download requires experiment.task_num")
        if self.replay.trace_type == "inferact_codex_swebenchpro" and self.backend.endpoint != "/v1/chat/completions":
            raise ValueError("inferact_codex_swebenchpro requires backend.endpoint=/v1/chat/completions")
        if self.replay.trace_type == "tracelab" and self.backend.endpoint != "/v1/chat/completions":
            raise ValueError("tracelab requires backend.endpoint=/v1/chat/completions")
        return self


def builtin_replay_config_path(trace_type: ReplayTraceType) -> Path:
    """Return the packaged Replay template selected by a trace adapter."""

    return Path(__file__).resolve().parents[1] / "configs" / _BUILTIN_REPLAY_CONFIGS[trace_type]


def _resolve_yaml_paths(raw: dict[str, object], base_dir: Path) -> None:
    """Resolve paths explicitly supplied by YAML before applying CLI overrides."""

    for section, name in (("experiment", "result_dir"), ("replay", "trace_path")):
        settings = raw.get(section)
        if not isinstance(settings, dict):
            continue
        value = settings.get(name)
        if value is None:
            continue
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value)
        if not path.is_absolute():
            settings[name] = (base_dir / path).resolve()


def _merge_replay_overrides(raw: dict[str, object], overrides: dict[str, dict[str, object]]) -> None:
    """Merge already-normalized CLI values over one YAML payload."""

    for section, values in overrides.items():
        settings = raw.setdefault(section, {})
        if not isinstance(settings, dict):
            raise ValueError(f"Replay configuration section {section!r} must be an object")
        settings.update(values)


def load_replay_config(
    path: Path,
    *,
    overrides: dict[str, dict[str, object]] | None = None,
) -> ReplayBenchConfig:
    """Load YAML, apply normalized CLI overrides, and validate the final Replay configuration."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Replay configuration root must be an object")
    _resolve_yaml_paths(raw, path.resolve().parent)
    if overrides:
        _merge_replay_overrides(raw, overrides)
    return ReplayBenchConfig.model_validate(raw)
