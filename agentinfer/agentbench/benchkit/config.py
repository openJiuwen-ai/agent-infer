# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Strict benchmark configuration loaded from YAML and CLI overrides."""

from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictModel):
    task_num: int | None = Field(None, ge=1, json_schema_extra={"cli": True})
    result_dir: Path = Field(Path("results"), json_schema_extra={"cli": True})
    max_concurrency: int = Field(1, ge=1, json_schema_extra={"cli": True})
    task_timeout_seconds: int = Field(3600, ge=1, json_schema_extra={"cli": {"flag": "--timeout"}})
    run_timeout_seconds: int | None = Field(None, ge=1)
    patch_flush_seconds: int = Field(15, ge=0)
    prompt_delivery_timeout_seconds: int = Field(30, ge=1)


class DatasetConfig(StrictModel):
    name: Literal["swebench_verified"] = Field(
        "swebench_verified", json_schema_extra={"cli": {"flag": "--dataset-name"}}
    )
    index_path: Path = Field(Path("data/swebench/instances.jsonl"), json_schema_extra={"cli": True})
    selection_path: Path = Field(Path("data/swebench/task-lists/default.txt"), json_schema_extra={"cli": True})
    cache_dir: Path = Path("data/swebench/repo-cache")


class AgentConfig(StrictModel):
    type: Literal["claude", "jiuwenswarm", "dsh"] = Field(
        "claude", json_schema_extra={"cli": {"flag": "--agent-type", "dest": "agent_type"}}
    )
    profile: str = Field("single", json_schema_extra={"cli": {"flag": "--agent-profile", "dest": "agent_profile"}})
    executable: Path = Field(
        Path("claude"), json_schema_extra={"cli": {"flag": "--agent-executable", "dest": "agent_executable"}}
    )
    tmux_startup_seconds: float = Field(2.0, ge=0)
    terminal_capture_interval_seconds: int = Field(30, ge=1)

    @model_validator(mode="after")
    def validate_runtime_profile(self) -> "AgentConfig":
        """Validate the profile against the selected runtime."""

        from ..agents.registry import get_runtime

        try:
            get_runtime(self.type).get_profile(self.profile)
        except ValueError as exc:
            raise ValueError(f"agent.profile {self.profile!r} is invalid for agent.type {self.type!r}") from exc
        return self


class BackendConfig(StrictModel):
    type: Literal["vllm"] = "vllm"
    base_url: str = Field("http://127.0.0.1:8000", json_schema_extra={"cli": True})
    metrics_url: str | None = Field(None, json_schema_extra={"cli": {"flag": "--metrics-url"}})
    model: str = Field("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", json_schema_extra={"cli": True})
    endpoint: str = Field("/v1/messages", json_schema_extra={"cli": True})
    api_key_env: str | None = None

    @property
    def effective_metrics_url(self) -> str:
        """Return the complete Prometheus endpoint used for vLLM evidence."""

        return self.metrics_url or f"{self.base_url.rstrip('/')}/metrics"


class RequestProxyConfig(StrictModel):
    listen_url: str = "http://127.0.0.1:0"
    request_timeout_seconds: int = Field(3600, ge=1)
    startup_timeout_seconds: float = Field(120.0, ge=0)
    shutdown_timeout_seconds: float = Field(30.0, gt=0)

    @model_validator(mode="after")
    def validate_loopback_listener(self) -> "RequestProxyConfig":
        """Require the Request Proxy to bind to an HTTP loopback address."""

        parsed = urlparse(self.listen_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("request_proxy.listen_url must be an HTTP loopback URL")
        try:
            loopback = ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.lower() == "localhost"
        if not loopback:
            raise ValueError("request_proxy.listen_url must bind to loopback")
        return self

    @property
    def host(self) -> str:
        """Return the configured listener hostname."""

        return urlparse(self.listen_url).hostname or "127.0.0.1"

    @property
    def port(self) -> int:
        """Return the configured listener port, including zero for auto-selection."""

        return urlparse(self.listen_url).port or 0


class AgentBenchConfig(StrictModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    request_proxy: RequestProxyConfig = Field(default_factory=RequestProxyConfig)

    @model_validator(mode="after")
    def validate_agent_endpoint(self) -> "AgentBenchConfig":
        """Require the model protocol used by the selected runtime."""

        from ..agents.registry import get_runtime

        required_endpoint = get_runtime(self.agent.type).required_endpoint
        if self.backend.endpoint != required_endpoint:
            raise ValueError(
                f"backend.endpoint {self.backend.endpoint!r} is invalid for "
                f"agent.type {self.agent.type!r}; expected {required_endpoint!r}"
            )
        return self


def resolve_config_paths(config: AgentBenchConfig, base_dir: Path) -> AgentBenchConfig:
    """Resolve filesystem paths relative to the configuration file directory."""

    for owner, name in (
        (config.dataset, "index_path"),
        (config.dataset, "selection_path"),
        (config.dataset, "cache_dir"),
        (config.experiment, "result_dir"),
    ):
        value = getattr(owner, name)
        if not value.is_absolute():
            setattr(owner, name, (base_dir / value).resolve())
    executable = config.agent.executable
    if not executable.is_absolute() and len(executable.parts) > 1:
        config.agent.executable = (base_dir / executable).resolve()
    return config


def load_config(path: Path | None = None) -> AgentBenchConfig:
    """Load, validate, and resolve a benchmark YAML configuration.

    When ``path`` is ``None``, build the config from model defaults and resolve
    relative paths against the current working directory. Passing a path keeps
    the YAML-anchored resolution where relative paths resolve from the file's
    directory.
    """

    if path is not None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = AgentBenchConfig.model_validate(raw)
        base_dir = path.resolve().parent
    else:
        config = AgentBenchConfig()
        base_dir = Path.cwd()
    return resolve_config_paths(config, base_dir)
