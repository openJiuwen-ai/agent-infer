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
    index_path: Path = Field(Path("../data/swebench/instances.jsonl"), json_schema_extra={"cli": True})
    selection_path: Path = Field(Path("../data/swebench/task-lists/default.txt"), json_schema_extra={"cli": True})
    cache_dir: Path = Path("repo-cache")


class AgentConfig(StrictModel):
    type: Literal["claude"] = Field("claude", json_schema_extra={"cli": {"flag": "--agent-type", "dest": "agent_type"}})
    profile: Literal["single", "plan-subagent"] = Field(
        "single", json_schema_extra={"cli": {"flag": "--agent-profile", "dest": "agent_profile"}}
    )
    executable: Path = Field(
        Path("claude"), json_schema_extra={"cli": {"flag": "--agent-executable", "dest": "agent_executable"}}
    )
    tmux_startup_seconds: float = Field(2.0, ge=0)
    terminal_capture_interval_seconds: int = Field(30, ge=1)


class BackendConfig(StrictModel):
    type: Literal["vllm"] = "vllm"
    base_url: str = Field("http://127.0.0.1:8000", json_schema_extra={"cli": True})
    model: str = Field("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", json_schema_extra={"cli": True})
    endpoint: Literal["/v1/messages"] = Field("/v1/messages", json_schema_extra={"cli": True})
    api_key_env: str | None = None


class RequestProxyConfig(StrictModel):
    listen_url: str = "http://127.0.0.1:0"
    request_timeout_seconds: int = Field(3600, ge=1)
    startup_timeout_seconds: float = Field(10.0, ge=0)
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


class RouterConfig(StrictModel):
    enabled: bool = Field(False, json_schema_extra={"cli": True})
    base_url: str | None = Field(None, json_schema_extra={"cli": {"flag": "--router-url", "dest": "router_url"}})
    control_timeout_seconds: float = Field(10.0, gt=0)

    @model_validator(mode="after")
    def validate_enabled_url(self) -> "RouterConfig":
        """Require Router enablement and its base URL to be configured together."""

        if self.enabled != bool(self.base_url):
            raise ValueError("router.enabled requires router.base_url, and a base_url requires enabled=true")
        return self


class AgentBenchConfig(StrictModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    request_proxy: RequestProxyConfig = Field(default_factory=RequestProxyConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)


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


def load_config(path: Path) -> AgentBenchConfig:
    """Load, validate, and resolve a benchmark YAML configuration."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return resolve_config_paths(AgentBenchConfig.model_validate(raw), path.resolve().parent)
