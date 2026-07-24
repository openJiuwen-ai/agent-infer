from pathlib import Path

import pytest
from pydantic import ValidationError

from agentinfer.agentbench.benchkit.config import (
    AgentBenchConfig,
    RequestProxyConfig,
    RouterConfig,
    load_config,
    resolve_config_paths,
)


def test_defaults_and_nested_overrides() -> None:
    default = AgentBenchConfig()
    overridden = AgentBenchConfig.model_validate(
        {
            "experiment": {"task_num": 4, "max_concurrency": 2},
            "agent": {"profile": "plan-subagent"},
            "backend": {"base_url": "http://127.0.0.1:9000"},
        }
    )

    assert default.experiment.task_num is None
    assert default.experiment.max_concurrency == 1
    assert overridden.experiment.task_num == 4
    assert overridden.experiment.max_concurrency == 2
    assert overridden.agent.profile == "plan-subagent"
    assert overridden.backend.base_url == "http://127.0.0.1:9000"


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": True},
        {"experiment": {"unknown": True}},
        {"request_proxy": {"router_url": "http://127.0.0.1:8400"}},
    ],
)
def test_unknown_fields_are_rejected(raw: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentBenchConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("enabled", "base_url"),
    [
        (True, None),
        (False, "http://127.0.0.1:8400"),
    ],
)
def test_router_rejects_inconsistent_settings(enabled: bool, base_url: str | None) -> None:
    with pytest.raises(ValidationError, match="router.enabled requires router.base_url"):
        RouterConfig(enabled=enabled, base_url=base_url)


@pytest.mark.parametrize(
    "listen_url",
    [
        "http://127.0.0.1:0",
        "http://[::1]:8000",
        "http://localhost:9000",
    ],
)
def test_request_proxy_accepts_loopback_listener(listen_url: str) -> None:
    assert RequestProxyConfig(listen_url=listen_url).port >= 0


@pytest.mark.parametrize(
    "listen_url",
    [
        "https://127.0.0.1:8000",
        "http://0.0.0.0:8000",
        "not-a-url",
    ],
)
def test_request_proxy_rejects_non_http_or_non_loopback_listener(listen_url: str) -> None:
    with pytest.raises(ValidationError, match="HTTP loopback|bind to loopback"):
        RequestProxyConfig(listen_url=listen_url)


def test_load_config_resolves_yaml_relative_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "benchmark.yaml"
    config_path.write_text(
        """experiment:
  result_dir: ../runs
dataset:
  index_path: ../data/instances.jsonl
  selection_path: ../data/tasks.txt
  cache_dir: ../cache
agent:
  executable: ../bin/claude
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.experiment.result_dir == (tmp_path / "runs").resolve()
    assert config.dataset.index_path == (tmp_path / "data/instances.jsonl").resolve()
    assert config.dataset.selection_path == (tmp_path / "data/tasks.txt").resolve()
    assert config.dataset.cache_dir == (tmp_path / "cache").resolve()
    assert config.agent.executable == (tmp_path / "bin/claude").resolve()


def test_bare_executable_remains_for_path_lookup(tmp_path: Path) -> None:
    config = resolve_config_paths(AgentBenchConfig(), tmp_path)

    assert config.agent.executable == Path("claude")


@pytest.mark.parametrize(
    ("filename", "router_enabled"),
    [
        ("swebench_vllm.yaml", False),
        ("swebench_agentinfer.yaml", True),
    ],
)
def test_sample_yaml_loads(filename: str, router_enabled: bool) -> None:
    config = load_config(Path("agentinfer/agentbench/configs") / filename)

    assert config.router.enabled is router_enabled
    assert bool(config.router.base_url) is router_enabled
    assert config.backend.endpoint == "/v1/messages"
