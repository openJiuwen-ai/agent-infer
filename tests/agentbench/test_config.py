# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentinfer.agentbench.benchkit.config import (
    AgentBenchConfig,
    RequestProxyConfig,
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
    assert default.backend.effective_metrics_url == "http://127.0.0.1:8000/metrics"
    assert overridden.backend.effective_metrics_url == "http://127.0.0.1:9000/metrics"


def test_explicit_metrics_url_is_used_as_complete_endpoint() -> None:
    config = AgentBenchConfig.model_validate(
        {"backend": {"base_url": "http://router:8400", "metrics_url": "http://vllm:8000/metrics"}}
    )

    assert config.backend.effective_metrics_url == "http://vllm:8000/metrics"


def test_jiuwenswarm_profile_and_openai_endpoint() -> None:
    config = AgentBenchConfig.model_validate(
        {
            "agent": {
                "type": "jiuwenswarm",
                "profile": "code.normal",
                "executable": "jiuwenswarm",
            },
            "backend": {"endpoint": "/v1/chat/completions"},
        }
    )

    assert config.agent.type == "jiuwenswarm"
    assert config.agent.profile == "code.normal"
    assert config.backend.endpoint == "/v1/chat/completions"


def test_dsh_profile_and_openai_endpoint() -> None:
    config = AgentBenchConfig.model_validate(
        {
            "agent": {
                "type": "dsh",
                "profile": "plan-subagent",
                "executable": "dsh",
            },
            "backend": {"endpoint": "/v1/chat/completions"},
        }
    )

    assert config.agent.type == "dsh"
    assert config.agent.profile == "plan-subagent"
    assert config.backend.endpoint == "/v1/chat/completions"


@pytest.mark.parametrize(
    ("agent_type", "profile"),
    [
        ("claude", "code.normal"),
        ("jiuwenswarm", "single"),
        ("jiuwenswarm", "code.team"),
        ("dsh", "code.normal"),
        ("claude", "autonomous"),
    ],
)
def test_agent_type_profile_mismatch_is_rejected(agent_type: str, profile: str) -> None:
    with pytest.raises(ValidationError, match="agent.profile"):
        AgentBenchConfig.model_validate({"agent": {"type": agent_type, "profile": profile}})


@pytest.mark.parametrize(
    ("agent_type", "profile", "endpoint"),
    [
        ("claude", "single", "/v1/chat/completions"),
        ("jiuwenswarm", "code.normal", "/v1/messages"),
        ("dsh", "single", "/v1/messages"),
    ],
)
def test_agent_endpoint_mismatch_is_rejected(
    agent_type: str,
    profile: str,
    endpoint: str,
) -> None:
    with pytest.raises(ValidationError, match="backend.endpoint"):
        AgentBenchConfig.model_validate(
            {
                "agent": {"type": agent_type, "profile": profile},
                "backend": {"endpoint": endpoint},
            }
        )


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": True},
        {"experiment": {"unknown": True}},
        {"request_proxy": {"router_url": "http://127.0.0.1:8400"}},
        {"router": {"enabled": True, "base_url": "http://127.0.0.1:8400"}},
    ],
)
def test_unknown_fields_are_rejected(raw: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentBenchConfig.model_validate(raw)


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


def test_load_config_none_resolves_cwd_relative_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config(None)

    assert config.dataset.index_path == (tmp_path / "data/swebench/instances.jsonl").resolve()
    assert config.dataset.selection_path == (tmp_path / "data/swebench/task-lists/default.txt").resolve()
    assert config.dataset.cache_dir == (tmp_path / "data/swebench/repo-cache").resolve()
    assert config.experiment.result_dir == (tmp_path / "results").resolve()
    # Bare executable stays a name for PATH lookup, not resolved against cwd.
    assert config.agent.executable == Path("claude")


def test_bare_executable_remains_for_path_lookup(tmp_path: Path) -> None:
    config = resolve_config_paths(AgentBenchConfig(), tmp_path)

    assert config.agent.executable == Path("claude")


@pytest.mark.parametrize(
    ("filename", "base_url", "metrics_url", "agent_type", "endpoint"),
    [
        ("swebench_vllm.yaml", "http://127.0.0.1:8000", None, "claude", "/v1/messages"),
        (
            "swebench_agentinfer.yaml",
            "http://127.0.0.1:8400",
            "http://127.0.0.1:8000/metrics",
            "claude",
            "/v1/messages",
        ),
    ],
)
def test_sample_yaml_loads(
    filename: str,
    base_url: str,
    metrics_url: str | None,
    agent_type: str,
    endpoint: str,
) -> None:
    config = load_config(Path("agentinfer/agentbench/configs") / filename)

    assert config.backend.base_url == base_url
    assert config.backend.metrics_url == metrics_url
    assert config.agent.type == agent_type
    assert config.backend.endpoint == endpoint
