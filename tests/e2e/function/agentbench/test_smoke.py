# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Case-driven local smoke task for all supported agents."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from agentinfer.agentbench.agents.preflight import check_agent_preflight
from agentinfer.agentbench.benchkit.compare import load_summary
from agentinfer.agentbench.benchkit.config import AgentBenchConfig, load_config
from tests.e2e.function.agentbench.artifact_validation import validate_agent_artifacts
from tests.e2e.function.agentbench.smoke_inputs import prepare_smoke_case
from tests.e2e.helpers.benchmark import run_benchmark
from tests.e2e.helpers.case_loader import E2EConfig, load_pytest_case, validate_benchmark_user


def resolve_agent_type(config: E2EConfig) -> str:
    """Validate case overrides together, as BenchKit does before execution."""
    bench = load_config(config.bench_config).model_dump()
    params = config.case.benchmark_params
    if "agent-type" in params:
        bench["agent"]["type"] = params["agent-type"]
    bench["agent"]["profile"] = config.agent_profile
    bench["agent"]["executable"] = config.agent_executable
    if "endpoint" in params:
        bench["backend"]["endpoint"] = params["endpoint"]
    return AgentBenchConfig.model_validate(bench).agent.type


def test_agent_smoke(
    pytestconfig: pytest.Config,
    case_file: Path,
    benchmark_index: int,
    agentbench_server: Callable[[E2EConfig], E2EConfig],
) -> None:
    """Require every selected task to finish and common artifacts to exist."""
    config = load_pytest_case(pytestconfig, case_file, benchmark_index=benchmark_index)
    validate_benchmark_user(config)
    agent_type = resolve_agent_type(config)
    if config.task_num <= 0 or config.max_concurrency <= 0:
        pytest.fail("task-num and max-concurrency must be positive")
    blockers = config.validate_prerequisites()
    if not shutil.which("git"):
        blockers.append("git executable is unavailable on PATH")
    if blockers:
        pytest.fail("; ".join(blockers))
    asyncio.run(check_agent_preflight(agent_type, config.agent_executable))

    config = prepare_smoke_case(config)
    config = agentbench_server(config)
    run_dir = run_benchmark(config, manage_server=False)
    print(f"{agent_type} functional artifacts: {run_dir}", flush=True)
    summary = load_summary(run_dir)
    assert summary["lifecycle"]["status"] == "completed", summary["lifecycle"]
    assert summary["tasks"]["completed"] == config.task_num, summary["tasks"]
    assert summary["tasks"]["failed"] == 0, summary["tasks"]
    validate_agent_artifacts(run_dir)
