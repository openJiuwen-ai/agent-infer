# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Case-driven SWE-bench functional E2E for all supported agents."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.compare import load_summary
from agentinfer.agentbench.benchkit.config import AgentBenchConfig, load_config
from tests.e2e.function.agentbench.artifact_validation import validate_agent_artifacts
from tests.e2e.helpers.benchmark import run_benchmark
from tests.e2e.helpers.case_loader import E2EConfig, load_case_file

_DEPENDENCIES = {
    "claude": ("git", "tmux"),
    "jiuwenswarm": ("git", "bwrap"),
    "dsh": ("git", "node"),
    "zcode": ("git", "node"),
}


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


@pytest.mark.e2e_function
@pytest.mark.gpu_test
def test_agent_swebench(pytestconfig: pytest.Config) -> None:
    """Require every selected task to finish and common artifacts to exist."""
    case_file = pytestconfig.getoption("--test-config-file")
    if not case_file:
        pytest.skip("pass --test-config-file with an AgentBench functional case")
    config = E2EConfig.from_case(load_case_file(Path(case_file))).with_benchmark_load(
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    benchmark_user = pytestconfig.getoption("--benchmark-run-as-user")
    if benchmark_user is not None:
        config = config.with_runtime_overrides(
            benchmark_run_as_user=str(benchmark_user).strip() or None,
            benchmark_run_as_user_set=True,
        )
    agent_type = resolve_agent_type(config)
    if config.task_num <= 0 or config.max_concurrency <= 0:
        pytest.fail("task-num and max-concurrency must be positive")
    blockers = config.validate_prerequisites()
    for executable in _DEPENDENCIES[agent_type]:
        if not shutil.which(executable):
            blockers.append(f"{executable} executable is unavailable on PATH")
    if blockers:
        pytest.fail("; ".join(blockers))

    run_dir = run_benchmark(config)
    print(f"{agent_type} functional artifacts: {run_dir}", flush=True)
    summary = load_summary(run_dir)
    assert summary["lifecycle"]["status"] == "completed", summary["lifecycle"]
    assert summary["tasks"]["completed"] == config.task_num, summary["tasks"]
    assert summary["tasks"]["failed"] == 0, summary["tasks"]
    validate_agent_artifacts(run_dir)
