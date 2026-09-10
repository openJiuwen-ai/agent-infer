# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest E2E performance benchmark driven by a JSON case file."""

from __future__ import annotations

from pathlib import Path

import pytest

from .helpers.benchmark import run_benchmark, validate_completed_run
from .helpers.case_loader import E2EPerfConfig, load_case_file


@pytest.mark.e2e_perf
def test_benchmark_completes(pytestconfig: pytest.Config) -> None:
    """Run one benchmark case selected by ``--test-config-file``."""

    test_config_file = pytestconfig.getoption("--test-config-file")
    if not test_config_file:
        pytest.skip("--test-config-file is required for e2e perf benchmarks")

    case = load_case_file(Path(test_config_file))
    config = E2EPerfConfig.from_case(case).with_benchmark_load(
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    benchmark_user = pytestconfig.getoption("--benchmark-run-as-user")
    if benchmark_user is not None:
        config = config.with_runtime_overrides(
            benchmark_run_as_user=str(benchmark_user).strip() or None,
            benchmark_run_as_user_set=True,
        )
    run_dir = run_benchmark(config)
    summary = validate_completed_run(run_dir)
    print(
        f"[benchmark:{config.scenario.value}] completed: "
        f"tasks={summary['tasks']['completed']} requests={summary['requests']['requests']} "
        f"dir={run_dir}",
        flush=True,
    )
