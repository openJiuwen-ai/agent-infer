# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Explicit pytest entry point for JSON-driven E2E performance benchmarks."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..helpers.benchmark import run_benchmark, validate_completed_run
from ..helpers.case_loader import E2EConfig, load_case_file


def test_benchmark_completes(pytestconfig: pytest.Config, case_file: Path) -> None:
    """Run one independently collected JSON benchmark case."""

    case = load_case_file(case_file)
    config = E2EConfig.from_case(case).with_benchmark_load(
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
