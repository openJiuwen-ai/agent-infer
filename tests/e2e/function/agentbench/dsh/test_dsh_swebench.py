# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""DSH + SWE-bench Verified functional E2E (BenchKit run + artifact checks)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.helpers.benchmark import run_benchmark, validate_completed_run
from tests.e2e.helpers.case_loader import E2EConfig, load_case_file


@pytest.mark.e2e_functional
def test_dsh_swebench(pytestconfig: pytest.Config) -> None:
    """Run one DSH SWE-bench case and assert standard artifacts plus DSH-specific checks."""

    test_config_file = pytestconfig.getoption("--test-config-file")
    if not test_config_file:
        raise pytest.UsageError("--test-config-file is required for DSH SWE-bench functional E2E")

    case = load_case_file(Path(test_config_file))
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
    summary = validate_completed_run(run_dir, functional_validation=config.case.functional_validation)
    print(
        f"[dsh-swebench:{config.scenario.value}] completed: "
        f"tasks={summary['tasks']['completed']} requests={summary['requests']['requests']} "
        f"dir={run_dir}",
        flush=True,
    )
