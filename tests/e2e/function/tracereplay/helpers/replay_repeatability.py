# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared repeatability test body for Trace Replay functional E2E."""

from __future__ import annotations

from pathlib import Path

import pytest

from ....helpers.case_loader import load_case_file
from .replay_compare import compare_replay_stability, format_stability_report
from .replay_e2e_config import ReplayE2EConfig, ReplaySuiteSpec
from .replay_executor import run_replay_cold_starts


def run_replay_repeatability_e2e(
    pytestconfig: pytest.Config,
    *,
    suite: ReplaySuiteSpec,
    suite_label: str,
) -> None:
    """Run N cold replays, then compare run 1 vs run 2 for repeatability."""

    test_config_file = pytestconfig.getoption("--test-config-file")
    if not test_config_file:
        raise pytest.UsageError(f"--test-config-file is required for {suite_label} replay E2E")

    case_path = Path(test_config_file)
    case = load_case_file(case_path)
    case_workload = case.benchmark_params.get("workload")
    if case_workload != suite.workload:
        pytest.skip(f"--test-config-file workload {case_workload!r} does not match {suite_label} ({suite.workload!r})")

    cold_runs = int(pytestconfig.getoption("--cold-runs"))
    if cold_runs < 2:
        raise pytest.UsageError("--cold-runs must be >= 2 (repeatability compares the first two runs)")

    config = ReplayE2EConfig.from_case_file(case_path, suite).with_benchmark_load(
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
        bench_config=pytestconfig.getoption("--bench-config"),
    )
    tolerance = float(pytestconfig.getoption("--stability-tolerance"))
    run_dirs = run_replay_cold_starts(config, cold_runs=cold_runs)
    report = compare_replay_stability(
        run_dirs[0],
        run_dirs[1],
        tolerance_ratio=tolerance,
    )
    print(format_stability_report(report), flush=True)
    assert report.passed, f"{suite_label} replay repeatability checks failed; see stdout for details"
