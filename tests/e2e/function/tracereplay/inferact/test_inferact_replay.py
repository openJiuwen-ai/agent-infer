# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest: Inferact replay functional E2E (cold starts + repeatability compare)."""

from __future__ import annotations

from pathlib import Path

import pytest

from .helpers.replay_compare import compare_replay_stability, format_stability_report
from .helpers.replay_config import ReplayE2EConfig
from .helpers.replay_executor import run_replay_cold_starts


def test_inferact_replay(pytestconfig: pytest.Config) -> None:
    """Run a case N times (cold vLLM each time), then compare run 1 vs run 2 for repeatability."""

    test_config_file = pytestconfig.getoption("--test-config-file")
    if not test_config_file:
        raise pytest.UsageError("--test-config-file is required for Inferact replay E2E")

    cold_runs = int(pytestconfig.getoption("--cold-runs"))
    if cold_runs < 2:
        raise pytest.UsageError("--cold-runs must be >= 2 (repeatability compares the first two runs)")

    config = ReplayE2EConfig.from_case_file(Path(test_config_file)).with_benchmark_load(
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    tolerance = float(pytestconfig.getoption("--stability-tolerance"))
    run_dirs = run_replay_cold_starts(config, cold_runs=cold_runs)
    report = compare_replay_stability(
        run_dirs[0],
        run_dirs[1],
        tolerance_ratio=tolerance,
    )
    print(format_stability_report(report), flush=True)
    assert report.passed, "Inferact replay repeatability checks failed; see stdout for details"
