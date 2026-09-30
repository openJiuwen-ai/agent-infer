# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest entrypoint for Inferact replay functional E2E."""

from __future__ import annotations

from pathlib import Path

import pytest

from .helpers.replay_config import ReplayE2EConfig
from .helpers.replay_run import run_replay_two_cold_starts
from .helpers.stability import compare_replay_stability, format_stability_report


def test_inferact_replay(pytestconfig: pytest.Config) -> None:
    """Run one case twice (cold vLLM each time), then assert replay repeatability."""

    test_config_file = pytestconfig.getoption("--test-config-file")
    if not test_config_file:
        pytest.skip("--test-config-file is required for Inferact replay E2E")

    config = ReplayE2EConfig.from_case_file(Path(test_config_file)).with_benchmark_load(
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    tolerance = float(pytestconfig.getoption("--stability-tolerance"))
    baseline_dir, candidate_dir = run_replay_two_cold_starts(config)
    report = compare_replay_stability(
        baseline_dir,
        candidate_dir,
        tolerance_ratio=tolerance,
    )
    print(format_stability_report(report), flush=True)
    assert report.passed, "Inferact replay repeatability checks failed; see stdout for details"
