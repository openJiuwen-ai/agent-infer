# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

from pathlib import Path

from .replay_compare import MetricCheck, StabilityReport, _process_start_check


def test_stability_report_ignores_informational_failures() -> None:
    report = StabilityReport(
        baseline_dir=Path("baseline"),
        candidate_dir=Path("candidate"),
        tolerance_ratio=0.05,
        checks=(
            MetricCheck("baseline.cold_start", False, False, False, "cold", informational=True),
            MetricCheck("requests.requests", 1, 1, True, "exact match"),
        ),
    )
    assert report.passed is True


def test_process_start_check_is_informational(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    for index, run_dir in enumerate((baseline_dir, candidate_dir)):
        evidence = run_dir / "evidence"
        evidence.mkdir(parents=True)
        start = 1_700_000_000.0 + index * 1000
        (evidence / "vllm_metrics_start.prom").write_text(
            f'process_start_time_seconds{{engine="0"}} {start}\n',
            encoding="utf-8",
        )

    check = _process_start_check(baseline_dir, candidate_dir)
    assert check.label == "evidence.process_start_time_seconds"
    assert check.ok is True
    assert check.informational is True
    assert check.baseline == "1700000000.000"
    assert check.candidate == "1700001000.000"
