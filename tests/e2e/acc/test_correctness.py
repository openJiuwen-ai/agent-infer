# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Real CLI E2E for PR #64; requires Docker and swebench==4.1.0."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.evaluation.compare import load_evaluation
from agentinfer.agentbench.benchkit.evaluation.inputs import digest
from tests.e2e.helpers.benchmark import run_benchmark, validate_completed_run
from tests.e2e.helpers.case_loader import BenchmarkCase, E2EConfig


def _cli(*args: str) -> str:
    completed = subprocess.run(
        ["vllm", "bench", "serve", "--agentinfer", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _validate(run_dir: Path, evaluation: Path, *, expected_task_num: int | None = None) -> None:
    for log in (run_dir / "tasks").glob("*/jiuwenbox-server.stderr.log"):
        sandbox_errors = [
            line
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if "bwrap:" in line and "Can't mount proc" in line
        ]
        assert not sandbox_errors, f"Jiuwen sandbox failed: {sandbox_errors}; see {log}"
    run_summary = validate_completed_run(run_dir)
    source = json.loads((run_dir / "manifest.json").read_bytes())
    index = json.loads((run_dir / "task_index.json").read_bytes())
    if expected_task_num is not None:
        failures = {
            row["instance_id"]: row.get("termination_reason") or row["outcome"]
            for row in index["tasks"]
            if row["outcome"] != "completed"
        }
        assert not failures, f"benchmark tasks did not complete: {failures}"
        assert len(index["tasks"]) == expected_task_num, "task results do not cover the configured task count"
        assert run_summary["tasks"]["completed"] == expected_task_num, "not all configured tasks completed"
        assert run_summary["tasks"]["failed"] == 0, "benchmark summary contains failed tasks"
    summary, manifest, results = load_evaluation(evaluation)
    expected = {row["instance_id"] for row in index["tasks"] if row["outcome"] == "completed"}
    assert set(results) == expected
    assert summary.source_run_id == source["run_id"]
    assert summary.counts.total == run_summary["tasks"]["completed"]
    assert summary.counts.error == 0
    assert manifest["status"] == "completed" and not manifest["errors"]
    for relative, expected_hash in manifest["source_hashes"].items():
        assert digest((run_dir / relative).read_bytes()) == expected_hash, relative
    for iid, result in results.items():
        assert not result["environment_failure"], iid
        patch = run_dir / "tasks" / iid / "model.patch"
        source_patch = patch.read_bytes() if patch.exists() else b""
        frozen_patch = evaluation.parent / "inputs" / "patches" / f"{iid}.patch"
        assert frozen_patch.read_bytes() == source_patch, iid
        assert result["patch_sha256"] == digest(source_patch), iid
    print(f"[correctness] counts={summary.counts.model_dump()} resolved_rate={summary.resolved_rate} {evaluation}")


@pytest.mark.usefixtures("correctness_environment")
def test_run_with_correctness(pytestconfig: pytest.Config, benchmark_case: BenchmarkCase, correctness_server) -> None:
    case = benchmark_case
    assert case.benchmark_params.get("with-correctness") is True
    evaluation_config = pytestconfig.getoption("--correctness-evaluation-config")
    if evaluation_config:
        case = replace(case, benchmark_params={**case.benchmark_params, "evaluation-config": evaluation_config})
    config = E2EConfig.from_case(case).with_benchmark_load(
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    user = pytestconfig.getoption("--benchmark-run-as-user")
    if user is not None:
        config = config.with_runtime_overrides(
            benchmark_run_as_user=str(user).strip() or None,
            benchmark_run_as_user_set=True,
        )
    agent = case.benchmark_params.get("agent-type", "claude")
    config = replace(
        config,
        result_root=config.result_root / agent / config.scenario.value,
        vllm_log_dir=config.result_root / config.scenario.value / "vllm-logs",
    )
    config = correctness_server(config)
    print(f"[correctness] agent={agent} scenario={config.scenario.value}", flush=True)
    run_dir = run_benchmark(config, start_server=False)
    evaluations = list((run_dir / "evaluations").glob("*/correctness.json"))
    assert len(evaluations) == 1, evaluations
    _validate(run_dir, evaluations[0], expected_task_num=config.task_num)


@pytest.mark.cpu
def test_evaluate_existing_run(pytestconfig: pytest.Config) -> None:
    raw = pytestconfig.getoption("--correctness-run-dir")
    if not raw:
        pytest.skip("provide --correctness-run-dir to score existing artifacts")
    run_dir = Path(raw).resolve()
    before = set((run_dir / "evaluations").glob("*/correctness.json"))
    args = ["evaluate", "--result-path", str(run_dir)]
    config = pytestconfig.getoption("--correctness-evaluation-config")
    if config:
        args.extend(["--config", config])
    _cli(*args)
    created = set((run_dir / "evaluations").glob("*/correctness.json")) - before
    assert len(created) == 1, created
    _validate(run_dir, created.pop())


@pytest.mark.cpu
def test_compare_correctness(pytestconfig: pytest.Config) -> None:
    baseline = pytestconfig.getoption("--correctness-baseline")
    candidate = pytestconfig.getoption("--correctness-candidate")
    if not baseline and not candidate:
        pytest.skip("provide baseline and candidate correctness.json paths")
    assert baseline and candidate, "both --correctness-baseline and --correctness-candidate are required"
    report = json.loads(_cli("compare-correctness", "--baseline", baseline, "--candidate", candidate))
    assert report["status"] == "passed", report
    assert report["regressions"] == [], report
