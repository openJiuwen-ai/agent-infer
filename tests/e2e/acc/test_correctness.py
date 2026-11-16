# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Real CLI E2E for PR #64; requires Docker and swebench==4.1.0."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.evaluation.compare import load_evaluation
from agentinfer.agentbench.benchkit.evaluation.inputs import digest
from tests.e2e.helpers.benchmark import run_benchmark, validate_completed_run
from tests.e2e.helpers.case_loader import E2EConfig, load_case_file

pytestmark = pytest.mark.e2e_function


def _cli(*args: str) -> str:
    completed = subprocess.run(
        ["vllm", "bench", "serve", "--agentinfer", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _validate(run_dir: Path, evaluation: Path) -> None:
    run_summary = validate_completed_run(run_dir)
    summary, manifest, results = load_evaluation(evaluation)
    source = json.loads((run_dir / "manifest.json").read_bytes())
    index = json.loads((run_dir / "task_index.json").read_bytes())
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


@pytest.mark.gpu_test
def test_run_with_correctness(pytestconfig: pytest.Config) -> None:
    raw = pytestconfig.getoption("--test-config-file")
    if not raw:
        pytest.skip("select a correctness case with --test-config-file")
    case = load_case_file(Path(raw))
    assert case.benchmark_params.get("with-correctness") is True
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
    run_dir = run_benchmark(config)
    evaluations = list((run_dir / "evaluations").glob("*/correctness.json"))
    assert len(evaluations) == 1, evaluations
    _validate(run_dir, evaluations[0])


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


def test_compare_correctness(pytestconfig: pytest.Config) -> None:
    baseline = pytestconfig.getoption("--correctness-baseline")
    candidate = pytestconfig.getoption("--correctness-candidate")
    if not baseline and not candidate:
        pytest.skip("provide baseline and candidate correctness.json paths")
    assert baseline and candidate, "both --correctness-baseline and --correctness-candidate are required"
    report = json.loads(_cli("compare-correctness", "--baseline", baseline, "--candidate", candidate))
    assert report["status"] == "passed", report
    assert report["regressions"] == [], report
