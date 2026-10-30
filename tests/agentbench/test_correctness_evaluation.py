# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Correctness contracts, fault handling and CLI integration without Docker."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.cli import main
from agentinfer.agentbench.benchkit.common import atomic_write_json
from agentinfer.agentbench.benchkit.evaluation import runner
from agentinfer.agentbench.benchkit.evaluation.compare import compare_correctness, load_evaluation
from agentinfer.agentbench.benchkit.evaluation.config import EvaluationConfig
from agentinfer.agentbench.benchkit.evaluation.inputs import digest, prepare_inputs
from agentinfer.agentcache.entrypoints.bench import _normalize_delegated_argv
from tests.agentbench.test_compare import write_run


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "source"
    write_run(run, completed=2, failed=1, status="failed")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    atomic_write_json(
        dataset / "manifest.json",
        {
            "source": "princeton-nlp/SWE-bench_Verified",
            "variant": "verified",
            "split": "test",
        },
    )
    manifest = json.loads((run / "manifest.json").read_bytes())
    manifest["config"] = {
        "dataset": {"name": "swebench_verified", "index_path": str(dataset / "instances.jsonl")},
        "agent": {"type": "dsh", "profile": "single"},
        "backend": {"type": "vllm", "model": "org/model", "endpoint": "/v1/chat/completions"},
        "experiment": {"task_num": 3, "max_concurrency": 2},
    }
    atomic_write_json(run / "manifest.json", manifest)
    rows, index = [], []
    for i, outcome in enumerate(["completed", "completed", "failed"]):
        iid = f"astropy__astropy-{i}"
        task = run / "tasks" / iid
        task.mkdir(parents=True)
        patch = f'diff {i}\r\nquoted "text"\n'.encode()
        (task / "model.patch").write_bytes(patch)
        (task / "prompt.txt").write_text(f"Instance: {iid}\nRepository: astropy/astropy\nBase commit: abc\n")
        atomic_write_json(
            task / "result.json",
            {
                "schema_version": "1",
                "instance_id": iid,
                "session_id": str(i),
                "agent": {"type": "dsh", "profile": "single"},
                "outcome": outcome,
                "termination_reason": None,
                "duration_seconds": 1,
                "has_patch": True,
                "patch_bytes": len(patch),
                "error": None,
                "task_position": i,
                "started_at": "",
                "finished_at": "",
                "topology": {"session_id": str(i), "agent_ids": [], "agent_roles": {}},
            },
        )
        index.append({"instance_id": iid, "session_id": str(i), "task_position": i, "outcome": outcome})
        rows.append(
            {
                "instance_id": iid,
                "repo": "astropy/astropy",
                "base_commit": "abc",
                "problem_statement": "fix bug",
                "version": "4.3",
                "test_patch": "test",
                "patch": "REFERENCE MUST NEVER BECOME PREDICTION",
                "FAIL_TO_PASS": '["test_bug"]',
                "PASS_TO_PASS": '["test_ok"]',
            }
        )
    atomic_write_json(run / "task_index.json", {"schema_version": "1", "run_id": run.name, "tasks": index})
    (dataset / "instances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


@pytest.fixture
def fake_harness(monkeypatch):
    outcomes = {"astropy__astropy-0": True, "astropy__astropy-1": False}

    def probe(config, dataset, timeout):
        rows = json.loads(dataset.read_bytes())
        return {"version": "4.1.0", "source_sha256": "source", "images": {r["instance_id"]: "image" for r in rows}}

    def execute(command, output, timeout, cleanup_timeout):
        run_id = command[command.index("--run_id") + 1]
        predictions = [json.loads(line) for line in (output / "inputs/predictions.jsonl").read_text().splitlines()]
        for prediction in predictions:
            iid = prediction["instance_id"]
            if not prediction["model_patch"] or outcomes.get(iid) == "missing":
                continue
            path = output / "raw/logs/run_evaluation" / run_id / "org__model" / iid
            path.mkdir(parents=True)
            (path / "patch.diff").write_bytes(prediction["model_patch"].encode())
            (path / "test_output.txt").write_text("actual tests")
            (path / "run_instance.log").write_text("log")
            if outcomes.get(iid) == "corrupt":
                (path / "report.json").write_text("{")
            else:
                atomic_write_json(
                    path / "report.json",
                    {
                        iid: {
                            "resolved": outcomes[iid],
                            "patch_successfully_applied": True,
                            "tests_status": {
                                "FAIL_TO_PASS": {"success": [], "failure": []},
                                "PASS_TO_PASS": {"success": [], "failure": []},
                            },
                        }
                    },
                )
        return 0

    monkeypatch.setattr(runner, "harness_identity", probe)
    monkeypatch.setattr(runner, "run_harness", execute)
    monkeypatch.setattr(runner, "cleanup_containers", lambda *a: [])
    monkeypatch.setattr(
        runner,
        "image_identities",
        lambda images, timeout: {k: {"name": v, "id": "sha256:image"} for k, v in images.items()},
    )
    return outcomes


def config():
    return EvaluationConfig(harness_python=Path(sys.executable), expected_harness_version="4.1.0")


def update(path, change):
    value = json.loads(path.read_bytes())
    change(value)
    atomic_write_json(path, value)


def test_select_only_completed_and_preserve_patch_bytes(source):
    prepared = prepare_inputs(source)
    assert prepared["instance_ids"] == ["astropy__astropy-0", "astropy__astropy-1"]
    for prediction in prepared["predictions"]:
        patch = (source / "tasks" / prediction["instance_id"] / "model.patch").read_bytes()
        assert prediction["model_patch"].encode() == patch
        assert "REFERENCE" not in prediction["model_patch"]


@pytest.mark.parametrize("change", ["count", "duplicate", "status", "base_commit", "running"])
def test_inconsistent_source_rejected(source, change):
    if change == "count":
        update(source / "summary.json", lambda d: d["tasks"].update(completed=1))
    elif change == "duplicate":
        update(source / "task_index.json", lambda d: d["tasks"].append(d["tasks"][0]))
    elif change == "status":
        update(source / "tasks/astropy__astropy-0/result.json", lambda d: d.update(outcome="failed"))
    elif change == "running":
        update(source / "manifest.json", lambda d: d.update(status="running"))
    else:
        (source / "tasks/astropy__astropy-0/prompt.txt").write_text("Base commit: wrong")
    with pytest.raises(ValueError):
        prepare_inputs(source)


def test_missing_patch_is_error_and_does_not_shrink_total(source, fake_harness):
    (source / "tasks/astropy__astropy-0/model.patch").unlink()
    output, code = runner.evaluate(source, config())
    summary, manifest, results = load_evaluation(output / "correctness.json")
    assert code == 1
    assert summary.counts.model_dump() == {"total": 2, "resolved": 0, "unresolved": 1, "empty_patch": 0, "error": 1}
    assert summary.resolved_rate == 0
    assert "input_error" in results["astropy__astropy-0"]["reason"]
    assert manifest["status"] == "failed"


def test_confirmed_no_changes_is_empty_not_error(source, fake_harness):
    task = source / "tasks/astropy__astropy-0"
    (task / "model.patch").unlink()
    update(task / "result.json", lambda d: d.update(has_patch=False, patch_bytes=0))
    atomic_write_json(task / "dsh-run-status.json", {"patch": "no_changes"})
    output, code = runner.evaluate(source, config())
    summary, _, _ = load_evaluation(output / "correctness.json")
    assert code == 0 and summary.counts.empty_patch == 1 and summary.counts.total == 2
    assert summary.counts.unresolved == 1


@pytest.mark.parametrize("fault", ["missing", "corrupt", "not_a_boolean"])
def test_exit_zero_does_not_hide_invalid_reports(source, fake_harness, fault):
    fake_harness["astropy__astropy-0"] = fault
    output, code = runner.evaluate(source, config())
    summary, _, _ = load_evaluation(output / "correctness.json")
    assert code == 1 and summary.counts.error == 1 and summary.counts.total == 2


def test_timeout_still_finalizes_all_selected_tasks(source, fake_harness, monkeypatch):
    def timeout(*args):
        raise subprocess.TimeoutExpired("harness", 1)

    monkeypatch.setattr(runner, "run_harness", timeout)
    output, code = runner.evaluate(source, config())
    summary, manifest, _ = load_evaluation(output / "correctness.json")
    assert code == 1 and summary.counts.error == 2
    assert manifest["finished_at"] and manifest["errors"]


def test_evaluation_preserves_source_and_refuses_overwrite(source, fake_harness, tmp_path):
    before = {str(p): digest(p.read_bytes()) for p in source.rglob("*") if p.is_file()}
    output, code = runner.evaluate(source, config(), tmp_path / "evaluation")
    assert code == 0
    assert {p: digest(Path(p).read_bytes()) for p in before} == before
    assert load_evaluation(output / "correctness.json")[0].resolved_rate == 0.5
    with pytest.raises(FileExistsError):
        runner.evaluate(source, config(), output)


def test_same_total_accuracy_still_detects_task_regression(source, fake_harness, tmp_path):
    baseline, _ = runner.evaluate(source, config(), tmp_path / "baseline")
    fake_harness.update({"astropy__astropy-0": False, "astropy__astropy-1": True})
    candidate, _ = runner.evaluate(source, config(), tmp_path / "candidate")
    report = compare_correctness(baseline / "correctness.json", candidate / "correctness.json")
    assert report["status"] == "failed" and report["delta_resolved_rate"] == 0
    assert report["regressions"] == ["astropy__astropy-0"]
    assert report["improvements"] == ["astropy__astropy-1"]


@pytest.mark.parametrize("fault", ["tamper", "running", "errors", "generation"])
def test_compare_rejects_incomplete_or_incompatible_evidence(source, fake_harness, tmp_path, fault):
    baseline, _ = runner.evaluate(source, config(), tmp_path / "baseline")
    candidate, _ = runner.evaluate(source, config(), tmp_path / "candidate")
    if fault == "tamper":
        (candidate / "inputs/patches/astropy__astropy-0.patch").write_text("changed")
    else:

        def alter(m):
            if fault == "running":
                m["status"] = "running"
            elif fault == "errors":
                m["errors"] = ["cleanup failed"]
            else:
                m["generation"]["backend"]["model"] = "another-model"

        update(candidate / "evaluation_manifest.json", alter)
    assert (
        compare_correctness(baseline / "correctness.json", candidate / "correctness.json")["status"]
        == "insufficient_evidence"
    )


def test_empty_patch_without_no_changes_evidence_is_error(source, fake_harness):
    task = source / "tasks/astropy__astropy-0"
    (task / "model.patch").unlink()
    update(task / "result.json", lambda d: d.update(has_patch=False, patch_bytes=0))
    output, code = runner.evaluate(source, config())
    summary, _, _ = load_evaluation(output / "correctness.json")
    assert code == 1 and summary.counts.error == 1 and summary.counts.empty_patch == 0


def test_all_empty_patches_need_no_docker(source, fake_harness, monkeypatch, tmp_path):
    for iid in ("astropy__astropy-0", "astropy__astropy-1"):
        task = source / "tasks" / iid
        (task / "model.patch").write_bytes(b"")
        update(task / "result.json", lambda d: d.update(has_patch=False, patch_bytes=0))

    def unexpected(*args):
        pytest.fail("empty patches must not start Docker")

    monkeypatch.setattr(runner, "run_harness", unexpected)
    monkeypatch.setattr(runner, "cleanup_containers", unexpected)
    output, code = runner.evaluate(source, config(), tmp_path / "all-empty")
    summary, _, _ = load_evaluation(output / "correctness.json")
    assert code == 0 and summary.counts.empty_patch == 2 and summary.resolved_rate == 0
    assert compare_correctness(output / "correctness.json", output / "correctness.json")["status"] == "passed"


def test_no_completed_tasks_does_not_create_output(source, tmp_path):
    update(source / "task_index.json", lambda d: [r.update(outcome="failed") for r in d["tasks"]])
    for p in (source / "tasks").glob("*/result.json"):
        update(p, lambda d: d.update(outcome="failed"))
    update(source / "summary.json", lambda d: d["tasks"].update(completed=0, failed=3))
    output = tmp_path / "no-tasks"
    with pytest.raises(ValueError, match="no_completed_tasks"):
        runner.evaluate(source, config(), output)
    assert not output.exists()


def test_actual_subprocess_timeout_preserves_logs(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_harness(
            [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"], tmp_path, 0.5, 1
        )
    assert "started" in (tmp_path / "harness.stdout.log").read_text()


def test_version_mismatch_fails_before_harness_import(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="5.0.2\n", stderr="")
    )
    with pytest.raises(ValueError, match="expected SWE-bench 4.1.0"):
        runner.harness_identity(config(), tmp_path / "dataset.json", 1)


def test_cli_delegation_and_evaluation(source, fake_harness, tmp_path, capsys):
    for command in ("evaluate", "compare-correctness"):
        argv, _ = _normalize_delegated_argv(["vllm", "bench", "serve", "--agentinfer", command])
        assert argv == [command]
    conf = tmp_path / "evaluation.yaml"
    conf.write_text(f"harness_python: {sys.executable}\nexpected_harness_version: '4.1.0'\n")
    output = tmp_path / "eval-cli"
    assert main(["evaluate", "--result-path", str(source), "--config", str(conf), "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["total"] == 2
    assert (
        main(
            [
                "compare-correctness",
                "--baseline",
                str(output / "correctness.json"),
                "--candidate",
                str(output / "correctness.json"),
            ]
        )
        == 0
    )
