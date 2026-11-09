# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Correctness contracts, fault handling and CLI integration without Docker."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentinfer.agentbench.benchkit.cli import main
from agentinfer.agentbench.benchkit.common import atomic_write_json
from agentinfer.agentbench.benchkit.evaluation import runner
from agentinfer.agentbench.benchkit.evaluation.compare import compare_correctness, load_evaluation
from agentinfer.agentbench.benchkit.evaluation.config import EvaluationConfig, load_evaluation_config
from agentinfer.agentbench.benchkit.evaluation.inputs import digest, prepare_inputs
from agentinfer.agentcache.entrypoints.bench import _normalize_delegated_argv, main as benchmark_main
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
        iid = prediction["instance_id"]
        patch = (source / "tasks" / iid / "model.patch").read_bytes()
        assert prediction["model_patch"].encode() == patch
        assert "REFERENCE" not in prediction["model_patch"]
        for name in ("result.json", "prompt.txt"):
            relative = f"tasks/{iid}/{name}"
            assert prepared["source_hashes"][relative] == digest((source / relative).read_bytes())
        assert prepared["generation"]["prompts"][iid] == prepared["source_hashes"][f"tasks/{iid}/prompt.txt"]


def test_evaluation_preserves_patch_bytes_through_harness(source, fake_harness):
    task = source / "tasks/astropy__astropy-0"
    # Exercise JSON escaping, UTF-8, mixed newlines and a missing final newline.
    patch = '--- a/file.py\r\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+print("中文", "\\\\path")\t '.encode()
    (task / "model.patch").write_bytes(patch)
    update(task / "result.json", lambda d: d.update(patch_bytes=len(patch)))
    originals = {p.parent.name: p.read_bytes() for p in (source / "tasks").glob("*/model.patch")}

    output, code = runner.evaluate(source, config())
    _, manifest, results = load_evaluation(output / "correctness.json")
    assert code == 0
    predictions = {
        p["instance_id"]: p["model_patch"].encode()
        for p in map(json.loads, (output / "inputs/predictions.jsonl").read_text().splitlines())
    }
    for iid, result in results.items():
        original = originals[iid]
        frozen = f"inputs/patches/{iid}.patch"
        harness_patch = str(Path(result["report_path"]).parent / "patch.diff")
        assert (source / "tasks" / iid / "model.patch").read_bytes() == original
        assert (output / frozen).read_bytes() == original
        assert predictions[iid] == original
        assert (output / harness_patch).read_bytes() == original
        expected = hashlib.sha256(original).hexdigest()
        assert result["patch_sha256"] == expected
        assert manifest["input_hashes"][frozen] == expected
        assert manifest["output_hashes"][harness_patch] == expected


@pytest.mark.parametrize("change", ["count", "duplicate", "status", "base_commit", "running", "directory"])
def test_inconsistent_source_rejected(source, change):
    if change == "count":
        update(source / "summary.json", lambda d: d["tasks"].update(completed=1))
    elif change == "duplicate":
        update(source / "task_index.json", lambda d: d["tasks"].append(d["tasks"][0]))
    elif change == "status":
        update(source / "tasks/astropy__astropy-0/result.json", lambda d: d.update(outcome="failed"))
    elif change == "running":
        update(source / "manifest.json", lambda d: d.update(status="running"))
    elif change == "directory":
        first = source / "tasks/astropy__astropy-0/result.json"
        second = source / "tasks/astropy__astropy-1/result.json"
        first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
        first.write_bytes(second_bytes)
        second.write_bytes(first_bytes)
    else:
        (source / "tasks/astropy__astropy-0/prompt.txt").write_text("Base commit: wrong")
    with pytest.raises(ValueError):
        prepare_inputs(source)


@pytest.mark.parametrize("fault", ["missing", "invalid_utf8"])
def test_invalid_patch_is_error_and_does_not_shrink_total(source, fake_harness, fault):
    task = source / "tasks/astropy__astropy-0"
    if fault == "missing":
        (task / "model.patch").unlink()
    else:
        (task / "model.patch").write_bytes(b"\xff")
        update(task / "result.json", lambda d: d.update(patch_bytes=1))
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


@pytest.mark.parametrize("resolved", [True, False])
@pytest.mark.parametrize(
    "original,replacement",
    [(b"diff 0", b"diff 9"), (b"\r\n", b"\n")],
    ids=["same-size-content", "line-endings"],
)
def test_harness_patch_mismatch_is_error(source, fake_harness, monkeypatch, resolved, original, replacement):
    iid = "astropy__astropy-0"
    fake_harness[iid] = resolved
    execute = runner.run_harness

    def execute_with_changed_patch(command, output, timeout, cleanup_timeout):
        code = execute(command, output, timeout, cleanup_timeout)
        run_id = command[command.index("--run_id") + 1]
        path = output / "raw/logs/run_evaluation" / run_id / "org__model" / iid / "patch.diff"
        patch = path.read_bytes()
        changed = patch.replace(original, replacement)
        assert changed != patch
        path.write_bytes(changed)
        return code

    monkeypatch.setattr(runner, "run_harness", execute_with_changed_patch)
    output, code = runner.evaluate(source, config())
    summary, manifest, results = load_evaluation(output / "correctness.json")
    assert code == 1 and manifest["returncode"] == 0 and manifest["status"] == "failed"
    assert summary.counts.model_dump() == {"total": 2, "resolved": 0, "unresolved": 1, "empty_patch": 0, "error": 1}
    assert results[iid]["status"] == "error"
    assert results[iid]["reason"] == "invalid_or_missing_report: report patch does not match submitted patch"


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
    assert "max_regressions" not in report
    assert (
        main(
            [
                "compare-correctness",
                "--baseline",
                str(baseline / "correctness.json"),
                "--candidate",
                str(candidate / "correctness.json"),
            ]
        )
        == 1
    )


def test_compare_cli_rejects_removed_regression_threshold():
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "compare-correctness",
                "--baseline",
                "baseline.json",
                "--candidate",
                "candidate.json",
                "--max-regressions",
                "1",
            ]
        )
    assert exc.value.code == 2


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


@pytest.mark.parametrize("count", [0, 500])
def test_container_cleanup_batches_only_this_run(monkeypatch, count):
    names = [f"sweb.eval.task-{i}.eval-current" for i in range(count)]
    listed = [*names, "sweb.eval.other.eval-other", "unrelated.eval-current", "sweb.eval.other.eval-current-extra"]
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="\n".join(listed), stderr="")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=iter([100, 103]).__next__))
    assert runner.cleanup_containers("eval-current", 30) == []
    assert len(calls) == (2 if names else 1)
    assert calls[0][1]["timeout"] == 30
    if names:
        assert calls[1][0] == ["docker", "rm", "-f", *names]
        assert calls[1][1]["timeout"] == 27


@pytest.mark.parametrize("fault", ["partial_failure", "timeout", "list_failure"])
def test_container_cleanup_reports_failure(monkeypatch, fault):
    names = ["sweb.eval.first.eval-current", "sweb.eval.second.eval-current"]

    def execute(command, **kwargs):
        if command[1] == "ps":
            if fault == "list_failure":
                raise subprocess.CalledProcessError(1, command, stderr="daemon unavailable")
            return subprocess.CompletedProcess(command, 0, stdout="\n".join(names), stderr="")
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 1, stdout=names[0], stderr=f"cannot remove {names[1]}")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    if fault == "partial_failure":
        errors = runner.cleanup_containers("eval-current", 30)
        assert len(errors) == 1
        assert all(name in errors[0] for name in names)
        assert "exit 1" in errors[0] and f"cannot remove {names[1]}" in errors[0]
    else:
        exception = subprocess.TimeoutExpired if fault == "timeout" else subprocess.CalledProcessError
        with pytest.raises(exception):
            runner.cleanup_containers("eval-current", 30)


@pytest.mark.parametrize("fault", ["partial_failure", "timeout"])
def test_cleanup_failure_keeps_scores_and_fails_evaluation(source, fake_harness, monkeypatch, fault):
    def cleanup(*args):
        if fault == "timeout":
            raise subprocess.TimeoutExpired("docker rm", 30)
        return ["docker rm: exit 1: cannot remove container"]

    monkeypatch.setattr(runner, "cleanup_containers", cleanup)
    output, code = runner.evaluate(source, config())
    summary, manifest, _ = load_evaluation(output / "correctness.json")
    assert code == 1 and manifest["status"] == "failed" and manifest["finished_at"]
    assert manifest["errors"] and "docker rm" in manifest["errors"][0]
    assert summary.counts.model_dump() == {"total": 2, "resolved": 1, "unresolved": 1, "empty_patch": 0, "error": 0}


def test_version_mismatch_fails_before_harness_import(monkeypatch, tmp_path):
    metadata = tmp_path / "swebench-5.0.2.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: swebench\nVersion: 5.0.2\n")
    (tmp_path / "swebench.py").write_text("raise AssertionError('harness imported before version check')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(ValueError, match="expected SWE-bench 4.1.0, got 5.0.2"):
        runner.harness_identity(config(), tmp_path / "dataset.json", 5)


def test_cli_evaluation_without_config_uses_current_python(source, fake_harness, monkeypatch, tmp_path, capsys):
    seen = []
    probe = runner.harness_identity

    def capture(config, *args):
        seen.append(config)
        return probe(config, *args)

    monkeypatch.setattr(runner, "harness_identity", capture)
    output = tmp_path / "without-config"
    assert main(["evaluate", "--result-path", str(source), "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["total"] == 2
    assert seen[0].harness_python == Path(sys.executable)
    assert seen[0].expected_harness_version == "4.1.0"
    assert seen[0].max_workers == 2


@pytest.mark.parametrize("outcome", ["unresolved", "report_error", "no_completed"])
def test_run_with_correctness_preserves_source_and_completed_scope(
    source, fake_harness, monkeypatch, capsys, caplog, outcome
):
    update(source / "manifest.json", lambda d: d.update(status="completed"))
    update(source / "summary.json", lambda d: d["lifecycle"].update(status="completed"))
    fake_harness["astropy__astropy-0"] = "missing" if outcome == "report_error" else False
    if outcome == "no_completed":
        for path in (source / "tasks").glob("*/result.json"):
            update(path, lambda d: d.update(outcome="failed"))
        index = json.loads((source / "task_index.json").read_bytes())
        for task in index["tasks"]:
            task["outcome"] = "failed"
        atomic_write_json(source / "task_index.json", index)
        update(source / "summary.json", lambda d: d["tasks"].update(completed=0, failed=3))
    originals = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    run = AsyncMock(return_value=source)
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.run_benchmark", run)

    code = benchmark_main(["vllm", "bench", "serve", "--agentinfer", "run", "--with-correctness"])

    run.assert_awaited_once()
    assert code == (0 if outcome == "unresolved" else 1)
    assert all(path.read_bytes() == data for path, data in originals.items())
    if outcome == "no_completed":
        assert not (source / "evaluations").exists()
        assert "no_completed_tasks" in caplog.text
        return
    outputs = list((source / "evaluations").glob("*/correctness.json"))
    assert len(outputs) == 1
    summary, manifest, results = load_evaluation(outputs[0])
    assert json.loads(capsys.readouterr().out) == summary.model_dump()
    assert summary.counts.model_dump() == {
        "total": 2,
        "resolved": 0,
        "unresolved": 1 if outcome == "report_error" else 2,
        "empty_patch": 0,
        "error": 1 if outcome == "report_error" else 0,
    }
    assert set(results) == set(manifest["instance_ids"]) == {"astropy__astropy-0", "astropy__astropy-1"}


def test_optional_config_defaults_and_relative_interpreter(tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text("harness_python: evaluator/bin/python\nmax_workers: 3\n")
    loaded = load_evaluation_config(config)
    assert loaded.harness_python == tmp_path / "evaluator/bin/python"
    assert loaded.expected_harness_version == "4.1.0"
    assert loaded.max_workers == 3
    config.write_text("")
    assert load_evaluation_config(config) == load_evaluation_config()


def test_optional_config_still_rejects_unknown_fields(tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text("unknown_setting: true\n")
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--result-path", str(tmp_path), "--config", str(config)])
    assert exc.value.code == 2


def test_image_identity_capture_batches_verified_scale_and_deduplicates_aliases(monkeypatch):
    images = {f"task-{i}": f"repo/image-{i}:latest" for i in range(500)}
    images["same-image"] = images["task-0"]
    calls = []

    def inspect(command, **kwargs):
        calls.append((command, kwargs))
        records = [
            {"Id": f"sha256:{i}", "RepoTags": [f"repo/image-{i}:latest"], "RepoDigests": [f"repo/image-{i}@sha256:{i}"]}
            for i in reversed(range(500))
        ]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(records), stderr="")

    monkeypatch.setattr(runner.subprocess, "run", inspect)
    result = runner.image_identities(images, 30)
    assert len(calls) == 1 and len(calls[0][0]) == 503
    assert calls[0][1]["timeout"] == 30
    assert result["task-499"]["id"] == "sha256:499"
    assert result["task-0"] == result["same-image"]


def test_image_identity_capture_keeps_partial_results_without_position_assumptions(monkeypatch):
    def inspect(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps([{"Id": "sha256:second", "RepoTags": ["second:latest"], "RepoDigests": None}]),
            stderr="No such image: first:latest",
        )

    monkeypatch.setattr(runner.subprocess, "run", inspect)
    result = runner.image_identities({"a": "first:latest", "b": "second:latest"}, 30)
    assert result == {
        "a": {"name": "first:latest", "id": None},
        "b": {"name": "second:latest", "id": "sha256:second", "digests": []},
    }


@pytest.mark.parametrize("stdout", ["not-json", "{}", "[]", '[{"RepoTags": ["image"]}]'])
def test_image_identity_capture_rejects_invalid_success_responses(monkeypatch, stdout):
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=stdout, stderr="")
    )
    with pytest.raises(ValueError):
        runner.image_identities({"task": "image"}, 30)


def test_image_identity_capture_skips_empty_input_and_propagates_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    assert runner.image_identities({}, 30) == {}
    with pytest.raises(subprocess.TimeoutExpired):
        runner.image_identities({"task": "image"}, 30)


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
