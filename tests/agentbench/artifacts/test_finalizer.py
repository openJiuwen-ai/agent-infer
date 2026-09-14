# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify the L2 finalization (task index -> summary -> manifest) assembly."""

import json
from datetime import timedelta
from pathlib import Path

from agentinfer.agentbench.benchkit.artifacts import build_run_manifest
from agentinfer.agentbench.benchkit.artifacts.finalizer import FinalizationContext, RunFinalizer
from agentinfer.agentbench.benchkit.common import atomic_write_json
from agentinfer.agentbench.benchkit.metrics.schema import EvidenceCapture
from agentinfer.agentbench.request_proxy.request_trace import RequestFact


def _manifest(run_id: str):
    return build_run_manifest(run_id, {})


def _facts(session_id: str = "s0") -> list[RequestFact]:
    return [
        RequestFact(
            "1",
            "run",
            "r0",
            session_id,
            "lead",
            "lead",
            "2026-08-13T00:00:00Z",
            "2026-08-13T00:00:01Z",
            "success",
            200,
            1.0,
            0.2,
            10,
            4,
            0,
            0,
            "http://backend",
            None,
        )
    ]


def _run_dir(tmp_path: Path, *, with_tasks: bool = True) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    if with_tasks:
        task_dir = run_dir / "tasks" / "instance"
        task_dir.mkdir(parents=True)
        result = {
            "schema_version": "1",
            "instance_id": "instance",
            "session_id": "s0",
            "agent": {"type": "claude", "profile": "single"},
            "outcome": "completed",
            "termination_reason": None,
            "task_position": 0,
            "started_at": "2026-08-13T00:00:00+00:00",
            "finished_at": "2026-08-13T00:01:00+00:00",
            "duration_seconds": 1.0,
            "has_patch": False,
            "patch_bytes": 0,
            "topology": {"session_id": "s0", "agent_ids": [], "agent_roles": {}},
            "error": None,
        }
        atomic_write_json(task_dir / "result.json", result)
    return run_dir


def test_finalize_writes_index_then_summary_then_manifest(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    manifest = _manifest("run")
    finished_at = manifest.created_at + timedelta(seconds=2)

    result = RunFinalizer(
        FinalizationContext(
            run_id="run",
            output_dir=run_dir,
            manifest=manifest,
            results=[],
            facts=_facts(),
            vllm_start=None,
            vllm_end=None,
            captures=[EvidenceCapture("request_trace", run_dir / "requests.jsonl", True, None, {})],
            correctness=None,
            close_result=None,
            run_exception=None,
            prior_finalization_errors=[],
            run_wall_time_seconds=2.0,
            finished_at=finished_at,
        )
    ).finalize()

    assert result.status == "completed"
    assert result.finalization_errors == ()
    index = json.loads((run_dir / "task_index.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert index["tasks"][0]["instance_id"] == "instance"
    assert summary["lifecycle"]["status"] == "completed"
    assert manifest_data["status"] == "completed"


def test_index_failure_marks_summary_and_manifest_failed(tmp_path: Path, monkeypatch) -> None:
    run_dir = _run_dir(tmp_path, with_tasks=True)

    real_write = atomic_write_json

    def fail_index(path: Path, value: object) -> None:
        if path.name == "task_index.json":
            raise OSError("index write exploded")
        real_write(path, value)

    monkeypatch.setattr("agentinfer.agentbench.benchkit.artifacts.finalizer.atomic_write_json", fail_index)
    manifest = _manifest("run")
    finished_at = manifest.created_at + timedelta(seconds=2)

    result = RunFinalizer(
        FinalizationContext(
            run_id="run",
            output_dir=run_dir,
            manifest=manifest,
            results=[],
            facts=_facts(),
            vllm_start=None,
            vllm_end=None,
            captures=[],
            correctness=None,
            close_result=None,
            run_exception=None,
            prior_finalization_errors=[],
            run_wall_time_seconds=2.0,
            finished_at=finished_at,
        )
    ).finalize()

    assert result.status == "failed"
    assert any("index write exploded" in e for e in result.finalization_errors)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # the core consistency: summary and manifest agree on failed status
    assert summary["lifecycle"]["status"] == "failed"
    assert manifest_data["status"] == "failed"


def test_summary_failure_still_finalizes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = _run_dir(tmp_path)
    real_write = atomic_write_json

    def fail_summary(path: Path, value: object) -> None:
        if path.name == "summary.json":
            raise OSError("summary write exploded")
        real_write(path, value)

    monkeypatch.setattr("agentinfer.agentbench.benchkit.artifacts.finalizer.atomic_write_json", fail_summary)
    manifest = _manifest("run")
    finished_at = manifest.created_at + timedelta(seconds=2)

    result = RunFinalizer(
        FinalizationContext(
            run_id="run",
            output_dir=run_dir,
            manifest=manifest,
            results=[],
            facts=_facts(),
            vllm_start=None,
            vllm_end=None,
            captures=[],
            correctness=None,
            close_result=None,
            run_exception=None,
            prior_finalization_errors=[],
            run_wall_time_seconds=2.0,
            finished_at=finished_at,
        )
    ).finalize()

    assert result.status == "failed"
    assert any("summary write exploded" in e for e in result.finalization_errors)
    manifest_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["status"] == "failed"
    assert any("summary write exploded" in (row["reason"] or "") for row in manifest_data["evidence"])
    # task index still written despite summary failure
    assert (run_dir / "task_index.json").exists()


def test_summary_build_failure_still_finalizes_manifest(tmp_path: Path, monkeypatch) -> None:
    """A summary *build* failure (aggregation/build_run_summary raising) must
    be recorded as a finalization error and must not skip manifest finalization.
    Mirrors main's summary_build guard: index -> summary(build) -> summary(write)
    -> manifest each independently guarded.
    """
    run_dir = _run_dir(tmp_path)

    def fail_build(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("summary build exploded")

    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.artifacts.finalizer.build_run_summary",
        fail_build,
    )
    manifest = _manifest("run")
    finished_at = manifest.created_at + timedelta(seconds=2)

    result = RunFinalizer(
        FinalizationContext(
            run_id="run",
            output_dir=run_dir,
            manifest=manifest,
            results=[],
            facts=_facts(),
            vllm_start=None,
            vllm_end=None,
            captures=[],
            correctness=None,
            close_result=None,
            run_exception=None,
            prior_finalization_errors=[],
            run_wall_time_seconds=2.0,
            finished_at=finished_at,
        )
    ).finalize()

    # (a) the build error lands in finalization_errors
    assert result.status == "failed"
    assert any("summary_build" in e and "summary build exploded" in e for e in result.finalization_errors)
    # (b) manifest is still written and reflects the failure
    assert (run_dir / "manifest.json").exists()
    manifest_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["status"] == "failed"
    assert any(
        row["source"] == "summary_build" and "summary build exploded" in (row["reason"] or "")
        for row in manifest_data["evidence"]
    )
    # summary.json is not written (build failed before the write)
    assert not (run_dir / "summary.json").exists()
    # task index still written despite the summary build failure
    assert (run_dir / "task_index.json").exists()


def test_vllm_aggregation_failure_is_isolated(tmp_path: Path, monkeypatch) -> None:
    """A vLLM aggregation failure must not prevent manifest finalization."""
    run_dir = _run_dir(tmp_path)

    def fail_vllm(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("vllm aggregation exploded")

    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.artifacts.finalizer.aggregate_vllm_metrics",
        fail_vllm,
    )
    manifest = _manifest("run")
    finished_at = manifest.created_at + timedelta(seconds=2)

    result = RunFinalizer(
        FinalizationContext(
            run_id="run",
            output_dir=run_dir,
            manifest=manifest,
            results=[],
            facts=_facts(),
            vllm_start=None,
            vllm_end=None,
            captures=[],
            correctness=None,
            close_result=None,
            run_exception=None,
            prior_finalization_errors=[],
            run_wall_time_seconds=2.0,
            finished_at=finished_at,
        )
    ).finalize()

    assert result.status == "failed"
    assert any(
        "vllm_aggregation" in error and "vllm aggregation exploded" in error for error in result.finalization_errors
    )
    assert not (run_dir / "summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    manifest_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["status"] == "failed"
    assert any(row["source"] == "vllm_aggregation" for row in manifest_data["evidence"])
    assert (run_dir / "task_index.json").exists()


def test_empty_results_produces_empty_index(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, with_tasks=False)
    manifest = _manifest("run")
    finished_at = manifest.created_at + timedelta(seconds=2)

    result = RunFinalizer(
        FinalizationContext(
            run_id="run",
            output_dir=run_dir,
            manifest=manifest,
            results=[],
            facts=_facts(),
            vllm_start=None,
            vllm_end=None,
            captures=[],
            correctness=None,
            close_result=None,
            run_exception=None,
            prior_finalization_errors=[],
            run_wall_time_seconds=2.0,
            finished_at=finished_at,
        )
    ).finalize()

    assert result.status == "completed"
    index = json.loads((run_dir / "task_index.json").read_text(encoding="utf-8"))
    assert index["tasks"] == []
    assert index["failure_summary"]["total_failed"] == 0


def test_skips_summary_when_facts_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    manifest = _manifest("run")
    finished_at = manifest.created_at + timedelta(seconds=2)

    RunFinalizer(
        FinalizationContext(
            run_id="run",
            output_dir=run_dir,
            manifest=manifest,
            results=[],
            facts=None,  # facts collection failed -> no summary
            vllm_start=None,
            vllm_end=None,
            captures=[],
            correctness=None,
            close_result=None,
            run_exception=None,
            prior_finalization_errors=[],
            run_wall_time_seconds=2.0,
            finished_at=finished_at,
        )
    ).finalize()

    # no facts -> summary skipped, but index + manifest still written
    assert (run_dir / "task_index.json").exists()
    assert not (run_dir / "summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    manifest_data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["status"] == "completed"
