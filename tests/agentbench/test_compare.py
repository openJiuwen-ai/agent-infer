# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.artifacts import build_run_manifest, build_run_summary, finalize_run_manifest
from agentinfer.agentbench.benchkit.compare import compare, load_summary
from agentinfer.agentbench.benchkit.metrics.request import LatencyStats, RequestMetrics
from agentinfer.agentbench.benchkit.metrics.schema import SourceHealth
from agentinfer.agentbench.benchkit.metrics.task import TaskMetrics
from agentinfer.agentbench.benchkit.metrics.vllm import VllmMetrics


def write_run(
    run_dir: Path,
    *,
    completed: int = 1,
    failed: int = 0,
    status: str = "completed",
    duration: float = 10.0,
    requests: int = 2,
    cache_creation: int | None = 0,
    cached_input: int | None = 5,
    router_events: dict[str, int] | None = None,
    vllm_available: bool = False,
    execution: dict | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    task_metrics = TaskMetrics(
        completed,
        failed,
        completed,
        {"mean": duration, "p50": duration, "p95": duration, "p99": duration},
    )
    request_metrics = RequestMetrics(
        requests,
        requests,
        0,
        10,
        4,
        cache_creation,
        cached_input,
        0.5 if cache_creation is not None and cached_input is not None else None,
        LatencyStats(1.0, 1.0, 1.0, 1.0),
        LatencyStats(0.2, 0.2, 0.2, 0.2),
    )
    vllm = VllmMetrics(
        vllm_available,
        None if vllm_available else "missing",
        False,
        False,
        {"queue_time": {"mean": 0.4, "count": 2, "sum": 0.8}} if vllm_available else {},
        {},
        {},
        None,
        None,
    )
    health = SourceHealth(1, 0, 1, (), {})
    summary = build_run_summary(
        run_dir.name,
        task_metrics,
        request_metrics,
        vllm,
        {"available": False, "reason": "artifact missing", "metadata": {}},
        health,
        {"status": status, "error": "Replay incomplete" if status == "failed" else None, "proxy_close": None},
        duration,
        execution=execution,
    )
    summary_data = summary.to_dict()
    if router_events is not None:
        # Compare continues to accept historical artifacts with Router metrics,
        # even though new AgentBench runs no longer produce them.
        summary_data["router"] = {"applicable": True, "events": router_events}
    (run_dir / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")
    manifest = build_run_manifest(run_dir.name, {})
    manifest = finalize_run_manifest(
        manifest,
        (),
        status=status,
        finished_at=manifest.created_at + timedelta(seconds=duration),
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")


def test_compare_finalized_artifacts_and_router_schema(tmp_path: Path) -> None:
    write_run(tmp_path / "baseline", duration=12)
    write_run(tmp_path / "candidate", duration=8, router_events={"admit": 2})
    result = json.loads(compare(tmp_path / "baseline", tmp_path / "candidate", as_json=True))
    assert result["metrics"]["mean_task_duration_seconds"]["delta_absolute"] == -4
    assert result["metrics"]["run_wall_time_seconds"]["baseline"] == 12
    assert result["metrics"]["run_wall_time_seconds"]["candidate"] == 8
    assert result["metrics"]["run_wall_time_seconds"]["delta_absolute"] == -4
    assert result["metrics"]["request_throughput_per_second"]["baseline"] == pytest.approx(2 / 12)
    assert result["metrics"]["request_throughput_per_second"]["candidate"] == pytest.approx(2 / 8)
    assert result["metrics"]["input_token_throughput_per_second"]["baseline"] == pytest.approx(10 / 12)
    assert result["metrics"]["output_token_throughput_per_second"]["candidate"] == pytest.approx(4 / 8)
    assert result["metrics"]["p99_task_duration_seconds"]["delta_absolute"] == -4
    assert result["metrics"]["cache_creation_input_tokens"]["baseline"] == 0
    assert result["metrics"]["latency_p99_seconds"]["baseline"] == 1.0
    assert result["metrics"]["ttft_p99_seconds"]["baseline"] == 0.2
    assert result["metadata"]["baseline_router"] == {"applicable": False, "events": {}}
    assert result["metadata"]["candidate_router"] == {"applicable": True, "events": {"admit": 2}}


def test_compare_accepts_cli_metadata_in_finalized_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["cli"] = {
        "entrypoint": "vllm bench serve --agentinfer",
        "argv": ["bench", "serve", "--agentinfer", "run"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    loaded = load_summary(run_dir)

    assert loaded["cli"] == summary["cli"]


def test_compare_falls_back_for_legacy_summary_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    write_run(run_dir, duration=8, requests=2)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("execution")
    for field in (
        "run_wall_time_seconds",
        "request_throughput_per_second",
        "input_token_throughput_per_second",
        "output_token_throughput_per_second",
    ):
        summary.pop(field)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="manifest fallback is deprecated"):
        loaded = load_summary(run_dir)

    assert loaded["run_wall_time_seconds"] == 8
    assert loaded["request_throughput_per_second"] == pytest.approx(2 / 8)
    assert loaded["execution"] is None


def test_compare_excludes_null_cache_creation_samples(tmp_path: Path) -> None:
    baseline = []
    for index, value in enumerate((None, 4)):
        path = tmp_path / f"baseline-{index}"
        write_run(path, cache_creation=value)
        baseline.append(path)
    candidate = tmp_path / "candidate"
    write_run(candidate, cache_creation=None)

    result = json.loads(compare(baseline, candidate, as_json=True))
    metric = result["metrics"]["cache_creation_input_tokens"]

    assert metric["baseline"] == 4
    assert metric["n_baseline"] == 1
    assert metric["candidate"] is None
    assert metric["n_candidate"] == 0


def test_compare_rejects_legacy_handwritten_shape(tmp_path: Path) -> None:
    path = tmp_path / "legacy"
    path.mkdir()
    (path / "summary.json").write_text('{"wall_seconds": 1}', encoding="utf-8")
    (path / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="finalized schemas"):
        load_summary(path)


def test_compare_rejects_invalid_artifact_identity_and_confidence(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_run(baseline)
    write_run(candidate)

    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    summary["run_id"] = "other-run"
    (baseline / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="different runs"):
        compare(baseline, candidate)
    with pytest.raises(ValueError, match="confidence=0.95"):
        compare(candidate, candidate, confidence=0.9)


def test_compare_accepts_failed_run_without_replay_metadata(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_run(baseline, status="failed", failed=1)
    write_run(candidate)

    report = json.loads(compare(baseline, candidate, as_json=True))
    run = report["metadata"]["baseline_runs"][0]
    assert report["metadata"]["diagnostic_only"] is True
    assert run["status"] == "failed"
    assert run["error"] == "Replay incomplete"
    assert run["planned_requests"] is None
    assert run["request_completion_ratio"] is None
    assert report["metrics"]["failed_tasks"]["baseline"] == 1


@pytest.mark.parametrize("as_json", [False, True])
def test_failed_replay_comparison_through_vllm_entrypoint(tmp_path: Path, capsys, as_json: bool) -> None:
    from agentinfer.agentcache.entrypoints.cli.main import main

    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    snapshots = {}
    for path in (baseline, candidate):
        write_run(
            path,
            status="failed",
            completed=23,
            failed=1,
            requests=767,
            execution={
                "available": True,
                "reason": None,
                "metadata": {
                    "planned_requests": 768,
                    "pre_send_failed_requests": 1,
                    "dependency_skipped_requests": 0,
                    "complete_replay": False,
                    "workload_fingerprint": "same-plan",
                },
            },
        )
        (path / "replay-execution.json").write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "task_id": "t1",
                            "nodes": [
                                {
                                    "status": "failed",
                                    "source_key": "root",
                                    "runtime_request_id": "r1",
                                    "error": "independent needs 12552 tokens, target is 3136",
                                }
                            ],
                        }
                    ]
                }
            )
        )
        snapshots.update({file: file.read_bytes() for file in path.iterdir()})

    args = [
        "vllm",
        "bench",
        "serve",
        "--agentinfer",
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ]
    if as_json:
        args.append("--json")
    assert main(args) == 0
    output = capsys.readouterr().out
    if as_json:
        report = json.loads(output)
        assert report["metadata"]["diagnostic_only"] is True
        for label in ("baseline", "candidate"):
            run = report["metadata"][f"{label}_runs"][0]
            assert run["status"] == "failed"
            assert run["request_completion_ratio"] == pytest.approx(767 / 768)
            assert run["pre_send_failed_requests"] == 1
            assert run["failed_nodes"][0]["error"] == "independent needs 12552 tokens, target is 3136"
        assert report["metrics"]["failed_tasks"]["baseline"] == 1
        assert report["metrics"]["requests"]["candidate"] == 767
    else:
        assert "diagnostic comparison only" in output
        assert "status=failed" in output
        assert "successful_requests=767/768" in output
        assert "pre_send_failed_requests=1" in output
        assert "independent needs 12552 tokens, target is 3136" in output
    assert all(file.read_bytes() == content for file, content in snapshots.items())


def test_failed_runs_suppress_significance_even_with_multiple_samples(tmp_path: Path) -> None:
    baseline, candidate = [], []
    for index in range(3):
        base, cand = tmp_path / f"b{index}", tmp_path / f"c{index}"
        write_run(base, duration=100 + index)
        write_run(cand, duration=1 + index, requests=1, status="failed", failed=1)
        baseline.append(base)
        candidate.append(cand)
    report = json.loads(compare(baseline, candidate, as_json=True))
    assert report["metadata"]["diagnostic_only"] is True
    assert len(report["metadata"]["baseline_runs"]) == 3
    assert len(report["metadata"]["candidate_runs"]) == 3
    for metric in report["metrics"].values():
        assert metric["baseline_ci"] is None
        assert metric["candidate_ci"] is None
        if "delta_absolute" in metric:
            assert metric["delta_ci"] is None
            assert metric["significant"] is None
    assert report["metrics"]["requests"]["delta_absolute"] == -1
    assert report["metadata"]["candidate_runs"][0]["planned_requests"] is None
    assert report["metadata"]["candidate_runs"][0]["request_completion_ratio"] is None


def test_successful_comparison_keeps_statistical_reporting(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_run(baseline)
    write_run(candidate)
    report = json.loads(compare(baseline, candidate, as_json=True))
    assert report["metadata"]["diagnostic_only"] is False
    assert not any("Failed runs included" in warning for warning in report["warnings"])
    assert report["metrics"]["requests"]["baseline_ci"] == [2, 2]
    assert report["metrics"]["requests"]["delta_ci"] == [0, 0]
    assert report["metrics"]["requests"]["significant"] is False


@pytest.mark.parametrize(
    "problem,expected",
    [
        ("running", "still running"),
        ("unfinished", "no finish timestamp"),
        ("identity", "identify different runs"),
        ("lifecycle", "inconsistent lifecycle"),
        ("schema", "finalized schemas"),
        ("missing_summary", "summary.json not found"),
    ],
)
def test_compare_still_validates_finalization_and_artifacts(tmp_path: Path, problem: str, expected: str) -> None:
    run = tmp_path / "run"
    write_run(run, status="failed", failed=1)
    manifest_path, summary_path = run / "manifest.json", run / "summary.json"
    manifest, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    if problem == "running":
        manifest["status"] = "running"
    elif problem == "unfinished":
        manifest["finished_at"] = None
    elif problem == "identity":
        summary["run_id"] = "other"
    elif problem == "lifecycle":
        summary["lifecycle"]["status"] = "completed"
    elif problem == "schema":
        summary["requests"] = None
    manifest_path.write_text(json.dumps(manifest))
    summary_path.write_text(json.dumps(summary))
    if problem == "missing_summary":
        summary_path.unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        load_summary(run)


@pytest.mark.parametrize("contents", ["not json", '{"tasks": [{"nodes": [null]}]}'])
def test_failed_comparison_reports_unreadable_optional_failure_details(tmp_path: Path, contents: str) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_run(baseline, status="failed")
    write_run(candidate)
    (baseline / "replay-execution.json").write_text(contents)
    report = json.loads(compare(baseline, candidate, as_json=True))
    assert report["metrics"]["requests"]["baseline"] == 2
    assert report["metadata"]["baseline_runs"][0]["error"] == "Replay incomplete"
    assert any("could not read" in warning for warning in report["warnings"])


def test_compare_accepts_same_mode_and_optional_router_metrics(tmp_path: Path) -> None:
    write_run(tmp_path / "first-baseline")
    write_run(tmp_path / "second-baseline")
    baseline_result = json.loads(compare(tmp_path / "first-baseline", tmp_path / "second-baseline", as_json=True))

    write_run(tmp_path / "first-candidate", router_events={"admit": 1})
    write_run(tmp_path / "second-candidate", router_events={"admit": 2})
    candidate_result = json.loads(compare(tmp_path / "first-candidate", tmp_path / "second-candidate", as_json=True))

    assert baseline_result["metadata"]["baseline_router"]["applicable"] is False
    assert baseline_result["metadata"]["candidate_router"]["applicable"] is False
    assert candidate_result["metadata"]["baseline_router"]["applicable"] is True
    assert candidate_result["metadata"]["candidate_router"]["applicable"] is True


def test_compare_rejects_empty_run_lists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline"):
        compare([], tmp_path / "candidate")
    with pytest.raises(ValueError, match="candidate"):
        compare(tmp_path / "baseline", [])


def test_compare_multi_run_variance(tmp_path: Path) -> None:
    baseline = []
    candidate = []
    for index, value in enumerate((10.0, 11.0, 12.0)):
        path = tmp_path / f"b{index}"
        write_run(path, duration=value)
        baseline.append(path)
    for index, value in enumerate((6.0, 7.0, 8.0)):
        path = tmp_path / f"c{index}"
        write_run(path, duration=value)
        candidate.append(path)
    metric = json.loads(compare(baseline, candidate, as_json=True))["metrics"]["mean_task_duration_seconds"]
    assert metric["n_baseline"] == metric["n_candidate"] == 3
    assert metric["significant"] is True
    assert metric["low_power"] is False


def test_compare_includes_vllm_only_when_available(tmp_path: Path) -> None:
    write_run(tmp_path / "baseline", vllm_available=True)
    write_run(tmp_path / "candidate", vllm_available=True)
    result = json.loads(compare(tmp_path / "baseline", tmp_path / "candidate", as_json=True))
    assert result["metrics"]["vllm_queue_time_mean_seconds"]["baseline"] == 0.4


def test_compare_excludes_metrics_when_execution_is_unavailable(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_run(baseline)
    write_run(candidate)
    summary_path = baseline / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["execution"] = {
        "available": False,
        "reason": "planner_only",
        "metadata": {},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = json.loads(compare(baseline, candidate, as_json=True))

    assert result["metrics"]["completed_tasks"]["baseline"] is None
    assert result["metrics"]["completed_tasks"]["n_baseline"] == 0
    assert result["metrics"]["completed_tasks"]["candidate"] == 1
    assert result["metrics"]["run_wall_time_seconds"]["baseline"] == 10
    assert result["metadata"]["baseline_execution"]["available"] is False
    assert any("baseline execution metrics unavailable" in warning for warning in result["warnings"])


def test_compare_treats_malformed_cold_evidence_as_unconfirmed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_run(baseline)
    write_run(candidate)
    for run_dir in (baseline, candidate):
        evidence = run_dir / "evidence"
        evidence.mkdir()
        (evidence / "vllm_metrics_start.prom").write_text(
            "vllm:prefix_cache_queries_total 0\nvllm:prefix_cache_queries_total 1\n",
            encoding="utf-8",
        )

    result = json.loads(compare(baseline, candidate, as_json=True))

    assert result["metadata"]["baseline_cold_confirmed"] is False
    assert result["metadata"]["candidate_cold_confirmed"] is False


def test_compare_requires_prefix_query_evidence_for_cold_confirmation(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    write_run(baseline)
    write_run(candidate)
    for run_dir in (baseline, candidate):
        evidence = run_dir / "evidence"
        evidence.mkdir()
        (evidence / "vllm_metrics_start.prom").write_text(
            "# TYPE vllm:prompt_tokens_total counter\nvllm:prompt_tokens_total 0\n",
            encoding="utf-8",
        )

    result = json.loads(compare(baseline, candidate, as_json=True))

    assert result["metadata"]["baseline_cold_confirmed"] is False
    assert result["metadata"]["candidate_cold_confirmed"] is False
