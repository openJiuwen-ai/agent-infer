# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.agents.contracts import AgentRunResult
from agentinfer.agentbench.agents.outcomes import AgentRunOutcome
from agentinfer.agentbench.benchkit.config import AgentBenchConfig
from agentinfer.agentbench.benchkit.dataset import Task
from agentinfer.agentbench.benchkit.runner import RunContext, _run_single_task, check_preflight
from agentinfer.agentbench.benchkit.session_registration import SessionRegistrationResult
from agentinfer.agentbench.request_proxy import RequestProxyCloseResult


def _context(tmp_path: Path, *, router: bool) -> RunContext:
    config = AgentBenchConfig.model_validate(
        {
            "experiment": {"result_dir": str(tmp_path)},
            "dataset": {"cache_dir": str(tmp_path / "cache")},
            "router": {"enabled": router, "base_url": "http://router" if router else None},
        }
    )
    return RunContext(config, tmp_path, tmp_path / "requests.jsonl", ())


def _task() -> Task:
    return Task("instance", "owner/repo", "abc", "fix it")


def _agent_result(session_id: str) -> AgentRunResult:
    return AgentRunResult(
        outcome=AgentRunOutcome.COMPLETED,
        termination_reason=None,
        session_id=session_id,
        instance_id="instance",
        agent_type="claude",
        profile_name="single",
        duration_seconds=1,
    )


class _Progress:
    def __init__(self, **kwargs) -> None:
        self.options = kwargs
        self.updates = []
        self.postfixes = []
        self.closed = False

    def update(self, value: int) -> None:
        self.updates.append(value)

    def set_postfix(self, **values: str) -> None:
        self.postfixes.append(values)

    def close(self) -> None:
        self.closed = True


def test_baseline_skips_router_control_and_serializes_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.prepare_workspace", lambda *_args: None)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.run_agent",
        lambda request: asyncio.sleep(0, result=_agent_result(request.session_id)),
    )

    async def forbidden(*_args):
        raise AssertionError("baseline must not call Router control")

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.register_session", forbidden)
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.cleanup_session", forbidden)
    result = asyncio.run(
        _run_single_task(_context(tmp_path, router=False), _task(), "http://proxy", tmp_path / "cache")
    )
    assert result.outcome == AgentRunOutcome.COMPLETED
    assert (tmp_path / "tasks" / "instance" / "result.json").exists()


def test_candidate_cleans_up_after_agent_exception_and_writes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.prepare_workspace", lambda *_args: None)

    async def register(_url, session_id, _timeout):
        calls.append("register")
        return SessionRegistrationResult("register", session_id, True, 200, 0, None)

    async def cleanup(_url, session_id, _timeout):
        calls.append("cleanup")
        return SessionRegistrationResult("cleanup", session_id, True, 200, 0, None)

    async def fail(_request):
        calls.append("agent")
        raise RuntimeError("boom")

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.register_session", register)
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.cleanup_session", cleanup)
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.run_agent", fail)
    with pytest.raises(RuntimeError, match="failed in benchmark harness"):
        asyncio.run(_run_single_task(_context(tmp_path, router=True), _task(), "http://proxy", tmp_path / "cache"))
    assert calls == ["register", "agent", "cleanup"]
    payload = json.loads((tmp_path / "tasks" / "instance" / "result.json").read_text(encoding="utf-8"))
    assert payload["termination_reason"] == "harness_error"
    assert payload["error"] == {"type": "RuntimeError", "message": "boom"}


def test_candidate_cleanup_failure_is_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.prepare_workspace", lambda *_args: None)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.run_agent",
        lambda request: asyncio.sleep(0, result=_agent_result(request.session_id)),
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.register_session",
        lambda _url, session_id, _timeout: asyncio.sleep(
            0, result=SessionRegistrationResult("register", session_id, True, 200, 0, None)
        ),
    )

    async def cleanup(*_args):
        raise RuntimeError("cleanup exploded")

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.cleanup_session", cleanup)
    with pytest.raises(RuntimeError, match="Router cleanup failed"):
        asyncio.run(_run_single_task(_context(tmp_path, router=True), _task(), "http://proxy", tmp_path / "cache"))
    payload = (tmp_path / "tasks" / "instance" / "result.json").read_text(encoding="utf-8")
    assert "cleanup exploded" in payload


def _run_config(tmp_path: Path) -> AgentBenchConfig:
    return AgentBenchConfig.model_validate(
        {
            "experiment": {"result_dir": str(tmp_path / "run")},
            "dataset": {"cache_dir": str(tmp_path / "cache")},
            "router": {"enabled": False, "base_url": None},
        }
    )


def _patch_successful_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit.metrics.schema import EvidenceCapture

    async def vllm(_url):
        return EvidenceCapture("vllm", None, False, "unavailable", {})

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.check_preflight", lambda _config: asyncio.sleep(0))
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.load_tasks", lambda *_args: [])
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.capture_vllm_metrics", vllm)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.create_request_proxy_lifecycle",
        lambda *_args: SimpleNamespace(
            start=lambda: asyncio.sleep(0, result=SimpleNamespace(base_url="http://proxy")),
            wait_ready=lambda _timeout: asyncio.sleep(0),
            close=lambda: asyncio.sleep(0, result=None),
        ),
    )
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner._run_tasks", lambda *_args: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.collect_environment",
        lambda: EvidenceCapture("environment", None, False, "unavailable", {}),
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.collect_source_control",
        lambda _source_path: EvidenceCapture("source_control", None, False, "unavailable", {}),
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.load_correctness_artifact",
        lambda path: EvidenceCapture("correctness", path, False, "missing", {}),
    )


def _manifest(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))


def test_collector_failure_finalizes_manifest_and_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    _patch_successful_run(monkeypatch)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.collect_environment",
        lambda: (_ for _ in ()).throw(RuntimeError("collector exploded")),
    )

    with pytest.raises(RuntimeError, match="collector exploded"):
        asyncio.run(run_benchmark(_run_config(tmp_path)))

    manifest = _manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert any("collector exploded" in (row["reason"] or "") for row in manifest["evidence"])


def test_request_fact_parser_failure_preserves_original_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    _patch_successful_run(monkeypatch)

    async def fail_run(*_args):
        raise ValueError("original run failure")

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner._run_tasks", fail_run)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.load_request_facts",
        lambda _path: (_ for _ in ()).throw(RuntimeError("facts exploded")),
    )

    with pytest.raises(ValueError, match="original run failure") as caught:
        asyncio.run(run_benchmark(_run_config(tmp_path)))

    assert any("facts exploded" in note for note in caught.value.__notes__)
    assert _manifest(tmp_path)["status"] == "failed"


def test_summary_artifact_write_failure_still_finalizes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    _patch_successful_run(monkeypatch)
    from agentinfer.agentbench.benchkit import runner

    real_write = runner.atomic_write_json

    def fail_summary(path, value):
        if path.name == "summary.json":
            raise OSError("summary write exploded")
        real_write(path, value)

    monkeypatch.setattr(runner, "atomic_write_json", fail_summary)

    with pytest.raises(RuntimeError, match="summary write exploded"):
        asyncio.run(run_benchmark(_run_config(tmp_path)))
    manifest = _manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert any("summary write exploded" in (row["reason"] or "") for row in manifest["evidence"])


def test_registration_failure_does_not_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.prepare_workspace", lambda *_args: None)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.register_session",
        lambda _url, session_id, _timeout: asyncio.sleep(
            0, result=SessionRegistrationResult("register", session_id, False, 409, 0, "conflict")
        ),
    )

    async def forbidden(*_args):
        raise AssertionError("cleanup must not run after failed registration")

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.cleanup_session", forbidden)

    with pytest.raises(RuntimeError, match="failed in benchmark harness"):
        asyncio.run(_run_single_task(_context(tmp_path, router=True), _task(), "http://proxy", tmp_path / "cache"))

    payload = json.loads((tmp_path / "tasks" / "instance" / "result.json").read_text(encoding="utf-8"))
    assert payload["error"]["message"].endswith("conflict")


def test_preflight_requires_tmux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="tmux executable is not available: tmux"):
        asyncio.run(check_preflight(_run_config(tmp_path)))


def test_preflight_requires_agent_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.shutil.which",
        lambda name: "/usr/bin/tmux" if name == "tmux" else None,
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    with pytest.raises(RuntimeError, match="Agent executable is not available: claude"):
        asyncio.run(check_preflight(_run_config(tmp_path)))


def test_preflight_rejects_broken_agent_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    results = iter(
        (
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=1, stderr=b"unknown option --version", stdout=b""),
        )
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.subprocess.run",
        lambda *_args, **_kwargs: next(results),
    )

    with pytest.raises(
        RuntimeError,
        match="Agent executable version check failed: claude\nunknown option --version",
    ):
        asyncio.run(check_preflight(_run_config(tmp_path)))


def test_preflight_failure_skips_remote_finalizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit import runner

    async def fail(_config):
        raise RuntimeError("preflight failed")

    async def forbidden(_url):
        raise AssertionError("remote finalizer must not run")

    monkeypatch.setattr(runner, "check_preflight", fail)
    monkeypatch.setattr(runner, "capture_vllm_metrics", forbidden)
    monkeypatch.setattr(runner, "capture_router_snapshot", forbidden)

    with pytest.raises(RuntimeError, match="preflight failed"):
        asyncio.run(runner.run_benchmark(_run_config(tmp_path)))


def test_run_timeout_covers_setup_and_still_finalizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    _patch_successful_run(monkeypatch)

    async def blocked_preflight(_config):
        await asyncio.Event().wait()

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.check_preflight", blocked_preflight)
    config = _run_config(tmp_path)
    config.experiment.run_timeout_seconds = 1

    with pytest.raises(TimeoutError):
        asyncio.run(run_benchmark(config))

    assert _manifest(tmp_path)["status"] == "failed"


def test_run_timeout_includes_cancelled_task_in_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit import runner

    run_tasks = runner._run_tasks
    _patch_successful_run(monkeypatch)
    monkeypatch.setattr(runner, "_run_tasks", run_tasks)
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.load_tasks", lambda *_args: [_task()])
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.ensure_repo_cache", lambda *_args: None)
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.prepare_workspace", lambda *_args: None)

    async def blocked_agent(_request):
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "run_agent", blocked_agent)
    config = _run_config(tmp_path)
    config.experiment.run_timeout_seconds = 1

    with pytest.raises(TimeoutError):
        asyncio.run(runner.run_benchmark(config))

    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    result = json.loads((tmp_path / "run" / "tasks" / "instance" / "result.json").read_text(encoding="utf-8"))
    assert summary["tasks"]["failed"] == 1
    assert result["termination_reason"] == "cancelled"


def test_run_tasks_reports_progress_and_closes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit import runner

    task = _task()
    context = _context(tmp_path, router=False)
    context = RunContext(context.config, context.output_dir, context.trace_path, (task,))
    progress = _Progress(total=0)
    monkeypatch.setattr(runner, "tqdm", lambda **_kwargs: progress)
    monkeypatch.setattr(runner, "ensure_repo_cache", lambda *_args: None)
    monkeypatch.setattr(runner, "_run_single_task", lambda *_args: asyncio.sleep(0))

    asyncio.run(runner._run_tasks(context, "http://proxy", []))

    assert progress.updates == [1]
    assert progress.postfixes == [{"last": "instance"}]
    assert progress.closed is True


def test_run_tasks_closes_progress_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit import runner

    task = _task()
    context = _context(tmp_path, router=False)
    context = RunContext(context.config, context.output_dir, context.trace_path, (task,))
    progress = _Progress(total=0)
    monkeypatch.setattr(runner, "tqdm", lambda **_kwargs: progress)
    monkeypatch.setattr(runner, "ensure_repo_cache", lambda *_args: None)

    async def fail(*_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "_run_single_task", fail)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(runner._run_tasks(context, "http://proxy", []))

    assert progress.updates == [1]
    assert progress.closed is True


def test_repository_and_workspace_preparation_use_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit import runner

    task = _task()
    context = _context(tmp_path, router=False)
    context = RunContext(context.config, context.output_dir, context.trace_path, (task,))
    calls = []

    async def to_thread(function, *args):
        calls.append(function)
        return function(*args)

    monkeypatch.setattr(runner.asyncio, "to_thread", to_thread)
    monkeypatch.setattr(runner, "ensure_repo_cache", lambda *_args: None)
    monkeypatch.setattr(runner, "prepare_workspace", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "run_agent",
        lambda request: asyncio.sleep(0, result=_agent_result(request.session_id)),
    )
    results = []

    asyncio.run(runner._run_tasks(context, "http://proxy", results))

    assert calls == [runner.ensure_repo_cache, runner.prepare_workspace]
    assert len(results) == 1


def test_run_artifacts_index_raw_evidence_without_embedding_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentinfer.agentbench.benchkit import runner
    from agentinfer.agentbench.benchkit.metrics.schema import EvidenceCapture

    _patch_successful_run(monkeypatch)
    samples = iter(("prometheus-start-payload", "prometheus-end-payload"))

    async def vllm(_url):
        return EvidenceCapture("vllm", None, True, None, {"text": next(samples)})

    monkeypatch.setattr(runner, "capture_vllm_metrics", vllm)
    monkeypatch.setattr(
        runner,
        "collect_environment",
        lambda: EvidenceCapture("environment", None, True, None, {"hostname": "raw-host"}),
    )
    monkeypatch.setattr(
        runner,
        "collect_source_control",
        lambda _path: EvidenceCapture("source_control", None, True, None, {"commit": "raw-commit"}),
    )

    asyncio.run(runner.run_benchmark(_run_config(tmp_path), cli_metadata={"entrypoint": "test"}))

    run_dir = tmp_path / "run"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    serialized = json.dumps({"manifest": manifest, "summary": summary})

    assert (run_dir / "evidence" / "vllm_metrics_start.prom").read_text() == "prometheus-start-payload"
    assert (run_dir / "evidence" / "vllm_metrics_end.prom").read_text() == "prometheus-end-payload"
    assert json.loads((run_dir / "evidence" / "environment.json").read_text()) == {"hostname": "raw-host"}
    assert json.loads((run_dir / "evidence" / "source_control.json").read_text()) == {"commit": "raw-commit"}
    assert "prometheus-start-payload" not in serialized
    assert "prometheus-end-payload" not in serialized
    assert "raw-host" not in serialized
    assert "raw-commit" not in serialized
    assert all(row["metadata"] == {} for row in manifest["evidence"])
    assert all(row["metadata"] == {} for rows in summary["source_health"]["sources"].values() for row in rows)
    assert summary["cli"] == {"entrypoint": "test"}


def test_non_graceful_proxy_close_fails_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_successful_run(monkeypatch)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.create_request_proxy_lifecycle",
        lambda *_args: SimpleNamespace(
            start=lambda: asyncio.sleep(0, result=SimpleNamespace(base_url="http://proxy")),
            wait_ready=lambda _timeout: asyncio.sleep(0),
            close=lambda: asyncio.sleep(
                0,
                result=RequestProxyCloseResult(None, None, False, "forced termination"),
            ),
        ),
    )

    from agentinfer.agentbench.benchkit.runner import run_benchmark

    with pytest.raises(RuntimeError, match="forced termination"):
        asyncio.run(run_benchmark(_run_config(tmp_path)))

    assert _manifest(tmp_path)["status"] == "failed"
