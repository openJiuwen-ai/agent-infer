# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.agents.contracts import AgentRunOutcome, AgentRunResult
from agentinfer.agentbench.benchkit.config import AgentBenchConfig
from agentinfer.agentbench.benchkit.dataset import Task
from agentinfer.agentbench.benchkit.runner import RunContext, _run_single_task, check_preflight
from agentinfer.agentbench.request_proxy import RequestProxyCloseResult


def _context(tmp_path: Path, *, router: bool = False) -> RunContext:
    config = AgentBenchConfig.model_validate(
        {
            "experiment": {"result_dir": str(tmp_path)},
            "dataset": {"cache_dir": str(tmp_path / "cache")},
            "backend": {
                "base_url": "http://router:8400" if router else "http://vllm:8000",
            },
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


def test_router_backend_skips_control_protocol_and_serializes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.prepare_workspace", lambda *_args: None)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.run_agent",
        lambda request: asyncio.sleep(0, result=_agent_result(request.session_id)),
    )

    result = asyncio.run(
        _run_single_task(_context(tmp_path, router=True), _task(), "http://proxy", tmp_path / "cache", 0)
    )
    assert result.outcome == AgentRunOutcome.COMPLETED
    payload = (tmp_path / "tasks" / "instance" / "result.json").read_text(encoding="utf-8")
    assert "registration" not in payload
    assert "cleanup" not in payload


def _run_config(tmp_path: Path) -> AgentBenchConfig:
    return AgentBenchConfig.model_validate(
        {
            "experiment": {"result_dir": str(tmp_path / "run")},
            "dataset": {"cache_dir": str(tmp_path / "cache")},
            "backend": {"base_url": "http://router:8400", "metrics_url": "http://vllm:8000/metrics"},
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

    async def run_tasks(context, *_args):
        context.trace_path.write_text("", encoding="utf-8")

    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner._run_tasks", run_tasks)
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


def test_run_writes_session_agent_csvs_as_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    _patch_successful_run(monkeypatch)
    asyncio.run(run_benchmark(_run_config(tmp_path)))

    run_dir = tmp_path / "run"
    assert (run_dir / "sessions.csv").exists()
    assert (run_dir / "agents.csv").exists()
    assert (run_dir / "distribution_samples.csv").exists()
    manifest = _manifest(tmp_path)
    assert manifest["status"] == "completed"
    assert any(row["source"] == "sessions" and row["available"] for row in manifest["evidence"])
    assert any(row["source"] == "agents" and row["available"] for row in manifest["evidence"])
    assert any(row["source"] == "distribution_samples" and row["available"] for row in manifest["evidence"])


def test_session_analysis_failure_is_nonfatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit.runner import run_benchmark

    _patch_successful_run(monkeypatch)
    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.runner.write_analysis_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("analysis exploded")),
    )

    asyncio.run(run_benchmark(_run_config(tmp_path)))

    manifest = _manifest(tmp_path)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert any(
        row["source"] == "session_agent_analysis" and "analysis exploded" in row["reason"]
        for row in manifest["evidence"]
    )
    assert any("analysis exploded" in reason for reason in summary["source_health"]["reasons"])


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
    from agentinfer.agentbench.benchkit.artifacts import finalizer

    real_write = finalizer.atomic_write_json

    def fail_summary(path, value):
        if path.name == "summary.json":
            raise OSError("summary write exploded")
        real_write(path, value)

    monkeypatch.setattr(finalizer, "atomic_write_json", fail_summary)

    with pytest.raises(RuntimeError, match="summary write exploded"):
        asyncio.run(run_benchmark(_run_config(tmp_path)))
    manifest = _manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert any("summary write exploded" in (row["reason"] or "") for row in manifest["evidence"])


def test_preflight_requires_tmux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentinfer.agentbench.agents.preflight.shutil.which", lambda _name: None)

    with pytest.raises(RuntimeError, match="tmux executable is not available: tmux"):
        asyncio.run(check_preflight(_run_config(tmp_path)))


def test_preflight_requires_agent_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.preflight.shutil.which",
        lambda name: "/usr/bin/tmux" if name == "tmux" else None,
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.preflight.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    with pytest.raises(RuntimeError, match="Agent executable is not available: claude"):
        asyncio.run(check_preflight(_run_config(tmp_path)))


def test_preflight_rejects_broken_agent_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.preflight.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    results = iter(
        (
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=1, stderr=b"unknown option --version", stdout=b""),
        )
    )
    monkeypatch.setattr(
        "agentinfer.agentbench.agents.preflight.subprocess.run",
        lambda *_args, **_kwargs: next(results),
    )

    with pytest.raises(
        RuntimeError,
        match="Agent executable preflight check failed: claude --version\nunknown option --version",
    ):
        asyncio.run(check_preflight(_run_config(tmp_path)))


def test_jiuwenswarm_preflight_does_not_require_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AgentBenchConfig.model_validate(
        {
            **_run_config(tmp_path).model_dump(mode="python"),
            "agent": {
                "type": "jiuwenswarm",
                "profile": "code.normal",
                "executable": "jiuwenswarm",
            },
            "backend": {
                **_run_config(tmp_path).backend.model_dump(mode="python"),
                "endpoint": "/v1/chat/completions",
            },
        }
    )
    calls = []
    urls = []

    def which(name: str):
        assert name != "tmux"
        return f"/usr/bin/{name}"

    def run(command, **_kwargs):
        calls.append(command)
        output = (
            b"usage: jiuwenswarm {chat}\n"
            if command[-1] == "--help" and "chat" not in command
            else b"--jsonl --mode --session --cwd --project-dir --gateway-url --timeout\n"
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr=b"")

    class Response:
        status_code = 200

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url):
            urls.append(url)
            return Response()

    monkeypatch.setattr("agentinfer.agentbench.agents.preflight.shutil.which", which)
    monkeypatch.setattr("agentinfer.agentbench.agents.preflight.subprocess.run", run)
    monkeypatch.setattr("agentinfer.agentbench.benchkit.runner.httpx.AsyncClient", lambda **_kwargs: Client())

    asyncio.run(check_preflight(config))

    assert calls == [
        ["/usr/bin/jiuwenswarm", "--help"],
        ["/usr/bin/jiuwenswarm", "chat", "--help"],
    ]
    assert urls == ["http://router:8400/v1/models"]


def test_preflight_failure_skips_remote_finalizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentinfer.agentbench.benchkit import runner

    async def fail(_config):
        raise RuntimeError("preflight failed")

    async def forbidden(_url):
        raise AssertionError("remote finalizer must not run")

    monkeypatch.setattr(runner, "check_preflight", fail)
    monkeypatch.setattr(runner, "capture_vllm_metrics", forbidden)

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
    metrics_urls = []
    proxy_args = []

    async def vllm(url):
        metrics_urls.append(url)
        return EvidenceCapture("vllm", None, True, None, {"text": next(samples)})

    monkeypatch.setattr(runner, "capture_vllm_metrics", vllm)
    monkeypatch.setattr(
        runner,
        "create_request_proxy_lifecycle",
        lambda *args: (
            proxy_args.append(args)
            or SimpleNamespace(
                start=lambda: asyncio.sleep(0, result=SimpleNamespace(base_url="http://proxy")),
                wait_ready=lambda _timeout: asyncio.sleep(0),
                close=lambda: asyncio.sleep(0, result=None),
            )
        ),
    )
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

    config = _run_config(tmp_path)
    asyncio.run(runner.run_benchmark(config, cli_metadata={"entrypoint": "test"}))

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
    assert summary["router"] == {"applicable": False, "events": {}}
    assert metrics_urls == ["http://vllm:8000/metrics", "http://vllm:8000/metrics"]
    assert proxy_args[0][1] == "http://router:8400"


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
