"""Functional orchestration for benchmark runs and task lifecycles."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import httpx

from ..agents import AgentRunRequest, AgentRunResult, run_agent
from ..agents.outcomes import AgentRunOutcome, TerminationReason
from ..request_proxy import create_request_proxy_lifecycle, load_request_facts
from .artifacts import RunSummary, build_run_manifest, build_run_summary, build_task_result, finalize_run_manifest
from .collectors.correctness import load_correctness_artifact
from .collectors.environment import collect_environment
from .collectors.router import capture_router_snapshot
from .collectors.source_control import collect_source_control
from .collectors.vllm import capture_vllm_metrics
from .common import atomic_write_json, write_text
from .config import AgentBenchConfig
from .dataset import Task, load_tasks
from .metrics.request import aggregate_request_metrics, derive_session_topology
from .metrics.router import aggregate_router_events, aggregate_router_window
from .metrics.schema import EvidenceCapture
from .metrics.source_health import evaluate_captures
from .metrics.task import aggregate_task_results
from .metrics.vllm import aggregate_vllm_metrics
from .session_registration import cleanup_session, register_session
from .workspace import WorkspaceProcessOwner, ensure_repo_cache, prepare_workspace

T = TypeVar("T")


@dataclass(frozen=True)
class RunContext:
    config: AgentBenchConfig
    output_dir: Path
    trace_path: Path
    tasks: tuple[Task, ...]


async def run_benchmark(config: AgentBenchConfig, *, cli_metadata: dict[str, object] | None = None) -> Path:
    output_dir = config.experiment.result_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = output_dir.name
    manifest = build_run_manifest(run_id, config.model_dump(mode="json"))
    atomic_write_json(output_dir / "manifest.json", manifest.to_dict())

    results: list[AgentRunResult] = []
    captures: list[EvidenceCapture] = []
    router_start: list[dict[str, object]] | None = None
    router_end: list[dict[str, object]] | None = None
    vllm_start: str | None = None
    vllm_end: str | None = None
    close_result = None
    run_exception: BaseException | None = None
    finalization_errors: list[str] = []
    lifecycle = None
    setup_owner = WorkspaceProcessOwner(
        time.monotonic() + config.experiment.run_timeout_seconds if config.experiment.run_timeout_seconds else None
    )
    context = RunContext(config, output_dir, output_dir / "requests.jsonl", ())

    async def execute() -> None:
        nonlocal context, lifecycle, router_start, vllm_start
        await check_preflight(config)
        tasks = tuple(load_tasks(config.dataset.index_path, config.dataset.selection_path, config.experiment.task_num))
        context = RunContext(config, output_dir, output_dir / "requests.jsonl", tasks)
        start_capture = await capture_vllm_metrics(config.backend.base_url)
        start_capture = _capture_with_path(
            _named_capture(start_capture, "vllm_start"),
            output_dir / "evidence" / "vllm_metrics_start.prom",
        )
        captures.append(start_capture)
        vllm_start = _capture_text(start_capture)
        if vllm_start is not None:
            write_text(start_capture.path, vllm_start)
        if config.router.enabled:
            assert config.router.base_url is not None
            router_capture = await capture_router_snapshot(config.router.base_url)
            router_capture = _capture_with_path(
                _named_capture(router_capture, "router_start"),
                output_dir / "evidence" / "router_start.json",
            )
            captures.append(router_capture)
            router_start = _router_events(router_capture) if router_capture.available else None
            if router_capture.available:
                atomic_write_json(router_capture.path, router_capture.metadata["raw"])
        else:
            captures.append(EvidenceCapture("router", None, False, None, {}, applicable=False))

        upstream = config.router.base_url if config.router.enabled else config.backend.base_url
        assert upstream is not None
        lifecycle = create_request_proxy_lifecycle(config.request_proxy, upstream, run_id, output_dir)
        handle = await lifecycle.start()
        await lifecycle.wait_ready(config.request_proxy.startup_timeout_seconds)
        await _run_tasks(context, handle.base_url, results, setup_owner)

    try:
        if config.experiment.run_timeout_seconds:
            await asyncio.wait_for(execute(), config.experiment.run_timeout_seconds)
        else:
            await execute()
    except BaseException as exc:
        setup_owner.terminate_all()
        run_exception = exc
    finally:

        async def finalize_async(source: str, operation: Callable[[], Awaitable[T]]) -> T | None:
            try:
                return await operation()
            except BaseException as exc:
                _record_finalization_error(finalization_errors, captures, source, exc)
                return None

        def finalize_sync(source: str, operation: Callable[[], T]) -> T | None:
            try:
                return operation()
            except BaseException as exc:
                _record_finalization_error(finalization_errors, captures, source, exc)
                return None

        if lifecycle is not None:
            close_result = await finalize_async("proxy_close", lifecycle.close)
            if close_result is not None and not close_result.graceful:
                error = close_result.error or "request proxy did not shut down gracefully"
                _record_finalization_error(
                    finalization_errors,
                    captures,
                    "proxy_close",
                    RuntimeError(error),
                )

        end_capture = await finalize_async("vllm_end", lambda: capture_vllm_metrics(config.backend.base_url))
        if end_capture is not None:
            end_capture = _capture_with_path(
                _named_capture(end_capture, "vllm_end"),
                output_dir / "evidence" / "vllm_metrics_end.prom",
            )
            captures.append(end_capture)
            vllm_end = _capture_text(end_capture)
            if vllm_end is not None:
                finalize_sync("vllm_end_write", lambda: write_text(end_capture.path, vllm_end))

        if config.router.enabled and config.router.base_url:
            router_capture = await finalize_async("router_end", lambda: capture_router_snapshot(config.router.base_url))
            if router_capture is not None:
                router_capture = _capture_with_path(
                    _named_capture(router_capture, "router_end"),
                    output_dir / "evidence" / "router_end.json",
                )
                captures.append(router_capture)
                router_end = _router_events(router_capture) if router_capture.available else None
                if router_capture.available:
                    finalize_sync(
                        "router_end_write",
                        lambda: atomic_write_json(router_capture.path, router_capture.metadata["raw"]),
                    )

        environment_capture = finalize_sync("environment", collect_environment)
        if environment_capture is not None:
            environment_capture = _capture_with_path(environment_capture, output_dir / "evidence" / "environment.json")
            captures.append(environment_capture)
            if environment_capture.available:
                finalize_sync(
                    "environment_write",
                    lambda: atomic_write_json(environment_capture.path, environment_capture.metadata),
                )

        source_capture = finalize_sync("source_control", lambda: collect_source_control(Path.cwd()))
        if source_capture is not None:
            source_capture = _capture_with_path(source_capture, output_dir / "evidence" / "source_control.json")
            captures.append(source_capture)
            if source_capture.available:
                finalize_sync(
                    "source_control_write",
                    lambda: atomic_write_json(source_capture.path, source_capture.metadata),
                )

        captures.append(_trace_capture(context.trace_path, close_result))
        correctness_capture = finalize_sync(
            "correctness", lambda: load_correctness_artifact(output_dir / "correctness.json")
        )
        if correctness_capture is not None:
            captures.append(correctness_capture)

        facts = finalize_sync("request_facts", lambda: load_request_facts(context.trace_path))
        vllm_metrics = finalize_sync("vllm_aggregation", lambda: aggregate_vllm_metrics(vllm_start, vllm_end))
        summary = None
        if facts is not None and vllm_metrics is not None:
            lifecycle_payload = {
                "status": "failed" if run_exception or finalization_errors else "completed",
                "error": _exception_text(run_exception),
                "finalization_errors": list(finalization_errors),
                "proxy_close": asdict(close_result) if close_result else None,
            }
            summary = finalize_sync(
                "summary_build",
                lambda: build_run_summary(
                    run_id,
                    "candidate" if config.router.enabled else "baseline",
                    aggregate_task_results(results),
                    aggregate_request_metrics(facts),
                    (
                        aggregate_router_window(router_start, router_end)
                        if config.router.enabled and router_start is not None and router_end is not None
                        else None
                    ),
                    vllm_metrics,
                    {
                        "available": correctness_capture.available if correctness_capture else False,
                        "reason": correctness_capture.reason if correctness_capture else "collection failed",
                        "metadata": dict(correctness_capture.metadata) if correctness_capture else {},
                    },
                    evaluate_captures(captures),
                    lifecycle_payload,
                ).to_dict(),
            )
        if summary is not None:
            if cli_metadata:
                summary["cli"] = cli_metadata
            finalize_sync("summary_write", lambda: atomic_write_json(output_dir / "summary.json", summary))

        status = "failed" if run_exception or finalization_errors else "completed"
        for attempt in range(2):
            try:
                evidence = tuple(_capture_dict(capture) for capture in captures)
                finalized_manifest = finalize_run_manifest(manifest, evidence, status=status)
                atomic_write_json(output_dir / "manifest.json", finalized_manifest.to_dict())
            except BaseException as exc:
                _record_finalization_error(finalization_errors, captures, "manifest_finalize", exc)
                status = "failed"
                if attempt == 0:
                    continue
            break

    if run_exception is not None:
        if finalization_errors:
            run_exception.add_note("Finalization errors: " + "; ".join(finalization_errors))
        raise run_exception
    if finalization_errors:
        raise RuntimeError("Benchmark finalization failed: " + "; ".join(finalization_errors))
    return output_dir


async def _run_tasks(
    context: RunContext,
    api_base_url: str,
    results: list[AgentRunResult],
    owner: WorkspaceProcessOwner | None = None,
) -> None:
    owner = owner or WorkspaceProcessOwner()
    config = context.config
    cache = config.dataset.cache_dir.resolve()
    for task in context.tasks:
        await asyncio.to_thread(ensure_repo_cache, task, cache, owner)
    semaphore = asyncio.Semaphore(config.experiment.max_concurrency)

    async def bounded(task: Task) -> None:
        async with semaphore:
            await _run_single_task(context, task, api_base_url, cache, results, owner)

    tasks = [asyncio.create_task(bounded(task)) for task in context.tasks]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        owner.terminate_all()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_single_task(
    context: RunContext,
    task: Task,
    api_base_url: str,
    cache: Path,
    results: list[AgentRunResult] | None = None,
    owner: WorkspaceProcessOwner | None = None,
) -> AgentRunResult:
    owner = owner or WorkspaceProcessOwner()
    config = context.config
    output_dir = _task_path(context.output_dir / "tasks", task.instance_id)
    workspace = _task_path(context.output_dir / "workspaces", task.instance_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    registration = None
    cleanup = None
    registered = False
    started = time.monotonic()
    result = _failed_result(task, session_id, config.agent.type, config.agent.profile, TerminationReason.HARNESS_ERROR)
    pending_error: BaseException | None = None
    error: dict[str, str] | None = None
    try:
        await asyncio.to_thread(prepare_workspace, task, cache, workspace, owner)
        if config.router.enabled:
            assert config.router.base_url is not None
            registration = await register_session(
                config.router.base_url, session_id, config.router.control_timeout_seconds
            )
            if not registration.success:
                raise RuntimeError(f"Router session registration failed: {registration.error}")
            registered = True
        result = await run_agent(
            AgentRunRequest(
                config.agent.type,
                task,
                config.agent.profile,
                config.agent.executable,
                config.backend.model,
                api_base_url,
                workspace,
                output_dir,
                session_id,
                config.experiment.task_timeout_seconds,
                config.experiment.patch_flush_seconds,
                config.experiment.prompt_delivery_timeout_seconds,
                config.agent.tmux_startup_seconds,
                config.agent.terminal_capture_interval_seconds,
            )
        )
    except asyncio.CancelledError as exc:
        result = _failed_result(task, session_id, config.agent.type, config.agent.profile, TerminationReason.CANCELLED)
        pending_error = exc
    except Exception as exc:
        result = _failed_result(
            task, session_id, config.agent.type, config.agent.profile, TerminationReason.HARNESS_ERROR
        )
        pending_error = RuntimeError(f"Task {task.instance_id} failed in benchmark harness")
        error = _exception_evidence(exc)
    finally:
        result.duration_seconds = result.duration_seconds or time.monotonic() - started
        if registered:
            assert config.router.base_url is not None
            try:
                cleanup = await cleanup_session(
                    config.router.base_url, session_id, config.router.control_timeout_seconds
                )
            except Exception as exc:
                from .session_registration import SessionRegistrationResult

                cleanup = SessionRegistrationResult(
                    "cleanup", session_id, False, None, time.monotonic() - started, f"{type(exc).__name__}: {exc}"
                )
            if not cleanup.success:
                cleanup_error = RuntimeError(cleanup.error or "Router session cleanup failed")
                error = _exception_evidence(cleanup_error)
                if pending_error is None:
                    pending_error = RuntimeError(f"Task {task.instance_id} Router cleanup failed")
        topology = derive_session_topology(load_request_facts(context.trace_path, session_id=session_id), session_id)
        atomic_write_json(
            output_dir / "result.json", build_task_result(result, registration, cleanup, topology, error).to_dict()
        )
        if results is not None:
            results.append(result)
    if pending_error is not None:
        raise pending_error
    return result


def _task_path(root: Path, instance_id: str) -> Path:
    root = root.resolve()
    path = (root / instance_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Task path escapes run directory: {instance_id!r}") from exc
    if path == root:
        raise ValueError(f"Task path must be below run directory: {instance_id!r}")
    return path


def _failed_result(
    task: Task, session_id: str, agent_type: str, profile: str, reason: TerminationReason
) -> AgentRunResult:
    return AgentRunResult(
        outcome=AgentRunOutcome.FAILED,
        termination_reason=reason,
        session_id=session_id,
        instance_id=task.instance_id,
        agent_type=agent_type,
        profile_name=profile,
    )


def _exception_evidence(exc: BaseException) -> dict[str, str]:
    message = " ".join(str(exc).split())
    if len(message) > 1000:
        message = message[:982] + "...<truncated>"
    return {"type": type(exc).__name__, "message": message}


def _exception_text(exc: BaseException | None) -> str | None:
    return f"{type(exc).__name__}: {exc}" if exc is not None else None


def _record_finalization_error(
    errors: list[str], captures: list[EvidenceCapture], source: str, exc: BaseException
) -> None:
    error = f"{source}: {type(exc).__name__}: {exc}"
    errors.append(error)
    captures.append(EvidenceCapture(source, None, False, error, {}))


def _named_capture(capture: EvidenceCapture, source: str) -> EvidenceCapture:
    return EvidenceCapture(
        source, capture.path, capture.available, capture.reason, capture.metadata, capture.applicable
    )


def _capture_with_path(capture: EvidenceCapture, path: Path) -> EvidenceCapture:
    return EvidenceCapture(
        capture.source,
        path if capture.available else None,
        capture.available,
        capture.reason,
        capture.metadata,
        capture.applicable,
    )


def _capture_text(capture: EvidenceCapture) -> str | None:
    value = capture.metadata.get("text")
    return value if isinstance(value, str) else None


def _router_events(capture: EvidenceCapture) -> list[dict[str, object]]:
    payload = capture.metadata.get("raw")
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
    return []


def _trace_capture(path: Path, close_result) -> EvidenceCapture:
    if close_result is None or close_result.trace_health is None:
        return EvidenceCapture("request_trace", path if path.exists() else None, False, "trace health unavailable", {})
    health = asdict(close_result.trace_health)
    available = health["writer_error"] is None and health["pending"] == 0
    reason = None if available else health["writer_error"] or f"{health['pending']} request facts pending"
    return EvidenceCapture("request_trace", path if path.exists() else None, available, reason, health)


def _capture_dict(capture: EvidenceCapture) -> dict:
    value = asdict(capture)
    value["path"] = str(capture.path) if capture.path is not None else None
    return value


async def check_preflight(config: AgentBenchConfig) -> None:
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        response = await client.get(f"{config.backend.base_url.rstrip('/')}/v1/models")
        if response.status_code != 200:
            raise RuntimeError(f"Backend {config.backend.base_url} is not healthy")
        if config.router.enabled:
            assert config.router.base_url is not None
            response = await client.get(f"{config.router.base_url.rstrip('/')}/health")
            if response.status_code != 200:
                raise RuntimeError(f"Router {config.router.base_url} is not healthy")


def summarize_run(run_dir: Path, config: AgentBenchConfig) -> RunSummary:
    facts = load_request_facts(run_dir / "requests.jsonl")
    return build_run_summary(
        run_dir.name,
        "candidate" if config.router.enabled else "baseline",
        aggregate_task_results([]),
        aggregate_request_metrics(facts),
        aggregate_router_events([]) if config.router.enabled else None,
        aggregate_vllm_metrics(None, None),
        {"available": False, "reason": "artifact missing", "metadata": {}},
        evaluate_captures([]),
        {"status": "summarized", "error": None, "proxy_close": None},
    )
