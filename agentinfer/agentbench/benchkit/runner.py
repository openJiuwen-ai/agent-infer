# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Functional orchestration for benchmark runs and task lifecycles."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import httpx
from tqdm import tqdm

from ..agents import AgentRunRequest, AgentRunResult, check_agent_preflight, run_agent
from ..agents.contracts import AgentRunOutcome, TerminationReason
from ..request_proxy import create_request_proxy_lifecycle, load_request_facts
from .artifacts import (
    FinalizationContext,
    RunFinalizer,
    build_run_manifest,
    build_task_result,
)
from .collectors.correctness import load_correctness_artifact
from .collectors.environment import collect_environment
from .collectors.source_control import collect_source_control
from .collectors.vllm import capture_vllm_metrics
from .common import atomic_write_json, utc_now, write_text
from .config import AgentBenchConfig
from .dataset import Task, load_tasks
from .metrics.request import derive_session_topology
from .metrics.schema import EvidenceCapture
from .session_analysis import write_analysis_artifacts
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
    vllm_start: str | None = None
    vllm_end: str | None = None
    close_result = None
    run_exception: BaseException | None = None
    finalization_errors: list[str] = []
    lifecycle = None
    preflight_succeeded = False
    setup_owner = WorkspaceProcessOwner(
        time.monotonic() + config.experiment.run_timeout_seconds if config.experiment.run_timeout_seconds else None
    )
    context = RunContext(config, output_dir, output_dir / "requests.jsonl", ())

    async def execute() -> None:
        nonlocal context, lifecycle, preflight_succeeded, vllm_start
        await check_preflight(config)
        preflight_succeeded = True
        tasks = tuple(load_tasks(config.dataset.index_path, config.dataset.selection_path, config.experiment.task_num))
        context = RunContext(config, output_dir, output_dir / "requests.jsonl", tasks)
        start_capture = await capture_vllm_metrics(config.backend.effective_metrics_url)
        start_capture = _capture_with_path(
            _named_capture(start_capture, "vllm_start"),
            output_dir / "evidence" / "vllm_metrics_start.prom",
        )
        captures.append(start_capture)
        vllm_start = _capture_text(start_capture)
        if vllm_start is not None:
            write_text(start_capture.path, vllm_start)
        lifecycle = create_request_proxy_lifecycle(
            config.request_proxy,
            config.backend.base_url,
            run_id,
            output_dir,
            config.backend.endpoint,
        )
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

        if preflight_succeeded:
            end_capture = await finalize_async(
                "vllm_end", lambda: capture_vllm_metrics(config.backend.effective_metrics_url)
            )
            if end_capture is not None:
                end_capture = _capture_with_path(
                    _named_capture(end_capture, "vllm_end"),
                    output_dir / "evidence" / "vllm_metrics_end.prom",
                )
                captures.append(end_capture)
                vllm_end = _capture_text(end_capture)
                if vllm_end is not None:
                    finalize_sync("vllm_end_write", lambda: write_text(end_capture.path, vllm_end))

        environment_capture = finalize_sync("environment", collect_environment)
        if environment_capture is not None:
            environment_capture = _capture_with_path(environment_capture, output_dir / "evidence" / "environment.json")
            captures.append(environment_capture)
            if environment_capture.available:
                finalize_sync(
                    "environment_write",
                    lambda: atomic_write_json(environment_capture.path, environment_capture.metadata),
                )

        source_capture = finalize_sync("source_control", lambda: collect_source_control(Path(__file__).resolve()))
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

        finished_at = utc_now()
        run_wall_time_seconds = (finished_at - manifest.created_at).total_seconds()
        facts = finalize_sync("request_facts", lambda: load_request_facts(context.trace_path))
        if facts is not None:
            try:
                analysis_artifacts = write_analysis_artifacts([output_dir], output_dir, plots=False)
            except Exception as exc:
                captures.append(
                    EvidenceCapture("session_agent_analysis", None, False, f"{type(exc).__name__}: {exc}", {})
                )
            else:
                captures.extend(
                    EvidenceCapture(source, path, True, None, {}) for source, path in analysis_artifacts.items()
                )
        finalization = RunFinalizer(
            FinalizationContext(
                run_id=run_id,
                output_dir=output_dir,
                manifest=manifest,
                results=results,
                facts=facts,
                vllm_start=vllm_start,
                vllm_end=vllm_end,
                captures=captures,
                correctness=correctness_capture,
                close_result=close_result,
                run_exception=run_exception,
                prior_finalization_errors=finalization_errors,
                run_wall_time_seconds=run_wall_time_seconds,
                finished_at=finished_at,
                cli_metadata=cli_metadata,
            )
        ).finalize()
        finalization_errors.extend(finalization.finalization_errors)

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

    progress = tqdm(total=len(context.tasks), desc="benchmark", unit="task")

    async def bounded(task: Task, task_position: int) -> None:
        async with semaphore:
            try:
                await _run_single_task(
                    context,
                    task,
                    api_base_url,
                    cache,
                    task_position,
                    results,
                    owner,
                )
            finally:
                progress.update(1)
                progress.set_postfix(last=task.instance_id)

    tasks = [asyncio.create_task(bounded(task, position)) for position, task in enumerate(context.tasks)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        owner.terminate_all()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        progress.close()


async def _run_single_task(
    context: RunContext,
    task: Task,
    api_base_url: str,
    cache: Path,
    task_position: int,
    results: list[AgentRunResult] | None = None,
    owner: WorkspaceProcessOwner | None = None,
) -> AgentRunResult:
    owner = owner or WorkspaceProcessOwner()
    config = context.config
    output_dir = _task_path(context.output_dir / "tasks", task.instance_id)
    workspace = _task_path(context.output_dir / "workspaces", task.instance_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    started = time.monotonic()
    started_at = utc_now().isoformat()
    result = _failed_result(
        task, session_id, config.agent.type, config.agent.profile, TerminationReason.HARNESS_ERROR, started_at
    )
    pending_error: BaseException | None = None
    error: dict[str, str] | None = None
    try:
        await asyncio.to_thread(prepare_workspace, task, cache, workspace, owner)
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
        result = _failed_result(
            task, session_id, config.agent.type, config.agent.profile, TerminationReason.CANCELLED, started_at
        )
        pending_error = exc
    except Exception as exc:
        result = _failed_result(
            task, session_id, config.agent.type, config.agent.profile, TerminationReason.HARNESS_ERROR, started_at
        )
        pending_error = RuntimeError(f"Task {task.instance_id} failed in benchmark harness")
        error = _exception_evidence(exc)
    finally:
        result.duration_seconds = result.duration_seconds or time.monotonic() - started
        if not result.finished_at:
            result.finished_at = utc_now().isoformat()
        if not result.started_at:
            result.started_at = started_at
        topology = derive_session_topology(load_request_facts(context.trace_path, session_id=session_id), session_id)
        atomic_write_json(
            output_dir / "result.json",
            build_task_result(result, topology, error, task_position=task_position).to_dict(),
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
    task: Task,
    session_id: str,
    agent_type: str,
    profile: str,
    reason: TerminationReason,
    started_at: str = "",
) -> AgentRunResult:
    return AgentRunResult(
        outcome=AgentRunOutcome.FAILED,
        termination_reason=reason,
        session_id=session_id,
        instance_id=task.instance_id,
        agent_type=agent_type,
        profile_name=profile,
        started_at=started_at,
    )


def _exception_evidence(exc: BaseException) -> dict[str, str]:
    message = " ".join(str(exc).split())
    if len(message) > 1000:
        message = message[:982] + "...<truncated>"
    return {"type": type(exc).__name__, "message": message}


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


def _trace_capture(path: Path, close_result) -> EvidenceCapture:
    if close_result is None or close_result.trace_health is None:
        return EvidenceCapture("request_trace", path if path.exists() else None, False, "trace health unavailable", {})
    health = asdict(close_result.trace_health)
    available = health["writer_error"] is None and health["pending"] == 0
    reason = None if available else health["writer_error"] or f"{health['pending']} request facts pending"
    return EvidenceCapture("request_trace", path if path.exists() else None, available, reason, health)


async def check_preflight(config: AgentBenchConfig) -> None:
    await check_agent_preflight(config.agent.type, config.agent.executable)
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        response = await client.get(f"{config.backend.base_url.rstrip('/')}/v1/models")
        if response.status_code != 200:
            raise RuntimeError(f"Backend {config.backend.base_url} is not healthy")
