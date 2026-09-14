# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run one benchmark task through the DeepSeek Harness headless CLI."""

import asyncio
import json
import os
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ...benchkit.workspace import export_patch
from ..contracts import AgentRunOutcome, AgentRunRequest, AgentRunResult, TerminationReason
from .instance import DshInstance
from .profiles import get_profile
from .transcript import (
    DshStuckDetector,
    DshTranscriptNormalizer,
    find_session_logs,
    is_turn_complete,
    load_session_log,
    session_is_root,
)


class _ProfileInitializationTimeout(Exception):
    """Raised when model-free DSH profile initialization exceeds the task deadline."""


class _ProfileInitializationError(Exception):
    """Raised when DSH rejects its task-local headless profile."""


class _ArgvLimitExceeded(Exception):
    """Raised when the launch command would exceed the host argv limit."""


def _check_argv_limit(command: list[str], environment: dict[str, str]) -> None:
    """Reject before spawn when the command would exceed the host argv limit.

    DSH headless takes its task prompt as a positional argument, so a large
    SWE-bench prompt can overflow ``E2BIG`` before DSH starts. Detecting here
    turns that host error into a classified startup failure with evidence.
    """

    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return
    arg_bytes = sum(len(argument) + 1 for argument in command) + len(command) + 1
    env_bytes = sum(len(key) + len(value) + 2 for key, value in environment.items())
    # Reserve room for argv/env pointer arrays and libc overhead.
    limit = sysconf("SC_ARG_MAX")
    if arg_bytes + env_bytes > limit * 3 // 4:
        raise _ArgvLimitExceeded(
            f"launch command exceeds the host argv limit: command+env {arg_bytes + env_bytes} bytes vs SC_ARG_MAX {limit}"
        )


async def run_dsh(request: AgentRunRequest) -> AgentRunResult:
    """Execute one task through one isolated headless DSH process."""

    profile = get_profile(request.profile_name)
    started_clock = time.monotonic()
    deadline = started_clock + request.timeout_seconds
    result = AgentRunResult(
        instance_id=request.task.instance_id,
        session_id=request.session_id,
        agent_type=request.agent_type,
        profile_name=profile.name,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    process: asyncio.subprocess.Process | None = None
    cancellation: asyncio.CancelledError | None = None
    try:
        # Every filesystem and process step is guarded so a failure here still
        # produces a classified result and the dsh-runner-error.log evidence.
        request.artifact_dir.mkdir(parents=True, exist_ok=True)
        transcript_artifact = request.artifact_dir / "transcript.jsonl"
        await asyncio.to_thread(transcript_artifact.write_text, "", encoding="utf-8")
        result.transcript = str(transcript_artifact)
        prompt = profile.build_prompt(request.task)
        await asyncio.to_thread((request.artifact_dir / "prompt.txt").write_text, prompt, encoding="utf-8")
        instance = DshInstance(
            artifact_dir=request.artifact_dir,
            api_base_url=request.api_base_url,
            model=request.model,
            session_id=request.session_id,
            policy_patch=profile.policy_patch,
            permission_mode=profile.permission_mode,
            enforce_plan_mode=profile.enforce_plan_mode,
        )
        await _initialize_profile(request, instance, deadline)
        await asyncio.to_thread(instance.bootstrap)
        command = [str(request.executable), "--profile", "headless", "--patch", str(instance.policy_patch_path), prompt]
        environment = instance.environment()
        _check_argv_limit(command, environment)
        await asyncio.to_thread(
            (request.artifact_dir / "dsh-launch-command.json").write_text,
            json.dumps(command, indent=2) + "\n",
            encoding="utf-8",
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=request.workspace,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        stdout, stderr, timed_out = await _communicate_with_deadline(process, deadline)
        await _write_outputs(request.artifact_dir, stdout, stderr)
        transcript_path = await _copy_transcript(instance.home_dir, request.artifact_dir, request.workspace)
        if transcript_path is not None:
            result.transcript = str(transcript_path)
        if timed_out:
            result.termination_reason = TerminationReason.TIMEOUT
        else:
            events = load_session_log(transcript_path) if transcript_path is not None else []
            reason = _classify(process.returncode or 0, stderr, events)
            if reason is None:
                result.outcome = AgentRunOutcome.COMPLETED
                result.termination_reason = None
            else:
                result.termination_reason = reason
    except asyncio.CancelledError as exc:
        cancellation = exc
        if process is not None:
            await asyncio.shield(_kill_process_group(process))
    except (_ProfileInitializationTimeout, _ProfileInitializationError) as exc:
        await asyncio.to_thread(
            (request.artifact_dir / "dsh-runner-error.log").write_text,
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        result.termination_reason = TerminationReason.AGENT_STARTUP_FAILED
    except (FileNotFoundError, PermissionError, _ArgvLimitExceeded) as exc:
        await asyncio.to_thread(
            (request.artifact_dir / "dsh-runner-error.log").write_text,
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        result.termination_reason = TerminationReason.AGENT_STARTUP_FAILED
    except Exception:
        await asyncio.to_thread(
            (request.artifact_dir / "dsh-runner-error.log").write_text,
            traceback.format_exc(),
            encoding="utf-8",
        )
        result.termination_reason = TerminationReason.HARNESS_ERROR

    if cancellation is not None:
        raise cancellation
    try:
        if request.patch_flush_seconds:
            await asyncio.sleep(request.patch_flush_seconds)
        patch = await asyncio.to_thread(export_patch, request.workspace)
        if patch:
            await asyncio.to_thread(
                (request.artifact_dir / "model.patch").write_text,
                patch,
                encoding="utf-8",
            )
            result.has_patch = True
            result.patch_bytes = len(patch.encode("utf-8"))
    except Exception:
        if result.outcome is AgentRunOutcome.COMPLETED:
            result.outcome = AgentRunOutcome.FAILED
            result.termination_reason = TerminationReason.HARNESS_ERROR
    result.finished_at = datetime.now(timezone.utc).isoformat()
    result.duration_seconds = time.monotonic() - started_clock
    return result


async def _initialize_profile(request: AgentRunRequest, instance: DshInstance, deadline: float) -> None:
    """Materialize DSH's task-local profile before writing adapter settings.

    DSH initializes a new ``DSH_HOME`` by replacing its contents on first boot.
    Running its model-free config dump first prevents that initialization from
    deleting the per-task settings and overlay written by ``bootstrap``.
    """

    process = await asyncio.create_subprocess_exec(
        str(request.executable),
        "--profile",
        "headless",
        "--dump-config",
        cwd=request.workspace,
        env=instance.environment(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    assert process.stderr is not None
    stderr_task = asyncio.create_task(_read_all(process.stderr))
    try:
        try:
            await asyncio.wait_for(asyncio.create_task(process.wait()), _remaining(deadline))
        except asyncio.TimeoutError as exc:
            await _kill_process_group(process)
            raise _ProfileInitializationTimeout from exc
    except asyncio.CancelledError:
        await asyncio.shield(_kill_process_group(process))
        await asyncio.shield(stderr_task)
        raise
    except BaseException:
        await _kill_process_group(process)
        await stderr_task
        raise
    stderr = await stderr_task
    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()
        raise _ProfileInitializationError(
            f"DSH profile initialization failed: {detail or f'exit {process.returncode}'}"
        )


async def _communicate_with_deadline(
    process: asyncio.subprocess.Process,
    deadline: float,
) -> tuple[bytes, bytes, bool]:
    """Collect stdout and stderr until exit or deadline, killing the group on timeout.

    DSH headless has no wall-clock timeout of its own, and its SIGTERM handler
    performs a graceful dispose that can exit 0, so deadline enforcement sends
    SIGKILL directly and reports timeout from deadline state, never from the
    exit code.
    """

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_read_all(process.stdout))
    stderr_task = asyncio.create_task(_read_all(process.stderr))
    timed_out = False
    try:
        try:
            await asyncio.wait_for(asyncio.create_task(process.wait()), _remaining(deadline))
        except asyncio.TimeoutError:
            timed_out = True
            await _kill_process_group(process)
    except asyncio.CancelledError:
        await asyncio.shield(_kill_process_group(process))
        await asyncio.shield(asyncio.gather(stdout_task, stderr_task))
        raise
    except BaseException:
        await _kill_process_group(process)
        await asyncio.gather(stdout_task, stderr_task)
        raise
    stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    return stdout, stderr, timed_out


async def _read_all(stream: asyncio.StreamReader) -> bytes:
    """Read a subprocess stream to completion."""

    captured = bytearray()
    while chunk := await stream.read(64 * 1024):
        captured.extend(chunk)
    return bytes(captured)


def _remaining(deadline: float) -> float:
    """Return seconds remaining until the deadline, or raise TimeoutError if expired."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return remaining


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """Force-stop the headless process group and wait for it to exit."""

    if process.returncode is not None:
        return
    if os.name == "nt":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


def _explicit_termination_reason(events: list[dict[str, object]]) -> TerminationReason | None:
    """Classify direct DSH lifecycle signals recorded in the session log."""

    for event in reversed(events):
        event_type = event.get("type")
        if event_type == "approval/asked":
            return TerminationReason.CONFIRMATION_HANG
        if event_type == "turn/end":
            data = event.get("data")
            reason = data.get("reason") if isinstance(data, dict) else None
            kind = reason.get("kind") if isinstance(reason, dict) else reason
            if kind == "blocked":
                return TerminationReason.CONFIRMATION_HANG
    return None


def _classify(
    return_code: int,
    stderr: bytes,
    events: list[dict[str, object]],
) -> TerminationReason | None:
    """Classify the termination reason from exit code, stderr, and the session log."""

    if is_turn_complete(events):
        return None
    explicit = _explicit_termination_reason(events)
    if explicit is not None:
        return explicit
    if return_code == 0:
        return None
    if not events and stderr.decode(errors="replace").strip():
        return TerminationReason.AGENT_STARTUP_FAILED
    stuck = DshStuckDetector().detect(DshTranscriptNormalizer().normalize(events))
    if stuck is not None:
        return stuck
    return TerminationReason.INTERRUPTED


async def _write_outputs(artifact_dir: Path, stdout: bytes, stderr: bytes) -> None:
    """Persist the headless stdout and stderr streams as task evidence."""

    await asyncio.gather(
        asyncio.to_thread((artifact_dir / "dsh.stdout.log").write_bytes, stdout),
        asyncio.to_thread((artifact_dir / "dsh.stderr.log").write_bytes, stderr),
    )


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


async def _copy_transcript(home: Path, artifact_dir: Path, workspace: Path | None = None) -> Path | None:
    """Decode the task's session logs and select the root transcript artifact.

    All durable session logs, including subagents, remain available under
    ``dsh-sessions/``. The largest root log is the task transcript; a subagent
    log is selected only when no root header is recognizable.
    """

    def copy() -> Path | None:
        logs = find_session_logs(home, workspace)
        if not logs:
            return None
        sessions_dir = artifact_dir / "dsh-sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        decoded: list[tuple[Path, list[dict[str, object]]]] = []
        for index, path in enumerate(logs):
            events = load_session_log(path)
            if not events:
                continue
            target = sessions_dir / f"session-{index}.jsonl"
            _write_events(target, events)
            decoded.append((target, events))
        if not decoded:
            return None
        root_logs = [item for item in decoded if session_is_root(item[1])] or decoded
        _main, main_events = max(root_logs, key=lambda item: len(item[1]))
        transcript_path = artifact_dir / "transcript.jsonl"
        _write_events(transcript_path, main_events)
        return transcript_path

    return await asyncio.to_thread(copy)
