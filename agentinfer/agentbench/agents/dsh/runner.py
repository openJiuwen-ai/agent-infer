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

import yaml

from ...benchkit.workspace import WorkspaceProcessOwner, export_patch
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

CLEANUP_SECONDS = 10.0


class _ProfileInitializationTimeout(Exception):
    """Raised when model-free DSH profile initialization exceeds the task deadline."""


class _ProfileInitializationError(Exception):
    """Raised when DSH rejects its task-local headless profile."""


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
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    instance = DshInstance(
        artifact_dir=request.artifact_dir,
        api_base_url=request.api_base_url,
        model=request.model,
        session_id=request.session_id,
        policy_patch=profile.policy_patch,
        permission_mode=profile.permission_mode,
        enforce_plan_mode=profile.enforce_plan_mode,
        context_window=request.context_window,
        max_tokens=request.max_tokens,
    )
    status = {"phase": "initialization", "collection": "not_started", "patch": "not_started"}
    cancellation = None
    return_code = None
    try:
        prompt = profile.build_prompt(request.task)
        (request.artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        await _initialize_profile(request, instance, deadline)
        instance.bootstrap()
        status["phase"] = "configuration"
        await _dump_config(request, instance, deadline)
        status["phase"] = "execution"
        command = [
            str(request.executable),
            "--profile",
            "headless",
            "--patch",
            str(instance.policy_patch_path),
            "--json",
        ]
        (request.artifact_dir / "dsh-launch-command.json").write_text(json.dumps(command, indent=2) + "\n")
        return_code = await _execute(
            request,
            instance,
            command,
            deadline,
            "dsh.stdout.log",
            "dsh.stderr.log",
            prompt=prompt,
        )
    except asyncio.CancelledError as exc:
        cancellation = exc
        result.termination_reason = TerminationReason.CANCELLED
    except _ProfileInitializationTimeout:
        result.termination_reason = TerminationReason.AGENT_STARTUP_TIMEOUT
    except (TimeoutError, asyncio.TimeoutError):
        result.termination_reason = TerminationReason.TIMEOUT
    except (_ProfileInitializationError, FileNotFoundError, PermissionError):
        result.termination_reason = TerminationReason.AGENT_STARTUP_FAILED
        (request.artifact_dir / "dsh-runner-error.log").write_text(traceback.format_exc())
    except Exception:
        result.termination_reason = TerminationReason.HARNESS_ERROR
        (request.artifact_dir / "dsh-runner-error.log").write_text(traceback.format_exc())

    cleanup_started = time.monotonic()
    cleanup_deadline = cleanup_started + CLEANUP_SECONDS
    owner = WorkspaceProcessOwner(cleanup_deadline)
    try:
        # Read/copy regular log files synchronously in bounded chunks. No detached
        # collector thread can continue writing after the result is published.
        root_id = _stdout_session(request.artifact_dir / "dsh.stdout.log")
        transcript, events, lineage = _copy_transcript(
            instance.home_dir,
            request.artifact_dir,
            request.workspace,
            root_id=root_id,
            task_id=request.session_id,
            allow_partial=return_code is None or return_code != 0,
            deadline=cleanup_deadline,
        )
        result.transcript = str(transcript)
        result.auto_plan_approvals = lineage["auto_plan_approvals"]
        if return_code == 0 and profile.enforce_plan_mode:
            children = {item["session_id"] for item in lineage["sessions"] if item["agent_role"] == "subagent"}
            if not children:
                raise ValueError("plan-subagent completed without a real subagent")
            if result.auto_plan_approvals == 0:
                raise ValueError("plan-subagent completed without an approved plan")
            if set(lineage["subagent_outcomes"]) != children or any(
                value != "completed" for value in lineage["subagent_outcomes"].values()
            ):
                raise ValueError("plan-subagent exploration failed")
        status["collection"] = "complete" if return_code == 0 else "partial"
        if return_code is not None:
            stderr = _read_tail(request.artifact_dir / "dsh.stderr.log")
            reason = _classify(return_code, stderr, events)
            if lineage.get("unanswered_question"):
                reason = TerminationReason.CONFIRMATION_HANG
            result.termination_reason = reason
            result.outcome = AgentRunOutcome.COMPLETED if reason is None else AgentRunOutcome.FAILED
    except Exception as exc:
        status["collection"] = f"failed: {type(exc).__name__}: {exc}"
        if return_code is not None:
            result.outcome = AgentRunOutcome.FAILED
            result.termination_reason = TerminationReason.HARNESS_ERROR
    try:
        # DSH already awaits child turns and durable writes before exiting.
        # Keep an optional flush delay within half the cleanup budget.
        flush_seconds = min(request.patch_flush_seconds, CLEANUP_SECONDS / 2)
        status["patch_flush_seconds"] = flush_seconds
        if flush_seconds:
            await asyncio.wait_for(asyncio.sleep(flush_seconds), _remaining(cleanup_deadline))
        # The owner bounds and reaps the actual Git subprocesses, rather than
        # timing out only the Python future that waits for them.
        patch = await asyncio.to_thread(export_patch, request.workspace, owner)
        if patch:
            (request.artifact_dir / "model.patch").write_text(patch, encoding="utf-8")
            result.has_patch = True
            result.patch_bytes = len(patch.encode("utf-8"))
        status["patch"] = "exported" if patch else "no_changes"
    except Exception as exc:
        status["patch"] = f"failed: {type(exc).__name__}: {exc}"
        if result.outcome is AgentRunOutcome.COMPLETED:
            result.outcome = AgentRunOutcome.FAILED
            result.termination_reason = TerminationReason.HARNESS_ERROR
    finally:
        owner.terminate_all()
    status.update(
        exit_code=return_code,
        cleanup_seconds=time.monotonic() - cleanup_started,
        cleanup_budget_seconds=CLEANUP_SECONDS,
        termination_reason=result.termination_reason,
    )
    (request.artifact_dir / "dsh-run-status.json").write_text(json.dumps(status, indent=2) + "\n")
    result.finished_at = datetime.now(timezone.utc).isoformat()
    result.duration_seconds = time.monotonic() - started_clock
    if cancellation is not None:
        raise cancellation
    return result


async def _initialize_profile(request: AgentRunRequest, instance: DshInstance, deadline: float) -> None:
    command = [str(request.executable), "--profile", "headless", "--dump-config"]
    try:
        code = await _execute(request, instance, command, deadline, "dsh-init-config.yaml", "dsh-init.stderr.log")
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise _ProfileInitializationTimeout from exc
    if code != 0:
        raise _ProfileInitializationError(
            _read_tail(request.artifact_dir / "dsh-init.stderr.log").decode(errors="replace")
        )


async def _dump_config(request: AgentRunRequest, instance: DshInstance, deadline: float) -> None:
    command = [
        str(request.executable),
        "--profile",
        "headless",
        "--patch",
        str(instance.policy_patch_path),
        "--dump-config",
    ]
    code = await _execute(request, instance, command, deadline, "dsh-effective-config.yaml", "dsh-config.stderr.log")
    if code != 0:
        raise _ProfileInitializationError(
            _read_tail(request.artifact_dir / "dsh-config.stderr.log").decode(errors="replace")
        )
    rows = yaml.load((request.artifact_dir / "dsh-effective-config.yaml").read_text(), Loader=yaml.BaseLoader)
    configs = {row["id"]: row.get("config", {}) for row in rows}
    if configs.get("agent-default-model") != {"provider": "agentbench-vllm", "model": request.model}:
        raise _ProfileInitializationError("DSH effective default model does not match the benchmark route")
    provider = configs.get("llm-pi-ai", {}).get("providers", {}).get("agentbench-vllm", {})
    if provider.get("baseURL") != f"{request.api_base_url.rstrip('/')}/v1":
        raise _ProfileInitializationError("DSH effective pi-ai route bypasses the Request Proxy")


async def _execute(
    request: AgentRunRequest,
    instance: DshInstance,
    command: list[str],
    deadline: float,
    stdout_name: str,
    stderr_name: str,
    *,
    prompt: str | None = None,
) -> int:
    _remaining(deadline)
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=request.workspace,
        env=instance.environment(),
        stdin=asyncio.subprocess.PIPE if prompt is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    tasks = [
        asyncio.create_task(_read_to_file(process.stdout, request.artifact_dir / stdout_name)),
        asyncio.create_task(_read_to_file(process.stderr, request.artifact_dir / stderr_name)),
        asyncio.create_task(process.wait()),
    ]
    if prompt is not None:
        tasks.append(asyncio.create_task(_send_prompt(process.stdin, prompt)))
    group = asyncio.gather(*tasks)
    try:
        with (request.artifact_dir / "dsh-processes.jsonl").open("a") as handle:
            handle.write(
                json.dumps({"pid": process.pid, "pgid": process.pid if os.name != "nt" else None, "command": command})
                + "\n"
            )
        await asyncio.wait_for(group, _remaining(deadline))
    finally:
        # Kill even after the group leader exits: descendants may still own pipes.
        await _kill_process_group(process)
        for task in tasks:
            task.cancel()
        group.cancel()
        await asyncio.gather(group, *tasks, return_exceptions=True)
    return process.returncode


async def _send_prompt(stream: asyncio.StreamWriter, prompt: str) -> None:
    try:
        data = prompt.encode("utf-8")
        for offset in range(0, len(data), 65536):
            stream.write(data[offset : offset + 65536])
            await stream.drain()
    finally:
        stream.close()
        await stream.wait_closed()


async def _read_to_file(stream: asyncio.StreamReader, path: Path) -> None:
    with path.open("wb", buffering=0) as handle:
        while chunk := await stream.read(65536):
            handle.write(chunk)


def _read_tail(path: Path) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 65536))
        return handle.read()


def _remaining(deadline: float) -> float:
    """Return seconds remaining until the deadline, or raise TimeoutError if expired."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("DSH task deadline exceeded")
    return remaining


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """Force-stop the headless process group and wait for it to exit."""

    if os.name == "nt":
        if process.returncode is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


def _stdout_session(path: Path) -> str:
    # The opening event is bounded by upstream. Never load the unbounded final
    # answer or the full stdout stream just to identify this run's root.
    roots = set()
    with path.open("rb") as handle:
        while line := handle.readline(65536):
            if not line.endswith(b"\n"):
                while line and not line.endswith(b"\n"):
                    line = handle.readline(65536)
                continue
            event = json.loads(line)
            if event.get("type") == "session":
                roots.add(event["sessionId"])
    if len(roots) != 1:
        raise ValueError(f"Expected one DSH stdout session identity, got {roots}")
    return roots.pop()


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


def _classify(return_code: int, stderr: bytes, events: list[dict[str, object]]) -> TerminationReason | None:
    """Classify the termination reason from exit code, stderr, and the session log."""

    if b"unanswered user question" in stderr:
        return TerminationReason.CONFIRMATION_HANG
    if not events:
        return TerminationReason.HARNESS_ERROR
    if return_code == 0 and is_turn_complete(events):
        return None
    explicit = _explicit_termination_reason(events)
    if explicit is not None:
        return explicit
    return DshStuckDetector().detect(DshTranscriptNormalizer().normalize(events)) or TerminationReason.INTERRUPTED


def _write_events(path: Path, events: list[dict[str, object]], deadline: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            _remaining(deadline)
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _copy_transcript(
    home: Path,
    artifact_dir: Path,
    workspace: Path | None = None,
    *,
    root_id: str,
    task_id: str,
    allow_partial: bool = False,
    deadline: float,
) -> tuple[Path, list[dict[str, object]], dict[str, object]]:
    logs = find_session_logs(home, workspace)
    if not logs:
        raise ValueError("DSH produced no durable root session log")
    sessions_dir = artifact_dir / "dsh-sessions"
    raw_dir = sessions_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    index = []
    sessions = {}
    main_events = None
    for path in logs:
        target = raw_dir / path.relative_to(home / "sessions")
        target.parent.mkdir(parents=True, exist_ok=True)
        generations = []
        for generation in sorted(path.parent.glob("session*.jsonl*")):
            raw_generation = target.parent / generation.name
            with generation.open("rb") as source, raw_generation.open("wb") as dest:
                while chunk := source.read(65536):
                    _remaining(deadline)
                    dest.write(chunk)
            generations.append(str(raw_generation))
        events = load_session_log(target, allow_partial=allow_partial, deadline=deadline)
        if not events or events[0].get("type") != "session":
            raise ValueError(f"Missing DSH session header: {path}")
        header = events[0]
        session_id = header["id"]
        if session_id in sessions:
            raise ValueError(f"Duplicate DSH session identity: {session_id}")
        sessions[session_id] = header
        decoded = sessions_dir / f"session-{session_id}.jsonl"
        _write_events(decoded, events, deadline)
        index.append(
            {
                "session_id": session_id,
                "source": str(path),
                "raw": str(target),
                "decoded": str(decoded),
                "partial": allow_partial,
                "generations": generations,
            }
        )
        if session_id == root_id:
            if not session_is_root(events):
                raise ValueError("DSH stdout root is a child session")
            main_events = events
    (raw_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    lineage_path = artifact_dir / "dsh-lineage.json"
    lineage = json.loads(lineage_path.read_text())
    if lineage["task_id"] != task_id or set(sessions) != {row["session_id"] for row in lineage["sessions"]}:
        raise ValueError("DSH bridge and durable session identities disagree")
    for row in lineage["sessions"]:
        header = sessions[row["session_id"]]
        if row.get("parent_program_id") != header.get("parentSession") or (
            row["agent_role"] == "lead"
        ) != session_is_root([header]):
            raise ValueError("DSH bridge and durable parent/role disagree")
    if main_events is None:
        raise ValueError(f"Missing root DSH log: {root_id}")
    transcript = artifact_dir / "transcript.jsonl"
    _write_events(transcript, main_events, deadline)
    return transcript, main_events, lineage
