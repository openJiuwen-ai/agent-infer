# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run one SWE-bench task through the JiuwenSwarm JSONL CLI."""

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
from .instance import JiuwenInstance
from .profiles import get_profile


async def run_jiuwenswarm(request: AgentRunRequest) -> AgentRunResult:
    """Execute one task in its own Jiuwen Gateway/AgentServer instance."""

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
    prompt = profile.build_prompt(request.task)
    await asyncio.to_thread((request.artifact_dir / "prompt.txt").write_text, prompt, encoding="utf-8")
    instance = JiuwenInstance(
        executable=request.executable,
        artifact_dir=request.artifact_dir,
        workspace=request.workspace,
        session_id=request.session_id,
        api_base_url=request.api_base_url,
        model=request.model,
        completion_timeout_seconds=request.timeout_seconds,
    )
    process: asyncio.subprocess.Process | None = None
    cancellation: asyncio.CancelledError | None = None
    try:
        await asyncio.wait_for(instance.start(), _remaining(deadline))
        command = [
            str(request.executable),
            "chat",
            "--jsonl",
            "--mode",
            profile.name,
            "--session",
            request.session_id,
            "--cwd",
            str(request.workspace),
            "--project-dir",
            str(request.workspace),
            "--gateway-url",
            instance.gateway_url,
            "--timeout",
            str(request.timeout_seconds),
        ]
        await asyncio.to_thread(
            (request.artifact_dir / "jiuwenswarm-launch-command.json").write_text,
            json.dumps(command, indent=2) + "\n",
            encoding="utf-8",
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=request.workspace,
            env=instance.environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        stdout, stderr, timed_out, jsonl_reason = await _communicate_with_timeout(
            process,
            prompt.encode("utf-8"),
            deadline,
            request.prompt_delivery_timeout_seconds,
        )
        await _write_outputs(request.artifact_dir, stdout, stderr)
        result.transcript = str(request.artifact_dir / "jiuwenswarm.jsonl")
        if timed_out:
            result.termination_reason = TerminationReason.TIMEOUT
            if not await instance.interrupt():
                result.termination_reason = TerminationReason.HARNESS_ERROR
        else:
            if jsonl_reason is None:
                result.outcome = AgentRunOutcome.COMPLETED
                result.termination_reason = None
            else:
                result.termination_reason = jsonl_reason
    except asyncio.TimeoutError:
        result.termination_reason = TerminationReason.AGENT_STARTUP_TIMEOUT
    except asyncio.CancelledError as exc:
        cancellation = exc
        if process is not None:
            await asyncio.shield(_kill_cli_process_group(process))
        await asyncio.shield(instance.interrupt())
    except (FileNotFoundError, PermissionError):
        result.termination_reason = TerminationReason.AGENT_STARTUP_FAILED
    except Exception:
        await asyncio.to_thread(
            (request.artifact_dir / "jiuwenswarm-runner-error.log").write_text,
            traceback.format_exc(),
            encoding="utf-8",
        )
        result.termination_reason = TerminationReason.HARNESS_ERROR
    finally:
        try:
            await asyncio.shield(instance.stop())
        except Exception:
            if cancellation is None:
                result.outcome = AgentRunOutcome.FAILED
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


async def _write_outputs(artifact_dir: Path, stdout: bytes, stderr: bytes) -> None:
    transcript_path = artifact_dir / "jiuwenswarm.jsonl"
    await asyncio.gather(
        asyncio.to_thread(transcript_path.write_bytes, stdout),
        asyncio.to_thread((artifact_dir / "jiuwenswarm.stderr.log").write_bytes, stderr),
    )


class _JsonlStatusTracker:
    """Track CLI status from every JSONL line without retaining the full stream."""

    def __init__(self) -> None:
        self._pending = bytearray()
        self._terminal = False
        self._reason: TerminationReason | None = None

    def feed(self, chunk: bytes) -> None:
        self._pending.extend(chunk)
        while (line_end := self._pending.find(b"\n")) >= 0:
            raw_line = bytes(self._pending[:line_end])
            del self._pending[: line_end + 1]
            self._observe_line(raw_line)

    def finish(self) -> None:
        if self._pending:
            raw_line = bytes(self._pending)
            self._pending.clear()
            self._observe_line(raw_line)

    def classify(self, return_code: int) -> TerminationReason | None:
        """Classify the task termination reason from JSONL events and process exit code.

        JSONL terminal status takes precedence over exit code, as the business-layer
        completion signal is more reliable than OS-level process termination.
        """
        if self._reason is not None:
            return self._reason
        if self._terminal:
            return None
        if return_code != 0:
            return TerminationReason.INTERRUPTED
        return TerminationReason.INTERRUPTED

    def _observe_line(self, raw_line: bytes) -> None:
        if self._reason is not None or not raw_line.strip():
            return
        try:
            frame = json.loads(raw_line)
            event = frame["event"]
            payload = frame["payload"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            self._reason = TerminationReason.HARNESS_ERROR
            return
        if event in {"chat.ask_user_question", "plan.approval_required"}:
            self._reason = TerminationReason.CONFIRMATION_HANG
            return
        if event == "chat.error":
            self._reason = TerminationReason.INTERRUPTED
            return
        if event == "chat.final":
            inner = payload.get("event_type", "")
            if inner == "team.error":
                self._reason = TerminationReason.INTERRUPTED
                return
            if not inner or inner == "chat.final":
                self._terminal = True
        elif event == "chat.processing_status" and not payload.get("is_processing", True):
            self._terminal = True


async def _kill_cli_process_group(process: asyncio.subprocess.Process) -> None:
    """Force-stop the CLI process group and wait for it to exit."""

    if process.returncode is not None:
        return
    if os.name == "nt":
        try:
            process.kill()
        except ProcessLookupError:
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
    await process.wait()


async def _communicate_with_timeout(
    process: asyncio.subprocess.Process,
    prompt: bytes,
    deadline: float,
    prompt_delivery_timeout_seconds: float,
) -> tuple[bytes, bytes, bool, TerminationReason | None]:
    """Deliver prompt to CLI stdin and collect stdout/stderr with timeout enforcement.

    Returns (stdout, stderr, timed_out, termination_reason).
    """
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_read_jsonl_and_track_status(process.stdout))
    stderr_task = asyncio.create_task(_read_all(process.stderr))
    timed_out = False
    try:
        try:
            prompt_timeout = min(_remaining(deadline), prompt_delivery_timeout_seconds)
            await asyncio.wait_for(
                _deliver_prompt(process.stdin, prompt),
                prompt_timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            process.stdin.close()
            await _kill_cli_process_group(process)
        if not timed_out:
            try:
                await asyncio.wait_for(process.wait(), _remaining(deadline))
            except asyncio.TimeoutError:
                timed_out = True
                await _kill_cli_process_group(process)
    except asyncio.CancelledError:
        await asyncio.shield(_kill_cli_process_group(process))
        await asyncio.shield(asyncio.gather(stdout_task, stderr_task))
        raise
    except BaseException:
        await _kill_cli_process_group(process)
        await asyncio.gather(stdout_task, stderr_task)
        raise
    (stdout, status_tracker), stderr = await asyncio.gather(stdout_task, stderr_task)
    return stdout, stderr, timed_out, status_tracker.classify(process.returncode or 0)


async def _deliver_prompt(stdin: asyncio.StreamWriter, prompt: bytes) -> None:
    """Write prompt to CLI stdin and close the stream to signal end of input."""
    try:
        stdin.write(prompt)
        await stdin.drain()
    finally:
        stdin.close()
    await stdin.wait_closed()


async def _read_jsonl_and_track_status(
    stream: asyncio.StreamReader,
) -> tuple[bytes, _JsonlStatusTracker]:
    """Read JSONL stream to completion while tracking status from events."""
    tracker = _JsonlStatusTracker()
    captured = bytearray()
    while chunk := await stream.read(64 * 1024):
        captured.extend(chunk)
        tracker.feed(chunk)
    tracker.finish()
    return bytes(captured), tracker


async def _read_all(stream: asyncio.StreamReader) -> bytes:
    """Read stream to completion."""
    captured = bytearray()
    while chunk := await stream.read(64 * 1024):
        captured.extend(chunk)
    return bytes(captured)


def _remaining(deadline: float) -> float:
    """Return seconds remaining until deadline, or raise TimeoutError if expired."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return remaining
