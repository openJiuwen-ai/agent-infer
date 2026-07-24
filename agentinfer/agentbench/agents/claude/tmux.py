"""Run Claude Code in a task-owned tmux server."""

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_TMUX_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class TmuxTarget:
    """Identify the tmux server owned by one benchmark task."""

    socket_name: str


_SESSION_NAME = "claude"


class TmuxCommandError(RuntimeError):
    """Report tmux failures without exposing process environment values."""


def control_environment() -> dict[str, str]:
    """Exclude the launch-only token from subsequent tmux commands."""

    env = dict(os.environ)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def _run(
    target: TmuxTarget,
    *args: str,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", "-L", target.socket_name, *args],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=dict(env) if env is not None else control_environment(),
    )


def _checked(
    target: TmuxTarget,
    *args: str,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    result = _run(target, *args, env=env, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise TmuxCommandError(f"tmux {args[0]} failed with rc={result.returncode}: {detail}")
    return result


def start_session(
    target: TmuxTarget,
    *,
    command: Sequence[str],
    cwd: Path,
    launch_env: Mapping[str, str],
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
) -> None:
    """Start Claude in the dedicated tmux server."""

    import shlex

    _checked(
        target,
        "new-session",
        "-d",
        "-s",
        _SESSION_NAME,
        "-c",
        str(cwd),
        ";",
        "set-option",
        "-p",
        "-t",
        _SESSION_NAME,
        "remain-on-exit",
        "on",
        ";",
        "respawn-pane",
        "-k",
        "-t",
        _SESSION_NAME,
        shlex.join(command),
        env=launch_env,
        timeout_seconds=timeout_seconds,
    )


def set_history_limit(
    target: TmuxTarget,
    limit: int,
    *,
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
) -> None:
    """Set pane history for the dedicated server."""

    _checked(
        target,
        "set-option",
        "-g",
        "history-limit",
        str(limit),
        timeout_seconds=timeout_seconds,
    )


def remove_global_environment(
    target: TmuxTarget,
    key: str,
    *,
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
) -> None:
    """Remove a launch-only key retained by the tmux server."""

    _checked(
        target,
        "set-environment",
        "-gu",
        key,
        timeout_seconds=timeout_seconds,
    )


def session_exists(target: TmuxTarget, *, timeout_seconds: float = _TMUX_TIMEOUT_SECONDS) -> bool:
    """Return whether the task session still exists within the command timeout."""

    return (
        _run(
            target,
            "has-session",
            "-t",
            _SESSION_NAME,
            timeout_seconds=timeout_seconds,
        ).returncode
        == 0
    )


def pane_has_exited(target: TmuxTarget, *, timeout_seconds: float = _TMUX_TIMEOUT_SECONDS) -> bool:
    """Return whether the agent process in the retained pane has exited."""

    result = _checked(
        target,
        "display-message",
        "-p",
        "-t",
        _SESSION_NAME,
        "#{pane_dead}",
        timeout_seconds=timeout_seconds,
    )
    return result.stdout.strip() == "1"


def capture_pane(target: TmuxTarget, *, timeout_seconds: float = _TMUX_TIMEOUT_SECONDS) -> str:
    """Capture the current task pane within the command timeout."""

    return _checked(
        target,
        "capture-pane",
        "-t",
        _SESSION_NAME,
        "-p",
        timeout_seconds=timeout_seconds,
    ).stdout


def send_keys(
    target: TmuxTarget,
    *keys: str,
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
) -> None:
    """Send keys to the task pane within the command timeout."""

    _checked(
        target,
        "send-keys",
        "-t",
        _SESSION_NAME,
        *keys,
        timeout_seconds=timeout_seconds,
    )


def load_and_paste_file(
    target: TmuxTarget,
    buffer_name: str,
    path: Path,
    *,
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> None:
    """Load and paste a prompt file within the shared deadline."""

    def remaining() -> float:
        if deadline is None:
            return timeout_seconds
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Claude task deadline expired")
        return min(timeout_seconds, value)

    _checked(
        target,
        "load-buffer",
        "-b",
        buffer_name,
        str(path),
        timeout_seconds=remaining(),
    )
    _checked(
        target,
        "paste-buffer",
        "-d",
        "-b",
        buffer_name,
        "-t",
        _SESSION_NAME,
        timeout_seconds=remaining(),
    )


def _server_exists(target: TmuxTarget, *, timeout_seconds: float) -> bool:
    """Return whether the dedicated tmux server still accepts commands."""

    return _run(target, "list-clients", timeout_seconds=timeout_seconds).returncode == 0


def kill_server(
    target: TmuxTarget,
    *,
    timeout_seconds: float = _TMUX_TIMEOUT_SECONDS,
) -> None:
    """Stop the dedicated server within the cleanup timeout."""

    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("tmux cleanup deadline expired")
        return value

    result = _run(target, "kill-server", timeout_seconds=remaining())
    if result.returncode != 0 and _server_exists(target, timeout_seconds=remaining()):
        detail = (result.stderr or result.stdout).strip()[:500]
        raise TmuxCommandError(f"tmux kill-server failed with rc={result.returncode}: {detail}")
