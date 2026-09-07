# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prepare benchmark task repositories and workspaces.

Tasks share a repo cache fetched from GitHub, then each task workspace is
materialized at its base commit and re-initialized as a small git repository so
agent edits can be exported as a patch.
"""

import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .dataset import Task

_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.-]*/[A-Za-z0-9_.][A-Za-z0-9_.-]*$")
_OUTPUT_SNIPPET_CHARS = 1000


class WorkspaceProcessOwner:
    """Run workspace subprocesses and terminate their process trees on demand."""

    def __init__(self, deadline: float | None = None) -> None:
        self.deadline = deadline
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._cancelled = False

    def run(
        self,
        args: list[str],
        *,
        cwd: Path,
        input: bytes | None = None,
        env: dict[str, str] | None = None,
        text: bool = False,
    ) -> subprocess.CompletedProcess:
        options = (
            {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        )
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdin=subprocess.PIPE if input is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=text,
            **options,
        )
        with self._lock:
            if self._cancelled:
                terminate = True
            else:
                self._processes.add(process)
                terminate = False
        if terminate:
            self._terminate(process)
            raise TimeoutError("Benchmark workspace setup was cancelled")
        try:
            timeout = None if self.deadline is None else max(0, self.deadline - time.monotonic())
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._terminate(process)
            raise TimeoutError(f"Workspace command timed out: {subprocess.list2cmdline(args)}") from exc
        finally:
            with self._lock:
                self._processes.discard(process)
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)

    def terminate_all(self) -> None:
        with self._lock:
            self._cancelled = True
            processes = tuple(self._processes)
        for process in processes:
            self._terminate(process)

    def check_active(self) -> None:
        with self._lock:
            if self._cancelled:
                raise TimeoutError("Benchmark workspace setup was cancelled")

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _output_snippet(value: bytes | str | None) -> str:
    """Decode and trim command output for compact failure messages."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    text = text.strip()
    if len(text) > _OUTPUT_SNIPPET_CHARS:
        return text[:_OUTPUT_SNIPPET_CHARS] + "...<truncated>"
    return text


def _command_failed(args: list[str], cwd: Path, result: subprocess.CompletedProcess) -> RuntimeError:
    """Build a command failure error with command, cwd, and output context."""

    details = [
        f"Command failed with exit code {result.returncode}: {subprocess.list2cmdline(args)}",
        f"cwd: {cwd}",
    ]
    stderr = _output_snippet(result.stderr)
    stdout = _output_snippet(result.stdout)
    if stderr:
        details.append(f"stderr: {stderr}")
    if stdout:
        details.append(f"stdout: {stdout}")
    return RuntimeError("\n".join(details))


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    input: bytes | None = None,
    env: dict[str, str] | None = None,
    text: bool = False,
    check: bool = True,
    owner: WorkspaceProcessOwner | None = None,
) -> subprocess.CompletedProcess:
    """Run a workspace command and raise a compact diagnostic on failure."""

    if owner is None:
        result = subprocess.run(args, cwd=cwd, capture_output=True, input=input, env=env, text=text)
    else:
        result = owner.run(args, cwd=cwd, input=input, env=env, text=text)
    if check and result.returncode != 0:
        raise _command_failed(args, cwd, result)
    return result


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    input: bytes | None = None,
    env: dict[str, str] | None = None,
    text: bool = False,
    check: bool = True,
    owner: WorkspaceProcessOwner | None = None,
) -> subprocess.CompletedProcess:
    """Run a git command in a repo or workspace directory."""

    return _run_command(["git", *args], cwd=cwd, input=input, env=env, text=text, check=check, owner=owner)


def validate_repo_name(repo: str) -> str:
    """Return a safe GitHub owner/repo name or raise for malformed task data."""

    if not _GITHUB_REPO_RE.fullmatch(repo):
        raise ValueError(f"Invalid benchmark repository name: {repo!r}")
    return repo


def ensure_repo_cache(task: Task, repo_cache: Path, owner: WorkspaceProcessOwner | None = None) -> Path:
    """Ensure the shared repo cache contains the task base commit."""

    if owner is not None:
        owner.check_active()
    repo = validate_repo_name(task.repo)
    repo_name = repo.replace("/", "__")
    cache_dir = repo_cache / repo_name

    if not (cache_dir / ".git").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        _run_git(["init"], cwd=cache_dir, owner=owner)
        _run_git(["remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=cache_dir, owner=owner)
        _run_git(["fetch", "--depth", "1", "origin", task.base_commit], cwd=cache_dir, owner=owner)

    if not _commit_available(cache_dir, task.base_commit, owner):
        _run_git(["fetch", "--depth", "1", "origin", task.base_commit], cwd=cache_dir, owner=owner)

    return cache_dir


def _commit_available(repo: Path, commit: str, owner: WorkspaceProcessOwner | None = None) -> bool:
    """Return whether a commit object is available in the cached repo."""

    result = _run_git(["cat-file", "-e", commit + "^{commit}"], cwd=repo, check=False, owner=owner)
    return result.returncode == 0


def prepare_workspace(
    task: Task,
    repo_cache: Path,
    destination: Path,
    owner: WorkspaceProcessOwner | None = None,
    *,
    clean: bool = False,
) -> Path:
    """Materialize a task workspace from the cached repository.

    Existing workspaces are reused unless ``clean`` is true. New workspaces are
    unpacked with ``git archive`` from the repo cache, then initialized as a
    fresh one-commit git repository so later agent edits can be collected with
    ``git diff``.
    """

    if owner is not None:
        owner.check_active()
    repo = validate_repo_name(task.repo)
    repo_name = repo.replace("/", "__")
    cache_dir = repo_cache / repo_name

    if destination.exists():
        if clean:
            shutil.rmtree(destination)
        else:
            return destination

    destination.mkdir(parents=True, exist_ok=True)

    archive_bytes = _run_git(["archive", task.base_commit], cwd=cache_dir, owner=owner).stdout

    _run_command(
        ["tar", "xf", "-"],
        cwd=destination,
        input=archive_bytes,
        owner=owner,
    )

    _run_git(["init"], cwd=destination, owner=owner)
    _run_git(["add", "-A", "-f"], cwd=destination, owner=owner)
    _run_git(
        ["commit", "-m", f"base {task.base_commit}"],
        cwd=destination,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "agentinfer",
            "GIT_AUTHOR_EMAIL": "agentinfer@bench",
            "GIT_COMMITTER_NAME": "agentinfer",
            "GIT_COMMITTER_EMAIL": "agentinfer@bench",
        },
        owner=owner,
    )

    return destination


def verify_workspace(workspace: Path) -> bool:
    """Return whether the workspace is the expected one-commit git repo."""

    result = _run_git(["rev-list", "--count", "HEAD"], cwd=workspace, text=True, check=False)
    return result.returncode == 0 and result.stdout.strip() == "1"


def workspace_has_changes(workspace: Path) -> bool:
    """Return whether tracked or untracked workspace changes exist."""

    result = _run_git(
        ["status", "--porcelain", "--untracked-files=normal"],
        cwd=workspace,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        text=True,
    )
    return bool(result.stdout)


def export_patch(workspace: Path) -> str:
    """Export tracked and untracked workspace changes from the base commit."""

    with tempfile.TemporaryDirectory() as temp_dir:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temp_dir) / "index")}
        _run_git(["read-tree", "HEAD"], cwd=workspace, env=env)
        _run_git(["add", "-N", "-A"], cwd=workspace, env=env)
        result = _run_git(["diff", "HEAD", "--binary"], cwd=workspace, env=env, text=True)
    return result.stdout
