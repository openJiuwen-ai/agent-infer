"""Prepare benchmark task repositories and workspaces.

Tasks share a repo cache fetched from GitHub, then each task workspace is
materialized at its base commit and re-initialized as a small git repository so
agent edits can be exported as a patch.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .dataset import Task

_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.-]*/[A-Za-z0-9_.][A-Za-z0-9_.-]*$")
_OUTPUT_SNIPPET_CHARS = 1000


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
) -> subprocess.CompletedProcess:
    """Run a workspace command and raise a compact diagnostic on failure."""

    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        input=input,
        env=env,
        text=text,
    )
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
) -> subprocess.CompletedProcess:
    """Run a git command in a repo or workspace directory."""

    return _run_command(["git", *args], cwd=cwd, input=input, env=env, text=text, check=check)


def validate_repo_name(repo: str) -> str:
    """Return a safe GitHub owner/repo name or raise for malformed task data."""

    if not _GITHUB_REPO_RE.fullmatch(repo):
        raise ValueError(f"Invalid benchmark repository name: {repo!r}")
    return repo


def ensure_repo_cache(task: Task, repo_cache: Path) -> Path:
    """Ensure the shared repo cache contains the task base commit."""

    repo = validate_repo_name(task.repo)
    repo_name = repo.replace("/", "__")
    cache_dir = repo_cache / repo_name

    if not (cache_dir / ".git").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        _run_git(["init"], cwd=cache_dir)
        _run_git(["remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=cache_dir)
        _run_git(["fetch", "--depth", "1", "origin", task.base_commit], cwd=cache_dir)

    if not _commit_available(cache_dir, task.base_commit):
        _run_git(["fetch", "--depth", "1", "origin", task.base_commit], cwd=cache_dir)

    return cache_dir


def _commit_available(repo: Path, commit: str) -> bool:
    """Return whether a commit object is available in the cached repo."""

    result = _run_git(["cat-file", "-e", commit + "^{commit}"], cwd=repo, check=False)
    return result.returncode == 0


def prepare_workspace(
    task: Task,
    repo_cache: Path,
    destination: Path,
    *,
    clean: bool = False,
) -> Path:
    """Materialize a task workspace from the cached repository.

    Existing workspaces are reused unless ``clean`` is true. New workspaces are
    unpacked with ``git archive`` from the repo cache, then initialized as a
    fresh one-commit git repository so later agent edits can be collected with
    ``git diff``.
    """

    repo = validate_repo_name(task.repo)
    repo_name = repo.replace("/", "__")
    cache_dir = repo_cache / repo_name

    if destination.exists():
        if clean:
            shutil.rmtree(destination)
        else:
            return destination

    destination.mkdir(parents=True, exist_ok=True)

    archive_bytes = _run_git(["archive", task.base_commit], cwd=cache_dir).stdout

    _run_command(
        ["tar", "xf", "-"],
        cwd=destination,
        input=archive_bytes,
    )

    _run_git(["init"], cwd=destination)
    _run_git(["add", "-A", "-f"], cwd=destination)
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
