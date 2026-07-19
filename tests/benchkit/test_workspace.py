import subprocess
from pathlib import Path

import pytest

from agentcache.benchmarks.benchkit.dataset import Task
from agentcache.benchmarks.benchkit.workspace import (
    _run_command,
    ensure_repo_cache,
    export_patch,
    prepare_workspace,
    validate_repo_name,
    verify_workspace,
    workspace_has_changes,
)


@pytest.mark.parametrize(
    "repo",
    [
        "django/django",
        "psf/requests",
        "owner.name/repo-name",
        "owner_name/repo.name",
    ],
)
def test_validate_repo_name_accepts_safe_owner_repo(repo: str) -> None:
    assert validate_repo_name(repo) == repo


@pytest.mark.parametrize(
    "repo",
    [
        "owner",
        "owner/repo/extra",
        "owner/repo\nextra",
        "--upload-pack=x/repo",
        "owner/--repo",
        "/repo",
        "owner/",
    ],
)
def test_validate_repo_name_rejects_malformed_or_option_like_values(repo: str) -> None:
    with pytest.raises(ValueError, match="Invalid benchmark repository name"):
        validate_repo_name(repo)


def test_ensure_repo_cache_rejects_invalid_repo_before_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("git should not run for invalid repo names")

    monkeypatch.setattr("agentcache.benchmarks.benchkit.workspace.subprocess.run", fail_if_called)

    task = Task(instance_id="task-a", repo="owner/repo\nextra", base_commit="abc", problem_statement="p")
    with pytest.raises(ValueError, match="Invalid benchmark repository name"):
        ensure_repo_cache(task, tmp_path / "repo-cache")


def test_ensure_repo_cache_initializes_and_fetches_missing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("agentcache.benchmarks.benchkit.workspace.subprocess.run", fake_run)
    task = Task(instance_id="task-a", repo="owner/repo", base_commit="abc123", problem_statement="p")

    cache = ensure_repo_cache(task, tmp_path / "repo-cache")

    assert cache == tmp_path / "repo-cache" / "owner__repo"
    assert ["git", "init"] in calls
    assert ["git", "remote", "add", "origin", "https://github.com/owner/repo.git"] in calls
    assert ["git", "fetch", "--depth", "1", "origin", "abc123"] in calls


def test_ensure_repo_cache_fetches_commit_missing_from_existing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "repo-cache" / "owner__repo"
    cache.mkdir(parents=True)
    (cache / ".git").mkdir()
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(args))
        returncode = 1 if args[:3] == ["git", "cat-file", "-e"] else 0
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=b"", stderr=b"")

    monkeypatch.setattr("agentcache.benchmarks.benchkit.workspace.subprocess.run", fake_run)
    task = Task(instance_id="task-a", repo="owner/repo", base_commit="abc123", problem_statement="p")

    assert ensure_repo_cache(task, tmp_path / "repo-cache") == cache
    assert calls.count(["git", "fetch", "--depth", "1", "origin", "abc123"]) == 1


def test_prepare_workspace_reuses_existing_destination_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("git should not run when workspace already exists")

    monkeypatch.setattr("agentcache.benchmarks.benchkit.workspace.subprocess.run", fail_if_called)

    destination = tmp_path / "workspace"
    destination.mkdir()
    task = Task(instance_id="task-a", repo="owner/repo", base_commit="abc", problem_statement="p")

    assert prepare_workspace(task, tmp_path / "repo-cache", destination) == destination


def test_prepare_workspace_clean_rebuilds_from_cached_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path, bytes | None]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        calls.append((list(args), Path(kwargs["cwd"]), kwargs.get("input")))
        stdout = b"archive-bytes" if args == ["git", "archive", "abc123"] else b""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr("agentcache.benchmarks.benchkit.workspace.subprocess.run", fake_run)

    destination = tmp_path / "workspace"
    destination.mkdir()
    (destination / "stale.txt").write_text("old", encoding="utf-8")
    task = Task(instance_id="task-a", repo="owner/repo", base_commit="abc123", problem_statement="p")

    assert prepare_workspace(task, tmp_path / "repo-cache", destination, clean=True) == destination
    assert not (destination / "stale.txt").exists()

    command_args = [call[0] for call in calls]
    assert ["git", "archive", "abc123"] in command_args
    assert ["git", "add", "-A", "-f"] in command_args
    assert ["git", "commit", "-m", "base abc123"] in command_args
    tar_call = next(call for call in calls if call[0] == ["tar", "xf", "-"])
    assert tar_call[2] == b"archive-bytes"


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "1\n", True),
        (0, "0\n", False),
        (0, "2\n", False),
        (1, "", False),
    ],
)
def test_verify_workspace_requires_one_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    expected: bool,
) -> None:
    def fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr("agentcache.benchmarks.benchkit.workspace.subprocess.run", fake_run)

    assert verify_workspace(tmp_path) is expected


def test_workspace_has_changes_is_read_only(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "base"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    index_before = (tmp_path / ".git" / "index").read_bytes()

    assert workspace_has_changes(tmp_path) is False
    (tmp_path / "new.txt").write_text("new\n", encoding="utf-8")
    assert workspace_has_changes(tmp_path) is True
    assert (tmp_path / ".git" / "index").read_bytes() == index_before


def test_export_patch_includes_untracked_binary_without_changing_index(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "base"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    tracked.write_text("changed\n", encoding="utf-8")
    (tmp_path / "new.bin").write_bytes(b"\x00\x01\x02")
    index_before = (tmp_path / ".git" / "index").read_bytes()

    patch = export_patch(tmp_path)

    assert "diff --git a/tracked.txt b/tracked.txt" in patch
    assert "diff --git a/new.bin b/new.bin" in patch
    assert "GIT binary patch" in patch
    assert (tmp_path / ".git" / "index").read_bytes() == index_before


def test_run_command_failure_includes_command_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(args=args, returncode=7, stdout=b"partial output", stderr=b"fatal: nope")

    monkeypatch.setattr("agentcache.benchmarks.benchkit.workspace.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as exc:
        _run_command(["git", "fetch"], cwd=tmp_path)

    message = str(exc.value)
    assert "git fetch" in message
    assert f"cwd: {tmp_path}" in message
    assert "exit code 7" in message
    assert "fatal: nope" in message
