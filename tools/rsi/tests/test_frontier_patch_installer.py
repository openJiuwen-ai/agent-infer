from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from integrations.frontier.ensure_patch import ensure_patch


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_patch_installer_is_idempotent(tmp_path):
    repo = tmp_path / "Frontier"
    (repo / "frontier").mkdir(parents=True)
    target = repo / "frontier" / "marker.txt"
    target.write_text("before\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", "frontier/marker.txt")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "base",
    )
    patch = tmp_path / "bridge.patch"
    patch.write_text(
        "diff --git a/frontier/marker.txt b/frontier/marker.txt\n"
        "index 90be1a7..3b18e51 100644\n"
        "--- a/frontier/marker.txt\n"
        "+++ b/frontier/marker.txt\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n",
        encoding="utf-8",
    )

    assert ensure_patch(repo, patch_path=patch) == "applied"
    assert target.read_text(encoding="utf-8") == "after\n"
    assert ensure_patch(repo, patch_path=patch) == "already_applied"


def test_patch_installer_rejects_partial_or_wrong_tree(tmp_path):
    repo = tmp_path / "Frontier"
    (repo / "frontier").mkdir(parents=True)
    _git(repo, "init")
    patch = tmp_path / "bridge.patch"
    patch.write_text(
        "diff --git a/frontier/missing.txt b/frontier/missing.txt\n"
        "--- a/frontier/missing.txt\n"
        "+++ b/frontier/missing.txt\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="neither cleanly applicable nor fully present"):
        ensure_patch(repo, patch_path=patch)
