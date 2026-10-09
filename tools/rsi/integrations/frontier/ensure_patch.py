#!/usr/bin/env python3
"""Idempotently apply the vllm-evolve bridge to a local Frontier checkout."""
from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run_git(
    repo: Path,
    *args: str,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def ensure_patch(
    frontier_repo: str | Path,
    *,
    patch_path: str | Path | None = None,
    runner: Runner = subprocess.run,
) -> str:
    """Return ``applied`` or ``already_applied``; reject drift/partial application."""
    repo = Path(frontier_repo).expanduser().resolve()
    patch = (
        Path(patch_path).expanduser().resolve()
        if patch_path
        else Path(__file__).with_name("ve_policy.patch").resolve()
    )
    if not (repo / "frontier").is_dir():
        raise RuntimeError(f"not a Frontier checkout (missing frontier/): {repo}")
    if not patch.is_file():
        raise RuntimeError(f"bridge patch not found: {patch}")

    forward = _run_git(repo, "apply", "--check", str(patch), runner=runner)
    if forward.returncode == 0:
        applied = _run_git(repo, "apply", str(patch), runner=runner)
        if applied.returncode != 0:
            raise RuntimeError(f"git apply failed after a clean check: {applied.stderr.strip()}")
        return "applied"

    reverse = _run_git(
        repo,
        "apply",
        "--check",
        "--reverse",
        str(patch),
        runner=runner,
    )
    if reverse.returncode == 0:
        return "already_applied"
    detail = forward.stderr.strip() or reverse.stderr.strip() or "unknown git apply error"
    raise RuntimeError(
        "ve_policy patch is neither cleanly applicable nor fully present; "
        f"the Frontier checkout may be on the wrong commit or partially patched: {detail}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontier_repo", help="Path to the local Frontier checkout")
    parser.add_argument("--patch", default=None, help="Override the ve_policy.patch path")
    args = parser.parse_args()
    try:
        status = ensure_patch(args.frontier_repo, patch_path=args.patch)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"ve_policy bridge: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
