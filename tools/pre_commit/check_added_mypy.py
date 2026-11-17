# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Run the configured mypy hook only for staged additions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath


def main() -> int:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=A", "-z"],
        capture_output=True,
        check=True,
    )
    files = [
        name
        for raw in result.stdout.split(b"\0")
        if raw
        and (name := raw.decode("utf-8", errors="surrogateescape"))
        and PurePosixPath(name).suffix in {".py", ".pyi"}
        and not name.startswith("inferrsi/")
    ]
    if not files:
        print("No staged Python additions to check with mypy.")
        return 0
    return subprocess.run(
        [sys.executable, "-m", "pre_commit", "run", "mypy", "--hook-stage", "manual", "--files", *files],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
