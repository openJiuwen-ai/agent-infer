#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validate Buildkite YAML after the same expansion upload_pipeline.py uploads.

Test pipelines use ``mirror_hardwares`` and ``host_only``. Those keys are
expanded or removed before upload, and the rendered YAML is checked against
the Buildkite schema. ``hardware.yml`` is the preset catalog, not a pipeline.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".buildkite"))

from upload_pipeline import render  # noqa: E402

SKIP_FILES = {
    ".buildkite/hardware.yml",
}
HARDWARE = ROOT / ".buildkite" / "hardware.yml"


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _relative(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel = path
    return _normalize(str(rel))


def _validate(path: Path, rendered: str) -> int:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(rendered)
        tmp = Path(handle.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "check_jsonschema",
                "--builtin-schema",
                "vendor.buildkite",
                str(tmp),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        tmp.unlink(missing_ok=True)

    if result.returncode == 0:
        return 0

    sys.stderr.write(f"{_relative(path)}: Buildkite schema validation failed\n")
    sys.stderr.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return 1


def main(argv: list[str]) -> int:
    rc = 0
    for raw in argv:
        path = Path(raw)
        if not path.is_file():
            path = ROOT / raw
        rel = _relative(path)
        if rel in SKIP_FILES:
            continue
        try:
            rendered = render(path, HARDWARE)
        except SystemExit as exc:
            message = exc.code if isinstance(exc.code, str) else exc
            sys.stderr.write(f"{rel}: {message}\n")
            rc = 1
            continue
        rc |= _validate(path, rendered)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
