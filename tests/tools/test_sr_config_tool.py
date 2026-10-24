# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Entrypoint smoke tests for the sr-config-tool archive.

The toolkit is standalone (dash in the directory name, not importable), so
these tests exercise its CLI surfaces via subprocess: a py_compile sweep over
all modules plus `pipeline.py --help` / `--list-tasks` against a temp copy of
the example config. No datasets or GPUs are required.
"""

from __future__ import annotations

import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "sr-config-tool"
SRC_DIR = TOOL_DIR / "src"


def _run(*cmd: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *cmd], capture_output=True, text=True, **kwargs)


def test_all_modules_compile(tmp_path: Path):
    modules = sorted(SRC_DIR.glob("*.py"))
    assert modules, f"no modules found under {SRC_DIR}"
    for module in modules:
        py_compile.compile(str(module), cfile=str(tmp_path / f"{module.stem}.pyc"), doraise=True)


def test_pipeline_help_lists_subcommands():
    result = _run(str(SRC_DIR / "pipeline.py"), "--help")
    assert result.returncode == 0, result.stderr
    for subcommand in ("build-tasks", "convert", "e2e"):
        assert subcommand in result.stdout


def test_list_tasks_with_missing_config_is_friendly(tmp_path: Path):
    result = _run(str(SRC_DIR / "pipeline.py"), "--config", str(tmp_path / "nope.yaml"), "--list-tasks")
    assert result.returncode == 1
    assert "config not found" in result.stderr
    assert "run_config.example.yaml" in result.stderr


def test_list_tasks_with_example_config(tmp_path: Path):
    config = tmp_path / "run_config.yaml"
    shutil.copyfile(TOOL_DIR / "config" / "run_config.example.yaml", config)
    result = _run(str(SRC_DIR / "pipeline.py"), "--config", str(config), "--list-tasks", cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "No tasks found." in result.stdout
