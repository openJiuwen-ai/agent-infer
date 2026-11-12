# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared pytest options for functional and performance E2E tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.helpers.hardware_marks import registered_names, resolve_pytest_marks, scene_for_path


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_file" not in metafunc.fixturenames:
        return
    suite_dir = Path(metafunc.definition.path).parent
    selected = metafunc.config.getoption("--test-config-file")
    paths = [Path(selected).resolve()] if selected else sorted((suite_dir / "cases").rglob("*.json"))
    params = []
    for path in paths:
        try:
            if not path.is_relative_to((suite_dir / "cases").resolve()):
                # A CLI override selects only the suite that owns the case.
                if not path.is_file():
                    raise ValueError("case file does not exist")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("case JSON must be an object")
            marks = resolve_pytest_marks(
                payload.get("mark", []), registered_names(metafunc.config), scene=scene_for_path(path)
            )
            case_id = path.relative_to(suite_dir / "cases").with_suffix("").as_posix()
            params.append(pytest.param(path, marks=marks, id=case_id))
        except (OSError, ValueError, TypeError) as exc:
            raise pytest.UsageError(f"{metafunc.definition.nodeid} [{path}]: {exc}") from exc
    metafunc.parametrize("case_file", params)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e_shared", "Shared E2E options")
    group.addoption(
        "--test-config-file",
        action="store",
        default=None,
        metavar="PATH",
        help="Case JSON path for functional or performance E2E tests",
    )
    group.addoption(
        "--task-num",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params task-num",
    )
    group.addoption(
        "--max-concurrency",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params max-concurrency",
    )
    group.addoption(
        "--benchmark-run-as-user",
        action="store",
        default=None,
        metavar="USER",
        help="For plan-subagent: run the benchmark as USER via sudo when pytest runs as root",
    )
