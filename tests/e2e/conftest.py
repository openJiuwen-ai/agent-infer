# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared pytest options for functional and performance E2E tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.helpers.hardware_marks import (
    registered_hardware_platforms,
    registered_names,
    resolve_pytest_marks,
)

_SELECTED_CASE = pytest.StashKey[Path | None]()


def pytest_configure(config: pytest.Config) -> None:
    raw = config.getoption("--test-config-file")
    selected = Path(raw).resolve() if raw else None
    config.stash[_SELECTED_CASE] = selected
    if selected is None:
        return
    root = Path(__file__).resolve().parent
    case_roots = [
        path.resolve()
        for path in root.rglob("cases")
        if path.is_dir() and (any(path.parent.glob("test_*.py")) or (path.parent / "run_benchmark.py").is_file())
    ]
    if (
        selected.suffix != ".json"
        or not selected.is_file()
        or not any(selected.is_relative_to(path) for path in case_roots)
    ):
        raise pytest.UsageError(f"--test-config-file must name an existing JSON in an E2E suite's cases tree: {raw}")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected = config.stash.get(_SELECTED_CASE, None)
    if selected is not None and not any(
        getattr(item, "callspec", None) is not None and item.callspec.params.get("case_file") == selected
        for item in items
    ):
        raise pytest.UsageError(f"--test-config-file is not handled by any collected suite: {selected}")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_file" not in metafunc.fixturenames:
        return
    cases_dir = (Path(metafunc.definition.path).parent / "cases").resolve()
    selected = metafunc.config.stash.get(_SELECTED_CASE, None)
    paths = [selected] if selected else sorted(path.resolve() for path in cases_dir.rglob("*.json"))
    registered = registered_names(metafunc.config)
    platforms = registered_hardware_platforms(metafunc.config)
    params = []
    for path in paths:
        try:
            if not path.is_relative_to(cases_dir):
                # A CLI override selects only the suite that owns the case.
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("case JSON must be an object")
            marks = resolve_pytest_marks(payload.get("mark", []), registered, platforms=platforms)
            case_id = path.relative_to(cases_dir).with_suffix("").as_posix()
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
