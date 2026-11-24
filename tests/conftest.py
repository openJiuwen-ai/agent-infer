# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from pathlib import Path

import pytest

from tests.helpers.marks import validate_marks


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    tests_root = Path(__file__).parent
    for item in items:
        path = Path(item.path)
        if not path.is_relative_to(tests_root):
            continue
        callspec = getattr(item, "callspec", None)
        if (
            path.is_relative_to(tests_root / "e2e")
            and callspec is not None
            and "case_file" in callspec.params
            and not isinstance(callspec.params["case_file"], Path)
            and item.get_closest_marker("skip") is not None
        ):
            # Pytest's empty-parameter placeholder for an unselected E2E suite.
            continue
        try:
            validate_marks(item.iter_markers(), config)
        except ValueError as exc:
            raise pytest.UsageError(f"{item.nodeid}: {exc}") from exc
