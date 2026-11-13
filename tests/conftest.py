# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from pathlib import Path

import pytest

from tests.e2e.helpers.hardware_marks import (
    CARDS_RE,
    LANES,
    is_hardware,
    registered_hardware_platforms,
    registered_names,
    validate_names,
)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    registered = registered_names(config)
    platforms = registered_hardware_platforms(config)
    tests_root = Path(__file__).parent
    for item in items:
        path = Path(item.path)
        if not path.is_relative_to(tests_root):
            continue
        names = [mark.name for mark in item.iter_markers()]
        if not path.is_relative_to(tests_root / "e2e") and not any(
            is_hardware(name, platforms) or CARDS_RE.fullmatch(name) for name in names
        ):
            defaults = ["cpu"]
            if not LANES.intersection(names):
                defaults.append("ready")
            for name in defaults:
                item.add_marker(name)
                names.append(name)
        try:
            validate_names(names, registered, platforms=platforms)
        except ValueError as exc:
            raise pytest.UsageError(f"{item.nodeid}: {exc}") from exc
