# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Apply the selected DSH case JSON ``mark`` field as pytest marks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.helpers.hardware_marks import resolve_pytest_marks


def pytest_itemcollected(item: pytest.Item) -> None:
    if item.name != "test_dsh_swebench":
        return
    raw = item.config.getoption("--test-config-file")
    if not raw:
        return
    path = Path(str(raw))
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for mark in resolve_pytest_marks(payload.get("mark")):
        item.add_marker(mark)
