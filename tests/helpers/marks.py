# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""One declaration format for model, card count and CI lane markers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pytest

LANES = frozenset({"ready", "nightly"})
CARDS_RE = re.compile(r"^cards_[1-9][0-9]*$")


def registered_names(config: pytest.Config) -> set[str]:
    return {line.split(":", 1)[0].split("(", 1)[0].strip() for line in config.getini("markers")}


def registered_hardware_names(config: pytest.Config) -> set[str]:
    names = set()
    for line in config.getini("markers"):
        name, _, description = line.partition(":")
        if "[hardware]" in description:
            names.add(name.strip())
    return names


def test_marks(*, hardware: str, num_cards: int | None = None, ci: str | None = None) -> list[pytest.MarkDecorator]:
    """Return standard pytest markers from the same fields used by case JSON."""
    if not isinstance(hardware, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", hardware):
        raise ValueError("hardware must be a model name starting with a letter, containing only letters and digits")
    if hardware == "cpu":
        if num_cards is not None:
            raise ValueError("CPU declarations require hardware='cpu' with no num_cards")
    elif type(num_cards) is not int or num_cards < 1:
        raise ValueError("accelerators require a non-CPU model and a positive integer num_cards")
    if ci is not None and (not isinstance(ci, str) or ci not in LANES):
        raise ValueError("ci must be ready, nightly, or omitted")
    marks = [getattr(pytest.mark, hardware)]
    if num_cards is not None:
        marks.append(getattr(pytest.mark, f"cards_{num_cards}"))
    if ci is not None:
        marks.append(getattr(pytest.mark, ci))
    return marks


# Imported by test modules as their declaration helper, not a test function.
test_marks.__test__ = False


def validate_marks(marks: Iterable[pytest.Mark], config: pytest.Config) -> None:
    registered = registered_names(config)
    hardware_names = registered_hardware_names(config)
    dimensions: dict[str, list[str]] = {"model": [], "cards": [], "lane": []}
    for mark in marks:
        name = mark.name
        if name not in registered:
            raise ValueError(f"unregistered marker {name!r}")
        if name in hardware_names:
            dimensions["model"].append(name)
        elif CARDS_RE.fullmatch(name):
            dimensions["cards"].append(name)
        elif name in LANES:
            dimensions["lane"].append(name)
    for dimension, values in dimensions.items():
        if len(values) > 1:
            raise ValueError(f"multiple {dimension} markers: {values}")
    if not dimensions["model"]:
        raise ValueError("missing model declaration; use test_marks(hardware=..., ci=...) or a case JSON mark object")
    if dimensions["model"] == ["cpu"]:
        if dimensions["cards"]:
            raise ValueError("cpu tests must not specify accelerator card counts")
    elif not dimensions["cards"]:
        raise ValueError("accelerator tests must declare num_cards")


def resolve_pytest_marks(mark_field: Any, config: pytest.Config) -> list[pytest.MarkDecorator]:
    if not isinstance(mark_field, dict) or not set(mark_field) <= {"hardware", "num_cards", "ci"}:
        raise ValueError("mark must be an object containing hardware and optional num_cards/ci")
    if "hardware" not in mark_field:
        raise ValueError("mark requires hardware")
    marks = test_marks(**mark_field)
    validate_marks([mark.mark for mark in marks], config)
    return marks
