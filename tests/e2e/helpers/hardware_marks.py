# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validate test labels and translate case hardware into one marker."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

SCENES = frozenset({"function", "perf", "acc"})
LANES = frozenset({"ready", "nightly"})
HARDWARE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[1-9][0-9]*$")


def registered_names(config: pytest.Config) -> set[str]:
    return {line.split(":", 1)[0].split("(", 1)[0].strip() for line in config.getini("markers")}


def is_hardware(name: str) -> bool:
    return name == "cpu" or bool(HARDWARE_RE.fullmatch(name))


def validate_names(names: Iterable[str], registered: set[str], *, scene: str | None = None) -> None:
    dimensions: dict[str, list[str]] = {"hardware": [], "scene": [], "lane": []}
    for name in names:
        if name not in registered:
            raise ValueError(f"unregistered marker {name!r}")
        if is_hardware(name):
            dimensions["hardware"].append(name)
        elif name in SCENES:
            dimensions["scene"].append(name)
        elif name in LANES:
            dimensions["lane"].append(name)
    for dimension, values in dimensions.items():
        if len(values) > 1:
            raise ValueError(f"multiple {dimension} markers: {values}")
    if dimensions["scene"] and dimensions["scene"] != [scene]:
        raise ValueError(f"scene {dimensions['scene'][0]!r} does not match directory {scene!r}")


def scene_for_path(path: Path) -> str | None:
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index : index + 2] == ("tests", "e2e"):
            return parts[index + 2] if parts[index + 2] in SCENES else None
    return None


def resolve_pytest_marks(
    mark_field: Any, registered: set[str], *, scene: str | None = None
) -> list[pytest.MarkDecorator]:
    if not isinstance(mark_field, list):
        raise ValueError("mark must be a list")
    names: list[str] = []
    hardware_seen = False
    for entry in mark_field:
        if isinstance(entry, str):
            if is_hardware(entry) and entry != "cpu":
                raise ValueError("accelerator markers must use a hardware_marks object")
            names.append(entry)
            continue
        if not isinstance(entry, dict) or set(entry) != {"hardware_marks"}:
            raise ValueError("mark entries must be strings or hardware_marks objects")
        if hardware_seen:
            raise ValueError("mark permits at most one hardware_marks object")
        hardware_seen = True
        hardware = entry["hardware_marks"]
        if not isinstance(hardware, dict) or set(hardware) != {"res", "num_cards"}:
            raise ValueError("hardware_marks requires only res and num_cards")
        resources, count = hardware["res"], hardware["num_cards"]
        if not isinstance(resources, dict) or len(resources) != 1:
            raise ValueError("res must contain exactly one platform and model")
        platform, model = next(iter(resources.items()))
        if platform not in {"cuda", "npu"}:
            raise ValueError(f"unsupported hardware platform {platform!r}")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", model):
            raise ValueError("hardware model must start with a letter and contain only letters and digits")
        if type(count) is not int or count < 1:
            raise ValueError("num_cards must be a positive integer")
        names.append(f"{model}_{count}")
    validate_names(names, registered, scene=scene)
    return [getattr(pytest.mark, name) for name in names]
