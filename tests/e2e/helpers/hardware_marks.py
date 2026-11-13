# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validate model, card count and CI lane labels from tests and case JSON."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pytest

LANES = frozenset({"ready", "nightly"})
CARDS_RE = re.compile(r"^cards_[1-9][0-9]*$")


def registered_names(config: pytest.Config) -> set[str]:
    return {line.split(":", 1)[0].split("(", 1)[0].strip() for line in config.getini("markers")}


def is_hardware(name: str, platforms: dict[str, str]) -> bool:
    return name == "cpu" or name in platforms


def registered_hardware_platforms(config: pytest.Config) -> dict[str, str]:
    platforms = {}
    for line in config.getini("markers"):
        name, _, description = line.partition(":")
        for platform in ("cuda", "npu"):
            if f"[{platform}]" in description:
                platforms[name.strip()] = platform
    return platforms


def validate_names(names: Iterable[str], registered: set[str], *, platforms: dict[str, str]) -> None:
    dimensions: dict[str, list[str]] = {"model": [], "cards": [], "lane": []}
    for name in names:
        if name not in registered:
            raise ValueError(f"unregistered marker {name!r}")
        if is_hardware(name, platforms):
            dimensions["model"].append(name)
        elif CARDS_RE.fullmatch(name):
            dimensions["cards"].append(name)
        elif name in LANES:
            dimensions["lane"].append(name)
    for dimension, values in dimensions.items():
        if len(values) > 1:
            raise ValueError(f"multiple {dimension} markers: {values}")
    if dimensions["model"] == ["cpu"] and dimensions["cards"]:
        raise ValueError("cpu tests must not specify accelerator card counts")


def resolve_pytest_marks(
    mark_field: Any, registered: set[str], *, platforms: dict[str, str]
) -> list[pytest.MarkDecorator]:
    if not isinstance(mark_field, list):
        raise ValueError("mark must be a list")
    names: list[str] = []
    hardware_seen = False
    for entry in mark_field:
        if isinstance(entry, str):
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
        if model in registered and platforms.get(model) != platform:
            raise ValueError(f"hardware model {model!r} is not registered for platform {platform!r}")
        names.extend([model, f"cards_{count}"])
    validate_names(names, registered, platforms=platforms)
    return [getattr(pytest.mark, name) for name in names]
