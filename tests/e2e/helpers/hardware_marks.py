# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Turn case JSON ``mark`` entries into pytest marks.

The JSON shape matches vllm-omni: ``res`` keys are accelerator platforms
(``cuda``), not the device-class mark ``gpu``. A registered SKU such as
``L20`` plus ``num_cards`` becomes ``pytest.mark.L20``, ``pytest.mark.cuda``,
``pytest.mark.gpu``, and ``pytest.mark.cards_2``, so ``pytest -m`` can filter.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

_HARDWARE_RESOURCE_MARKER_TAG = "[hardware-resource]"
_HARDWARE_CARDS_MARKER_TAG = "[hardware-cards]"
_HARDWARE_PLATFORM_MARKER_TAG = "[hardware-platform]"
_GPU_DEVICE_CLASS_TAG = "[gpu]"
_CARDS_MARK_NAME_RE = re.compile(r"^cards_(\d+)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _marker_lines() -> list[str]:
    text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    return re.findall(r'^\s*"([^"]+)"\s*,?\s*$', text, flags=re.M)


@lru_cache(maxsize=8)
def _marker_names_with_tag(tag: str) -> frozenset[str]:
    names: set[str] = set()
    for line in _marker_lines():
        if tag not in line:
            continue
        name = line.split(":", 1)[0].strip()
        if name:
            names.add(name)
    return frozenset(names)


def get_hardware_mark_list() -> frozenset[str]:
    return _marker_names_with_tag(_HARDWARE_RESOURCE_MARKER_TAG)


def get_supported_card_counts() -> frozenset[int]:
    counts: set[int] = set()
    for name in _marker_names_with_tag(_HARDWARE_CARDS_MARKER_TAG):
        matched = _CARDS_MARK_NAME_RE.fullmatch(name)
        if matched:
            counts.add(int(matched.group(1)))
    return frozenset(counts)


def get_supported_platforms() -> frozenset[str]:
    return _marker_names_with_tag(_HARDWARE_PLATFORM_MARKER_TAG)


def get_skus_for_platform(platform: str) -> frozenset[str]:
    return get_hardware_mark_list() & _marker_names_with_tag(f"[{platform}]")


def _res_platforms() -> frozenset[str]:
    return frozenset(platform for platform in get_supported_platforms() if get_skus_for_platform(platform))


def _cards_mark(num_cards: int) -> pytest.MarkDecorator:
    if not isinstance(num_cards, int) or isinstance(num_cards, bool) or num_cards < 1:
        raise ValueError(f"num_cards must be a positive int, got {num_cards!r}")
    if num_cards not in get_supported_card_counts():
        supported = ", ".join(f"cards_{count}" for count in sorted(get_supported_card_counts()))
        supported = supported or "(none tagged [hardware-cards])"
        raise ValueError(f"num_cards={num_cards} has no registered marker. Supported: {supported}.")
    return getattr(pytest.mark, f"cards_{num_cards}")


def _cuda_marks(*, res: str, num_cards: int) -> list[pytest.MarkDecorator]:
    allowed = get_skus_for_platform("cuda")
    if res not in allowed:
        supported = ", ".join(sorted(allowed)) or "(none tagged [hardware-resource] [cuda])"
        raise ValueError(f"Invalid cuda resource type: {res}. Supported: {supported}")
    return [getattr(pytest.mark, res), pytest.mark.cuda, pytest.mark.gpu, _cards_mark(num_cards)]


def hardware_marks(*, res: dict[str, str], num_cards: int = 1) -> list[pytest.MarkDecorator]:
    """Return pytest marks for one case. ``res`` keys are platforms, not ``gpu``."""

    allowed = _res_platforms()
    device_class = get_supported_platforms() - allowed
    if not isinstance(res, dict) or not res:
        raise ValueError(f"hardware_marks res must be a non-empty dict, got {res!r}")
    for platform, resource in res.items():
        if platform in device_class:
            raise ValueError(
                f"{platform!r} is not a res dict key; use an accelerator key ({', '.join(sorted(allowed))})."
            )
        if platform not in allowed:
            supported = ", ".join(sorted(allowed)) or "(none)"
            raise ValueError(f"Unsupported platform: {platform}. res keys: {supported}.")
        if not isinstance(resource, str):
            raise ValueError(f"{platform} resource must be a string, got {resource!r}")
    marks: list[pytest.MarkDecorator] = []
    for platform, resource in res.items():
        if platform != "cuda":
            raise ValueError(f"Unsupported platform: {platform}")
        marks.extend(_cuda_marks(res=resource, num_cards=num_cards))
    return marks


def resolve_pytest_marks(mark_field: Any) -> list[pytest.MarkDecorator]:
    """Convert a JSON ``mark`` list into pytest mark decorators."""

    if mark_field is None:
        return []
    if not isinstance(mark_field, list):
        raise ValueError(f"mark must be a list; got {type(mark_field).__name__}")

    marks: list[pytest.MarkDecorator] = []
    seen_hardware = False
    for item in mark_field:
        if isinstance(item, dict) and "hardware_marks" in item:
            if seen_hardware:
                raise ValueError("mark array must contain at most one hardware_marks object")
            seen_hardware = True
            unknown = set(item) - {"hardware_marks"}
            if unknown:
                raise ValueError(
                    f"mark hardware_marks object only allows hardware_marks; unknown keys: {sorted(unknown)}"
                )
            hardware = item["hardware_marks"]
            if not isinstance(hardware, dict):
                raise ValueError(f"mark.hardware_marks must be a dict, got {type(hardware).__name__}")
            res_raw = hardware.get("res")
            if not isinstance(res_raw, dict) or not res_raw:
                raise ValueError(f"hardware_marks res must be a non-empty dict, got {res_raw!r}")
            res: dict[str, str] = {}
            for platform, resource in res_raw.items():
                if not isinstance(platform, str) or not isinstance(resource, str):
                    raise ValueError(f"hardware_marks res entries must be strings, got {platform!r}: {resource!r}")
                res[platform] = resource
            num_cards = hardware.get("num_cards", 1)
            if isinstance(num_cards, bool) or not isinstance(num_cards, int):
                raise ValueError(f"num_cards must be a positive int, got {num_cards!r}")
            marks.extend(hardware_marks(res=res, num_cards=num_cards))
        elif isinstance(item, str) and item.strip():
            marks.append(getattr(pytest.mark, item.strip()))
        else:
            raise ValueError(
                f"mark array entries must be hardware_marks objects or marker name strings; got {type(item).__name__}"
            )
    if not seen_hardware:
        raise ValueError("mark array must contain a hardware_marks object")
    return marks
