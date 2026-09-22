# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validated evidence records for offline ingestion of external validator results."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any

from .taxonomy import Backend, Layer

SCOPE_KEYS = ("engine_version", "model_revision", "workload_id")


class Category(str, Enum):
    CORRECTNESS = "correctness"
    SYSTEM_BEHAVIOR = "system_behavior"
    PERFORMANCE = "performance"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


def require_text(value: object, name: str) -> str:
    """Require a nonblank string rather than coercing missing identifiers."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def validate_scope(scope: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(scope, Mapping):
        raise ValueError("scope must be an object")
    result = {require_text(key, "scope key"): require_text(value, f"scope.{key}") for key, value in scope.items()}
    for key in SCOPE_KEYS:
        require_text(result.get(key), f"scope.{key}")
    return result


@dataclass(frozen=True)
class FeedbackRecord:
    """Record a measured check; a verdict is evidence from a validator, not an LLM rating.

    Scope identifies the exact tested stack, model revision, and workload. Extra
    scope dimensions (shape, dtype, topology, and so on) must also match during
    evaluation. ``diagnostic`` means profiling or instrumentation may perturb
    performance. Evidence references are provenance pointers, not authenticated
    attestations; this module does not execute or verify external artifacts.
    """

    check_id: str
    hypothesis_id: str
    candidate_id: str
    baseline_id: str
    backend: Backend
    layer: Layer
    category: Category
    verdict: Verdict
    metric: str
    value: float | None
    unit: str
    scope: Mapping[str, str]
    diagnostic: bool
    evidence_refs: tuple[str, ...]
    next_test: str
    synthetic: bool = False

    def __post_init__(self) -> None:
        for name in ("check_id", "hypothesis_id", "candidate_id", "baseline_id", "metric", "unit", "next_test"):
            require_text(getattr(self, name), name)
        for name, enum_type in (("backend", Backend), ("layer", Layer), ("category", Category), ("verdict", Verdict)):
            try:
                object.__setattr__(self, name, enum_type(getattr(self, name)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid {name}: {getattr(self, name)!r}") from exc
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError("value must be a finite number or null")
            try:
                finite = math.isfinite(self.value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("value must be finite")
        if self.verdict in (Verdict.PASS, Verdict.FAIL) and self.value is None:
            raise ValueError("pass/fail records require a measured value")
        if self.verdict is Verdict.UNAVAILABLE and self.value is not None:
            raise ValueError("unavailable records must use null, never a fabricated measurement")
        if not isinstance(self.diagnostic, bool) or not isinstance(self.synthetic, bool):
            raise ValueError("diagnostic and synthetic must be booleans")
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError("evidence_refs must be a list or tuple")
        refs = tuple(require_text(ref, "evidence_refs item") for ref in self.evidence_refs)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "scope", MappingProxyType(validate_scope(self.scope)))

    def to_dict(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        for name in ("backend", "layer", "category", "verdict"):
            result[name] = result[name].value
        result["scope"] = dict(self.scope)
        result["evidence_refs"] = list(self.evidence_refs)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeedbackRecord":
        if not isinstance(data, Mapping):
            raise ValueError("feedback record must be an object")
        try:
            return cls(**dict(data))
        except TypeError as exc:
            raise ValueError(f"invalid feedback record fields: {exc}") from exc


def to_dict(record: FeedbackRecord) -> dict[str, Any]:
    return record.to_dict()


def from_dict(data: Mapping[str, Any]) -> FeedbackRecord:
    return FeedbackRecord.from_dict(data)
