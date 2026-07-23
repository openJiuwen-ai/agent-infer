"""Shared metric contracts."""

from collections.abc import Mapping
from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class EvidenceCapture:
    """Describe raw evidence availability without interpreting its metrics."""

    source: str
    path: Path | None
    available: bool
    reason: str | None
    metadata: Mapping[str, object]
    applicable: bool = True


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SourceHealth:
    """Summarize availability across applicable raw evidence sources."""

    available: int
    unavailable: int
    not_applicable: int
    reasons: tuple[str, ...]
    sources: Mapping[str, tuple[Mapping[str, object], ...]]
