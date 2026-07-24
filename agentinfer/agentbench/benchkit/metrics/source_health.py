"""Evaluate availability and applicability of raw evidence captures."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict

from .schema import EvidenceCapture, SourceHealth


def evaluate_captures(captures: Iterable[EvidenceCapture]) -> SourceHealth:
    """Summarize available, unavailable, and inapplicable evidence captures."""

    rows = list(captures)
    applicable = [row for row in rows if row.applicable]
    reasons = tuple(row.reason for row in applicable if not row.available and row.reason)
    sources: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        sources.setdefault(row.source, []).append(
            {
                **asdict(row),
                "path": str(row.path) if row.path is not None else None,
            }
        )
    return SourceHealth(
        sum(row.available for row in applicable),
        sum(not row.available for row in applicable),
        sum(not row.applicable for row in rows),
        reasons,
        {source: tuple(capture_list) for source, capture_list in sources.items()},
    )
