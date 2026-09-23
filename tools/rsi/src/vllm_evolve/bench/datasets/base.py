"""Shared types + helpers for real-trace adapters.

Every adapter normalizes a real-world trace into a ``TraceDataset`` (a list of
``TraceRequest``), so the replay load generator and the backend consume one
uniform shape regardless of source format. All file reads use
``encoding="utf-8"`` (this repo is developed on a GBK-default Windows host where
the default encoding silently corrupts non-ASCII bytes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TraceRequest:
    """One request from a trace, in canonical units.

    ``arrival_s`` is a relative offset in **seconds** from the start of the
    trace. ``prefix_hashes`` carries KV-reuse block hashes when the source has
    them (Mooncake); it is empty otherwise. Two requests sharing a leading run
    of ``prefix_hashes`` share that token-block prefix.
    """

    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    request_id: str = ""
    prefix_hashes: tuple[int, ...] = ()


@dataclass
class TraceDataset:
    """A loaded trace, sorted ascending by arrival time."""

    name: str
    requests: list[TraceRequest] = field(default_factory=list)
    parse_errors: int = 0

    @property
    def total_requests(self) -> int:
        return len(self.requests)

    @property
    def duration_s(self) -> float:
        if len(self.requests) < 2:
            return 0.0
        return self.requests[-1].arrival_s - self.requests[0].arrival_s

    @property
    def avg_prompt_tokens(self) -> float:
        if not self.requests:
            return 0.0
        return sum(r.prompt_tokens for r in self.requests) / len(self.requests)

    @property
    def avg_output_tokens(self) -> float:
        if not self.requests:
            return 0.0
        return sum(r.output_tokens for r in self.requests) / len(self.requests)

    @property
    def avg_qps(self) -> float:
        d = self.duration_s
        return (len(self.requests) / d) if d > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Trace '{self.name}': {self.total_requests} reqs, {self.duration_s:.1f}s, "
            f"avg_prompt={self.avg_prompt_tokens:.0f} tok, "
            f"avg_output={self.avg_output_tokens:.0f} tok, avg_qps={self.avg_qps:.2f}"
        )

    def sorted_by_arrival(self) -> TraceDataset:
        self.requests.sort(key=lambda r: r.arrival_s)
        return self


def char_token_estimate(text: str) -> int:
    """Fallback token count when no tokenizer is available (~4 chars/token).

    Used by the ShareGPT adapter when ``transformers`` / a real tokenizer is not
    supplied. Deliberately crude; callers that need accurate counts must pass a
    real tokenizer-backed counter.
    """
    return max(1, math.ceil(len(text) / 4))


def normalize_header(name: str) -> str:
    """Canonicalize a CSV header cell for case/space/underscore-insensitive match."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_column(header: list[str], aliases: set[str]) -> int:
    """Return the index of the first header cell matching any alias.

    Matching is done on normalized names so 'Request tokens', 'request_tokens',
    and 'REQUEST TOKENS' all resolve. Raises ``KeyError`` if none match.
    """
    norm = [normalize_header(h) for h in header]
    for i, h in enumerate(norm):
        if h in aliases:
            return i
    raise KeyError(f"none of {sorted(aliases)} found in header {header}")
