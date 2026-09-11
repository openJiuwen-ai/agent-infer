# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Aggregate raw request facts into request metrics and observed topology."""

from collections.abc import Iterable

from numpy import quantile
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from ...request_proxy.request_trace import RequestFact


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LatencyStats:
    """Store typed latency distribution summaries."""

    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RequestMetrics:
    """Store request-edge counts, token totals, and latency statistics."""

    requests: int
    successful_requests: int
    failed_requests: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None
    cached_input_tokens: int | None
    prefix_cache_hit_rate: float | None
    latency_seconds: LatencyStats
    ttft_seconds: LatencyStats


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ObservedTopology:
    """Store actors observed for one Claude session."""

    session_id: str
    agent_ids: tuple[str, ...]
    agent_roles: dict[str, str]


def latency_stats(values: list[float]) -> LatencyStats:
    """Summarize latency samples using the benchmark percentiles."""

    if not values:
        return LatencyStats(None, None, None, None)
    return LatencyStats(
        sum(values) / len(values),
        float(quantile(values, 0.5)),
        float(quantile(values, 0.95)),
        float(quantile(values, 0.99)),
    )


def request_tpot(fact: RequestFact) -> tuple[float | None, str | None]:
    """Return average post-first-token latency and an unavailable reason."""

    if fact.status != "success":
        return None, "request_failed"
    if fact.ttft_seconds is None:
        return None, "missing_ttft"
    if fact.output_tokens is None:
        return None, "missing_output_tokens"
    if fact.output_tokens <= 1:
        return None, "output_tokens_le_1"
    if fact.latency_seconds < fact.ttft_seconds:
        return None, "latency_before_ttft"
    return (fact.latency_seconds - fact.ttft_seconds) / (fact.output_tokens - 1), None


def aggregate_request_metrics(facts: Iterable[RequestFact]) -> RequestMetrics:
    """Aggregate finalized request facts into request-edge metrics."""

    rows = list(facts)
    success = [row for row in rows if row.status == "success"]
    input_tokens = sum(row.input_tokens or 0 for row in success)
    creation_rows = [row for row in success if row.cache_creation_tokens is not None]
    cached_rows = [row for row in success if row.cached_tokens is not None]
    cache_rows = [row for row in success if row.cache_creation_tokens is not None or row.cached_tokens is not None]
    cache_input = sum(row.input_tokens or 0 for row in cache_rows)
    cache_creation = sum(row.cache_creation_tokens or 0 for row in creation_rows) if creation_rows else None
    cached = sum(row.cached_tokens or 0 for row in cached_rows) if cached_rows else None
    eligible_input = cache_input + (cache_creation or 0) + (cached or 0)
    return RequestMetrics(
        len(rows),
        len(success),
        len(rows) - len(success),
        input_tokens,
        sum(row.output_tokens or 0 for row in success),
        cache_creation,
        cached,
        cached / eligible_input if cache_creation is not None and cached is not None and eligible_input else None,
        latency_stats([row.latency_seconds for row in rows]),
        latency_stats([row.ttft_seconds for row in rows if row.ttft_seconds is not None]),
    )


def derive_session_topology(facts: Iterable[RequestFact], session_id: str) -> ObservedTopology:
    """Derive deterministic actor identity and roles for one session."""

    actors = {row.actor_id: row.actor_role for row in facts if row.session_id == session_id}
    return ObservedTopology(session_id, tuple(sorted(actors)), actors)
