# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from dataclasses import replace

from agentinfer.agentbench.benchkit.metrics.request import (
    LatencyStats,
    aggregate_request_metrics,
    derive_session_topology,
)
from agentinfer.agentbench.request_proxy.request_trace import RequestFact


def _fact(**changes: object) -> RequestFact:
    fact = RequestFact(
        schema_version="1",
        run_id="run",
        request_id="r1",
        session_id="session-a",
        actor_id="lead",
        actor_role="lead",
        started_at="start",
        finished_at="finish",
        status="success",
        status_code=200,
        latency_seconds=1.0,
        ttft_seconds=0.2,
        input_tokens=10,
        output_tokens=4,
        cache_creation_tokens=2,
        cached_tokens=6,
        upstream="upstream",
        error=None,
    )
    return replace(fact, **changes)


def test_aggregate_request_metrics_counts_status_tokens_and_percentiles() -> None:
    metrics = aggregate_request_metrics(
        [
            _fact(),
            _fact(
                request_id="r2",
                latency_seconds=3.0,
                ttft_seconds=None,
                input_tokens=5,
                output_tokens=2,
                cache_creation_tokens=1,
                cached_tokens=1,
            ),
            _fact(
                request_id="r3",
                status="error",
                status_code=500,
                latency_seconds=2.0,
                input_tokens=100,
                output_tokens=100,
                cache_creation_tokens=100,
                cached_tokens=100,
            ),
        ]
    )

    assert metrics.requests == 3
    assert metrics.successful_requests == 2
    assert metrics.failed_requests == 1
    assert metrics.input_tokens == 15
    assert metrics.output_tokens == 6
    assert metrics.cache_creation_input_tokens == 3
    assert metrics.cached_input_tokens == 7
    assert metrics.prefix_cache_hit_rate == 7 / 25
    assert metrics.latency_seconds == LatencyStats(2.0, 2.0, 2.9, 2.98)
    assert metrics.ttft_seconds == LatencyStats(0.2, 0.2, 0.2, 0.2)


def test_aggregate_request_metrics_handles_empty_and_zero_input() -> None:
    empty = aggregate_request_metrics([])
    zero = aggregate_request_metrics([_fact(input_tokens=0, cache_creation_tokens=0, cached_tokens=0)])

    assert empty.requests == 0
    assert empty.cache_creation_input_tokens is None
    assert empty.cached_input_tokens is None
    assert empty.latency_seconds == LatencyStats(None, None, None, None)
    assert empty.ttft_seconds == LatencyStats(None, None, None, None)
    assert empty.prefix_cache_hit_rate is None
    assert zero.cache_creation_input_tokens == 0
    assert zero.cached_input_tokens == 0
    assert zero.prefix_cache_hit_rate is None


def test_aggregate_request_metrics_preserves_cache_telemetry_availability() -> None:
    missing = _fact(cache_creation_tokens=None, cached_tokens=None)
    read_only = _fact(request_id="r2", input_tokens=5, cache_creation_tokens=None, cached_tokens=4)
    creation_only = _fact(request_id="r3", input_tokens=3, cache_creation_tokens=1, cached_tokens=None)

    missing_metrics = aggregate_request_metrics([missing])
    assert missing_metrics.cache_creation_input_tokens is None
    assert missing_metrics.cached_input_tokens is None
    assert missing_metrics.prefix_cache_hit_rate is None

    read_metrics = aggregate_request_metrics([read_only])
    assert read_metrics.cache_creation_input_tokens is None
    assert read_metrics.cached_input_tokens == 4
    assert read_metrics.prefix_cache_hit_rate is None

    creation_metrics = aggregate_request_metrics([creation_only])
    assert creation_metrics.cache_creation_input_tokens == 1
    assert creation_metrics.cached_input_tokens is None
    assert creation_metrics.prefix_cache_hit_rate is None


def test_derive_session_topology_filters_and_sorts_actors() -> None:
    topology = derive_session_topology(
        [
            _fact(actor_id="sub-b", actor_role="subagent"),
            _fact(request_id="r2", actor_id="lead", actor_role="lead"),
            _fact(request_id="r3", session_id="session-b", actor_id="other"),
            _fact(request_id="r4", actor_id="sub-b", actor_role="subagent"),
        ],
        "session-a",
    )

    assert topology.session_id == "session-a"
    assert topology.agent_ids == ("lead", "sub-b")
    assert topology.agent_roles == {"sub-b": "subagent", "lead": "lead"}
