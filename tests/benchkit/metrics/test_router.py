from agentcache.benchmarks.benchkit.metrics.router import aggregate_router_events


def test_aggregate_router_events_counts_event_names() -> None:
    metrics = aggregate_router_events(
        [
            {"event": "route_selected"},
            {"event": "route_selected"},
            {"event": "cache_hit"},
            {},
        ]
    )

    assert metrics.events == {"route_selected": 2, "cache_hit": 1, "unknown": 1}
