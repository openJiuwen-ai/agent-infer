from agentcache.benchmarks.benchkit.metrics.router import aggregate_router_events, aggregate_router_window


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


def test_aggregate_router_window_excludes_cumulative_start_events() -> None:
    metrics = aggregate_router_window(
        [{"event": "route_selected"}, {"event": "existing"}],
        [
            {"event": "route_selected"},
            {"event": "existing"},
            {"event": "route_selected"},
            {"event": "cache_hit"},
        ],
    )

    assert metrics.events == {"route_selected": 1, "cache_hit": 1}
