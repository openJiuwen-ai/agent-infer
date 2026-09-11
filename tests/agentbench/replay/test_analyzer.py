# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentinfer.agentbench.replay.analyzer import analyze_replay_trace, replay_analysis_to_dict

_REQUIRED_ROW_ANALYSIS_COUNTERS = {
    "invalid_json",
    "not_an_object",
    "missing_session_id",
    "missing_agent_id",
    "invalid_timestamps",
    "retry_matched_failures",
    "rows_without_replayable_tokens",
    "sessions_without_replayable_requests",
}

_TRACE_ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(
    request_id: str,
    *,
    session_id: str = "s1",
    actor_id: str = "lead",
    started: int,
    finished: int,
    input_tokens: int | None = 10,
    output_tokens: int | None = 2,
    status: str = "success",
    use_agent_id: bool = False,
    actor_role: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "request_id": request_id,
        "session_id": session_id,
        "started_at": (_TRACE_ORIGIN + timedelta(seconds=started)).isoformat(),
        "finished_at": (_TRACE_ORIGIN + timedelta(seconds=finished)).isoformat(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": status,
    }
    row["agent_id" if use_agent_id else "actor_id"] = actor_id
    if actor_role is not None:
        row["actor_role"] = actor_role
    return row


def _write(path: Path, rows: list[object], *, invalid_prefix: bool = False) -> None:
    lines = ["{not-json}\n"] if invalid_prefix else []
    lines.extend(json.dumps(row) + "\n" for row in rows)
    path.write_text("".join(lines), encoding="utf-8")


def test_clean_trace_serializes_all_required_quality_counters(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    _write(source, [_row("lead", started=0, finished=1)])

    artifact = replay_analysis_to_dict(analyze_replay_trace(source))
    row_analysis = artifact["row_analysis"]

    assert isinstance(row_analysis, dict)
    assert _REQUIRED_ROW_ANALYSIS_COUNTERS <= row_analysis.keys()
    assert all(row_analysis[name] == 0 for name in _REQUIRED_ROW_ANALYSIS_COUNTERS)


def test_empty_trace_preserves_required_quality_counter_schema(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    _write(source, [])

    analysis = analyze_replay_trace(source)

    assert analysis.total_rows == 0
    assert analysis.sessions == ()
    assert _REQUIRED_ROW_ANALYSIS_COUNTERS <= analysis.row_analysis.keys()
    assert all(analysis.row_analysis[name] == 0 for name in _REQUIRED_ROW_ANALYSIS_COUNTERS)


def test_analyzer_matches_retry_and_separates_send_and_context_dependencies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row("lead-1", started=0, finished=1),
            _row(
                "child-failed",
                actor_id="child",
                started=2,
                finished=10,
                input_tokens=None,
                output_tokens=None,
                status="failed",
                use_agent_id=True,
            ),
            _row(
                "child-retry",
                actor_id="child",
                started=4,
                finished=6,
                input_tokens=30,
                output_tokens=4,
                use_agent_id=True,
            ),
            _row(
                "child-next",
                actor_id="child",
                started=12,
                finished=13,
                input_tokens=40,
                use_agent_id=True,
            ),
            _row("lead-2", started=14, finished=15, input_tokens=20),
        ],
    )

    analysis = analyze_replay_trace(source)
    requests = {request.source_request_id: request for request in analysis.sessions[0].requests}

    failed = requests["child-failed"]
    retry = requests["child-retry"]
    assert failed.replay_kind == "request"
    assert failed.retry_match_key == retry.key
    assert failed.prompt_recipe_key == retry.key
    assert (failed.input_tokens, failed.output_tokens) == (30, 4)
    assert requests["child-next"].context_after == retry.key
    assert requests["child-next"].same_agent_gap_seconds == 2
    assert requests["lead-2"].send_after == requests["child-next"].key
    assert requests["lead-2"].context_after == requests["lead-1"].key
    assert requests["lead-2"].dependency_kind == "blocking_subagent_to_parent"
    assert analysis.row_analysis["retry_matched_failures"] == 1


def test_analyzer_isolates_bad_rows_and_counts_pruned_timing_nodes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            {"session_id": "missing-actor"},
            _row("lead", started=0, finished=1),
            _row(
                "timing-ancestor",
                actor_id="orphan",
                started=2,
                finished=3,
                input_tokens=None,
                output_tokens=None,
                status="failed",
            ),
            _row("lead-next", started=4, finished=5),
            _row(
                "terminal-timing",
                actor_id="tail",
                started=6,
                finished=7,
                input_tokens=None,
                output_tokens=None,
                status="failed",
            ),
        ],
        invalid_prefix=True,
    )

    analysis = analyze_replay_trace(source)

    assert analysis.total_rows == 6
    assert analysis.valid_rows == 4
    assert analysis.row_analysis["invalid_json"] == 1
    assert analysis.row_analysis["missing_agent_id"] == 1
    assert analysis.row_analysis["rows_without_replayable_tokens"] == 2
    assert analysis.row_analysis["pruned_terminal_timing_dependency_rows"] == 1


def test_parallel_subagents_and_blocking_return_are_inferred(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row("lead-1", started=0, finished=2),
            _row("child-a", actor_id="a", started=3, finished=5),
            _row("child-b", actor_id="b", started=3, finished=6),
            _row("lead-2", started=7, finished=8),
        ],
    )

    requests = {request.source_request_id: request for request in analyze_replay_trace(source).sessions[0].requests}

    assert requests["child-a"].parent_actor_id == "lead"
    assert requests["child-b"].key in requests["child-a"].parallel_with
    assert requests["lead-2"].dependency_kind == "blocking_subagent_to_parent"


def test_parallel_lead_title_and_main_requests_use_separate_prompt_kinds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row("title", started=0, finished=4, input_tokens=20, output_tokens=4),
            _row("main", started=0, finished=2, input_tokens=200, output_tokens=40),
            _row("lead-next", started=5, finished=6, input_tokens=260, output_tokens=40),
        ],
    )

    requests = {request.source_request_id: request for request in analyze_replay_trace(source).sessions[0].requests}

    assert requests["title"].prompt_kind == "lead_title"
    assert requests["title"].prompt_kind_source == "inferred"
    assert requests["main"].prompt_kind == "lead_main"
    assert requests["title"].context_after is None
    assert requests["main"].context_after is None
    assert requests["main"].key in requests["title"].parallel_with
    assert requests["lead-next"].send_after == requests["title"].key
    assert requests["lead-next"].context_after == requests["main"].key


def test_late_overlapping_tiny_lead_request_is_inferred_as_independent_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row("lead-1", started=0, finished=1, input_tokens=20_000, output_tokens=100),
            _row("lead-2", started=2, finished=3, input_tokens=21_000, output_tokens=100),
            _row("late-name", started=4, finished=6, input_tokens=352, output_tokens=13),
            _row("lead-main", started=5, finished=7, input_tokens=21_500, output_tokens=200),
        ],
    )

    analysis = analyze_replay_trace(source)
    requests = {request.source_request_id: request for request in analysis.sessions[0].requests}

    assert requests["late-name"].prompt_kind == "lead_name"
    assert requests["late-name"].prompt_kind_source == "inferred"
    assert requests["late-name"].prompt_kind_inference_rule == "overlapping_lead_name"
    assert requests["late-name"].context_after is None
    assert requests["lead-main"].prompt_kind == "continuation"
    assert requests["lead-main"].context_after == requests["lead-2"].key
    assert analysis.row_analysis["inferred_late_title_requests"] == 0
    assert analysis.row_analysis["inferred_lead_name_requests"] == 1


def test_late_overlapping_larger_lead_request_remains_independent_title(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row("lead-1", started=0, finished=1, input_tokens=20_000, output_tokens=100),
            _row("lead-2", started=2, finished=3, input_tokens=21_000, output_tokens=100),
            _row("late-title", started=4, finished=6, input_tokens=1_000, output_tokens=14),
            _row("lead-main", started=5, finished=7, input_tokens=21_500, output_tokens=200),
        ],
    )

    analysis = analyze_replay_trace(source)
    requests = {request.source_request_id: request for request in analysis.sessions[0].requests}

    assert requests["late-title"].prompt_kind == "lead_title"
    assert requests["late-title"].prompt_kind_inference_rule == "overlapping_late_lead_title"
    assert requests["late-title"].context_after is None
    assert requests["lead-main"].context_after == requests["lead-2"].key
    assert analysis.row_analysis["inferred_late_title_requests"] == 1
    assert analysis.row_analysis["inferred_lead_name_requests"] == 0


def test_explicit_continuation_is_not_overwritten_by_late_auxiliary_inference(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    explicit = _row("explicit", started=4, finished=6, input_tokens=352, output_tokens=13)
    explicit["request_purpose"] = "continuation"
    _write(
        source,
        [
            _row("lead-1", started=0, finished=1, input_tokens=20_000, output_tokens=100),
            _row("lead-2", started=2, finished=3, input_tokens=21_000, output_tokens=100),
            explicit,
            _row("overlap", started=5, finished=7, input_tokens=21_500, output_tokens=200),
        ],
    )

    requests = {request.source_request_id: request for request in analyze_replay_trace(source).sessions[0].requests}

    assert requests["explicit"].prompt_kind == "continuation"
    assert requests["explicit"].prompt_kind_source == "explicit"
    assert requests["explicit"].prompt_kind_inference_rule is None


def test_explicit_lead_role_with_nonstandard_actor_id_has_no_parent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row(
                "lead-1",
                actor_id="main-agent",
                actor_role="lead",
                started=0,
                finished=1,
            ),
            _row(
                "child",
                actor_id="child",
                actor_role="subagent",
                started=2,
                finished=3,
            ),
            _row(
                "lead-2",
                actor_id="main-agent",
                actor_role="lead",
                started=4,
                finished=5,
            ),
        ],
    )

    analysis = analyze_replay_trace(source)
    requests = {request.source_request_id: request for request in analysis.sessions[0].requests}
    relationships = replay_analysis_to_dict(analysis)["sessions"][0]["agent_relationships"]

    assert requests["lead-1"].parent_actor_id is None
    assert requests["lead-2"].parent_actor_id is None
    assert requests["child"].parent_actor_id == "main-agent"
    assert requests["lead-2"].dependency_kind == "blocking_subagent_to_parent"
    assert relationships["main-agent"]["parent_inferred"] is False
    assert relationships["child"]["parent_inferred"] is True


def test_sweep_tracks_completed_and_overlapping_requests_independently(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row("long", actor_id="long", started=0, finished=10),
            _row("short", actor_id="short", started=1, finished=2),
            _row("during-long", actor_id="during", started=3, finished=4),
            _row("at-boundary", actor_id="boundary", started=10, finished=11),
        ],
    )

    requests = {request.source_request_id: request for request in analyze_replay_trace(source).sessions[0].requests}

    assert requests["during-long"].send_after == requests["short"].key
    assert requests["during-long"].parallel_with == (requests["long"].key,)
    assert requests["long"].parallel_with == (
        requests["short"].key,
        requests["during-long"].key,
    )
    assert requests["at-boundary"].send_after == requests["long"].key
    assert requests["at-boundary"].parallel_with == ()


def test_completion_sweep_preserves_key_tiebreak_across_iterations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row("later-seen", started=10, finished=10),
            _row("larger-key", started=0, finished=10),
            _row("current", started=10, finished=11),
        ],
    )

    requests = {request.source_request_id: request for request in analyze_replay_trace(source).sessions[0].requests}

    assert requests["larger-key"].key > requests["later-seen"].key
    assert requests["current"].send_after == requests["larger-key"].key
    assert requests["current"].context_after == requests["larger-key"].key


def test_large_sparse_session_has_no_parallel_relationships(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            _row(
                f"request-{index}",
                started=index * 2,
                finished=index * 2 + 1,
            )
            for index in range(2_000)
        ],
    )

    requests = analyze_replay_trace(source).sessions[0].requests

    assert len(requests) == 2_000
    assert all(request.parallel_with == () for request in requests)
    assert requests[-1].send_after == requests[-2].key


def test_fractional_and_boolean_token_counts_are_not_accepted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.jsonl"
    _write(
        source,
        [
            {
                **_row("fractional", started=0, finished=1),
                "input_tokens": 10.5,
            },
            {
                **_row("boolean", started=2, finished=3),
                "output_tokens": True,
            },
        ],
    )

    analysis = analyze_replay_trace(source)

    assert analysis.row_analysis["invalid_token_counts"] == 2
    assert all(request.replay_kind == "timing_dependency" for request in analysis.sessions[0].requests)


def test_cached_tokens_preserve_zero_and_reject_invalid_values(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    rows = [
        {**_row("cached", started=0, finished=1), "cached_tokens": 7},
        {**_row("zero", started=2, finished=3), "cached_tokens": 0},
        {**_row("negative", started=4, finished=5), "cached_tokens": -1},
        {**_row("fractional", started=6, finished=7), "cached_tokens": 1.5},
        {**_row("boolean", started=8, finished=9), "cached_tokens": True},
    ]
    _write(source, rows)

    analysis = analyze_replay_trace(source)
    requests = {request.source_request_id: request for request in analysis.sessions[0].requests}
    artifact_requests = {
        request["source_request_id"]: request
        for request in replay_analysis_to_dict(analysis)["sessions"][0]["requests"]
    }

    assert requests["cached"].source_cached_tokens == 7
    assert requests["zero"].source_cached_tokens == 0
    assert requests["negative"].source_cached_tokens is None
    assert requests["fractional"].source_cached_tokens is None
    assert requests["boolean"].source_cached_tokens is None
    assert artifact_requests["cached"]["source_cached_tokens"] == 7
    assert artifact_requests["zero"]["source_cached_tokens"] == 0
    assert analysis.row_analysis["invalid_token_counts"] == 3
    assert all(request.replay_kind == "request" for request in requests.values())
