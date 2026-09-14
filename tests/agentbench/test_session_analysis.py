# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agentinfer.agentbench.benchkit.session_analysis import analyze_sessions, main
from agentinfer.agentbench.request_proxy.request_trace import RequestFact


def _fact(
    request_id: str,
    *,
    run_id: str = "run",
    session_id: str | None = "session-1",
    actor_id: str = "lead",
    actor_role: str = "lead",
    started_at: str = "2026-07-28T00:00:00+00:00",
    finished_at: str = "2026-07-28T00:00:02+00:00",
    cached_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> RequestFact:
    return RequestFact(
        "1",
        run_id,
        request_id,
        session_id,
        actor_id,
        actor_role,
        started_at,
        finished_at,
        "success",
        200,
        2.0,
        0.5,
        10,
        4,
        cache_creation_tokens,
        cached_tokens,
        "http://localhost",
        None,
    )


def _write_run(tmp_path: Path, facts: list[RequestFact], tasks: list[dict] | None = None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "requests.jsonl").write_text(
        "".join(json.dumps(asdict(fact)) + "\n" for fact in facts), encoding="utf-8"
    )
    for index, task in enumerate(tasks or []):
        task_dir = run_dir / "tasks" / f"task-{index}"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(json.dumps(task), encoding="utf-8")
    return run_dir


def _task(session_id: str = "session-1") -> dict:
    return {
        "schema_version": "1",
        "instance_id": "instance-1",
        "session_id": session_id,
        "agent": {"type": "claude", "profile": "plan-subagent"},
        "outcome": "completed",
        "termination_reason": None,
        "task_position": 0,
        "started_at": "2026-07-28T00:00:00+00:00",
        "finished_at": "2026-07-28T00:00:20+00:00",
        "duration_seconds": 20.0,
        "has_patch": True,
        "patch_bytes": 10,
        "registration": None,
        "cleanup": None,
        "topology": {"session_id": session_id, "agent_ids": ["lead"], "agent_roles": {"lead": "lead"}},
        "error": None,
    }


def _legacy_task(session_id: str = "session-1") -> dict:
    """A pre-change (schema-v1) task result lacking task_position/started_at/
    finished_at, exercising the backward-compatible load path in _load_tasks."""
    return {
        "schema_version": "1",
        "instance_id": "instance-1",
        "session_id": session_id,
        "agent": {"type": "claude", "profile": "plan-subagent"},
        "outcome": "completed",
        "termination_reason": None,
        "duration_seconds": 20.0,
        "has_patch": True,
        "patch_bytes": 10,
        "registration": None,
        "cleanup": None,
        "topology": {"session_id": session_id, "agent_ids": ["lead"], "agent_roles": {"lead": "lead"}},
        "error": None,
        # task_position / started_at / finished_at intentionally absent
    }


def test_analyze_sessions_loads_legacy_task_result(tmp_path: Path) -> None:
    """Historical runs whose ``result.json`` predates the task-position/timestamp
    fields must still load and analyze without raising (Codex P1: the new
    required dataclass fields broke ``_load_tasks`` for schema-v1 artifacts).
    """
    facts = [_fact("r1")]
    run_dir = _write_run(tmp_path, facts, [_legacy_task()])

    analysis = analyze_sessions(run_dir)

    session = analysis.sessions[0]
    assert session.task is not None
    assert session.task.instance_id == "instance-1"
    assert session.task.outcome == "completed"
    assert session.requests.requests == 1


def test_analyze_sessions_groups_agents_tasks_and_gaps(tmp_path: Path) -> None:
    facts = [
        _fact("r1", cached_tokens=5, cache_creation_tokens=5),
        _fact(
            "r2",
            actor_id="sub-1",
            actor_role="subagent",
            started_at="2026-07-28T00:00:01+00:00",
            finished_at="2026-07-28T00:00:03+00:00",
            cached_tokens=2,
            cache_creation_tokens=3,
        ),
        _fact(
            "r3",
            started_at="2026-07-28T00:00:05+00:00",
            finished_at="2026-07-28T00:00:06+00:00",
            cached_tokens=3,
            cache_creation_tokens=2,
        ),
    ]

    analysis = analyze_sessions(_write_run(tmp_path, facts, [_task()]))
    session = analysis.sessions[0]

    assert session.task is not None and session.task.instance_id == "instance-1"
    assert session.requests.requests == 3
    assert session.requests.cached_input_tokens == 10
    assert [agent.actor_id for agent in session.agents] == ["lead", "sub-1"]
    assert session.inter_request_gap_seconds.count == 2
    assert session.inter_request_gap_seconds.overlap_count == 1
    assert session.inter_request_gap_seconds.minimum == -1
    assert session.inter_request_gap_seconds.maximum == 2
    assert session.agents[0].inter_request_gap_seconds.p50 == 3


def test_analysis_preserves_null_cache_and_partial_sessions(tmp_path: Path) -> None:
    facts = [
        _fact("unassigned", session_id=None),
        _fact("request-only", session_id="request-only"),
    ]
    run_dir = _write_run(tmp_path, facts, [_task("task-only")])

    analysis = analyze_sessions(run_dir)

    assert analysis.unassigned_facts[0].request_id == "unassigned"
    assert [session.session_id for session in analysis.sessions] == ["request-only", "task-only"]
    request_only, task_only = analysis.sessions
    assert request_only.task is None
    assert request_only.requests.cached_input_tokens is None
    assert request_only.requests.prefix_cache_hit_rate is None
    assert task_only.task is not None
    assert task_only.requests.requests == 0
    assert task_only.inter_request_gap_seconds.count == 0


def test_main_writes_deterministic_csvs(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_fact("r1")], [_task()])
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert main([str(run_dir), str(run_dir), "--output-dir", str(first)]) == 0
    assert main([str(run_dir), str(run_dir), "--output-dir", str(second)]) == 0

    assert (first / "sessions.csv").read_bytes() == (second / "sessions.csv").read_bytes()
    assert (first / "agents.csv").read_bytes() == (second / "agents.csv").read_bytes()
    assert (first / "distribution_samples.csv").read_bytes() == (second / "distribution_samples.csv").read_bytes()
    sessions = list(csv.DictReader((first / "sessions.csv").open(encoding="utf-8", newline="")))
    agents = list(csv.DictReader((first / "agents.csv").open(encoding="utf-8", newline="")))
    assert len(sessions) == len(agents) == 2
    assert [row["run_id"] for row in sessions] == ["run", "run"]
    assert sessions[0]["session_id"] == "session-1"
    assert sessions[0]["requests"] == "1"
    assert sessions[0]["termination_reason"] == "null"
    assert sessions[0]["cached_input_tokens"] == "null"
    assert agents[0]["actor_id"] == "lead"
    assert agents[0]["requests"] == "1"
    assert agents[0]["cached_input_tokens"] == "null"
    assert all(value != "" for row in (*sessions, *agents) for value in row.values())
    assert "adjacent_gap_minimum" in sessions[0]
    assert sessions[0]["adjacent_gap_overlap_ratio"] == "null"
    assert agents[0]["tpot_seconds_p50"] == "0.5"
    assert "session_gap_minimum" not in sessions[0]
    samples = list(csv.DictReader((first / "distribution_samples.csv").open(encoding="utf-8", newline="")))
    assert {row["level"] for row in samples} == {"task", "session", "agent", "request"}
    request_tpot = next(row for row in samples if row["level"] == "request" and row["metric"] == "tpot_seconds")
    assert float(request_tpot["value"]) == pytest.approx(0.5)
    assert request_tpot["unit"] == "seconds_per_token"
    assert request_tpot["value_status"] == "valid"
    assert all(value != "" for row in samples for value in row.values())
    expected_plots = {"distribution_summary.png", "task_request_breakdown.png"}
    assert {path.name for path in (first / "plots").iterdir()} == expected_plots
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in (first / "plots").iterdir())


def test_task_only_session_stays_in_sessions_csv(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [], [_task("task-only")])
    output_dir = tmp_path / "analysis"

    assert main([str(run_dir), "--output-dir", str(output_dir)]) == 0

    sessions = list(csv.DictReader((output_dir / "sessions.csv").open(encoding="utf-8", newline="")))
    agents = list(csv.DictReader((output_dir / "agents.csv").open(encoding="utf-8", newline="")))
    assert len(sessions) == 1
    assert agents == []
    assert sessions[0]["session_id"] == "task-only"
    assert sessions[0]["requests"] == "0"
    assert sessions[0]["cached_input_tokens"] == "null"


def test_analysis_rejects_invalid_inputs_before_writing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="request trace"):
        analyze_sessions(missing)

    run_dir = _write_run(tmp_path, [_fact("r1", run_id="other")])
    output_dir = tmp_path / "analysis"
    with pytest.raises(ValueError, match="run_id does not match"):
        main([str(run_dir), "--output-dir", str(output_dir)])
    assert not output_dir.exists()


def test_analysis_rejects_duplicate_task_sessions_and_naive_timestamps(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_fact("r1")], [_task(), _task()])
    with pytest.raises(ValueError, match="duplicate task session_id"):
        analyze_sessions(run_dir)

    timestamp_dir = tmp_path / "timestamp" / "run"
    timestamp_dir.mkdir(parents=True)
    fact = _fact("r1", started_at="2026-07-28T00:00:00")
    (timestamp_dir / "requests.jsonl").write_text(json.dumps(asdict(fact)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks timezone"):
        analyze_sessions(timestamp_dir)


def test_analysis_rejects_inconsistent_actor_roles_and_chronology(tmp_path: Path) -> None:
    roles_dir = _write_run(
        tmp_path,
        [
            _fact("r1"),
            _fact(
                "r2",
                actor_role="subagent",
                started_at="2026-07-28T00:00:03+00:00",
                finished_at="2026-07-28T00:00:04+00:00",
            ),
        ],
    )
    with pytest.raises(ValueError, match="inconsistent roles"):
        analyze_sessions(roles_dir)

    chronology_dir = tmp_path / "chronology" / "run"
    chronology_dir.mkdir(parents=True)
    fact = _fact(
        "r1",
        started_at="2026-07-28T00:00:02+00:00",
        finished_at="2026-07-28T00:00:01+00:00",
    )
    (chronology_dir / "requests.jsonl").write_text(json.dumps(asdict(fact)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finishes before it starts"):
        analyze_sessions(chronology_dir)


def test_distribution_samples_preserve_unavailable_tpot_reason(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, [_fact("r1")], [_task()])
    payload = json.loads((run_dir / "requests.jsonl").read_text(encoding="utf-8"))
    payload["output_tokens"] = 1
    (run_dir / "requests.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output_dir = tmp_path / "analysis"

    assert main([str(run_dir), "--output-dir", str(output_dir)]) == 0

    samples = list(csv.DictReader((output_dir / "distribution_samples.csv").open(encoding="utf-8", newline="")))
    tpot = next(row for row in samples if row["level"] == "request" and row["metric"] == "tpot_seconds")
    assert tpot["value"] == "null"
    assert tpot["value_status"] == "unavailable"
    assert tpot["exclusion_reason"] == "output_tokens_le_1"
    assert (output_dir / "plots" / "distribution_summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output_dir / "plots" / "task_request_breakdown.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_distribution_samples_include_unassigned_and_exclude_failed_latency(tmp_path: Path) -> None:
    facts = [
        _fact("unassigned", session_id=None),
        _fact("failed", session_id=None),
    ]
    facts[1] = RequestFact(**{**asdict(facts[1]), "status": "error", "status_code": 500})
    run_dir = _write_run(tmp_path, facts)
    output_dir = tmp_path / "analysis"

    assert main([str(run_dir), "--output-dir", str(output_dir)]) == 0

    samples = list(csv.DictReader((output_dir / "distribution_samples.csv").open(encoding="utf-8", newline="")))
    request_ids = {row["request_id"] for row in samples if row["level"] == "request"}
    assert request_ids == {"unassigned", "failed"}
    failed_latency = next(
        row for row in samples if row["request_id"] == "failed" and row["metric"] == "latency_seconds"
    )
    assert failed_latency["value"] == "null"
    assert failed_latency["exclusion_reason"] == "request_failed"


def test_session_agent_count_includes_topology_without_requests(tmp_path: Path) -> None:
    task = _task()
    task["topology"] = {
        "session_id": "session-1",
        "agent_ids": ["lead", "silent-subagent"],
        "agent_roles": {"lead": "lead", "silent-subagent": "subagent"},
    }
    run_dir = _write_run(tmp_path, [_fact("r1")], [task])
    output_dir = tmp_path / "analysis"

    assert main([str(run_dir), "--output-dir", str(output_dir)]) == 0

    session = next(csv.DictReader((output_dir / "sessions.csv").open(encoding="utf-8", newline="")))
    assert session["agent_count"] == "2"
