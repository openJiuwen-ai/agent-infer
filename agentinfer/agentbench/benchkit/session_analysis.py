# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Analyze task, session, agent, and request distributions from run artifacts."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean

from numpy import quantile
from pydantic import TypeAdapter, ValidationError

from ..request_proxy.request_trace import RequestFact, load_request_facts
from .artifacts.task_result import TaskResultArtifact
from .common import atomic_write_csv
from .metrics.request import LatencyStats, RequestMetrics, aggregate_request_metrics, latency_stats, request_tpot

_TASK_RESULT_ADAPTER = TypeAdapter(TaskResultArtifact)
_REQUEST_FIELDS = (
    "requests",
    "successful_requests",
    "failed_requests",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cached_input_tokens",
    "prefix_cache_hit_rate",
)
_REQUEST_DISTRIBUTIONS = ("latency_seconds", "ttft_seconds", "tpot_seconds")
_GAP_FIELDS = ("count", "overlap_count", "overlap_ratio", "minimum", "mean", "p50", "p90", "p95", "p99", "maximum")
_SAMPLE_COLUMNS = (
    "run_id",
    "level",
    "metric",
    "value",
    "unit",
    "instance_id",
    "session_id",
    "actor_id",
    "actor_role",
    "request_id",
    "task_outcome",
    "has_patch",
    "request_status",
    "value_status",
    "exclusion_reason",
)


@dataclass(frozen=True)
class GapDistribution:
    count: int
    overlap_count: int
    overlap_ratio: float | None
    minimum: float | None
    mean: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    maximum: float | None


@dataclass(frozen=True)
class TaskDiagnostics:
    instance_id: str
    outcome: str
    termination_reason: str | None
    duration_seconds: float
    has_patch: bool
    agent_ids: tuple[str, ...]


@dataclass(frozen=True)
class AgentDiagnostics:
    actor_id: str
    actor_role: str
    requests: RequestMetrics
    tpot_seconds: LatencyStats
    inter_request_gap_seconds: GapDistribution
    facts: tuple[RequestFact, ...]


@dataclass(frozen=True)
class SessionDiagnostics:
    session_id: str
    task: TaskDiagnostics | None
    requests: RequestMetrics
    tpot_seconds: LatencyStats
    inter_request_gap_seconds: GapDistribution
    agents: tuple[AgentDiagnostics, ...]
    facts: tuple[RequestFact, ...]


@dataclass(frozen=True)
class SessionsAnalysis:
    run_id: str
    unassigned_facts: tuple[RequestFact, ...]
    sessions: tuple[SessionDiagnostics, ...]


def analyze_sessions(run_dir: Path) -> SessionsAnalysis:
    request_path = run_dir / "requests.jsonl"
    if not request_path.is_file():
        raise FileNotFoundError(f"request trace not found: {request_path}")
    facts = load_request_facts(request_path)
    run_ids = {fact.run_id for fact in facts}
    if len(run_ids) > 1:
        raise ValueError(f"request trace contains multiple run_id values: {request_path}")
    if run_ids and run_ids != {run_dir.name}:
        raise ValueError(f"request trace run_id does not match directory: {request_path}")

    tasks = _load_tasks(run_dir / "tasks")
    grouped: dict[str, list[RequestFact]] = defaultdict(list)
    unassigned = []
    for fact in facts:
        if fact.session_id is None:
            unassigned.append(fact)
        else:
            grouped[fact.session_id].append(fact)

    session_ids = sorted(set(grouped) | set(tasks))
    sessions = tuple(
        _build_session(session_id, grouped.get(session_id, []), tasks.get(session_id)) for session_id in session_ids
    )
    return SessionsAnalysis(
        next(iter(run_ids), run_dir.name),
        tuple(sorted(unassigned, key=lambda fact: fact.request_id)),
        sessions,
    )


def write_analysis_artifacts(run_dirs: list[Path], output_dir: Path, *, plots: bool) -> dict[str, Path]:
    analyses = [analyze_sessions(run_dir.resolve()) for run_dir in run_dirs]
    session_rows = [
        _session_csv_row(analysis.run_id, session) for analysis in analyses for session in analysis.sessions
    ]
    agent_rows = [
        _agent_csv_row(analysis.run_id, session, agent)
        for analysis in analyses
        for session in analysis.sessions
        for agent in session.agents
    ]
    sample_rows = [row for analysis in analyses for row in _distribution_rows(analysis)]
    artifacts = {
        "sessions": output_dir / "sessions.csv",
        "agents": output_dir / "agents.csv",
        "distribution_samples": output_dir / "distribution_samples.csv",
    }
    atomic_write_csv(artifacts["sessions"], session_rows, _session_columns())
    atomic_write_csv(artifacts["agents"], agent_rows, _agent_columns())
    atomic_write_csv(artifacts["distribution_samples"], sample_rows, list(_SAMPLE_COLUMNS))
    if plots:
        from .distribution_plot import write_distribution_plots

        write_distribution_plots(artifacts["distribution_samples"], output_dir / "plots")
    return artifacts


def _session_columns() -> list[str]:
    return [
        *_TASK_COLUMNS,
        "agent_count",
        *_request_column_names(),
        *_gap_column_names("adjacent_gap"),
    ]


def _agent_columns() -> list[str]:
    return [
        *_TASK_COLUMNS,
        "actor_id",
        "actor_role",
        *_request_column_names(),
        *_gap_column_names("adjacent_gap"),
    ]


_TASK_COLUMNS = (
    "run_id",
    "session_id",
    "instance_id",
    "outcome",
    "termination_reason",
    "task_duration_seconds",
    "has_patch",
)


def _task_columns(session: SessionDiagnostics) -> dict[str, object]:
    task = session.task
    return {
        "run_id": "",
        "session_id": session.session_id,
        "instance_id": task.instance_id if task else "N/A",
        "outcome": task.outcome if task else "N/A",
        "termination_reason": _csv_value(task.termination_reason) if task else "N/A",
        "task_duration_seconds": task.duration_seconds if task else "N/A",
        "has_patch": task.has_patch if task else "N/A",
    }


def _agent_count(session: SessionDiagnostics) -> int:
    observed = len(session.agents)
    if session.task is None:
        return observed
    return len(set(session.task.agent_ids) | {agent.actor_id for agent in session.agents})


def _session_csv_row(run_id: str, session: SessionDiagnostics) -> dict[str, object]:
    row = _task_columns(session)
    row["run_id"] = run_id
    row["agent_count"] = _agent_count(session)
    row.update(_request_columns(session.requests, session.tpot_seconds))
    row.update(_gap_columns("adjacent_gap", session.inter_request_gap_seconds))
    return row


def _agent_csv_row(run_id: str, session: SessionDiagnostics, agent: AgentDiagnostics) -> dict[str, object]:
    row = _task_columns(session)
    row["run_id"] = run_id
    row.update({"actor_id": agent.actor_id, "actor_role": agent.actor_role})
    row.update(_request_columns(agent.requests, agent.tpot_seconds))
    row.update(_gap_columns("adjacent_gap", agent.inter_request_gap_seconds))
    return row


def _request_column_names() -> list[str]:
    return [
        *_REQUEST_FIELDS,
        *(f"{name}_{percentile}" for name in _REQUEST_DISTRIBUTIONS for percentile in ("mean", "p50", "p95", "p99")),
    ]


def _request_columns(metrics: RequestMetrics, tpot_seconds: LatencyStats) -> dict[str, object]:
    result = {field: _csv_value(getattr(metrics, field)) for field in _REQUEST_FIELDS}
    for name, stats in (
        ("latency_seconds", metrics.latency_seconds),
        ("ttft_seconds", metrics.ttft_seconds),
        ("tpot_seconds", tpot_seconds),
    ):
        for percentile in ("mean", "p50", "p95", "p99"):
            result[f"{name}_{percentile}"] = _csv_value(getattr(stats, percentile))
    return result


def _csv_value(value: object) -> object:
    return "null" if value is None else value


def _gap_column_names(prefix: str) -> list[str]:
    return [f"{prefix}_{field}" for field in _GAP_FIELDS]


def _gap_columns(prefix: str, gaps: GapDistribution) -> dict[str, object]:
    return {f"{prefix}_{field}": _csv_value(getattr(gaps, field)) for field in _GAP_FIELDS}


def _distribution_rows(analysis: SessionsAnalysis) -> list[dict[str, object]]:
    rows = []
    for fact in analysis.unassigned_facts:
        rows.extend(_request_sample_rows(analysis.run_id, None, None, (fact,)))
    for session in analysis.sessions:
        if session.task is not None:
            rows.append(
                _sample_row(analysis.run_id, "task", "task_duration_seconds", session.task.duration_seconds, session)
            )
        rows.extend(_aggregate_sample_rows(analysis.run_id, "session", session, None))
        for agent in session.agents:
            rows.extend(_aggregate_sample_rows(analysis.run_id, "agent", session, agent))
            rows.extend(_request_sample_rows(analysis.run_id, session, agent, agent.facts))
    return rows


def _aggregate_sample_rows(
    run_id: str,
    level: str,
    session: SessionDiagnostics,
    agent: AgentDiagnostics | None,
) -> list[dict[str, object]]:
    metrics = agent.requests if agent else session.requests
    facts = agent.facts if agent else session.facts
    successful = tuple(fact for fact in facts if fact.status == "success")
    latency = latency_stats([fact.latency_seconds for fact in successful])
    ttft = latency_stats([fact.ttft_seconds for fact in successful if fact.ttft_seconds is not None])
    tpot = agent.tpot_seconds if agent else session.tpot_seconds
    gaps = agent.inter_request_gap_seconds if agent else session.inter_request_gap_seconds
    values = (
        ("request_count", metrics.requests, "count", None),
        ("agent_count", _agent_count(session), "count", None) if level == "session" else None,
        ("input_tokens", metrics.input_tokens, "tokens", None),
        ("output_tokens", metrics.output_tokens, "tokens", None),
        *(
            (
                f"request_{name}_{percentile}",
                getattr(
                    tpot if name == "tpot_seconds" else latency if name == "latency_seconds" else ttft,
                    percentile,
                ),
                "seconds_per_token" if name == "tpot_seconds" else "seconds",
                "no_valid_samples",
            )
            for name in _REQUEST_DISTRIBUTIONS
            for percentile in ("mean", "p50", "p95")
        ),
        ("adjacent_gap_mean_seconds", gaps.mean, "seconds", "no_adjacent_requests"),
        ("adjacent_gap_p50_seconds", gaps.p50, "seconds", "no_adjacent_requests"),
        ("adjacent_gap_overlap_ratio", gaps.overlap_ratio, "ratio", "no_adjacent_requests"),
    )
    return [
        _sample_row(run_id, level, metric, value, session, agent, unit=unit, unavailable_reason=reason)
        for item in values
        if item is not None
        for metric, value, unit, reason in (item,)
    ]


def _request_sample_rows(
    run_id: str,
    session: SessionDiagnostics | None,
    agent: AgentDiagnostics | None,
    facts: tuple[RequestFact, ...],
) -> list[dict[str, object]]:
    rows = []
    for fact in facts:
        tpot, tpot_reason = request_tpot(fact)
        values = (
            (
                "latency_seconds",
                fact.latency_seconds if fact.status == "success" else None,
                "seconds",
                "request_failed" if fact.status != "success" else None,
            ),
            (
                "ttft_seconds",
                fact.ttft_seconds if fact.status == "success" else None,
                "seconds",
                "request_failed" if fact.status != "success" else "missing_ttft",
            ),
            ("tpot_seconds", tpot, "seconds_per_token", tpot_reason),
            ("input_tokens", fact.input_tokens, "tokens", "missing_input_tokens"),
            ("output_tokens", fact.output_tokens, "tokens", "missing_output_tokens"),
        )
        rows.extend(
            _sample_row(
                run_id,
                "request",
                metric,
                value,
                session,
                agent,
                fact,
                unit=unit,
                unavailable_reason=reason,
            )
            for metric, value, unit, reason in values
        )
    return rows


def _sample_row(
    run_id: str,
    level: str,
    metric: str,
    value: int | float | None,
    session: SessionDiagnostics | None,
    agent: AgentDiagnostics | None = None,
    fact: RequestFact | None = None,
    *,
    unit: str = "seconds",
    unavailable_reason: str | None = None,
) -> dict[str, object]:
    task = session.task if session else None
    available = value is not None
    return {
        "run_id": run_id,
        "level": level,
        "metric": metric,
        "value": value if available else "null",
        "unit": unit,
        "instance_id": task.instance_id if task else "N/A",
        "session_id": session.session_id if session else "N/A",
        "actor_id": agent.actor_id if agent else (fact.actor_id if fact else "N/A"),
        "actor_role": agent.actor_role if agent else (fact.actor_role if fact else "N/A"),
        "request_id": fact.request_id if fact else "N/A",
        "task_outcome": task.outcome if task else "N/A",
        "has_patch": task.has_patch if task else "N/A",
        "request_status": fact.status if fact else "N/A",
        "value_status": "valid" if available else "unavailable",
        "exclusion_reason": "N/A" if available else (unavailable_reason or "unavailable"),
    }


def _load_tasks(tasks_dir: Path) -> dict[str, TaskResultArtifact]:
    tasks: dict[str, TaskResultArtifact] = {}
    if not tasks_dir.is_dir():
        return tasks
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        path = task_dir / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"task result not found: {path}")
        try:
            task = _TASK_RESULT_ADAPTER.validate_json(path.read_bytes())
        except ValidationError as exc:
            raise ValueError(f"invalid task result: {path}") from exc
        if task.session_id in tasks:
            raise ValueError(f"duplicate task session_id: {task.session_id}")
        tasks[task.session_id] = task
    return tasks


def _build_session(
    session_id: str,
    facts: list[RequestFact],
    task: TaskResultArtifact | None,
) -> SessionDiagnostics:
    ordered = _ordered_facts(facts)
    by_actor: dict[str, list[RequestFact]] = defaultdict(list)
    for fact in ordered:
        by_actor[fact.actor_id].append(fact)
    agents = tuple(_build_agent(actor_id, by_actor[actor_id]) for actor_id in sorted(by_actor))
    return SessionDiagnostics(
        session_id,
        (
            TaskDiagnostics(
                task.instance_id,
                task.outcome,
                task.termination_reason,
                task.duration_seconds,
                task.has_patch,
                task.topology.agent_ids,
            )
            if task
            else None
        ),
        aggregate_request_metrics(ordered),
        _tpot_stats(ordered),
        _gap_distribution(ordered),
        agents,
        tuple(ordered),
    )


def _build_agent(actor_id: str, facts: list[RequestFact]) -> AgentDiagnostics:
    ordered = _ordered_facts(facts)
    roles = {fact.actor_role for fact in ordered}
    if len(roles) != 1:
        raise ValueError(f"actor {actor_id} has inconsistent roles: {sorted(roles)}")
    return AgentDiagnostics(
        actor_id,
        next(iter(roles)),
        aggregate_request_metrics(ordered),
        _tpot_stats(ordered),
        _gap_distribution(ordered),
        tuple(ordered),
    )


def _ordered_facts(facts: list[RequestFact]) -> list[RequestFact]:
    for fact in facts:
        started_at = _timestamp(fact.started_at, fact.request_id)
        finished_at = _timestamp(fact.finished_at, fact.request_id)
        if finished_at < started_at:
            raise ValueError(f"request {fact.request_id} finishes before it starts")
    return sorted(
        facts,
        key=lambda fact: (
            _timestamp(fact.started_at, fact.request_id),
            _timestamp(fact.finished_at, fact.request_id),
            fact.request_id,
        ),
    )


def _timestamp(value: str, request_id: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp for request {request_id}: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone for request {request_id}: {value}")
    return parsed


def _tpot_stats(facts: list[RequestFact]) -> LatencyStats:
    return latency_stats([value for fact in facts if (value := request_tpot(fact)[0]) is not None])


def _gap_distribution(facts: list[RequestFact]) -> GapDistribution:
    gaps = [
        (
            _timestamp(current.started_at, current.request_id) - _timestamp(previous.finished_at, previous.request_id)
        ).total_seconds()
        for previous, current in zip(facts, facts[1:], strict=False)
    ]
    if not gaps:
        return GapDistribution(0, 0, None, None, None, None, None, None, None, None)
    overlap_count = sum(gap < 0 for gap in gaps)
    return GapDistribution(
        len(gaps),
        overlap_count,
        overlap_count / len(gaps),
        min(gaps),
        fmean(gaps),
        float(quantile(gaps, 0.50)),
        float(quantile(gaps, 0.90)),
        float(quantile(gaps, 0.95)),
        float(quantile(gaps, 0.99)),
        max(gaps),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze distributions from existing benchmark runs")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    write_analysis_artifacts(args.run_dirs, args.output_dir, plots=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
