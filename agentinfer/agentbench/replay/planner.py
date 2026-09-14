# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Build and validate deterministic structural Replay plans."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .config import ReplayBenchConfig
from .sampler import sample_replay_sessions
from .schema import PromptKind, ReplayAnalysis, ReplayRequest, ReplaySession, SampledSession, required_request_keys
from .timing import (
    IntervalModel,
    backend_sampling_seed,
    build_interval_model,
    effective_interval_seconds,
)
from .unified_trace_ir import PromptReference, UnifiedTraceIR

ContextMode = Literal["none", "independent", "append", "trim", "reset"]
_CONTEXT_FREE_PROMPT_KINDS = frozenset({"lead_title", "lead_name"})


@dataclass(frozen=True)
class ReplayPlanNode:
    """One request or timing dependency in a Replay task plan."""

    runtime_request_id: str
    source_key: str
    source_status: str
    prompt_recipe_key: str
    prompt_ref: PromptReference | None
    node_type: Literal["request", "timing_dependency"]
    prompt_kind: PromptKind
    actor_id: str
    actor_role: Literal["lead", "subagent", "unknown"]
    parent_actor_id: str | None
    send_after: str | None
    context_after: str | None
    context_mode: ContextMode
    effective_interval_seconds: float
    effective_duration_seconds: float | None
    planned_input_tokens: int | None
    planned_output_tokens: int | None
    backend_sampling_seed: int | None

    def to_dict(self) -> dict[str, object]:
        """Serialize only the materialized fields required to execute the node."""

        value: dict[str, object] = {
            "runtime_request_id": self.runtime_request_id,
            "source_key": self.source_key,
            "node_type": self.node_type,
            "send_after": self.send_after,
            "effective_interval_seconds": self.effective_interval_seconds,
        }
        if self.node_type == "timing_dependency":
            value["effective_duration_seconds"] = self.effective_duration_seconds
            return value

        value.update(
            {
                "prompt_recipe_key": self.prompt_recipe_key,
                "prompt_kind": self.prompt_kind,
                "actor_id": self.actor_id,
                "actor_role": self.actor_role,
                "parent_actor_id": self.parent_actor_id,
                "context_after": self.context_after,
                "context_mode": self.context_mode,
                "planned_input_tokens": self.planned_input_tokens,
                "planned_output_tokens": self.planned_output_tokens,
                "backend_sampling_seed": self.backend_sampling_seed,
            }
        )
        if self.prompt_ref is not None:
            value["prompt_ref"] = self.prompt_ref.to_dict()
        return value


@dataclass(frozen=True)
class ReplayTaskPlan:
    """One sampled source session materialized as a Replay task."""

    task_id: str
    task_index: int
    sample_ordinal: int
    runtime_session_id: str
    source_session_id: str
    source_task_id: str | None
    requests: tuple[ReplayPlanNode, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the task and its request nodes for JSON artifacts."""

        return {
            "task_id": self.task_id,
            "task_index": self.task_index,
            "sample_ordinal": self.sample_ordinal,
            "runtime_session_id": self.runtime_session_id,
            "source_session_id": self.source_session_id,
            "source_task_id": self.source_task_id,
            "requests": [request.to_dict() for request in self.requests],
        }


@dataclass(frozen=True)
class ReplayPlan:
    """A validated deterministic Replay workload plan."""

    schema_version: Literal["1"]
    plan_kind: Literal["structural"]
    planner_version: str
    execution_ready: bool
    execution_unavailable_reason: str | None
    source: str
    source_sha256: str
    source_bundle_sha256: str | None
    source_sessions: int
    replayable_source_sessions: int
    sample_seed: int
    plan_namespace: str
    workload_fingerprint: str
    prompt_calibration: dict[str, str | None]
    interval_model: IntervalModel
    tasks: tuple[ReplayTaskPlan, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete plan while preserving its artifact schema."""

        return {
            "schema_version": self.schema_version,
            "plan_kind": self.plan_kind,
            "planner_version": self.planner_version,
            "execution_ready": self.execution_ready,
            "execution_unavailable_reason": self.execution_unavailable_reason,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "source_sessions": self.source_sessions,
            "replayable_source_sessions": self.replayable_source_sessions,
            "sample_seed": self.sample_seed,
            "plan_namespace": self.plan_namespace,
            "workload_fingerprint": self.workload_fingerprint,
            "prompt_calibration": dict(self.prompt_calibration),
            "interval_model": self.interval_model.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
        }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _workload_config(config: ReplayBenchConfig) -> dict[str, object]:
    replay = config.replay.model_dump(mode="json")
    replay.pop("trace_path", None)
    return {
        "task_num": config.experiment.task_num,
        "max_concurrency": config.experiment.max_concurrency,
        "model": config.backend.model,
        "replay": replay,
        "sampler_version": "agentinfer-replay-sampler/v1",
        "planner_version": "agentinfer-replay-structural/v9",
    }


def _required_requests(session: ReplaySession) -> tuple[ReplayRequest, ...]:
    required = required_request_keys(session)
    return tuple(request for request in session.requests if request.key in required)


def _bounded_tokens(value: int | None, maximum: int | None) -> int | None:
    if value is None:
        return None
    return min(value, maximum) if maximum is not None else value


def _runtime_request_id(runtime_session_id: str, source_key: str) -> str:
    return str(uuid.uuid5(uuid.UUID(runtime_session_id), source_key))


def _context_plan(
    config: ReplayBenchConfig,
    request: ReplayRequest,
    token_targets: dict[str, tuple[int | None, int | None]],
) -> tuple[str | None, ContextMode]:
    """Choose the context predecessor and mode using bounded token targets."""

    source_context = request.context_after
    if request.replay_kind != "request":
        return source_context, "none"
    if config.replay.prompt_shape == "trace_record":
        if source_context is None:
            return None, "independent"
        return source_context, "append"
    if source_context is None or request.prompt_kind in _CONTEXT_FREE_PROMPT_KINDS:
        return None, "independent"
    if config.replay.context_adjustment_mode == "strict":
        return source_context, "append"

    target_input, _ = token_targets[request.key]
    context_input, context_output = token_targets[source_context]
    assert target_input is not None
    assert context_input is not None
    assert context_output is not None
    deficit = max(0, context_input + context_output - target_input)
    if deficit == 0:
        return source_context, "append"
    if deficit <= config.replay.context_micro_trim_limit(target_input):
        return source_context, "trim"
    return None, "reset"


def _task_plan(
    config: ReplayBenchConfig,
    sampled: SampledSession,
    interval_model: IntervalModel,
) -> ReplayTaskPlan:
    requests = _required_requests(sampled.source)
    token_targets = {
        request.key: (
            _bounded_tokens(request.input_tokens, config.replay.max_input_tokens),
            _bounded_tokens(request.output_tokens, config.replay.max_output_tokens),
        )
        for request in requests
    }
    nodes: list[ReplayPlanNode] = []
    for request in requests:
        context_after, context_mode = _context_plan(
            config,
            request,
            token_targets,
        )
        planned_input_tokens, planned_output_tokens = token_targets[request.key]
        nodes.append(
            ReplayPlanNode(
                runtime_request_id=_runtime_request_id(
                    sampled.runtime_session_id,
                    request.key,
                ),
                source_key=request.key,
                source_status=request.historical_status,
                prompt_recipe_key=request.prompt_recipe_key,
                prompt_ref=request.prompt_ref,
                node_type=request.replay_kind,
                prompt_kind=request.prompt_kind,
                actor_id=request.actor_id,
                actor_role=request.actor_role,
                parent_actor_id=request.parent_actor_id,
                send_after=request.send_after,
                context_after=context_after,
                context_mode=context_mode,
                effective_interval_seconds=effective_interval_seconds(
                    config.replay,
                    request,
                    interval_model,
                    sampled.runtime_session_id,
                ),
                effective_duration_seconds=(
                    (request.finished_at - request.started_at).total_seconds()
                    if request.replay_kind == "timing_dependency"
                    else None
                ),
                planned_input_tokens=planned_input_tokens,
                planned_output_tokens=planned_output_tokens,
                backend_sampling_seed=(
                    backend_sampling_seed(
                        config.replay,
                        request,
                        sampled.runtime_session_id,
                    )
                    if request.replay_kind == "request"
                    else None
                ),
            )
        )
    return ReplayTaskPlan(
        task_id=f"replay-{sampled.task_index:06d}",
        task_index=sampled.task_index,
        sample_ordinal=sampled.sample_ordinal,
        runtime_session_id=sampled.runtime_session_id,
        source_session_id=sampled.source.source_session_id,
        source_task_id=sampled.source.source_task_id,
        requests=tuple(nodes),
    )


def _validate_acyclic(nodes: Iterable[ReplayPlanNode]) -> None:
    by_key = {node.source_key: node for node in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visited:
            return
        if key in visiting:
            raise ValueError(f"send_after cycle detected at {key}")
        visiting.add(key)
        predecessor = by_key[key].send_after
        if predecessor is not None:
            visit(predecessor)
        visiting.remove(key)
        visited.add(key)

    for key in by_key:
        visit(key)


def _validate_task(task: ReplayTaskPlan) -> None:
    nodes = task.requests
    keys = [node.source_key for node in nodes]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate source request key in {task.task_id}")
    runtime_ids = [node.runtime_request_id for node in nodes]
    if len(runtime_ids) != len(set(runtime_ids)):
        raise ValueError(f"duplicate runtime request id in {task.task_id}")
    by_key = {node.source_key: node for node in nodes}
    for node in nodes:
        for field, reference in (
            ("send_after", node.send_after),
            ("context_after", node.context_after),
            ("prompt_recipe_key", node.prompt_recipe_key),
        ):
            if reference is not None and reference not in by_key:
                raise ValueError(f"{field} reference {reference} is missing in {task.task_id}")
        if node.node_type == "request":
            if not isinstance(node.planned_input_tokens, int) or node.planned_input_tokens <= 0:
                raise ValueError("real Replay request has no positive input target")
            if not isinstance(node.planned_output_tokens, int) or node.planned_output_tokens <= 0:
                raise ValueError("real Replay request has no positive output target")
        interval = node.effective_interval_seconds
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("Replay interval must be finite and non-negative")
        context_after = node.context_after
        if context_after is not None:
            context = by_key[context_after]
            if context.actor_id != node.actor_id:
                raise ValueError("context_after must reference the same actor")
            if context.source_status != "success":
                raise ValueError("context_after must reference a source success")
        if node.context_mode in {"independent", "reset"} and context_after is not None:
            raise ValueError(f"{node.context_mode} context must not reference context_after")
        if node.context_mode in {"append", "trim"} and context_after is None:
            raise ValueError(f"{node.context_mode} context requires context_after")
    _validate_acyclic(nodes)


def _validate_plan(tasks: Iterable[ReplayTaskPlan]) -> None:
    runtime_sessions: set[str] = set()
    for task in tasks:
        runtime_session_id = task.runtime_session_id
        if runtime_session_id in runtime_sessions:
            raise ValueError("runtime session ids must be unique")
        runtime_sessions.add(runtime_session_id)
        _validate_task(task)


def build_replay_plan(
    config: ReplayBenchConfig,
    analysis: ReplayAnalysis,
    trace_ir: UnifiedTraceIR | None = None,
) -> ReplayPlan:
    """Build a deterministic structural plan without sending Backend requests."""

    if config.replay.prompt_shape == "trace_record":
        if trace_ir is None:
            raise ValueError("trace_record planning requires a validated unified Trace IR")
    elif trace_ir is not None:
        raise ValueError("a unified Trace IR was supplied for a non-trace-record Replay mode")
    interval_model = build_interval_model(config.replay)
    replayable_count = sum(session.replayable for session in analysis.sessions)
    total_tasks = config.experiment.task_num or replayable_count
    if total_tasks <= 0:
        raise ValueError("the Replay trace has no usable sessions")

    workload_config = _workload_config(config)
    plan_namespace = _sha256_json(
        {
            "source_sha256": analysis.source_sha256,
            "source_bundle_sha256": trace_ir.bundle_sha256 if trace_ir is not None else None,
            "workload_config": workload_config,
            "interval_model": interval_model.to_dict(),
        }
    )
    sampled = sample_replay_sessions(
        analysis.sessions,
        total_tasks=total_tasks,
        seed=config.replay.sample_seed,
        plan_namespace=plan_namespace,
    )
    tasks = tuple(_task_plan(config, sampled_session, interval_model) for sampled_session in sampled)
    _validate_plan(tasks)
    workload = {
        "source_sha256": analysis.source_sha256,
        "source_bundle_sha256": trace_ir.bundle_sha256 if trace_ir is not None else None,
        "workload_config": workload_config,
        "interval_model": interval_model.to_dict(),
        "tasks": [task.to_dict() for task in tasks],
    }
    return ReplayPlan(
        schema_version="1",
        plan_kind="structural",
        planner_version="agentinfer-replay-structural/v9",
        execution_ready=False,
        execution_unavailable_reason="structural plan requires runtime Prompt calibration",
        source=str(config.replay.trace_path),
        source_sha256=analysis.source_sha256,
        source_bundle_sha256=trace_ir.bundle_sha256 if trace_ir is not None else None,
        source_sessions=len(analysis.sessions),
        replayable_source_sessions=replayable_count,
        sample_seed=config.replay.sample_seed,
        plan_namespace=plan_namespace,
        workload_fingerprint=_sha256_json(workload),
        prompt_calibration={
            "status": "pending",
            "model": config.backend.model,
            "tokenizer_base_url": config.backend.resolved_tokenizer_base_url,
        },
        interval_model=interval_model,
        tasks=tasks,
    )
