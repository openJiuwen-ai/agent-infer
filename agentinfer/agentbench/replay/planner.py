# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Build and validate deterministic structural Replay plans."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
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

# Enumerated values a loaded plan must accept. These mirror the Literal
# contracts on the dataclass fields and are the fail-closed source of truth
# for ``from_dict`` so a corrupted artifact raises instead of round-tripping.
_CONTEXT_MODES = frozenset({"none", "independent", "append", "trim", "reset"})
_ACTOR_ROLES = frozenset({"lead", "subagent", "unknown"})
_NODE_TYPES = frozenset({"request", "timing_dependency"})
_PROMPT_KINDS = frozenset({"lead_title", "lead_name", "lead_main", "subagent_first", "continuation", "none"})
_RESPONSE_VALIDATIONS = frozenset({"exact_tokens"})
_PLAN_KINDS = frozenset({"structural"})
_SCHEMA_VERSIONS = frozenset({"1", "2"})


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
    response_validation: Literal["exact_tokens"] | None = None
    source_prefix_tokens: int | None = None
    source_output_tokens: int | None = None

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
                "source_output_tokens": self.source_output_tokens,
                "backend_sampling_seed": self.backend_sampling_seed,
            }
        )
        if self.prompt_ref is not None:
            value["prompt_ref"] = self.prompt_ref.to_dict()
        if self.response_validation is not None:
            value["response_validation"] = self.response_validation
            if self.planned_output_tokens == 0:
                value["effective_output_tokens"] = self.planned_output_tokens
        if self.source_prefix_tokens is not None:
            value["source_prefix_tokens"] = self.source_prefix_tokens
        return value

    @classmethod
    def from_dict(cls, value: object) -> ReplayPlanNode:
        """Reconstruct a node from a serialized plan artifact.

        Mirrors :meth:`to_dict` presence rules exactly: a ``timing_dependency``
        carries only its clock/duration, while a ``request`` carries the full
        materialized prompt/actor/context set. Fields the build path populates
        but ``to_dict`` intentionally omits (``source_status``) are restored to
        execution-neutral defaults; nothing downstream reads them for a loaded
        plan, and :func:`_validate_plan` is not re-run on load (it needs the
        original source).
        """

        if not isinstance(value, dict):
            raise ValueError("plan node must be an object")
        node_type = value.get("node_type")
        if node_type not in _NODE_TYPES:
            raise ValueError(f"node node_type must be one of {sorted(_NODE_TYPES)}")
        runtime_request_id = _node_str(value, "runtime_request_id")
        source_key = _node_str(value, "source_key")
        send_after = _node_optional_str(value, "send_after")
        effective_interval = _node_nonneg_finite(value, "effective_interval_seconds")

        if node_type == "timing_dependency":
            return cls(
                runtime_request_id=runtime_request_id,
                source_key=source_key,
                source_status="unknown",
                prompt_recipe_key="",
                prompt_ref=None,
                node_type=node_type,
                prompt_kind="none",
                actor_id="",
                actor_role="unknown",
                parent_actor_id=None,
                send_after=send_after,
                context_after=None,
                context_mode="none",
                effective_interval_seconds=effective_interval,
                effective_duration_seconds=_node_nonneg_finite(value, "effective_duration_seconds"),
                planned_input_tokens=None,
                planned_output_tokens=None,
                backend_sampling_seed=None,
            )

        prompt_ref = value.get("prompt_ref")
        context_after = _node_optional_str(value, "context_after")
        planned_input_tokens = _node_optional_nonneg_int(value, "planned_input_tokens")
        planned_output_tokens = _node_optional_nonneg_int(value, "planned_output_tokens")
        backend_sampling_seed = _node_optional_nonneg_int(value, "backend_sampling_seed")
        source_prefix_tokens = _node_optional_nonneg_int(value, "source_prefix_tokens")
        source_output_tokens = _node_optional_nonneg_int(value, "source_output_tokens")
        response_validation = value.get("response_validation")
        if response_validation is not None and response_validation not in _RESPONSE_VALIDATIONS:
            raise ValueError(f"node response_validation must be one of {sorted(_RESPONSE_VALIDATIONS)}")

        return cls(
            runtime_request_id=runtime_request_id,
            source_key=source_key,
            source_status="unknown",
            prompt_recipe_key=_node_str(value, "prompt_recipe_key"),
            prompt_ref=PromptReference.from_dict(prompt_ref) if prompt_ref is not None else None,
            node_type=node_type,
            prompt_kind=_node_enum(value, "prompt_kind", _PROMPT_KINDS),
            actor_id=_node_str(value, "actor_id"),
            actor_role=_node_enum(value, "actor_role", _ACTOR_ROLES),
            parent_actor_id=_node_optional_str(value, "parent_actor_id"),
            send_after=send_after,
            context_after=context_after,
            context_mode=_node_enum(value, "context_mode", _CONTEXT_MODES),
            effective_interval_seconds=effective_interval,
            effective_duration_seconds=None,
            planned_input_tokens=planned_input_tokens,
            planned_output_tokens=planned_output_tokens,
            backend_sampling_seed=backend_sampling_seed,
            response_validation=response_validation,
            source_prefix_tokens=source_prefix_tokens,
            source_output_tokens=source_output_tokens,
        )


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

    @classmethod
    def from_dict(cls, value: object) -> ReplayTaskPlan:
        """Reconstruct one task plan from its serialized form."""

        if not isinstance(value, dict):
            raise ValueError("task plan must be an object")
        requests = value.get("requests")
        if not isinstance(requests, list):
            raise ValueError(f"task {value.get('task_id')!r} requests must be a list")
        return cls(
            task_id=_node_str(value, "task_id"),
            task_index=_nonneg_int_field(value, "task_index"),
            sample_ordinal=_nonneg_int_field(value, "sample_ordinal"),
            runtime_session_id=_node_str(value, "runtime_session_id"),
            source_session_id=_node_str(value, "source_session_id"),
            source_task_id=_node_optional_str(value, "source_task_id"),
            requests=tuple(ReplayPlanNode.from_dict(node) for node in requests),
        )


@dataclass(frozen=True)
class ReplayPlan:
    """A validated deterministic Replay workload plan."""

    schema_version: Literal["1", "2"]
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

        value = {
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
        return value

    @classmethod
    def from_dict(cls, value: object) -> ReplayPlan:
        """Reconstruct a plan from its serialized artifact.

        Fail-closed on every field; callers must still run
        :func:`validate_loaded_plan` for cross-field invariants (reference
        closure, acyclicity, session uniqueness) that per-field parsing cannot
        see.
        """

        if not isinstance(value, dict):
            raise ValueError("replay plan must be an object")
        schema_version = value.get("schema_version")
        if schema_version not in _SCHEMA_VERSIONS:
            raise ValueError(f"plan schema_version must be one of {sorted(_SCHEMA_VERSIONS)}")
        plan_kind = value.get("plan_kind")
        if plan_kind not in _PLAN_KINDS:
            raise ValueError(f"plan plan_kind must be one of {sorted(_PLAN_KINDS)}")
        execution_ready = value.get("execution_ready")
        if not isinstance(execution_ready, bool):
            raise ValueError("plan execution_ready must be a boolean")
        prompt_calibration = value.get("prompt_calibration")
        if not isinstance(prompt_calibration, dict):
            raise ValueError("plan prompt_calibration must be an object")
        tasks = value.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("plan tasks must be a list")
        return cls(
            schema_version=schema_version,
            plan_kind=plan_kind,
            planner_version=_str_field(value, "planner_version"),
            execution_ready=execution_ready,
            execution_unavailable_reason=_opt_str_field(value, "execution_unavailable_reason"),
            source=_str_field(value, "source"),
            source_sha256=_str_field(value, "source_sha256"),
            source_bundle_sha256=_opt_str_field(value, "source_bundle_sha256"),
            source_sessions=_nonneg_int_field(value, "source_sessions"),
            replayable_source_sessions=_nonneg_int_field(value, "replayable_source_sessions"),
            sample_seed=_nonneg_int_field(value, "sample_seed"),
            plan_namespace=_str_field(value, "plan_namespace"),
            workload_fingerprint=_str_field(value, "workload_fingerprint"),
            prompt_calibration=dict(prompt_calibration),
            interval_model=_interval_model_from_dict(value.get("interval_model")),
            tasks=tuple(ReplayTaskPlan.from_dict(task) for task in tasks),
        )

    @classmethod
    def load(cls, path: Path) -> ReplayPlan:
        """Load and structurally validate a plan from a ``replay-plan.json`` file.

        This is the convenience entry for resuming from disk; it reads the
        JSON, reconstructs the plan, and runs :func:`validate_loaded_plan`.
        The plan it returns is the structural artifact -- execution readiness
        and Prompt calibration are still established by the runner, not here.
        """

        data = json.loads(path.read_text(encoding="utf-8"))
        plan = cls.from_dict(data)
        validate_loaded_plan(plan)
        return plan


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _workload_config(config: ReplayBenchConfig, planner_version: str) -> dict[str, object]:
    replay = config.replay.model_dump(mode="json")
    replay.pop("trace_path", None)
    if config.replay.trace_type != "agentX":
        replay.pop("max_inflight_requests", None)
    return {
        "task_num": config.experiment.task_num,
        "max_concurrency": config.experiment.max_concurrency,
        "model": config.backend.model,
        "chat_template_kwargs": config.backend.chat_template_kwargs,
        "replay": replay,
        "sampler_version": "agentinfer-replay-sampler",
        "planner_version": planner_version,
    }


def _required_requests(session: ReplaySession) -> tuple[ReplayRequest, ...]:
    required = required_request_keys(session)
    return tuple(request for request in session.requests if request.key in required)


def _runtime_request_id(runtime_session_id: str, source_key: str) -> str:
    return str(uuid.uuid5(uuid.UUID(runtime_session_id), source_key))


def _context_plan(
    config: ReplayBenchConfig,
    request: ReplayRequest,
    token_targets: dict[str, tuple[int | None, int | None]],
) -> tuple[str | None, ContextMode]:
    """Choose the context predecessor and mode using source token targets."""

    source_context = request.context_after
    if request.replay_kind != "request":
        return source_context, "none"
    if config.replay.prompt_shape in {"inferact_synthetic", "tracelab_synthetic"}:
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
    extra_tokens = (
        config.replay.extra_system_tokens(request.key, request.actor_role) if config.replay.uses_token_ids else 0
    )
    deficit = max(0, context_input + context_output + extra_tokens - target_input)
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
    token_targets = {request.key: (request.input_tokens, request.output_tokens) for request in requests}
    nodes: list[ReplayPlanNode] = []
    for request in requests:
        context_after, context_mode = _context_plan(
            config,
            request,
            token_targets,
        )
        planned_input_tokens, planned_output_tokens = token_targets[request.key]
        if config.replay.prompt_shape == "agentX_snapshot" and planned_output_tokens == 0:
            planned_output_tokens = 1
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
                source_output_tokens=request.source_output_tokens,
                backend_sampling_seed=(
                    backend_sampling_seed(
                        config.replay,
                        request,
                        sampled.runtime_session_id,
                    )
                    if request.replay_kind == "request"
                    else None
                ),
                response_validation="exact_tokens" if config.replay.uses_token_ids else None,
                source_prefix_tokens=(
                    request.source_cached_tokens if config.replay.prompt_shape == "tracelab_synthetic" else None
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


def _validate_task(
    task: ReplayTaskPlan, *, allow_unknown_context: bool = False, allow_hash_snapshot: bool = False
) -> None:
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
            if field == "prompt_recipe_key" and allow_hash_snapshot:
                if not reference.isdecimal():
                    raise ValueError("hash snapshot prompt_recipe_key must be a byte offset")
                continue
            if reference is not None and reference not in by_key:
                raise ValueError(f"{field} reference {reference} is missing in {task.task_id}")
        if node.node_type == "request":
            if not isinstance(node.planned_input_tokens, int) or node.planned_input_tokens <= 0:
                raise ValueError("real Replay request has no positive input target")
            minimum_output = 0 if allow_unknown_context else 1
            if not isinstance(node.planned_output_tokens, int) or node.planned_output_tokens < minimum_output:
                raise ValueError("real Replay request has an invalid output target")
            if node.source_prefix_tokens is not None and (
                node.source_prefix_tokens < 0 or node.source_prefix_tokens > node.planned_input_tokens
            ):
                raise ValueError("TraceLab prefix_tokens must fit the input target")
        interval = node.effective_interval_seconds
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("Replay interval must be finite and non-negative")
        context_after = node.context_after
        if context_after is not None:
            context = by_key[context_after]
            if context.actor_id != node.actor_id:
                raise ValueError("context_after must reference the same actor")
            # TraceLab has no historical success evidence; execution still requires a live success.
            allowed_statuses = {"success", "unknown"} if allow_unknown_context else {"success"}
            if context.source_status not in allowed_statuses:
                raise ValueError("context_after must reference a source success")
        if node.context_mode in {"independent", "reset"} and context_after is not None:
            raise ValueError(f"{node.context_mode} context must not reference context_after")
        if node.context_mode in {"append", "trim"} and context_after is None:
            raise ValueError(f"{node.context_mode} context requires context_after")
    _validate_acyclic(nodes)


def _node_str(node: dict[str, object], field: str) -> str:
    value = node.get(field)
    if not isinstance(value, str):
        raise ValueError(f"node {field} must be a string")
    return value


def _node_optional_str(node: dict[str, object], field: str) -> str | None:
    value = node.get(field)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"node {field} must be a string or null")


def _node_enum(node: dict[str, object], field: str, allowed: frozenset[str]) -> str:
    value = node.get(field)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"node {field} must be one of {sorted(allowed)}")
    return value


def _node_finite(node: dict[str, object], field: str) -> float:
    value = node.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"node {field} must be a finite number")
    return float(value)


def _node_nonneg_finite(node: dict[str, object], field: str) -> float:
    value = _node_finite(node, field)
    if value < 0:
        raise ValueError(f"node {field} must be non-negative")
    return value


def _node_optional_nonneg_int(node: dict[str, object], field: str) -> int | None:
    value = node.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"node {field} must be a non-negative integer or null")
    return value


def _str_field(plan: dict[str, object], field: str) -> str:
    value = plan.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"plan {field} must be a non-empty string")
    return value


def _opt_str_field(plan: dict[str, object], field: str) -> str | None:
    value = plan.get(field)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"plan {field} must be a string or null")


def _nonneg_int_field(plan: dict[str, object], field: str) -> int:
    value = plan.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"plan {field} must be a non-negative integer")
    return value


def _interval_model_from_dict(value: object) -> IntervalModel:
    """Reconstruct an :class:`IntervalModel` from ``IntervalModel.to_dict()``.

    ``IntervalModel.to_dict`` is ``asdict(self)``; any field not declared on the
    dataclass (or a mistyped value) is rejected so a drifted artifact cannot
    load silently.
    """

    if not isinstance(value, dict):
        raise ValueError("plan interval_model must be an object")
    known = {item.name for item in fields(IntervalModel)}
    for key in value:
        if key not in known:
            raise ValueError(f"interval_model has unknown field {key!r}")
    return IntervalModel(
        mode=_str_field(value, "mode"),
        fit_version=_str_field(value, "fit_version"),
        mu=value.get("mu"),
        sigma=value.get("sigma"),
        anchors=value.get("anchors"),
        anchor_predictions=value.get("anchor_predictions"),
        anchor_residual_ratios=value.get("anchor_residual_ratios"),
        log_space_rmse=value.get("log_space_rmse"),
        mean_predicted=value.get("mean_predicted"),
        u_clamp=value.get("u_clamp"),
        cdf_version=value.get("cdf_version"),
    )


def validate_loaded_plan(plan: ReplayPlan) -> None:
    """Structurally validate a plan reconstructed via ``ReplayPlan.from_dict``.

    This is the load-time counterpart to :func:`_validate_plan`. It checks the
    facts recoverable from the serialized artifact alone -- type/enum
    well-formedness (already enforced by ``from_dict``), reference closure,
    ``send_after`` acyclicity, and node target sanity. It intentionally does
    **not** re-run :func:`_validate_plan`: that pass needs the original source
    (``source_status``, context token targets, hash-snapshot offsets) which a
    standalone artifact does not carry. Likewise it does not re-derive
    ``workload_fingerprint``/``plan_namespace`` -- both embed the build-time
    workload config, so they are checked for presence only; build-side stability
    is already covered by the existing planner tests.
    """

    if plan.plan_kind not in _PLAN_KINDS:
        raise ValueError(f"plan plan_kind must be one of {sorted(_PLAN_KINDS)}")
    if plan.schema_version not in _SCHEMA_VERSIONS:
        raise ValueError(f"plan schema_version must be one of {sorted(_SCHEMA_VERSIONS)}")
    if not isinstance(plan.workload_fingerprint, str) or not plan.workload_fingerprint:
        raise ValueError("plan workload_fingerprint must be a non-empty string")
    if not isinstance(plan.plan_namespace, str) or not plan.plan_namespace:
        raise ValueError("plan plan_namespace must be a non-empty string")
    if not isinstance(plan.prompt_calibration, dict):
        raise ValueError("plan prompt_calibration must be an object")

    runtime_sessions: set[str] = set()
    for task in plan.tasks:
        if task.task_index < 0:
            raise ValueError(f"task {task.task_id} task_index must be non-negative")
        if task.runtime_session_id in runtime_sessions:
            raise ValueError("runtime session ids must be unique")
        runtime_sessions.add(task.runtime_session_id)
        nodes = task.requests
        by_key = {node.source_key: node for node in nodes}
        if len(by_key) != len(nodes):
            raise ValueError(f"duplicate source request key in {task.task_id}")
        runtime_ids = [node.runtime_request_id for node in nodes]
        if len(runtime_ids) != len(set(runtime_ids)):
            raise ValueError(f"duplicate runtime request id in {task.task_id}")
        for node in nodes:
            for field, reference in (("send_after", node.send_after), ("context_after", node.context_after)):
                if reference is not None and reference not in by_key:
                    raise ValueError(f"{field} reference {reference} is missing in {task.task_id}")
            if node.node_type == "request" and not (
                isinstance(node.planned_input_tokens, int) and node.planned_input_tokens > 0
            ):
                raise ValueError(f"request {node.source_key} has no positive input target")
        _validate_acyclic(nodes)


def _validate_plan(
    tasks: Iterable[ReplayTaskPlan], *, allow_unknown_context: bool = False, allow_hash_snapshot: bool = False
) -> None:
    runtime_sessions: set[str] = set()
    for task in tasks:
        runtime_session_id = task.runtime_session_id
        if runtime_session_id in runtime_sessions:
            raise ValueError("runtime session ids must be unique")
        runtime_sessions.add(runtime_session_id)
        _validate_task(task, allow_unknown_context=allow_unknown_context, allow_hash_snapshot=allow_hash_snapshot)


def build_replay_plan(
    config: ReplayBenchConfig,
    analysis: ReplayAnalysis,
    trace_ir: UnifiedTraceIR | None = None,
) -> ReplayPlan:
    """Build a deterministic structural plan without sending Backend requests."""

    if config.replay.prompt_shape in {"inferact_synthetic", "tracelab_synthetic", "agentX_snapshot"}:
        if trace_ir is None:
            raise ValueError(f"{config.replay.prompt_shape} planning requires a validated unified Trace IR")
    elif trace_ir is not None:
        raise ValueError("a unified Trace IR was supplied for a non-trace-record Replay mode")
    interval_model = build_interval_model(config.replay)
    replayable_count = sum(session.replayable for session in analysis.sessions)
    total_tasks = config.experiment.task_num
    if total_tasks is None:  # Defensive for unchecked model construction.
        raise ValueError("experiment.task_num is required for Replay")
    if total_tasks <= 0:
        raise ValueError("the Replay trace has no usable sessions")

    if config.replay.prompt_shape == "tracelab_synthetic":
        if trace_ir is None or trace_ir.prompt_source_kind != "token_recipe":
            raise ValueError("token_recipe planning requires token_recipe unified Trace IR")
    if config.replay.prompt_shape == "agentX_snapshot":
        if trace_ir is None or trace_ir.prompt_source_kind != "hash_snapshot":
            raise ValueError("AgentX planning requires hash_snapshot unified Trace IR")
    planner_version = "agentinfer-replay-structural"
    workload_config = _workload_config(config, planner_version)
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
    _validate_plan(
        tasks,
        allow_unknown_context=config.replay.prompt_shape == "tracelab_synthetic",
        allow_hash_snapshot=config.replay.prompt_shape == "agentX_snapshot",
    )
    workload = {
        "source_sha256": analysis.source_sha256,
        "source_bundle_sha256": trace_ir.bundle_sha256 if trace_ir is not None else None,
        "workload_config": workload_config,
        "interval_model": interval_model.to_dict(),
        "tasks": [task.to_dict() for task in tasks],
    }
    return ReplayPlan(
        schema_version="2" if config.replay.prompt_shape in {"tracelab_synthetic", "agentX_snapshot"} else "1",
        plan_kind="structural",
        planner_version=planner_version,
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
