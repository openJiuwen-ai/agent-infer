# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Build finalized run summary artifacts from normalized metrics."""

from dataclasses import asdict
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from ..metrics.request import RequestMetrics
from ..metrics.schema import SourceHealth
from ..metrics.task import TaskMetrics
from ..metrics.vllm import VllmMetrics


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CorrectnessSummary:
    available: bool
    reason: str | None
    metadata: dict[str, object]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LifecycleSummary:
    status: Literal["completed", "failed", "summarized"]
    error: str | None
    finalization_errors: tuple[str, ...] = ()
    proxy_close: dict[str, object] | None = None


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RouterSummary:
    applicable: bool
    events: dict[str, int]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ExecutionSummary:
    """Describe execution-metric availability, its reason, and supporting metadata."""

    available: bool
    reason: str | None
    metadata: dict[str, object]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RunSummary:
    schema_version: Literal["1"]
    run_id: str
    tasks: TaskMetrics
    requests: RequestMetrics
    router: RouterSummary
    vllm: VllmMetrics
    correctness: CorrectnessSummary
    source_health: SourceHealth
    lifecycle: LifecycleSummary
    run_wall_time_seconds: float
    request_throughput_per_second: float
    input_token_throughput_per_second: float
    output_token_throughput_per_second: float
    execution: ExecutionSummary | None = None
    cli: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_execution(
    execution: ExecutionSummary | dict[str, object] | None,
) -> ExecutionSummary:
    """Normalize an execution summary and provide the default state."""

    if execution is None:
        return ExecutionSummary(
            available=True,
            reason=None,
            metadata={},
        )
    if isinstance(execution, ExecutionSummary):
        return execution
    return ExecutionSummary(**execution)


def build_run_summary(
    run_id: str,
    tasks: TaskMetrics,
    requests: RequestMetrics,
    vllm: VllmMetrics,
    correctness: CorrectnessSummary | dict[str, object],
    health: SourceHealth,
    lifecycle: LifecycleSummary | dict[str, object],
    run_wall_time_seconds: float,
    execution: ExecutionSummary | dict[str, object] | None = None,
) -> RunSummary:
    return RunSummary(
        schema_version="1",
        run_id=run_id,
        tasks=tasks,
        requests=requests,
        # Retain the field for historical artifact and compare compatibility.
        # Transparent Router use has no AgentBench-side metric collection.
        router=RouterSummary(False, {}),
        vllm=vllm,
        correctness=(correctness if isinstance(correctness, CorrectnessSummary) else CorrectnessSummary(**correctness)),
        source_health=health,
        lifecycle=lifecycle if isinstance(lifecycle, LifecycleSummary) else LifecycleSummary(**lifecycle),
        run_wall_time_seconds=run_wall_time_seconds,
        request_throughput_per_second=requests.requests / run_wall_time_seconds,
        input_token_throughput_per_second=requests.input_tokens / run_wall_time_seconds,
        output_token_throughput_per_second=requests.output_tokens / run_wall_time_seconds,
        execution=_coerce_execution(execution),
    )
