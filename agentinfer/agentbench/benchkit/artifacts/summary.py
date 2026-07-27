"""Build finalized run summary artifacts from normalized metrics."""

from dataclasses import asdict
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from ..metrics.request import RequestMetrics
from ..metrics.router import RouterMetrics
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
    cli: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_run_summary(
    run_id: str,
    tasks: TaskMetrics,
    requests: RequestMetrics,
    router: RouterMetrics | None,
    vllm: VllmMetrics,
    correctness: CorrectnessSummary | dict[str, object],
    health: SourceHealth,
    lifecycle: LifecycleSummary | dict[str, object],
) -> RunSummary:
    return RunSummary(
        "1",
        run_id,
        tasks,
        requests,
        RouterSummary(router is not None, dict(router.events) if router else {}),
        vllm,
        correctness if isinstance(correctness, CorrectnessSummary) else CorrectnessSummary(**correctness),
        health,
        lifecycle if isinstance(lifecycle, LifecycleSummary) else LifecycleSummary(**lifecycle),
    )
