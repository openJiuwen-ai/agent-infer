# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Assemble and write run-level benchmark artifacts.

The runner collects evidence and executes tasks. This module writes
``task_index.json``, ``summary.json``, and ``manifest.json`` in that order,
recording individual failures so the final manifest reflects run status.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar

from ...agents.contracts import AgentRunResult
from ...request_proxy.lifecycle import RequestProxyCloseResult
from ...request_proxy.request_trace import RequestFact
from ..common import atomic_write_json
from ..metrics.request import aggregate_request_metrics
from ..metrics.schema import EvidenceCapture
from ..metrics.source_health import evaluate_captures
from ..metrics.task import aggregate_task_results
from ..metrics.vllm import aggregate_vllm_metrics
from .manifest import RunManifest, finalize_run_manifest
from .summary import build_run_summary
from .task_index import build_run_task_index

_T = TypeVar("_T")


@dataclass(frozen=True)
class RunFinalizationResult:
    """Return value of finalization: errors accumulated and final status."""

    finalization_errors: tuple[str, ...]
    status: Literal["completed", "failed"]


@dataclass
class FinalizationContext:
    """Bundle of inputs a run carries into finalization."""

    run_id: str
    output_dir: Path
    manifest: RunManifest
    results: list[AgentRunResult]
    facts: list[RequestFact] | None
    vllm_start: str | None
    vllm_end: str | None
    captures: list[EvidenceCapture]
    correctness: EvidenceCapture | None
    close_result: RequestProxyCloseResult | None
    run_exception: BaseException | None
    prior_finalization_errors: list[str]
    run_wall_time_seconds: float
    finished_at: datetime
    cli_metadata: dict[str, object] | None = None


class RunFinalizer:
    """Assemble run-level artifacts and return finalization status."""

    def __init__(self, ctx: FinalizationContext) -> None:
        self._ctx = ctx
        self._errors: list[str] = list(ctx.prior_finalization_errors)
        self._captured: list[EvidenceCapture] = list(ctx.captures)
        self._status: Literal["completed", "failed"] = "failed" if ctx.run_exception or self._errors else "completed"

    def finalize(self) -> RunFinalizationResult:
        """Build and write task_index, summary, manifest in that order."""
        self._write_task_index()
        self._write_summary()
        self._write_manifest()
        # Return only errors added during this finalization.
        new_errors = tuple(self._errors[len(self._ctx.prior_finalization_errors) :])
        return RunFinalizationResult(new_errors, self._status)

    # -- steps ---------------------------------------------------------------

    def _write_task_index(self) -> None:
        self._guard(
            "task_index_build",
            lambda: atomic_write_json(
                self._ctx.output_dir / "task_index.json",
                build_run_task_index(self._ctx.run_id, self._ctx.output_dir).to_dict(),
            ),
        )
        self._recompute_status()

    def _write_summary(self) -> None:
        vllm_metrics = self._guard(
            "vllm_aggregation",
            lambda: aggregate_vllm_metrics(self._ctx.vllm_start, self._ctx.vllm_end),
        )
        self._recompute_status()
        if self._ctx.facts is None or vllm_metrics is None:
            return
        lifecycle_payload = {
            "status": self._status,
            "error": self._exception_text(self._ctx.run_exception),
            "finalization_errors": list(self._errors),
            "proxy_close": asdict(self._ctx.close_result) if self._ctx.close_result else None,
        }
        # Keep manifest finalization independent when summary construction fails.
        summary = self._guard(
            "summary_build",
            lambda: build_run_summary(
                self._ctx.run_id,
                aggregate_task_results(self._ctx.results),
                aggregate_request_metrics(self._ctx.facts),
                vllm_metrics,
                {
                    "available": self._ctx.correctness.available if self._ctx.correctness else False,
                    "reason": self._ctx.correctness.reason if self._ctx.correctness else "collection failed",
                    "metadata": {},
                },
                evaluate_captures(self._captured),
                lifecycle_payload,
                self._ctx.run_wall_time_seconds,
            ).to_dict(),
        )
        if summary is not None:
            if self._ctx.cli_metadata:
                summary["cli"] = self._ctx.cli_metadata
            self._guard("summary_write", lambda: atomic_write_json(self._ctx.output_dir / "summary.json", summary))
        self._recompute_status()

    def _write_manifest(self) -> None:
        try:
            evidence = tuple(self._capture_dict(capture) for capture in self._captured)
            finalized_manifest = finalize_run_manifest(
                self._ctx.manifest,
                evidence,
                status=self._status,
                finished_at=self._ctx.finished_at,
            )
            atomic_write_json(self._ctx.output_dir / "manifest.json", finalized_manifest.to_dict())
        except BaseException as exc:  # noqa: BLE001 - finalization must not crash
            # Record interrupts so finalization can still write the manifest.
            self._errors.append(f"manifest_finalize: {type(exc).__name__}: {exc}")
            self._captured.append(EvidenceCapture("manifest_finalize", None, False, f"{type(exc).__name__}: {exc}", {}))
            self._status = "failed"

    # -- helpers -------------------------------------------------------------

    def _guard(self, source: str, operation: Callable[[], _T]) -> _T | None:
        """Run ``operation``; on any raise record a finalization error and return None."""
        try:
            return operation()
        except BaseException as exc:  # noqa: BLE001 - finalization must not crash
            self._errors.append(f"{source}: {type(exc).__name__}: {exc}")
            self._captured.append(EvidenceCapture(source, None, False, f"{type(exc).__name__}: {exc}", {}))
            return None

    def _recompute_status(self) -> None:
        self._status = "failed" if self._ctx.run_exception or self._errors else "completed"

    @staticmethod
    def _capture_dict(capture: EvidenceCapture) -> dict[str, object]:
        return {
            "source": capture.source,
            "path": str(capture.path) if capture.path is not None else None,
            "available": capture.available,
            "reason": capture.reason,
            "metadata": {},
            "applicable": capture.applicable,
        }

    @staticmethod
    def _exception_text(exc: BaseException | None) -> str | None:
        return f"{type(exc).__name__}: {exc}" if exc is not None else None
