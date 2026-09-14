# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Orchestrate Trace Replay execution and finalize reviewer-facing artifacts.

This module owns run lifecycle, evidence aggregation, and summary metadata; it
delegates source analysis, planning, Prompt construction, and graph execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from pathlib import Path

from ..agents.contracts import AgentRunOutcome, AgentRunResult, TerminationReason
from ..benchkit.artifacts import build_run_manifest, build_run_summary, finalize_run_manifest
from ..benchkit.collectors.vllm import capture_vllm_metrics
from ..benchkit.common import atomic_write_json, utc_now, write_text
from ..benchkit.metrics.request import aggregate_request_metrics
from ..benchkit.metrics.schema import EvidenceCapture
from ..benchkit.metrics.source_health import evaluate_captures
from ..benchkit.metrics.task import aggregate_task_results
from ..benchkit.metrics.vllm import aggregate_vllm_metrics
from ..request_proxy.request_trace import RequestTraceWriter, load_request_facts
from .analyzer import analyze_replay_trace, replay_analysis_to_dict
from .config import ReplayBenchConfig
from .converters.codex_swebenchpro import CodexSwebenchProConverter
from .executor import ReplayExecutor, ReplayTaskExecution
from .planner import ReplayPlan, build_replay_plan
from .prompt import PromptBuilder, TokenizerClient
from .transport import ReplayTransport
from .unified_trace_ir import UnifiedTraceIR, validate_trace_ir

logger = logging.getLogger(__name__)


def _capture(source: str, path: Path, available: bool = True, reason: str | None = None) -> EvidenceCapture:
    return EvidenceCapture(source, path, available, reason, {})


def _capture_dict(capture: EvidenceCapture) -> dict[str, object]:
    """Serialize one Evidence capture for the run Manifest."""

    return {
        "source": capture.source,
        "path": str(capture.path) if capture.path else None,
        "available": capture.available,
        "reason": capture.reason,
        "metadata": dict(capture.metadata),
        "applicable": capture.applicable,
    }


async def _close_replay_resources(
    writer: RequestTraceWriter | None,
    tokenizer: TokenizerClient | None,
    transport: ReplayTransport | None,
    *,
    writer_started: bool,
) -> tuple[str, ...]:
    """Best-effort close initialized Replay resources without masking the run error."""

    errors: list[str] = []
    resources = (
        ("transport", transport, transport is not None),
        ("tokenizer", tokenizer, tokenizer is not None),
        ("writer", writer, writer is not None and writer_started),
    )
    for name, resource, should_close in resources:
        if not should_close or resource is None:
            continue
        try:
            await resource.close()
        except BaseException as exc:
            errors.append(f"{name} close failed: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _task_result(result: ReplayTaskExecution, endpoint: str) -> AgentRunResult:
    completed = result.status == "completed"
    return AgentRunResult(
        agent_type="replay",
        profile_name=endpoint,
        outcome=AgentRunOutcome.COMPLETED if completed else AgentRunOutcome.FAILED,
        termination_reason=None if completed else TerminationReason.HARNESS_ERROR,
        duration_seconds=result.duration_seconds,
        session_id=result.runtime_session_id,
        instance_id=result.task_id,
    )


def _execution_metadata(
    plan: ReplayPlan,
    tasks: tuple[ReplayTaskExecution, ...],
    tolerance_tokens: int = 0,
) -> dict[str, object]:
    """Aggregate planned coverage, node outcomes, and Prompt calibration evidence."""

    nodes = [node for task in tasks for node in task.nodes]
    planned_nodes = [node for task in plan.tasks for node in task.requests]
    calibrations = [node.prompt_calibration for node in nodes if node.prompt_calibration is not None]
    adjustments = [calibration for calibration in calibrations if calibration.get("adjustment") != "none"]
    request_nodes = [node for node in nodes if node.node_type == "request"]
    attempted_request_nodes = [node for node in request_nodes if node.actual_send_offset_seconds is not None]
    absolute_residuals = [abs(int(item.get("residual_tokens", 0))) for item in calibrations]
    return {
        "workload_fingerprint": plan.workload_fingerprint,
        "planned_tasks": len(plan.tasks),
        "completed_tasks": sum(task.status == "completed" for task in tasks),
        "failed_tasks": sum(task.status != "completed" for task in tasks),
        "planned_requests": sum(node.node_type == "request" for task in plan.tasks for node in task.requests),
        "planned_timing_dependency_nodes": sum(
            node.node_type == "timing_dependency" for task in plan.tasks for node in task.requests
        ),
        "materialized_request_nodes": len(request_nodes),
        "attempted_requests": len(attempted_request_nodes),
        "successful_requests": sum(node.status == "success" for node in attempted_request_nodes),
        "failed_requests": sum(node.status != "success" for node in attempted_request_nodes),
        "pre_send_failed_requests": sum(
            node.status == "failed" and node.actual_send_offset_seconds is None for node in request_nodes
        ),
        "dependency_skipped_requests": sum(node.status == "skipped_dependency_failed" for node in request_nodes),
        "timing_dependency_nodes": sum(node.node_type == "timing_dependency" for node in nodes),
        "lead_title_requests": sum(node.prompt_kind == "lead_title" for node in nodes),
        "lead_name_requests": sum(node.prompt_kind == "lead_name" for node in nodes),
        "lead_main_requests": sum(node.prompt_kind == "lead_main" for node in nodes),
        "planned_context_modes": {
            mode: sum(node.context_mode == mode for node in planned_nodes)
            for mode in ("none", "independent", "append", "trim", "reset")
        },
        "prompt_calibration_adjustments": {
            adjustment: sum(item.get("adjustment") == adjustment for item in adjustments)
            for adjustment in ("pad", "trim", "trim_and_pad", "reset", "reset_and_pad")
        },
        "prompt_calibration_exact_requests": sum(bool(item.get("target_met")) for item in calibrations),
        "prompt_calibration_tolerated_requests": sum(
            bool(item.get("accepted_with_tolerance")) for item in calibrations
        ),
        "prompt_calibration_repair_attempts": sum(int(item.get("repair_attempts", 0)) for item in calibrations),
        "prompt_calibration_signed_residual_tokens": sum(int(item.get("residual_tokens", 0)) for item in calibrations),
        "prompt_calibration_absolute_residual_tokens": sum(absolute_residuals),
        "prompt_calibration_max_absolute_residual_tokens": max(absolute_residuals, default=0),
        "prompt_calibration_mean_absolute_residual_tokens": (
            sum(absolute_residuals) / len(absolute_residuals) if absolute_residuals else 0.0
        ),
        "prompt_calibration_requests_over_tolerance": sum(value > tolerance_tokens for value in absolute_residuals),
        "requested_filler_tokens": sum(int(item.get("requested_filler_tokens", 0)) for item in calibrations),
        "actual_prompt_token_gain": sum(int(item.get("actual_prompt_token_gain", 0)) for item in calibrations),
        "trimmed_filler_tokens": sum(int(item.get("trimmed_filler_tokens", 0)) for item in adjustments),
    }


def _prepare_replay_source(
    config: ReplayBenchConfig,
    output_dir: Path,
) -> tuple[Path, UnifiedTraceIR | None, tuple[EvidenceCapture, ...]]:
    """Resolve the Analyzer input, validating converted IR once before returning it."""

    trace_type = config.replay.trace_type
    if trace_type == "agentinfer":
        logger.info("Using AgentInfer Replay trace directly: trace_path=%s", config.replay.trace_path)
        return config.replay.trace_path, None, ()
    if trace_type in {"agentX", "tracelab"}:
        raise NotImplementedError(f"trace_type={trace_type} is reserved for future integration")

    convert_dir = output_dir / "convert_result"
    started = time.monotonic()
    logger.info(
        "Converting Replay trace: trace_type=%s trace_path=%s output_dir=%s",
        trace_type,
        config.replay.trace_path,
        convert_dir,
    )
    converter = CodexSwebenchProConverter.from_backend(config)
    try:
        summary = converter.convert(config.replay.trace_path, convert_dir)
    finally:
        converter.close()
    trace_ir = validate_trace_ir(convert_dir / "requests.jsonl", convert_dir / "texts")
    captures = (_capture("replay_conversion_manifest", trace_ir.manifest_path),)
    logger.info(
        "Replay trace conversion completed: trace_type=%s sessions=%d requests=%d duration_seconds=%.3f bundle_sha256=%s",
        trace_type,
        summary.sessions,
        summary.requests,
        time.monotonic() - started,
        trace_ir.bundle_sha256,
    )
    return trace_ir.requests_path, trace_ir, captures


async def _run_replay(
    config: ReplayBenchConfig,
    cli_metadata: dict[str, object] | None,
) -> Path:
    output_dir = config.experiment.result_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir()
    manifest_config = config.model_dump(mode="json")
    manifest_config["implementation_stage"] = "executable"
    if cli_metadata is not None:
        manifest_config["cli"] = cli_metadata
    manifest = build_run_manifest(output_dir.name, manifest_config)
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest.to_dict())

    requests_path = output_dir / "requests.jsonl"
    captures: list[EvidenceCapture] = []
    writer: RequestTraceWriter | None = None
    tokenizer: TokenizerClient | None = None
    transport: ReplayTransport | None = None
    writer_started = False
    trace_ir: UnifiedTraceIR | None = None
    try:
        write_text(requests_path, "")
        captures.append(_capture("request_trace", requests_path))

        analysis_source, trace_ir, conversion_captures = _prepare_replay_source(config, output_dir)
        captures.extend(conversion_captures)

        analysis = analyze_replay_trace(analysis_source)
        analysis_path = output_dir / "replay-source-analysis.json"
        atomic_write_json(analysis_path, replay_analysis_to_dict(analysis))
        captures.append(_capture("replay_source_analysis", analysis_path))

        structural_plan = build_replay_plan(config, analysis, trace_ir)
        plan = replace(
            structural_plan,
            execution_ready=True,
            execution_unavailable_reason=None,
            prompt_calibration={
                "status": "active",
                "model": config.backend.model,
                "tokenizer_base_url": config.backend.resolved_tokenizer_base_url,
            },
        )
        plan_path = output_dir / "replay-plan.json"
        atomic_write_json(plan_path, plan.to_dict())
        captures.append(_capture("replay_plan", plan_path))

        vllm_start_capture = await capture_vllm_metrics(config.backend.effective_metrics_url)
        vllm_start = str(vllm_start_capture.metadata.get("text")) if vllm_start_capture.available else None
        vllm_start_path = evidence_dir / "vllm_metrics_start.prom"
        if vllm_start is not None:
            write_text(vllm_start_path, vllm_start)
        captures.append(
            _capture("vllm_start", vllm_start_path, vllm_start_capture.available, vllm_start_capture.reason)
        )

        writer = RequestTraceWriter(requests_path)
        await writer.start()
        writer_started = True
        tokenizer = TokenizerClient(config)
        transport = ReplayTransport(config, output_dir.name, writer)
        execution = ReplayExecutor(config, plan, PromptBuilder(config, tokenizer, trace_ir), transport).execute()
        if config.experiment.run_timeout_seconds:
            task_results = await asyncio.wait_for(execution, config.experiment.run_timeout_seconds)
        else:
            task_results = await execution

        await transport.close()
        transport = None
        await tokenizer.close()
        tokenizer = None
        writer_health = await writer.close()
        writer = None
        writer_started = False
        if writer_health.writer_error is not None:
            raise RuntimeError(f"request trace writer failed: {writer_health.writer_error}")

        plan = replace(plan, prompt_calibration={**plan.prompt_calibration, "status": "completed"})
        atomic_write_json(plan_path, plan.to_dict())
        execution_path = output_dir / "replay-execution.json"
        execution_metadata = _execution_metadata(
            plan,
            task_results,
            config.replay.prompt_calibration_tolerance_tokens,
        )
        atomic_write_json(
            execution_path,
            {
                "schema_version": "1",
                "summary": execution_metadata,
                "trace_writer": writer_health.__dict__,
                "tasks": [task.to_dict() for task in task_results],
            },
        )
        captures.append(_capture("replay_execution", execution_path))

        if config.replay.prompt_shape == "trace_record":
            validation_path = output_dir / "trace-record-validation.json"
            atomic_write_json(
                validation_path,
                {
                    "schema_version": "1",
                    "workload_fingerprint": plan.workload_fingerprint,
                    "source_bundle_sha256": plan.source_bundle_sha256,
                    "tolerance_tokens": config.replay.prompt_calibration_tolerance_tokens,
                    "max_absolute_residual_tokens": execution_metadata[
                        "prompt_calibration_max_absolute_residual_tokens"
                    ],
                    "mean_absolute_residual_tokens": execution_metadata[
                        "prompt_calibration_mean_absolute_residual_tokens"
                    ],
                    "sum_absolute_residual_tokens": execution_metadata["prompt_calibration_absolute_residual_tokens"],
                    "requests_over_tolerance": execution_metadata["prompt_calibration_requests_over_tolerance"],
                    "wire_fidelity": "content_order_and_token_length_not_byte_identical",
                    "assistant_source": "live_backend_length_constrained",
                },
            )
            captures.append(_capture("trace_record_validation", validation_path))

        vllm_end_capture = await capture_vllm_metrics(config.backend.effective_metrics_url)
        vllm_end = str(vllm_end_capture.metadata.get("text")) if vllm_end_capture.available else None
        vllm_end_path = evidence_dir / "vllm_metrics_end.prom"
        if vllm_end is not None:
            write_text(vllm_end_path, vllm_end)
        captures.append(_capture("vllm_end", vllm_end_path, vllm_end_capture.available, vllm_end_capture.reason))

        facts = load_request_facts(requests_path)
        finished_at = utc_now()
        run_wall_time = max((finished_at - manifest.created_at).total_seconds(), 1e-9)
        summary = build_run_summary(
            output_dir.name,
            aggregate_task_results(_task_result(result, config.backend.endpoint) for result in task_results),
            aggregate_request_metrics(facts),
            aggregate_vllm_metrics(vllm_start, vllm_end),
            {"available": False, "reason": "not applicable to Replay", "metadata": {}},
            evaluate_captures(captures),
            {"status": "completed", "error": None, "proxy_close": None},
            run_wall_time,
            execution={"available": True, "reason": None, "metadata": execution_metadata},
        ).to_dict()
        if cli_metadata is not None:
            summary["cli"] = cli_metadata
        atomic_write_json(output_dir / "summary.json", summary)
        finalized = finalize_run_manifest(
            manifest,
            tuple(_capture_dict(capture) for capture in captures),
            status="completed",
            finished_at=finished_at,
        )
        atomic_write_json(manifest_path, finalized.to_dict())
        return output_dir
    except BaseException as exc:
        logger.exception(
            "Replay run failed: result_dir=%s trace_type=%s error_type=%s",
            output_dir,
            config.replay.trace_type,
            type(exc).__name__,
        )
        cleanup_errors = await _close_replay_resources(
            writer,
            tokenizer,
            transport,
            writer_started=writer_started,
        )
        error_path = output_dir / "replay-error.json"
        atomic_write_json(
            error_path,
            {
                "schema_version": "1",
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
                "cleanup_errors": list(cleanup_errors),
            },
        )
        captures.append(_capture("replay_error", error_path))
        finalized = finalize_run_manifest(
            manifest,
            tuple(_capture_dict(capture) for capture in captures),
            status="failed",
            finished_at=utc_now(),
        )
        atomic_write_json(manifest_path, finalized.to_dict())
        raise


def run_replay(
    config: ReplayBenchConfig,
    *,
    cli_metadata: dict[str, object] | None = None,
) -> Path:
    """Analyze a source trace and execute its deterministic Replay plan."""

    return asyncio.run(_run_replay(config, cli_metadata))
