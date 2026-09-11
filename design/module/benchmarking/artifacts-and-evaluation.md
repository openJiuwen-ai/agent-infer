---
title: Benchmark artifacts and evaluation
kind: module
status: draft
owners:
  - TBD
primary_code_paths:
  - agentinfer/agentbench/benchkit/artifacts/**
  - agentinfer/agentbench/benchkit/metrics/**
  - agentinfer/agentbench/benchkit/compare.py
related_code_paths:
  - agentinfer/agentbench/benchkit/collectors/**
  - agentinfer/agentbench/replay/**
  - agentinfer/agentbench/request_proxy/request_trace.py
depends_on:
  - index.md
  - benchkit-orchestration.md
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/8
  - https://github.com/JiusiServe/AgentInfer/issues/27
validation_paths:
  - tests/agentbench/collectors/**
  - tests/agentbench/metrics/**
  - tests/agentbench/test_compare.py
  - tests/agentbench/test_runner_lifecycle.py
upstream_refs:
  - vLLM 0.23.0 metrics
  - SWE-bench external evaluation contract
last_reviewed: 2026-08-26
---

## Benchmark artifacts and evaluation

## Data boundary

Collectors capture raw external evidence and availability. Metrics convert source-owned facts into normalized values.
Artifacts validate and serialize task/run contracts. Comparison reads finalized summaries. These layers do not call
serving services outside collectors and do not duplicate each other's formulas.

Raw request traces, service captures, task outcomes, patches, and environment/source-control evidence remain
authoritative. Missing evidence is represented as unavailable or not applicable, not silently converted to a successful
zero.

## Artifact ownership

| Artifact | Owner and purpose |
| --- | --- |
| `tasks/<instance_id>/result.json` | Per-task execution result, patch presence, topology, timestamps, and errors. |
| `task_index.json` | Run-level task position, task-to-session mapping, and failed-task roll-up. |
| `summary.json` | Normalized run metrics and lifecycle state for comparison. |
| `manifest.json` | Run identity, configuration, provenance, and captured-evidence inventory. |
| `requests.jsonl` | Immutable request trace; the authoritative request-level evidence. |
| `sessions.csv`, `agents.csv`, `distribution_samples.csv` | Derived diagnostics for session, agent, and distribution analysis. |
| `replay-source-analysis.json`, `replay-plan.json`, `replay-execution.json` | Trace Replay source analysis and inference, context/execution plan, and runtime calibration/outcomes. |

The [run artifacts reference](../../../docs/en/reference/run-artifacts.md) explains the output files.
The implementation remains authoritative for `summary.json` fields and calculation inputs;
this document intentionally does not repeat metric formulas.

Correctness is external: BenchKit exports `model.patch` and imports evaluator availability. Agent completion and patch
presence are not SWE-bench resolution claims.

Replay starts from `requests.jsonl` and adds `replay-source-analysis.json`, `replay-plan.json`,
and `replay-execution.json` before finalizing the common request and service evidence.
For current Replay validation, two runs of the same source trace must use independently restarted vLLM services and
must compare the resulting summaries for cold-start reproducibility.

Replay artifacts do not establish equivalence with an actual Claude Code run. The synthetic workload does not yet
fully reconstruct original Prompt semantics, complete Tool schemas, structured Tool exchanges, natural stopping, or
task correctness; persistent metric differences must be reported rather than interpreted as scheduler regressions or
exact Claude parity.

### BENCH-INV-008: Comparison consumes finalized evidence

**Rule:** Comparison MUST reject incomplete, incompatible, or incorrectly paired baseline/candidate summaries.

**Enforced by:** `tests/agentbench/test_compare.py`.

**Approved alternative:** Finalize or explicitly migrate the source artifacts before comparison.

### BENCH-INV-009: Cold status is evidence-based

**Rule:** A run MUST NOT be described as cold solely because commands were executed in order; cold-state claims require
matching service-start evidence or an explicit warning.

**Enforced by:** compare cold-evidence tests and the manual cold E2E checklist.

**Approved alternative:** Report cold status as unavailable and exclude it from release-gate claims.

## Evaluation guide

Issue #8 defines directional and release-gate metrics. A fair pair uses the same commit, model, tasks/order, profile,
concurrency, timeouts, and hardware, with fresh service state for each arm. Runtime/cache comparison and external
correctness evidence must both be present before a release claim.
