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
  - agentinfer/agentbench/request_proxy/request_trace.py
depends_on:
  - index.md
  - benchkit-orchestration.md
validation_paths:
  - tests/agentbench/collectors/**
  - tests/agentbench/metrics/**
  - tests/agentbench/test_compare.py
  - tests/agentbench/test_runner_lifecycle.py
upstream_refs:
  - vLLM 0.23.0 metrics
  - SWE-bench external evaluation contract
last_reviewed: 2026-07-21
---

## Benchmark artifacts and evaluation

## Data boundary

Collectors capture raw external evidence and availability. Metrics convert source-owned facts into normalized values.
Artifacts validate and serialize task/run contracts. Comparison reads finalized summaries. These layers do not call
serving services outside collectors and do not duplicate each other's formulas.

Raw request traces, service captures, task outcomes, patches, and environment/source-control evidence remain
authoritative. Missing evidence is represented as unavailable or not applicable, not silently converted to a successful
zero.

Correctness is external: BenchKit exports `model.patch` and imports evaluator evidence. Agent completion and patch
presence are not SWE-bench resolution claims.

### BENCH-INV-008: Comparison consumes finalized evidence

**Rule:** Comparison MUST reject incomplete, incompatible, or incorrectly paired baseline/candidate summaries.

**Enforced by:** `tests/agentbench/test_compare.py` after PR-06/07 integration.

**Approved alternative:** Finalize or explicitly migrate the source artifacts before comparison.

### BENCH-INV-009: Cold status is evidence-based

**Rule:** A run MUST NOT be described as cold solely because commands were executed in order; cold-state claims require
matching service-start evidence or an explicit warning.

**Enforced by:** compare cold-evidence tests and the manual cold E2E checklist.

**Approved alternative:** Report cold status as unavailable and exclude it from release-gate claims.

## Evaluation guide

Issue #8 defines the directional and release-gate metrics. A fair pair uses the same commit, model, tasks/order,
profile, concurrency, timeouts, and hardware, with fresh service state for each arm. Runtime/cache comparison and
external correctness evidence must both be present before a release claim.
