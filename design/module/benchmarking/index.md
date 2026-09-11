---
title: Benchmark subsystem
kind: module
status: draft
owners:
  - TBD
primary_code_paths:
  - agentinfer/agentbench/**
related_code_paths:
  - agentinfer/scheduling/headers.py
  - agentinfer/agentcache/entrypoints/**
depends_on:
  - ../entrypoints.md
validation_paths:
  - tests/agentbench/**
upstream_refs:
  - vLLM 0.23.0 Anthropic Messages API
last_reviewed: 2026-07-21
---

## Benchmark subsystem

## Purpose

The benchmark compares a pinned vLLM Prefix Cache baseline with an AgentCache Router candidate using the same agent
workload and transparent request-observation path. It records execution, request, cache, service, and reproducibility
evidence. External SWE-bench evaluation owns correctness.

## Module map

| Document | Primary owner |
| --- | --- |
| [BenchKit orchestration](benchkit-orchestration.md) | Configuration, datasets, workspaces, run lifecycle, CLI, and Router control orchestration |
| [Agent runtime adapters](agent-runtime-adapters.md) | Shared execution contracts and Claude Code runtime |
| [Request proxy and hints](request-proxy-and-hints.md) | Transparent request forwarding and immutable request facts |
| [Artifacts and evaluation](artifacts-and-evaluation.md) | Evidence normalization, artifact schemas, comparison, and correctness handoff |

```text
BenchKit -> agent runtime -> Request Proxy -> vLLM             (baseline)
                                      \-----> Router -> vLLM    (candidate)
```

Dependencies flow from orchestration to adapters and evidence modules. Production serving code does not import BenchKit.

### BENCH-INV-001: Production runtime does not depend on BenchKit

**Rule:** Production serving modules MUST NOT import `agentinfer.agentbench`.

**Rationale:** Benchmark orchestration is not a serving dependency.

**Enforced by:** package-boundary review and `tests/agentbench/test_target_contracts.py` once the complete target is
stacked.

**Approved alternative:** Move a genuinely shared contract to a source-owned neutral module.

## Status

This document remains draft until PR-06/07 and PR-08/09 merge and owners review the routed paths. Historical results and
superseded hint/midlayer designs in issue #8 are context, not current contracts.
