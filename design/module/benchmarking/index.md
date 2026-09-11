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
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/8
  - https://github.com/JiusiServe/AgentInfer/issues/27
validation_paths:
  - tests/agentbench/**
upstream_refs:
  - vLLM 0.23.0 Anthropic and OpenAI-compatible APIs
last_reviewed: 2026-09-03
---

## Benchmark subsystem

## Purpose

BenchKit runs agent workloads through a transparent request-observation path and records execution, request, cache,
service, and reproducibility evidence. It also supports Trace Replay analysis and execution from historical request
traces. External SWE-bench evaluation owns correctness.

The benchmark supports two candidate transport modes. Both can use any supported agent runtime and the same vLLM
backend configuration.

```text
Direct AgentCache:  Agent -> Request Proxy -> AgentCache-enabled vLLM
Router-assisted:    Agent -> Request Proxy -> Router -> vLLM
```

Direct AgentCache mode sets `backend.base_url` to vLLM. Router-assisted mode sets it to Router, while
`backend.metrics_url` remains the direct vLLM Prometheus endpoint. Router is a transparent data-plane upstream:
BenchKit has no Router control-plane or task-session protocol. For Router-assisted JiuwenSwarm/OpenAI-compatible
runs, Router can opt into `agent_hint_affinity` to use an already-present `agent_hint.session_id` for cache-aware
data-parallel affinity; this remains Router data-plane behavior. A direct upstream-vLLM run remains the normal
baseline for either candidate mode.

Trace Replay is the privacy-safe reconstruction path for historical `requests.jsonl` artifacts. It recovers replayable
Session/Actor topology, token targets, dependencies, historical timing, context evolution, and synthetic shared-prefix
shape without recovering Prompt plaintext. The current validation contract emphasizes two independently cold-started
Replay runs of the same trace and compares their finalized metrics for reproducibility.

Trace Replay does not rerun Claude Code and does not currently reproduce complete Tool schemas,
`tool_use`/`tool_result` blocks, natural stopping behavior, semantic task execution, or correctness. Its metrics may
remain different from an actual Claude Code run; Replay evidence measures reconstructed-workload fidelity rather than
exact Claude execution equivalence.

## Module map

| Document | Primary owner |
| --- | --- |
| [BenchKit orchestration](benchkit-orchestration.md) | Configuration, datasets, workspaces, run lifecycle, and transparent Router upstream selection |
| [Agent runtime adapters](agent-runtime-adapters.md) | Shared contracts plus Claude Code and JiuwenSwarm execution |
| [Request proxy and hints](request-proxy-and-hints.md) | Runtime-selected transparent forwarding and immutable request facts |
| [Artifacts and evaluation](artifacts-and-evaluation.md) | Artifact ownership, evidence normalization, comparison, and correctness handoff |
| [Benchmark quickstart](../../../docs/en/how-to/run-benchmark.md) | Runnable direct and Router-assisted recipes |
| [Metrics guide](../../../docs/en/reference/run-artifacts.md) | `summary.json` field semantics and evidence needed to recompute a value |

Dependencies flow from orchestration to adapters and evidence modules. Production serving code does not import
BenchKit.

### BENCH-INV-001: Production runtime does not depend on BenchKit

**Rule:** Production serving modules MUST NOT import `agentinfer.agentbench`.

**Rationale:** Benchmark orchestration is not a serving dependency.

**Enforced by:** `tests/agentbench/test_module_contracts.py`.

**Approved alternative:** Move a genuinely shared contract to a source-owned neutral module.

## Status

This document describes the current benchmark contracts. Historical hint and midlayer proposals remain context rather
than implementation contracts.
