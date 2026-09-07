---
title: BenchKit orchestration
kind: module
status: draft
owners:
  - TBD
primary_code_paths:
  - agentinfer/agentbench/benchkit/**
related_code_paths:
  - agentinfer/agentbench/configs/**
  - agentinfer/agentcache/entrypoints/bench.py
depends_on:
  - index.md
  - agent-runtime-adapters.md
  - request-proxy-and-hints.md
  - artifacts-and-evaluation.md
validation_paths:
  - tests/agentbench/test_config.py
  - tests/agentbench/test_dataset.py
  - tests/agentbench/test_cli.py
  - tests/agentbench/test_runner_lifecycle.py
upstream_refs:
  - vLLM 0.23.0 service endpoints
last_reviewed: 2026-07-21
---

## BenchKit orchestration

## Boundary

BenchKit loads strict configuration, selects deterministic tasks, prepares workspaces, manages one run and its workers,
controls the Request Proxy lifecycle, performs candidate-only Router registration and cleanup, collects evidence, and
finalizes artifacts. It does not implement agent runtimes, serving policy, protocol conversion, or physical KV
ownership.

The CLI exposes `prepare`, `run`, `summarize`, and `compare`. Handlers delegate to dataset, runner, and comparison APIs
rather than duplicating their algorithms.

## Lifecycle

```text
load config -> select tasks -> create run -> preflight/capture
-> start proxy -> workers(register candidate -> agent -> cleanup)
-> close proxy -> capture -> aggregate -> finalize
```

Baseline runs target `backend.base_url` and never call Router control APIs. Candidate runs target `router.base_url`;
registration and cleanup remain separate evidence from the agent result.

### BENCH-INV-002: CLI handlers remain thin

**Rule:** CLI handlers MUST delegate orchestration, summarization, and comparison to their owning modules.

**Enforced by:** `tests/agentbench/test_cli.py` delegation tests.

**Approved alternative:** Extend the owning API before adding logic to the CLI.

### BENCH-INV-003: Configuration precedence is deterministic

**Rule:** Configuration MUST apply defaults, then YAML, then explicit CLI overrides, followed by complete-model
validation.

**Enforced by:** `tests/agentbench/test_config.py` and `tests/agentbench/test_cli.py`.

**Approved alternative:** Add schema metadata to expose another override; do not create a second option registry.

### BENCH-INV-004: Router control is candidate-only

**Rule:** Baseline runs MUST NOT register or clean up Router sessions. Candidate cleanup MUST be attempted after
registration even when agent execution fails or is cancelled.

**Enforced by:** `tests/agentbench/test_runner_lifecycle.py` after PR-06/07 integration.

**Approved alternative:** None within the baseline/candidate comparison contract.
