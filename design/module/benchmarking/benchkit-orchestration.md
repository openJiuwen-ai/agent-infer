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
  - agentinfer/agentbench/replay/**
  - agentinfer/agentcache/entrypoints/bench.py
depends_on:
  - index.md
  - agent-runtime-adapters.md
  - request-proxy-and-hints.md
  - artifacts-and-evaluation.md
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/8
  - https://github.com/JiusiServe/AgentInfer/issues/27
validation_paths:
  - tests/agentbench/test_config.py
  - tests/agentbench/test_dataset.py
  - tests/agentbench/test_cli.py
  - tests/agentbench/test_runner_lifecycle.py
  - tests/agentbench/test_benchmark_smoke.py
upstream_refs:
  - vLLM 0.23.0 service endpoints
last_reviewed: 2026-09-03
---

## BenchKit orchestration

## Boundary

BenchKit loads strict configuration, selects deterministic tasks, prepares workspaces, manages one run and its workers,
controls the Request Proxy lifecycle, collects evidence, and finalizes artifacts. It does not implement agent runtimes,
serving policy, protocol conversion, or physical KV ownership.

`backend.base_url` selects the proxy upstream: set it directly to vLLM or to a transparent Router. In Router-assisted
runs, `backend.metrics_url` remains the direct vLLM Prometheus endpoint. The CLI exposes `prepare`, `replay`, `run`,
`summarize`, and `compare`, delegating work to the owning dataset, Replay, runner, and comparison APIs.

## Lifecycle

```text
load config -> select tasks -> create run -> preflight/capture -> start proxy
-> workers(agent) -> close proxy -> capture -> aggregate -> finalize
```

Direct AgentCache and Router-assisted runs use the same BenchKit lifecycle. Router-assisted mode changes only the
proxy's request upstream; BenchKit does not call Router control APIs, create Router sessions, or write Router-control
evidence.

## Trace Replay boundary

Replay consumes a historical `requests.jsonl`, freezes source analysis and a deterministic structural plan, calibrates
privacy-safe synthetic Prompts to recorded token targets, executes the dependency graph, and finalizes Replay-specific
evidence. The recommended reproducibility workflow repeats that plan against two independently cold-started vLLM
services with zero Prefix Cache counters at each start.

Replay does not launch Claude Code or claim semantic equivalence with the source workflow. Missing original Prompt
plaintext, complete Tool schemas, structured Tool Use/Result exchanges, natural stop behavior, and task correctness
remain outside the current Replay boundary, so Replay and actual Claude Code metrics may differ.

### BENCH-INV-002: CLI handlers remain thin

**Rule:** CLI handlers MUST delegate orchestration, summarization, and comparison to their owning modules.

**Enforced by:** `tests/agentbench/test_cli.py` delegation tests.

**Approved alternative:** Extend the owning API before adding logic to the CLI.

### BENCH-INV-003: Configuration precedence is deterministic

**Rule:** Configuration MUST apply defaults, then YAML, then explicit CLI overrides, followed by complete-model
validation.

**Enforced by:** `tests/agentbench/test_config.py` and `tests/agentbench/test_cli.py`.

**Approved alternative:** Add schema metadata to expose another override; do not create a second option registry.

### BENCH-INV-004: Router integration remains transparent

**Rule:** Router-assisted mode MUST select Router through `backend.base_url`; BenchKit MUST NOT invoke Router control
APIs, create task sessions, or synthesize Router metrics.

**Enforced by:** `tests/agentbench/test_config.py`, `tests/agentbench/test_cli.py`, and
`tests/agentbench/test_runner_lifecycle.py`.

**Approved alternative:** Collect Router-specific evidence with AgentRouter-owned tooling outside BenchKit.
