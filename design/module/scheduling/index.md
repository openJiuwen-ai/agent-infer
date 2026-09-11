---
title: Scheduling subsystem
kind: module
status: normative
owners:
  - warriorsniu
primary_code_paths:
  - agentinfer/scheduling/**
related_code_paths:
  - agentinfer/agentcache/core/**
depends_on: []
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/20
  - https://github.com/JiusiServe/AgentInfer/issues/27
validation_paths:
  - tests/scheduling/**
  - tests/agentcache/core/**
upstream_refs:
  - vLLM V1 Scheduler
  - vLLM V1 AsyncScheduler
last_reviewed: 2026-08-06
---

## Scheduling subsystem

## Purpose

The scheduling subsystem treats one continuous agent context as a Program and controls when its requests may enter a
host inference engine. It owns protocol-neutral identity, Program lifecycle, request retention, typed scheduling
decisions, and the Progress-TTL policy. A host adapter supplies normalized request metadata and backend facts, then
transfers admitted requests into its native queue.

The subsystem does not replace token-level scheduling, execute model work, retry failed requests, perform cross-backend
load balancing, or parse endpoint-specific response bodies.

## Module map

| Document | Primary responsibility |
| --- | --- |
| [Program identity](program-identity.md) | Maps requests from canonical fields, supported framework hints, and headers to stable Program identity and protocol-neutral lifecycle facts. |
| [Program state machine](program-state-machine.md) | Defines runtime-owned Program fields, request retention, status/state transitions, snapshots, events, and update points. |
| [Progress-TTL scheduling](progress-ttl-scheduling.md) | Defines segment continuity, admission/resume capacity, pause ordering, adaptive acting TTL, privilege, and cleanup policy. |
| [vLLM runtime integration](vllm-runtime-integration.md) | Adapts vLLM V1 requests, scheduler hooks, DP-rank capacity, API middleware, lifecycle transport, configuration, and startup. |

Read the documents in table order. Identity defines which requests belong together; the state machine defines the
policy-neutral lifecycle of that Program; Progress-TTL defines how transitions are selected; the vLLM integration shows
how those contracts are hosted without replacing native token scheduling.

## Architecture

```text
framework request
  -> host identity adapter
  -> AgentIdentity(program_id, task_id, relationships)
  -> ProgramScheduler + RequestPool
  -> Progress-TTL transition requests
  -> host runtime adapter
  -> native inference-engine queue and token scheduler
```

| Owner | Owned data and behavior |
| --- | --- |
| Host/API adapter | Wire protocol, framework headers, response semantics, native request objects, backend observation, and serialization. |
| `agentinfer.scheduling.identity` | Canonical `AgentIdentity` derivation and validation. |
| `ProgramScheduler` | Request-to-Program bindings, RequestPool, live Program records, snapshots, transition validation, and event dispatch. |
| Scheduling strategy | Policy-owned factors, ordering, capacity formulas, deadlines, and typed transition requests. |
| Native inference engine | Native waiting/running queues, batching, token budget, KV allocation, preemption, model execution, and outputs. |

## Stable boundaries

### SCHED-INV-001: Requests and Programs are different identities

**Rule:** Every request attempt has a unique `request_id`; requests that continue the same agent context map to one
stable `program_id`.

**Rationale:** Request completion must not erase continuity, cache value, or scheduling history that survives into the
next agent round.

**Enforced by:** identity tests, generation-safe request bindings, and duplicate-request validation.

### SCHED-INV-002: Status and forwarding state remain independent

**Rule:** `REASONING`/`ACTING` describe model demand, while `ACTIVE`/`PAUSED` describe scheduler admission. Neither
dimension is derived from the other.

**Rationale:** A reasoning request may be retained, native-waiting, or executing, and an acting Program may remain
active to protect its KV cache.

**Enforced by:** runtime transition preconditions and state-machine contract tests.

### SCHED-INV-003: Strategies do not mutate runtime records

**Rule:** A strategy reads immutable snapshots and requests typed transitions against exact `(program_id, generation)`
references.

**Rationale:** Runtime ownership centralizes lifecycle validation and prevents stale callbacks or same-cycle decisions
from corrupting a replacement Program generation.

**Enforced by:** `TransitionController`, `TransitionResult`, snapshot advancement, and strategy contract tests.

### SCHED-INV-004: Request ownership is singular

**Rule:** A request is retained by RequestPool or owned by the host-native queue, never both, and admission transfers it
at most once.

**Rationale:** Duplicate ownership would duplicate execution, cancellation, unfinished counts, and queue-time
accounting.

**Enforced by:** one-shot recent-admit storage and vLLM bridge tests.

### SCHED-INV-005: Host-specific behavior stays in adapters

**Rule:** Protocol parsing, vLLM request fields, DP-rank discovery, lifecycle sockets, and native cache APIs must not
enter the host-neutral scheduling package.

**Rationale:** The Program model and strategy must remain reusable by routers and other inference engines.

**Enforced by:** package-boundary review and focused adapter tests.

## Current scope and limitations

The main-branch Progress-TTL implementation is validated as a rank-local HBM scheduling policy embedded in vLLM V1.
Generic contracts can describe DRAM, SSD, and remote cache capacity, but tier-aware residency prediction, reload-cost
estimation, and offloading-aware admission remain future work. Cross-DP load balancing and cache-affinity routing are
also outside this module; an upstream router selects the rank and each EngineCore schedules its own queue.

## Status

This document and its four child module documents are normative for the merged scheduling contracts and current
implementation. Issue #20 remains the decision and enhancement tracker; later proposals do not make the reviewed current
contract draft again. Repository-level discoverability links may evolve independently of this module status.

## Validation

Run the host-neutral and vLLM integration suites with:

```bash
pytest -q tests/scheduling tests/agentcache/core
```

Performance-sensitive changes additionally require matched native-vLLM and AgentInfer experiments with the same model,
task list, concurrency, sampling settings, hardware, and backend startup options.
