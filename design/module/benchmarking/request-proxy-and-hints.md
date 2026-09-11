---
title: Benchmark request proxy and hints
kind: module
status: draft
owners:
  - TBD
primary_code_paths:
  - agentinfer/agentbench/request_proxy/**
related_code_paths:
  - agentinfer/scheduling/headers.py
  - tests/agentbench/request_proxy/**
depends_on:
  - index.md
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/8
  - https://github.com/JiusiServe/AgentInfer/issues/27
validation_paths:
  - tests/agentbench/request_proxy/**
  - tests/agentbench/test_headers.py
upstream_refs:
  - vLLM 0.23.0 Anthropic and OpenAI-compatible APIs
last_reviewed: 2026-08-25
---

## Benchmark request proxy and hints

## Current boundary

The Request Proxy is a benchmark-owned observation boundary. It transparently forwards the endpoint selected by the
active runtime, relays raw response bytes, records immutable `RequestFact` rows, and exposes loopback-only health and
authenticated shutdown endpoints.

```text
Direct AgentCache:  Claude Code or JiuwenSwarm -> Request Proxy -> vLLM
Router-assisted:    Claude Code or JiuwenSwarm -> Request Proxy -> Router -> vLLM
```

Claude Code uses `POST /v1/messages`; JiuwenSwarm uses `POST /v1/chat/completions`. Selecting the endpoint is
configuration validation against the active runtime, not protocol translation by the proxy.

The proxy owns its HTTP client, in-flight admission, subprocess lifecycle, trace queue/file, and trace health.
BenchKit owns neither its server internals nor request-level metric formulas. Trace rows are immutable; adapters must attach
any scheduling identity before forwarding so the proxy can observe it while requests are in flight.

## Hint status

The current proxy does not construct or inject hints, `agentic_context`, Programs, scheduling policy, or protocol
transformations. For Router-assisted OpenAI-compatible requests, Router can opt into its
`agent_hint_affinity` middleware, which copies an already-present `agent_hint.session_id` into its routing session
parameter when the request does not set one. That Router-side operation does not change the proxy's transparent relay
and does not apply to Claude Code's `/v1/messages`. Router-owned serving metadata is outside this benchmark module.
Historical hint and midlayer designs are not current implementation contracts.

### BENCH-INV-006: Proxy forwarding remains transparent

**Rule:** The proxy MUST preserve request semantics and relayed response status, headers, body, and stream bytes while
observation failures remain isolated from forwarding where specified.

**Enforced by:** `tests/agentbench/request_proxy/test_server.py`.

**Approved alternative:** Protocol adaptation belongs at the serving protocol boundary, not in this proxy.

### BENCH-INV-007: The control surface remains local and bounded

**Rule:** The proxy MUST bind to loopback, authenticate shutdown, stop admission, wait boundedly for in-flight requests,
and report trace/process health.

**Enforced by:** `tests/agentbench/test_config.py`, `tests/agentbench/request_proxy/test_lifecycle.py`, and
`test_server.py`.

**Approved alternative:** A separately reviewed deployment boundary with equivalent authentication and lifecycle
evidence.
