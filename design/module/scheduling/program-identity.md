---
title: Program identity
kind: module
status: normative
owners:
  - warriorsniu
primary_code_paths:
  - agentinfer/scheduling/identity.py
  - agentinfer/scheduling/lifecycle.py
related_code_paths:
  - agentinfer/agentcache/core/api_adapter.py
  - agentinfer/agentcache/core/scheduler.py
depends_on:
  - index.md
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/20
validation_paths:
  - tests/scheduling/test_identity.py
  - tests/agentcache/core/test_vllm_api_adapter.py
upstream_refs:
  - vLLM OpenAI-compatible request extensions
  - vLLM Anthropic Messages API
last_reviewed: 2026-08-06
---

## Program identity

## Purpose and boundary

Program identity defines how a host adapter maps agent requests to one protocol-neutral `AgentIdentity`. Identity
normalization belongs to `agentinfer.scheduling.identity`; API and engine adapters call this shared parser instead of
maintaining their own field mappings.

This contract does not perform HTTP I/O, mutate a Program, infer scheduling state, or decide admission.
Endpoint-specific response parsing is also separate: API adapters reduce it to the closed `ProgramLifecycle` vocabulary
defined below.

## Agent, Program, and request

An **agent** is an application-level actor that repeatedly alternates between model calls and local actions such as tool
calls. AgentInfer treats one continuous agent context as a **Program**: model requests that continue the same agent
context share one `program_id`, even though every request attempt has a different `request_id`.

```text
agent context / Program P
  request r1 -> tool call -> request r2 -> tool call -> request r3
```

`request_id` identifies one transport or inference attempt. `program_id` identifies the longer-lived scheduling unit
whose context, state, progress, and cache value survive across those attempts. A task or session may contain several
agents; when their identities are available, each agent becomes a separate Program under the same `task_id`. When only a
session identifier is available, all requests in that session are conservatively grouped into one lead Program.

Adapters must derive the same `program_id` for every request that continues the same agent context. A client retry
remains a new request attempt and therefore needs a new `request_id`, but it still maps to the same Program when it
continues that context.

## Canonical identity

`AgentIdentity` is the only identity object accepted by Program scheduling. The runtime binds the host request ID to the
current generation of `program_id` at arrival; later stream, completion, cancellation, and lifecycle callbacks resolve
through that exact binding.

| Field | Required | Meaning |
| --- | --- | --- |
| `program_id` | Yes | Stable serving-side identifier for one continuous agent context. |
| `task_id` | No | Grouping key for related Programs. It is not a request ID or queue ID. |
| `session_id` | No | Original upstream session identifier. |
| `agent_id` | No | Agent identifier within a task or session. |
| `parent_program_id` | No | Stable parent Program identifier. |
| `blocks_parent` | Yes | Whether this Program prevents its parent from progressing. Defaults depend on the input contract. |
| `expected_resume` | Yes | Whether another model request is expected after the current round. |
| `agent_role` | No | Explicit or inferred role such as `lead` or `subagent`. |
| `spawn_reason` | No | Open-ended framework hint describing why or how the Program was created. |
| `request_id` | No | Upstream correlation identifier; it never participates in Program identity. |

Identifiers must be non-empty after trimming. `parent_program_id` must differ from `program_id`, and
`blocks_parent=true` requires a parent.

The recommended request representation is:

```json
{
  "vllm_xargs": {
    "agentic_context": {
      "program_id": "task-42:researcher",
      "task_id": "task-42",
      "session_id": "session-42",
      "agent_id": "researcher",
      "parent_program_id": "task-42:lead",
      "blocks_parent": false,
      "expected_resume": true,
      "agent_role": "subagent"
    }
  }
}
```

`agentic_context` may be an object or a JSON-encoded object no larger than 16 KiB. New frameworks should produce this
canonical representation rather than add another header schema.

## Input precedence

`parse_agent_identity()` accepts exactly three input groups:

1. `vllm_xargs.agentic_context`;
2. the supported `agent_hint` object;
3. framework headers.

When canonical `agentic_context` exists, its explicit values win and supported headers fill only missing session, agent,
or parent-agent fields. `agent_hint` is considered only when canonical context is absent; a valid `agent_hint` returns a
complete identity without mixing in headers. If neither body contract exists, the parser derives identity from the first
matching header schema.

Malformed metadata in a recognized contract raises `MetadataError`. Absence of every supported identity returns `None`,
which tells an adapter to preserve compatibility traffic on its native path.

## Canonical derivation

When `program_id` is not explicit, the parser derives it from task and agent identity. A session-only request becomes
the `lead` Program for that session. A session plus agent ID creates a distinct Program under the same task.

Body values support these compatibility aliases:

| Canonical field | Accepted aliases |
| --- | --- |
| `parent_program_id` | `parent_id` |
| `parent_agent_id` | Used to derive `parent_program_id` within the task. |
| `blocks_parent` | `blocking_parent` |
| `spawn_reason` | `subagent_type`, `agent_type` |

If `agent_role` is absent, `agent_id=lead` implies `lead`; any other agent ID implies `subagent`. An inferred subagent
defaults to `blocks_parent=true`. `expected_resume` is used exactly when explicitly supplied; otherwise it defaults to
false for a blocking subagent and true for other Programs.

To keep an explicit `program_id` outside any task, send `"task_id": null`. If `task_id` is omitted, an available session
ID remains the default task grouping.

## Framework mappings

Header matching is case-insensitive and chooses one complete schema by its session header. Fields from different schemas
are never combined.

| Framework contract | Session | Agent | Parent | Result |
| --- | --- | --- | --- | --- |
| Claude Code | `x-claude-code-session-id` | `x-claude-code-agent-id` | `x-claude-code-parent-agent-id` | Session is the default task. Missing agent means `lead`; a present agent creates a subagent Program. |
| Codex/OpenCode session-thread form | `session-id` | `thread-id` | unavailable | Session is the task and thread distinguishes Programs. A non-lead thread is inferred as a subagent of the lead. |
| Generic session form | `x-session-id` | unavailable | unavailable | Session becomes one lead Program. |

DeepSeek Harness must supply identity through canonical `vllm_xargs.agentic_context` (or `agent_hint`
where that contract applies); its provider-private headers are not a supported framework mapping.

Stock vLLM trace-header propagation is not part of this contract and should not be extended with framework identity
headers. The vLLM identity middleware normalizes API metadata into `vllm_xargs.agentic_context` before EngineCore
transport.

## `agent_hint` mapping

The supported session-oriented framework may send:

```json
{
  "agent_hint": {
    "session_id": "child-session",
    "parent_session_id": "parent-session",
    "blocks_parent": true,
    "expected_resume": false
  }
}
```

The mapping is deliberately minimal:

| `agent_hint` field | Canonical result |
| --- | --- |
| `session_id` | Required; copied to both `program_id` and `session_id`. |
| `parent_session_id` | Copied to `parent_program_id`. |
| `blocks_parent` | Defaults to true. A true value requires `parent_session_id`. |
| `expected_resume` | Explicit value wins; otherwise defaults to `not blocks_parent`. |
| task grouping | `task_id` remains `None`. |

Because the default is blocking, a root request without `parent_session_id` must explicitly send `blocks_parent=false`.

## Minimal identity feature levels

| Available information | Scheduling result |
| --- | --- |
| No identity | Native compatibility traffic; no Program is created. |
| Session only | Every request in the session belongs to one lead Program. Different agents in that session are indistinguishable. |
| Session and agent ID | Agents become separate Programs under one task, allowing independent continuity, waiting, and capacity accounting. |
| Parent and blocking hints | The registry can represent parent blocking state. These hints describe workflow intent, not proof that either Program is executing. |
| Explicit `expected_resume` | Overrides inferred demand semantics, including the default for a blocking subagent. |

## Lifecycle result contract

`ProgramLifecycle` contains only:

| Value | Meaning |
| --- | --- |
| `CONTINUE` | API parsing indicates that the Program is expected to continue, such as a tool-call response. |
| `TERMINAL` | API parsing proves that the idle Program can be released. |
| `UNKNOWN` | The response does not provide a safe terminal conclusion. |

The scheduling package never parses OpenAI finish reasons, Anthropic stop reasons, HTTP success, or tool-call payloads.
API adapters own those wire-specific mappings and report only this normalized result. `TERMINAL` may release only the
current idle Program and cannot interrupt an unfinished request. Because the optional lifecycle channel carries only
`program_id`, it cannot distinguish a delayed signal for an older generation from an idle replacement with the same ID.

## Normative invariants

- **PI-INV-001:** Equivalent metadata at different host boundaries resolves to the same canonical `AgentIdentity`.
- **PI-INV-002:** `request_id` never substitutes for `program_id`, `task_id`, or a queue identifier.
- **PI-INV-003:** Explicit canonical values take precedence over inferred defaults, and an explicit `expected_resume` is
  never overwritten.
- **PI-INV-004:** Unsupported or absent identity produces `None`; malformed recognized metadata produces
  `MetadataError`.
- **PI-INV-005:** Scheduling policy code branches only on canonical identity and lifecycle fields, never on endpoint or
  header names.
- **PI-INV-006:** A terminal lifecycle fact cannot release a Program with an unfinished request.

## Safe changes and validation

Adding a new framework mapping requires parser tests for precedence, malformed values, default inference, and canonical
encoding. If an API middleware is involved, test both streaming and non-streaming body transport without changing
response bytes.

```bash
pytest -q tests/scheduling/test_identity.py tests/agentcache/core/test_vllm_api_adapter.py
```
