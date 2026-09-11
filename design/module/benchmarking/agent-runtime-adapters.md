---
title: Benchmark agent runtime adapters
kind: module
status: draft
owners:
  - TBD
primary_code_paths:
  - agentinfer/agentbench/agents/**
related_code_paths:
  - tests/agentbench/agents/**
depends_on:
  - index.md
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/8
  - https://github.com/JiusiServe/AgentInfer/issues/27
validation_paths:
  - tests/agentbench/agents/**
upstream_refs:
  - Claude Code CLI
  - JiuwenSwarm CLI
last_reviewed: 2026-08-20
---

## Benchmark agent runtime adapters

## Purpose and boundary

The agent layer receives an `AgentRunRequest`, executes one task-owned runtime, and returns an `AgentRunResult`. It owns
runtime process control and runtime-specific evidence, not Router control, request aggregation, benchmark comparison,
or SWE-bench correctness.

Claude Code, JiuwenSwarm, and DeepSeek Harness are supported runtimes, dispatched through the explicit `AgentRuntime`
registry. Each runtime owns its profiles, preflight check, required endpoint, and task execution:

| Runtime | Required endpoint | Task-process isolation |
| --- | --- | --- |
| Claude Code | `POST /v1/messages` | One dedicated tmux server per task, including terminal capture and transcript handling. |
| JiuwenSwarm | `POST /v1/chat/completions` | Gateway, AgentServer, and CLI processes launched directly for the task; tmux is not used. |
| DeepSeek Harness | `POST /v1/chat/completions` | One headless process, task-private `DSH_HOME`, and DSH-native `workspace-write` filesystem isolation; durable session logs provide post-run topology evidence. |

Each task receives a dedicated workspace, artifact directory, and session identity. Credentials enter through the launch
environment and are excluded from settings, command artifacts, and logs. Completion means the runtime execution
contract completed; it does not mean the generated patch is correct.

### BENCH-INV-005: Runtime evidence is secret-safe

**Rule:** Agent artifacts MUST NOT contain configured credentials or secret environment values.

**Rationale:** Benchmark artifacts are intended for review and reproduction.

**Enforced by:** `tests/agentbench/agents/claude/test_settings.py`, `test_artifacts.py`, and `test_runner.py`.

**Approved alternative:** Record only the environment-key policy or redacted presence metadata.

## Extension guide

A new runtime must implement the existing request/result boundary and prove process cleanup, timeout/cancellation
semantics, artifact safety, endpoint validation, and patch export. Runtime-specific prompts and permissions remain
inside that adapter; BenchKit must not branch on runtime internals.
