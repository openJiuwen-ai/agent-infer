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
validation_paths:
  - tests/agentbench/agents/**
upstream_refs:
  - Claude Code CLI
last_reviewed: 2026-07-21
---

## Benchmark agent runtime adapters

## Purpose and boundary

The agent layer receives an `AgentRunRequest`, executes one task-owned agent runtime, and returns an `AgentRunResult`.
It owns runtime process control and runtime-specific evidence, not Router control, request aggregation, benchmark
comparison, or SWE-bench correctness.

Claude Code is the only current runtime. Dispatch remains an explicit conditional rather than a plugin registry. The
Claude adapter owns profiles, settings, tmux isolation, terminal interaction, transcript normalization, patch export,
and cleanup.

## Lifecycle and resources

Each task gets a dedicated workspace, artifact directory, session identity, and tmux server. Credentials enter through
the launch environment and are excluded from settings, command artifacts, and logs. Completion means the agent execution
contract completed; it does not mean the generated patch is correct.

### BENCH-INV-005: Runtime evidence is secret-safe

**Rule:** Agent artifacts MUST NOT contain configured credentials or secret environment values.

**Rationale:** Benchmark artifacts are intended for review and reproduction.

**Enforced by:** `tests/agentbench/agents/claude/test_settings.py`, `test_artifacts.py`, and `test_runner.py`.

**Approved alternative:** Record only the environment-key policy or redacted presence metadata.

## Extension guide

A new runtime must implement the existing request/result boundary and prove process cleanup, timeout/cancellation
semantics, artifact safety, and patch export. Runtime-specific prompts and permissions remain inside that adapter;
BenchKit must not branch on runtime internals.
