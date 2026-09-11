---
title: Agent runtime execution isolation
kind: module
status: draft
owners:
  - TBD
primary_code_paths:
  - agentinfer/agentbench/agents/claude/**
  - agentinfer/agentbench/agents/jiuwenswarm/**
  - agentinfer/agentbench/agents/dsh/**
related_code_paths:
  - agentinfer/agentbench/agents/preflight.py
  - tests/agentbench/test_isolation_*.py
  - tests/agentbench/agents/**
depends_on:
  - agent-runtime-adapters.md
decision_refs:
  - https://github.com/JiusiServe/AgentInfer/issues/8
  - https://github.com/JiusiServe/AgentInfer/issues/27
validation_paths:
  - tests/agentbench/test_isolation_config.py
  - tests/agentbench/test_isolation_preflight.py
  - tests/agentbench/test_isolation_boundary.py
  - tests/agentbench/agents/dsh/test_dsh_sandbox_integration.py
upstream_refs:
  - Claude Code sandbox and PreToolUse hooks
  - JiuwenSwarm SysOperation SANDBOX
  - jiuwenbox
  - DeepSeek Harness workspace-write sandbox (Bubblewrap or Landlock)
last_reviewed: 2026-09-03
---

## Agent runtime execution isolation

## Purpose and boundary

Every AgentBench task executes in a dedicated workspace. Filesystem isolation is a mandatory runtime invariant rather
than a BenchKit option: BenchKit keeps the shared request and orchestration contracts unchanged, while each agent adapter
configures and verifies its own isolation mechanism. Missing isolation prerequisites fail preflight or runtime startup;
the adapters never fall back to unsandboxed host execution.

The boundary protects benchmark artifacts and neighboring tasks from agent writes outside the task workspace. It is not
a general-purpose hostile-code sandbox and does not provide network isolation.

## Runtime design

| Runtime | Isolation mechanism | Writable host paths |
| --- | --- | --- |
| Claude Code | Native Bash sandbox plus a `PreToolUse` path fence for `Write`, `Edit`, `MultiEdit`, and `NotebookEdit` | Task workspace |
| JiuwenSwarm | Task-local external `jiuwenbox-server` used by `SysOperation SANDBOX`; `mcp_exec_command` is disabled because it bypasses `SysOperation` | Task workspace, task-local Jiuwen home and data, and the task-local `/tmp` backing directory |
| DeepSeek Harness | DSH-native `workspace-write` sandbox; the adapter loads the installed DSH `LocalSandboxProvider`, runs a confined no-op preflight, and fails closed when no backend works | Task workspace; `/tmp` is private under Bubblewrap but may be host-writable under Landlock |

Claude enables `failIfUnavailable` and disables unsandboxed Bash commands. The path fence resolves canonical paths before
allowing file tools, so absolute paths, parent traversal, and symlinks cannot escape the workspace.

Jiuwen uses `network.mode: host` because network namespaces require privileges outside the benchmark contract. Its
sandbox has a private `/tmp`, read-only runtime mounts, and explicit read-write binds only for task-owned paths. The
adapter owns the task-local jiuwenbox server lifecycle and removes it at task shutdown.

DSH owns its backend selection and path confinement: both profiles select `workspace-write`, and the adapter verifies
the installed composition functionally before task dispatch rather than duplicating DSH's path handling. The Linux
integration test drives a real headless process through workspace-write, relative, absolute, and symlink escape canaries.

### BENCH-INV-006: Agent filesystem isolation is mandatory

**Rule:** Every supported agent runtime MUST confine filesystem writes to task-owned paths and MUST fail closed when its
isolation mechanism is unavailable.

**Rationale:** A working directory alone does not prevent absolute, parent-directory, or symlink writes. Such writes are
not represented by `model.patch` and can pollute other benchmark tasks.

**Enforced by:** runtime preflight checks, Claude sandbox/fence configuration, Jiuwen sandbox policy and command sealing,
DSH's functional sandbox probe and `workspace-write` profile policy, plus the isolation unit and live-boundary tests
listed in `validation_paths`.

**Approved alternative:** None at the BenchKit layer. A runtime may replace its internal mechanism only if it preserves
the same fail-closed filesystem boundary.

## Runtime prerequisites

This change adds no Python package dependency to `pyproject.toml`. It does require the following host/runtime tools:

| Runtime | Required prerequisite | Why |
| --- | --- | --- |
| Claude Code | `bwrap` (Bubblewrap) | Linux filesystem sandbox used by Claude's native Bash sandbox |
| Claude Code | `socat` | Network proxy helper required by Claude's sandbox initialization |
| JiuwenSwarm | `jiuwenbox-server` | External server backing Jiuwen `SysOperation SANDBOX` |
| JiuwenSwarm | `bwrap` (Bubblewrap) | Linux filesystem sandbox used by `jiuwenbox-server` |
| DeepSeek Harness | `dsh` 0.1.1-rc.2 with Node >= 22 | Installed CLI providing the `workspace-write` sandbox provider and its backend selection |

AgentBench checks for each runtime's required executables during preflight. Failure to initialize the selected runtime's
sandbox is a hard startup failure. Windows and hosts without these runtime prerequisites cannot execute AgentBench agent
workloads under this contract.
