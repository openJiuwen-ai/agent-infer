# Changelog

This file records notable changes to AgentInfer. When a version is released, move its entries from `Unreleased` into
a versioned section.

## Unreleased

### Added

- Added the `vllm serve MODEL --agentinfer` single-flag serving path: the delegating `vllm` console script takes over
  serve parsing, injects the agent-aware scheduler (async or sync, following `--async-scheduling`/`--no-async-scheduling`),
  the identity and lifecycle middleware, and the embedded Progress-TTL controller, then dispatches upstream vLLM. The
  lifecycle socket defaults when unset and conflicting `--scheduler-cls` options exit with code 2. Explicit long-form
  commands remain supported. Verified end-to-end on vLLM 0.29 with a
  GPU e2e test (`tests/agentcache/entrypoints/test_serve_flag_e2e.py`); the serve parser factory import, scheduler
  `schedule()` pass-through, and prefix-lookup observer tolerate both current and older vLLM layouts.
- Added the AgentRouter native middleware patch for `agent_hint_affinity` / `agent_hint_token_offsets` under `agentinfer/agentrouter/patches` (apply onto upstream Router; do not vendor the full router tree).
- Added mirrored Chinese and English documentation organized with the Diátaxis framework.

### Changed

- Inferact replay counts complete conversations with the configured chat template and its parameters, using a
 validated local tokenizer when available. Current-turn calibration preserves historical messages and requires exact
 input lengths; clipping checks prefixes from longest to shortest to handle nonmonotone token counts.
- Converted Trace IR manifests identify the converter by name without a separate converter version label.
- Inferact rejects nonzero calibration tolerance instead of silently normalizing it. Local incremental counting is
 limited to probed message counts, and tokenizer discovery errors close the HTTP client before propagating.
- Reorganized the root README into project overview, core features, related documentation, requirements, installation,
  Quick Start, and license sections.

### Removed

- Removed the legacy `AgentAwareScheduler`/`AgentAwareQueue` compatibility surfaces, the import-time `EngineArgs`
  patch that defaulted `scheduler_cls`, and the `agentinfer.LLM` passthrough subclass. Agent-aware serving is enabled
  explicitly through `vllm serve MODEL --agentinfer` or the long-form scheduler and middleware options; other commands
  delegate to upstream vLLM unchanged.
- Refreshed release-package installation and made the explicit agent-aware scheduler, middleware, and Progress-TTL
 controller the primary Quick Start path, based on PR #62.

## 0.1.0 — Initial Release

**AgentInfer** provides efficient KV-cache management for multi-agent LLM serving on vLLM. It replaces vLLM's
default FCFS scheduler with an agent-aware scheduling policy that optimizes GPU KV-cache utilization when serving
concurrent multi-turn agent workloads.

### Core Subsystems

#### AgentCache

Production runtime plugin for vLLM:

- `AgentAwareScheduler` as a drop-in replacement for vLLM's default scheduler.
- ASGI middleware for agent identity resolution across OpenAI Chat and Anthropic Messages, and for lifecycle signal
 observation of continue versus terminal states.
- Admission retention bridges that intercept requests before native vLLM admission.
- Unix datagram signaling between API and EngineCore layers.

#### Progress-TTL Scheduling Policy

The concrete scheduling strategy:

- Guarantees short consecutive service segments per program.
- Retains acting programs while the avoided cold-prefill cost justifies it through an adaptive TTL.
- Resumes paused reasoning programs before idle acting work.
- Repairs capacity by segment-tenure priority.
- Hands off privilege between related programs, including parent and child task relationships.
- Supports three modes: `ON`, `OFF`, and `AUTO`, with automatic enablement based on continuity utility.
- Maintains rolling statistics for adaptive TTL estimation.
- Uses a host-neutral design with no vLLM dependency in the scheduling package.

#### AgentBench

Benchmarking harness for SWE-bench and Claude Code workloads:

- CLI with `prepare`, `run`, `summarize`, and `compare` subcommands.
- Transparent Anthropic request proxy with JSONL request tracing.
- Concurrent agent lifecycle management through tmux.
- Baseline-versus-candidate comparison with throughput, latency, and cache-hit metrics.
