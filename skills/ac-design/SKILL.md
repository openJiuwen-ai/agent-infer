---
name: ac-design
description: Use when analyzing cache patterns in coding-agent workloads or writing RFC/design docs for AgentInfer optimization strategies; produces measurable success criteria and PR evidence checklists.
allowed-tools:
  - Bash
  - Read
  - Write
  - Grep
  - Glob
  - WebFetch
---

# ac-design

Analyze agentic workload cache patterns and write a structured RFC/design document
for an AgentInfer cache or scheduling optimization strategy.

## WHEN TO INVOKE

- The user asks to "analyze cache patterns", "design a cache optimization",
  "write a design doc", "write an RFC", or "analyze tool-call timing" for
  AgentInfer.
- Starting work on single-agent tool-call optimization or subagent concurrency optimization.
- The user asks to evaluate existing benchmark data to inform a cache strategy.

Do NOT invoke for: implementing already-designed features, writing a PR description
only, or non-AgentInfer design work.

## STEPS

1. **Gather current evidence**

   Read the current repo state and any user-provided traces, benchmark artifacts,
   RFCs, or PR comments before relying on older numbers. If using historical
   measurements, label them as historical and cite their source.

2. **Scope the optimization**

   Determine which scenario the design addresses:

   - **Single-agent**: consecutive LLM requests interleaved with tool calls made
     by one agent. Focus: tool-aware KV retention, prefix-aware scheduling, or
     predictor-guided eviction.
   - **Subagent concurrency**: multiple agents spawned by a parent. Focus:
     task/agent lifecycle tracking, value/cost scoring, KV lifecycle management,
     and isolation.
   - **Both**: design for both scenarios with a shared metadata schema.

3. **Write the RFC/design contract**

   Use `.github/ISSUE_TEMPLATE/750-RFC.yml` as a guide. Include:

   - **Motivation**: the problem, user need, or roadmap goal.
   - **Proposed change**: architecture overview, components, and data flow.
   - **Alternatives and tradeoffs**: simpler options and rejected designs.
   - **Configuration**: define typed/config-file and CLI knobs using verified
     current repo paths; distinguish runtime scheduler config from benchmark
     workload config.
   - **Metrics and success criteria**: measurable targets and verification steps.
   - **Correctness guardrails**: pass-rate, prefix alignment, subagent isolation,
     tool-call integrity, or other relevant safety constraints.
   - **Phased roadmap**: incremental, mergeable milestones.

4. **Produce a PR acceptance checklist**

   Convert RFC success criteria into reviewer-facing PR evidence requirements:

   ```md
   ## PR acceptance checklist

   | RFC criterion | Required PR evidence | Test / Artifact |
   | --- | --- | --- |
   | | | |
   ```

   The checklist should make clear which evidence comes from tests, benchmark
   artifacts, logs, config files, or explicit out-of-scope deferrals.

5. **Verify current paths before citing them**

   Before naming benchmark/config/script paths, inventory the current repo with
   read-only search. Prefer verified paths or generic patterns over stale path
   examples.

## DON'T

- Don't copy code or design from CacheWise, Mooncake, or other projects without
  license clearance; use them as inspiration, not source material.
- Don't define success criteria without measurable, verifiable evidence.
- Don't make E2E benchmark CI the acceptance source; AgentInfer relies on
  reviewer-readable PR evidence and artifacts for now.
- Don't design a monolithic solution; prefer incremental, mergeable phases.
- Don't freeze benchmark host, environment, or path facts in repo-level agent
  guidance when they belong in artifacts or handoff docs.

## AFTER

Once the design is approved, open or link the implementation PR and invoke
`ac-review` for quick or full readiness checks.
