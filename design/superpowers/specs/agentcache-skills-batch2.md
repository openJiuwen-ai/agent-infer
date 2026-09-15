# AgentCache Skills — Batch 2 Design

**Date:** 2026-07-03
**Status:** Draft
**Scope:** Batch-2 extensions to the AgentCache skill catalog

## Context

The batch-1 skills defined four workflow-stage skills:
`ac-bootstrap`, `ac-benchmark`, `ac-integrate`, and `ac-review`. Since then,
AgentCache work has moved from scaffolding toward benchmark harness integration,
cache-aware design work, and reviewer rules for architecture-sensitive code.

This batch records the additions made after the first batch while preserving the
batch-1 spec as a dated historical snapshot.

## Goals

- Add a design-stage skill for cache optimization proposals.
- Extend `ac-review` with workflow, regression, and architecture criteria.
- Keep the skill catalog aligned with the emerging benchkit layout.

## Non-Goals

- Rewriting the batch-1 spec or plan retroactively.
- Adding a generic Python coding-standards skill; `ac-review` remains the
  project-specific review surface.
- Replacing benchmark runbooks with skill text.

## Catalog additions

| Skill / criterion | Purpose |
| --- | --- |
| `ac-design` | Analyze cache patterns and write RFC-style design documents for AgentCache optimization strategies. |
| `60-workflow.md` | Review CI/workflow configuration, pre-commit compatibility, CI time, and DCO checks. |
| `70-regression.md` | Review benchmark and performance-regression evidence against baseline runs. |
| `80-architecture.md` | Review file ownership boundaries, contracts, utility placement, and key class surface area. |

## `ac-design`

`ac-design` is a producer skill for design work before implementation. It is
triggered by requests to analyze AgentCache cache patterns, write design docs,
or evaluate benchmark data for optimization strategy.

The skill is grounded in project RFCs and benchmark context:

- Single-agent tool-call optimization from RFC #10.
- Subagent concurrency optimization from RFC #9.
- Correctness guardrails from RFC #2.

It requires designs to state measurable success criteria and correctness
guardrails before implementation begins.

## `ac-review` criteria extensions

### Workflow

`60-workflow.md` covers GitHub Actions, pre-commit configuration, tool-version
compatibility, CI runtime impact, and DCO. This keeps workflow changes out of
style-only review and makes build-system ownership explicit.

### Regression

`70-regression.md` is stricter than the general performance bar in `40-perf.md`.
It requires baseline/candidate comparisons for benchmark-sensitive changes,
blocks pass-rate regressions, and calls out TTFT checks for scheduling work.

### Architecture

`80-architecture.md` adds reviewer-level gates that apply across later phases:

- File headers define ownership boundaries and reader entry points.
- Public/core contracts are documented and typed.
- Reusable helpers live in domain utilities rather than private sprawl.
- Key class method surfaces are justified by ownership and call-site evidence.
- Public APIs appear before private helpers.

## Benchmark skill alignment

`ac-benchmark` now uses a discovery-first workflow instead of freezing a
particular benchmark layout in the skill text. It should inventory the current
repo before naming:

- benchmark config files or templates,
- task lists or instance indexes,
- launch scripts or CLI entry points,
- status/runbook docs,
- output/result directories.

Concrete experiment state should be produced from verified current templates or
configs, not from stale hardcoded path assumptions.
