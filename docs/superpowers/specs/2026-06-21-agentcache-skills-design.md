# AgentCache Skills — Design

**Date:** 2026-06-21
**Status:** Approved (pending spec review)
**Scope:** First batch — 4 ZCode/agent skills for the AgentCache project

## Context

AgentCache is a Python 3.10+ project (library + service) for "efficient cache
management for agent workflows, designed to plug into LLM serving engines such
as vLLM." At the time of this design the repo has no source code yet — only a
README, `pyproject.toml` (ruff config, no deps), pre-commit hooks, DCO sign-off
enforcement, GitHub issue/PR templates, and CI.

This design defines the first batch of **ZCode/agent skills** (SKILL.md
instruction files invoked via the `Skill` tool) that guide AI agents working on
AgentCache. Two audiences are served:

- **Contributors** — agents that add code to AgentCache.
- **Integrators** — agents that plug AgentCache into an external deployment.

## Goals

- Ship a small set of skills that cover the most frequent, highest-value agent
  actions across both audiences.
- Make at least one skill (`ac-review`) able to evolve as the repo grows,
  without requiring rewrites.
- Be useful on day one (no source code yet) and remain useful as code lands.

## Non-Goals

- A complete catalog of every conceivable skill (deferred to future batches).
- An in-codebase "skill" feature type (these are agent-instruction files, not
  Python feature code).
- Integrator-facing skills shipped as published package docs (the integrator
  skills here live in-repo and are consumed by agents working in this repo's
  ZCode session).

## Approach

Selected approach: **Workflow-stage skills** (lifecycle-oriented), with one
extra skill (`ac-review`) added to satisfy the "evolves as the repo develops"
requirement. Alternative approaches considered (capability-axis taxonomy; one
comprehensive skill with helpers) were rejected — see _Alternatives_.

## Catalog (first batch)

| Skill           | Audience                | Purpose                                                       |
| --------------- | ----------------------- | ------------------------------------------------------------- |
| `ac-bootstrap`  | Contributor             | Scaffold new backends/policies/entry points per conventions.  |
| `ac-benchmark`  | Contributor + Integrator| Run the benchmark harness vs vLLM, capture + compare metrics. |
| `ac-integrate`  | Integrator              | Plug AgentCache into a vLLM deployment.                       |
| `ac-review`     | Contributor             | Review changes against repo conventions that grow over time.  |

## Section 1 — Location, Packaging, Invocation

**Location:** `skills/` at the repo root, one directory per skill:

```
skills/
  README.md              # index: one line per skill
  ac-bootstrap/
    SKILL.md
  ac-benchmark/
    SKILL.md
  ac-integrate/
    SKILL.md
  ac-review/
    SKILL.md
    criteria/
      README.md          # evolution protocol
      00-meta.md
      10-style.md
      20-api.md
      30-testing.md
      40-perf.md
      50-safety.md
```

This mirrors the convention used by existing skills
(`/Users/elle/.zcode/skills/<name>/SKILL.md`) and the superpowers plugin layout.

**Naming:** `ac-` prefix (AgentCache) to avoid collisions with general skills
(e.g. `benchmark`, `bootstrap`) and to make them greppable.

**Invocation:** via the `Skill` tool by an agent whose ZCode session has this
repo's skills on its search path. Each `SKILL.md` opens with a `description:`
frontmatter line tuned for auto-discovery — the description must let the agent
reliably decide "yes, invoke this" from the user's request alone.

**Frontmatter contract** (per skill):

```yaml
---
name: <skill-name>
description: <one sentence: when to invoke>
---
```

A `skills/README.md` index lists all four skills with one-line purposes.

## Section 2 — `ac-review` Architecture (the evolvable skill)

### Problem

At today's state, there is almost nothing to review against (no code; only ruff
config, DCO, and the PR template). If review criteria were hardcoded into
`SKILL.md`, they would be wrong within a week. The design must separate the
**stable review process** from the **evolving review criteria**.

### Structure

```
skills/ac-review/
  SKILL.md              # the PROCESS (stable, rarely changes)
  criteria/
    00-meta.md          # always-true repo rules: DCO, ruff clean, PR template
    10-style.md         # formatting, imports, naming — grows as conventions land
    20-api.md           # public API stability — grows as API firms up
    30-testing.md       # test expectations — grows as test infra lands
    40-perf.md          # cache-performance bar — grows as benchmark harness lands
    50-safety.md        # cache poisoning, eviction correctness — grows as threats emerge
    README.md           # evolution protocol + update cadence
```

### What is stable (lives in `SKILL.md`)

The review _process_, not the criteria:

1. Gather the diff + list of changed files.
2. Read every applicable `criteria/*.md` file (each file's `## TRIGGER`
   header decides whether it applies to this diff).
3. For each criterion, check it against the diff; record pass / fail / NA with
   `file:line` references.
4. Report findings grouped by severity (blocker / warning / nit). Never
   auto-fix beyond what the criteria explicitly authorize.
5. **Suggest new criteria** when a review notices a recurring issue not yet
   captured — this is the evolution input (feeds the weekly refresh).

### What evolves (lives in `criteria/`)

The actual checks. Per the "strong template now + weekly collection" model
agreed with the user:

- **Strong starting template:** the `criteria/*.md` files are authored with
  real, opinionated defaults drawn from what already exists (ruff rules, DCO,
  PR template) plus sensible cache-project defaults (e.g. public API function
  = must have a docstring; cache mutation = must have a test). So the skill is
  useful on day one, not after a month of accretion.
- **Weekly collection:** `criteria/README.md` documents a weekly update
  protocol. Because the repo already has a `750-RFC.yml` issue template, the
  natural mechanism is a recurring **"criteria refresh" issue** opened weekly
  where reviewers batch up the conventions that have landed in the last 7 days.
  Organic suggestions emitted by `ac-review` runs feed into that issue, so the
  weekly update is just emptying the inbox — not fresh archaeology. Weekly
  (rather than monthly) matches the rate at which conventions land in an
  early-stage repo; the cadence can be revisited once the codebase matures.

The numbered prefix (`00-`, `10-` …) keeps criteria evaluated in deterministic
order and makes gaps obvious (you can see at a glance that `30-testing.md` is
still a stub).

### Why this shape

- _Single big SKILL.md with all criteria_ → edits get noisy, hard to see what
  changed, discourages small additive commits. **Rejected.**
- _Criteria generated by scanning repo files_ → fragile; conventions that
  aren't machine-detectable (e.g. "public API functions need a docstring
  example") never get captured. **Rejected.**
- _Process + criteria files (this design)_ → each criterion addition is a tiny
  focused commit; the process never churns; gaps are visible. **Chosen.**

### Trade-off / risk

There is a small risk the criteria files drift stale if nobody runs the
weekly refresh. Mitigation: the skill's final step proposes new criteria when
warranted, feeding the recurring refresh issue. A staleness check ("flag
criteria untouched in N commits") is explicitly **out of scope for v1** (YAGNI)
and deferred to a future batch if drift proves real.

## Section 3 — The Three Simpler Skills

These share a common structure (none evolve), so each `SKILL.md` has: frontmatter
description, a `## WHEN TO INVOKE` block (the trigger, mirroring how the
superpowers skills do it), a `## STEPS` checklist, and a `## DON'T` block for
the common mistakes that matter on this repo.

### `ac-bootstrap`

_Use when scaffolding a new AgentCache cache backend, eviction policy, or public
entry point._

**Steps:**

1. Pick the kind: backend / policy / entry point.
2. Read existing layout under `src/agentcache/` (or create it if this is the
   first).
3. Place the file in the convention-correct location.
4. Write the `__init__.py` export.
5. Add a smoke test that imports the new symbol.
6. Run `pre-commit run --all-files`.
7. Confirm the DCO sign-off line is in the commit message.

**Don't:**

- Add deps to `pyproject.toml` without surfacing it for review.
- Put backends and policies in the same module.
- Skip the `pre-commit` run.

### `ac-benchmark`

_Use when measuring AgentCache's cache hit rate / latency / throughput against a
target engine (vLLM or stub)._

**Steps:**

1. Locate (or create, if absent) the benchmark harness under `benchmarks/`.
   - _Bootstrap branch:_ if no harness exists yet, create the layout
     (`benchmarks/run.py`, `benchmarks/baseline.json`,
     `benchmarks/results/.gitkeep`) before proceeding.
2. Confirm a target is reachable (vLLM endpoint URL or in-process stub).
3. Run a warmup pass.
4. Run the measured pass, writing JSON metrics to
   `benchmarks/results/<timestamp>.json`.
5. Compare against the latest baseline in `benchmarks/baseline.json`.
6. Report hit-rate, p50/p99 latency, throughput deltas.

**Don't:**

- Compare against a non-baseline run.
- Delete old result files.
- Trust a single iteration.

### `ac-integrate`

_Use when plugging AgentCache into an existing vLLM serving deployment._

**Steps:**

1. Detect the target engine and version
   (`python -c "import vllm; print(vllm.__version__)"` or read the deployment's
   declared version).
2. Locate the integration adapter under `src/agentcache/adapters/` (create if
   missing).
3. Wire the adapter per the engine's prefix-cache API for that version.
4. Run a cache-hit validation (send identical prefix, assert second call is
   faster / hits cache).
5. Document the deployment specifics in the integration notes.

**Don't:**

- Assume a vLLM version's cache API without checking.
- Mutate the user's deployment config without confirmation.

### Cross-skill consistency

All four skills link to `ac-review` as the final pre-merge step —
bootstrap/benchmark/integrate produce changes, then `ac-review` validates them.
This makes the catalog coherent rather than four isolated scripts.

## Alternatives Considered

- **Capability-axis skills** (`ac-cache-strategy`, `ac-vllm-adapter`,
  `ac-metrics`): cleaner taxonomy, but these overlap heavily (a cache strategy
  change immediately needs metrics + an adapter touch), so agents would invoke
  several at once and get conflicting guidance. Rejected.
- **Single comprehensive `agentcache-dev` skill with internal branches + 2 thin
  helpers:** easiest to discover, but violates the skill-design principle of
  small focused units, and the existing superpowers catalog favours many
  focused skills over few large ones. Rejected.

## Open Questions Resolved During Design

- _Q: Where do skills live?_ A: `skills/` at repo root, one dir per skill.
- _Q: How does `ac-review` evolve?_ A: stable process in `SKILL.md` + evolving
  `criteria/*.md` files, refreshed weekly via a recurring issue that drains
  suggestions emitted by `ac-review` runs.
- _Q: What if the benchmark harness doesn't exist yet?_ A: `ac-benchmark` has
  an explicit bootstrap branch in its steps.

## Out of Scope (future batches)

- Additional skills beyond these four (e.g. release, triage, debug).
- A staleness check for `ac-review` criteria (deferred; revisit if drift proves
  real).
- Shipped-as-package integrator docs (separate effort from in-repo skills).
