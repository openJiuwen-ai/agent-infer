# AgentCache Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 4 ZCode/agent skills (`ac-bootstrap`, `ac-benchmark`, `ac-integrate`, `ac-review`) under `skills/` that guide AI agents working on AgentCache, with `ac-review` designed to evolve via weekly criteria refresh.

**Architecture:** Each skill is a `skills/<name>/SKILL.md` file with `name` + `description` frontmatter (the description is the auto-discovery trigger). `ac-review` separates stable process (`SKILL.md`) from evolving criteria (`criteria/*.md` with a `## TRIGGER` header each), refreshed weekly via a recurring issue. A `skills/README.md` indexes all four. All three producer skills link to `ac-review` as the final pre-merge step.

**Tech Stack:** Markdown only; no runtime code. Conventions: ruff-formatted where applicable, DCO sign-off on every commit, markdownlint-clean (CI runs `markdownlint-cli2`).

---

## File Structure

```
skills/
  README.md                          # index: one line per skill (Task 1)
  ac-bootstrap/
    SKILL.md                         # scaffold skill (Task 2)
  ac-benchmark/
    SKILL.md                         # benchmark skill (Task 3)
  ac-integrate/
    SKILL.md                         # integration skill (Task 4)
  ac-review/
    SKILL.md                         # review PROCESS (stable) (Task 5)
    criteria/
      README.md                      # evolution protocol + weekly cadence (Task 5)
      00-meta.md                     # DCO, ruff, PR template (Task 6)
      10-style.md                    # formatting, imports, naming (Task 6)
      20-api.md                      # public API docstring rule (Task 6)
      30-testing.md                  # cache mutation = test required (Task 6)
      40-perf.md                     # performance bar (Task 6)
      50-safety.md                   # cache poisoning, eviction correctness (Task 6)
      _template.md                   # template for adding a new criterion file (Task 6)
```

Each file has a single, clear responsibility:

- `SKILL.md` files = the skill the agent invokes (process + steps).
- `criteria/*.md` files = one category of review checks each, with a `## TRIGGER` header that says when it applies.
- `criteria/_template.md` = copy-and-fill template so new criteria stay consistent.
- `criteria/README.md` = the weekly refresh protocol (the "evolve" mechanism).

**Verification gate (Task 7):** run pre-commit on the whole skills tree and fix any markdownlint/ruff/typos findings before the final commit.

**Decision rules applied:**

- DRY: the cross-skill "final step = run ac-review" link is written once in each SKILL.md but points at the same skill rather than duplicated logic.
- YAGNI: no staleness check for criteria in v1; no benchmark harness code (only the skill describing how to use one once it exists).
- TDD: these are doc-only skills, so "tests" = verification that each skill is markdownlint-clean and that its frontmatter matches the contract. Task 7 is the verification pass.

---

## Task 1: Create the skills index

**Files:**

- Create: `skills/README.md`

- [ ] **Step 1: Write the index**

Create `skills/README.md` with this exact content:

```markdown
# AgentCache Skills

ZCode/agent skills for working on AgentCache. Each skill lives in its own
directory as a `SKILL.md` file and is invoked by an AI agent via the Skill tool.

## Skills

| Skill          | Audience                 | Purpose                                                          |
| -------------- | ------------------------ | ---------------------------------------------------------------- |
| `ac-bootstrap`  | Contributor              | Scaffold a new cache backend, eviction policy, or entry point.   |
| `ac-benchmark`  | Contributor + Integrator | Run the benchmark harness vs vLLM, capture and compare metrics.  |
| `ac-integrate`  | Integrator               | Plug AgentCache into a vLLM serving deployment.                  |
| `ac-review`     | Contributor              | Review changes against repo conventions that grow over time.     |

All three producer skills (`ac-bootstrap`, `ac-benchmark`, `ac-integrate`) end
by invoking `ac-review` as the final pre-merge validation step.

## Conventions

- Each skill directory contains a `SKILL.md` with `name` and `description`
  frontmatter. The `description` is the auto-discovery trigger — it must let an
  agent decide "yes, invoke this" from the user's request alone.
- `ac-review` stores its evolving criteria in `criteria/*.md`; see
  `ac-review/criteria/README.md` for the weekly refresh protocol.
```

- [ ] **Step 2: Verify markdownlint is clean on the new file**

Run: `pre-commit run markdownlint-cli2 --files skills/README.md`
Expected: PASS (no output, exit 0). If it reports issues, fix the flagged lines and re-run.

- [ ] **Step 3: Commit**

```bash
git add skills/README.md
git commit -m "docs(skills): add skills index

Signed-off-by: $(git config user.name) <$(git config user.email)>"
```

---

## Task 2: Create `ac-bootstrap` skill

**Files:**

- Create: `skills/ac-bootstrap/SKILL.md`

- [ ] **Step 1: Write the skill**

Create `skills/ac-bootstrap/SKILL.md` with this exact content:

```markdown
---
name: ac-bootstrap
description: Use when scaffolding a new AgentCache cache backend, eviction policy, or public entry point; places files per project conventions and runs pre-commit.
---

# ac-bootstrap

Scaffold a new AgentCache component (cache backend, eviction policy, or public
entry point) following the project's conventions, then validate locally before
committing.

## WHEN TO INVOKE

- The user asks to "add a new cache backend" / "add an eviction policy" /
  "add a public API entry point" for AgentCache.
- The user asks to scaffold or stub a new AgentCache module.

Do NOT invoke for: editing an existing component, adding tests only, or
non-AgentCache scaffolding.

## STEPS

1. Identify the kind: **backend**, **policy**, or **entry point**.
2. Read the existing layout under `src/agentcache/`. If `src/agentcache/` does
   not yet exist, this is the first component: create the package skeleton:
   ```bash
   mkdir -p src/agentcache
   printf '"""AgentCache: efficient cache management for agent workflows."""\n' > src/agentcache/__init__.py
   ```

1. Place the new file by kind:
   - **backend** -> `src/agentcache/backends/<name>.py`
   - **policy** -> `src/agentcache/policies/<name>.py`
   - **entry point** -> `src/agentcache/__init__.py` (add the public export)
   Never put backends and policies in the same module.
2. Add the symbol to the relevant `__init__.py` `__all__` / import so it is
   importable from the package top level (for entry points and public backends).
3. Add a smoke test that imports the new symbol:
   - File: `tests/test_<kind>_<name>.py`
   - Minimal body: `from agentcache import <symbol>` inside a test that asserts
     the import succeeds.
4. Run the full pre-commit pass on the changed files:

   ```bash
   pre-commit run --files src/agentcache/** tests/**
   ```

   Fix every reported issue before continuing.
5. Confirm the commit message will carry the DCO `Signed-off-by:` line. The
   repo's `commit-msg` hook adds it automatically if missing; do not strip it.

## DON'T

- Don't add dependencies to `pyproject.toml` without surfacing it for review.
- Don't place backends and policies in the same module.
- Don't skip the `pre-commit` run "to save time".
- Don't commit without the `Signed-off-by:` trailer (DCO check will fail in CI).

## AFTER

Once the new component compiles and pre-commit passes, invoke `ac-review` to
validate the diff against repo conventions before opening the PR.

```

- [ ] **Step 2: Verify markdownlint is clean**

Run: `pre-commit run markdownlint-cli2 --files skills/ac-bootstrap/SKILL.md`
Expected: PASS. Fix and re-run on any finding.

- [ ] **Step 3: Commit**

```bash
git add skills/ac-bootstrap/SKILL.md
git commit -m "docs(skills): add ac-bootstrap skill

Signed-off-by: $(git config user.name) <$(git config user.email)>"
```

---

## Task 3: Create `ac-benchmark` skill

**Files:**

- Create: `skills/ac-benchmark/SKILL.md`

- [ ] **Step 1: Write the skill**

Create `skills/ac-benchmark/SKILL.md` with this exact content:

```markdown
---
name: ac-benchmark
description: Use when measuring AgentCache cache hit rate, latency, or throughput against a target engine (vLLM or in-process stub); bootstraps the harness layout if absent and compares against the baseline.
---

# ac-benchmark

Run the AgentCache benchmark harness against a target LLM serving engine,
capture metrics, and compare against the recorded baseline.

## WHEN TO INVOKE

- The user asks to "benchmark AgentCache", "measure cache hit rate", "compare
  latency/throughput vs vLLM", or "run the benchmark suite".
- A change to cache behaviour needs evidence before merge.

Do NOT invoke for: unit-testing a function, profiling unrelated code, or
comparing two non-baseline runs against each other.

## STEPS

1. Locate the benchmark harness under `benchmarks/`.
   - **Bootstrap branch (harness does not exist yet):** create the layout first:
     ```bash
     mkdir -p benchmarks/results
     touch benchmarks/results/.gitkeep
     printf '{\n  "hit_rate": null,\n  "p50_latency_ms": null,\n  "p99_latency_ms": null,\n  "throughput_rps": null,\n  "note": "populate with the first measured run"\n}\n' > benchmarks/baseline.json
     printf '"""AgentCache benchmark harness entry point."""\n' > benchmarks/run.py
     ```
     Then continue to step 2.
2. Confirm a target is reachable:
   - vLLM: a base URL is set (`export AGENTCACHE_VLLM_URL=...`) and reachable.
   - Stub: an in-process stub target is configured (no external dependency).
3. Run a warmup pass to populate the cache:
   ```bash
   python benchmarks/run.py --target "$AGENTCACHE_VLLM_URL" --warmup
   ```

1. Run the measured pass, writing JSON metrics to a timestamped result file:

   ```bash
   python benchmarks/run.py --target "$AGENTCACHE_VLLM_URL" \
     --out "benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ).json"
   ```

2. Compare the new result against `benchmarks/baseline.json` (hit-rate, p50/p99
   latency, throughput). Report each metric's delta and direction.
3. If the run regresses any metric relative to baseline, state so explicitly
   (do not silently commit a worse number as the new baseline).

## DON'T

- Don't compare against a non-baseline run as if it were the baseline.
- Don't delete old `benchmarks/results/*.json` files — they are the history.
- Don't trust a single iteration; the harness must take multiple samples and
  report percentiles.

## AFTER

Once metrics are captured and compared, invoke `ac-review` to validate any
harness or baseline changes before opening the PR.

```

- [ ] **Step 2: Verify markdownlint is clean**

Run: `pre-commit run markdownlint-cli2 --files skills/ac-benchmark/SKILL.md`
Expected: PASS. Fix and re-run on any finding.

- [ ] **Step 3: Commit**

```bash
git add skills/ac-benchmark/SKILL.md
git commit -m "docs(skills): add ac-benchmark skill

Signed-off-by: $(git config user.name) <$(git config user.email)>"
```

---

## Task 4: Create `ac-integrate` skill

**Files:**

- Create: `skills/ac-integrate/SKILL.md`

- [ ] **Step 1: Write the skill**

Create `skills/ac-integrate/SKILL.md` with this exact content:

```markdown
---
name: ac-integrate
description: Use when plugging AgentCache into an existing vLLM (or compatible) serving deployment; detects engine version, wires the prefix-cache adapter, and validates cache hits.
---

# ac-integrate

Wire AgentCache into an existing LLM serving deployment (vLLM or compatible),
version-check the engine's prefix-cache API, and validate that caching actually
fires.

## WHEN TO INVOKE

- The user asks to "integrate AgentCache with vLLM", "plug AgentCache into my
  deployment", or "wire the adapter" for a serving engine.
- A new engine adapter is being added under `src/agentcache/adapters/`.

Do NOT invoke for: adding a cache backend (use `ac-bootstrap`), benchmarking
(use `ac-benchmark`), or editing integration docs only.

## STEPS

1. Detect the target engine and version:
   ```bash
   python -c "import vllm; print(vllm.__version__)" 2>/dev/null \
     || echo "vllm not importable locally; read the deployment's declared version"
   ```

   Record the version — the prefix-cache API differs across vLLM releases.
2. Locate the integration adapter under `src/agentcache/adapters/`. If the
   directory or adapter does not exist, create it:

   ```bash
   mkdir -p src/agentcache/adapters
   ```

1. Wire the adapter per the engine's prefix-cache API **for the detected
   version**. Do not assume a cache API shape without checking that version's
   docs.
2. Run a cache-hit validation: send the same prompt prefix twice and assert the
   second call is faster (or reports a cache hit). Example shape:

   ```bash
   python -c "
   from agentcache.adapters import vllm as acv
   ac = acv.connect('$AGENTCACHE_VLLM_URL')
   t1 = ac.time_call('Once upon a time,')
   t2 = ac.time_call('Once upon a time,')
   assert t2 < t1, f'no cache speedup: {t2=}, {t1=}'
   print('cache hit validated')
   "
   ```

3. Document the deployment specifics (engine, version, endpoint shape, any
   non-default config) in the integration notes so the next integration is
   reproducible.

## DON'T

- Don't assume a vLLM version's cache API without checking that version.
- Don't mutate the user's deployment config files without explicit confirmation.
- Don't declare integration done until the cache-hit validation passes.

## AFTER

Once the adapter is wired and cache hits are validated, invoke `ac-review` to
validate the diff against repo conventions before opening the PR.

```

- [ ] **Step 2: Verify markdownlint is clean**

Run: `pre-commit run markdownlint-cli2 --files skills/ac-integrate/SKILL.md`
Expected: PASS. Fix and re-run on any finding.

- [ ] **Step 3: Commit**

```bash
git add skills/ac-integrate/SKILL.md
git commit -m "docs(skills): add ac-integrate skill

Signed-off-by: $(git config user.name) <$(git config user.email)>"
```

---

## Task 5: Create `ac-review` skill and its criteria README

This task creates the stable review process (`SKILL.md`) and the evolution
protocol (`criteria/README.md`). The criteria *content* files come in Task 6.

**Files:**

- Create: `skills/ac-review/SKILL.md`
- Create: `skills/ac-review/criteria/README.md`

- [ ] **Step 1: Write the stable review process**

Create `skills/ac-review/SKILL.md` with this exact content:

```markdown
---
name: ac-review
description: Use when reviewing a diff or PR against AgentCache conventions; reads each applicable criteria/*.md file, reports findings by severity, and proposes new criteria when a recurring issue is not yet captured.
---

# ac-review

Review a change (diff or PR) against AgentCache's review criteria. The review
**process** is stable and lives here. The **criteria** evolve and live in
`criteria/*.md`, refreshed weekly.

## WHEN TO INVOKE

- After `ac-bootstrap`, `ac-benchmark`, or `ac-integrate` produces a diff.
- The user asks to "review this change", "check this PR", or "what conventions
  does this break".
- As the final pre-merge validation step.

Do NOT invoke for: writing new criteria (that is the weekly refresh), or
reviewing non-AgentCache code.

## STEPS

1. Gather the diff and the list of changed files:
   ```bash
   git diff --name-only origin/main...HEAD
   git diff origin/main...HEAD
   ```

1. Read every file in `criteria/` (in numeric-prefix order: `00-`, `10-`, ...).
   For each file, read its `## TRIGGER` block. If the trigger applies to this
   diff, apply every `## CRITERION` in that file; otherwise skip it.
2. For each criterion, check it against the diff and record one of:
   - **PASS** — criterion satisfied.
   - **FAIL** — criterion violated; cite `file:line` and the criterion id.
   - **NA** — criterion does not apply to this diff.
3. Report findings grouped by severity:
   - **blocker** — must fix before merge.
   - **warning** — should fix; surface to the author.
   - **nit** — optional; mention but don't block.
   Do not auto-fix beyond what a criterion explicitly authorizes.
4. Propose new criteria. If you found yourself repeating a check that is not in
   any `criteria/*.md`, or the PR establishes a new convention, draft a
   `## CRITERION` block and point the author at `criteria/README.md` to add it
   in the next weekly refresh. This is how the skill evolves.

## DON'T

- Don't invent criteria on the fly and apply them as blockers — unwritten rules
  are suggestions, recorded only as proposals in step 5.
- Don't auto-fix violations you weren't asked to fix.
- Don't skip reading `criteria/README.md`'s trigger rules.

## AFTER

Hand the findings back to the author. Blockers must be resolved before merge;
proposed new criteria go into the next weekly refresh (see
`criteria/README.md`).

```

- [ ] **Step 2: Write the evolution protocol**

Create `skills/ac-review/criteria/README.md` with this exact content:

```markdown
# `ac-review` criteria — evolution protocol

The review **process** lives in `../SKILL.md` and is stable. The actual checks
live in this directory, one category per file, and evolve as the repo develops.

## File conventions

- One category per file, named `<NN>-<topic>.md` where `NN` is a zero-padded
  number. The number sets evaluation order and makes gaps visible.
- Every file starts with a `## TRIGGER` block describing which diffs it applies
  to, followed by one or more `## CRITERION` blocks.
- A `## CRITERION` block has the shape:
  ```markdown
  ## CRITERION <id>: <one-line summary>
  - **Severity:** blocker | warning | nit
  - **Check:** <how to evaluate it against a diff>
  - **Fix:** <what the author should do if it fails>
  ```

- Use `_template.md` to start a new category file.

## Weekly refresh

The criteria are refreshed **weekly**. Mechanism:

1. Every `ac-review` run may propose new `## CRITERION` blocks (step 5 of the
   skill). Those proposals are collected as comments on a single recurring
   GitHub issue titled `criteria refresh <YYYY-MM-DD>` (use the `750-RFC.yml`
   issue template).
2. Once a week, the maintainer opens (or reuses) that issue, triages the
   collected proposals, and lands accepted ones as small commits editing the
   relevant `criteria/*.md` file. Each accepted criterion is its own focused
   commit.
3. Close the issue once the batch is landed; open the next week's issue.

Weekly (rather than monthly) matches the rate at which conventions land in an
early-stage repo. Revisit the cadence once the codebase matures.

## Current categories

- `00-meta.md` — always-true repo rules: DCO, ruff, PR template.
- `10-style.md` — formatting, imports, naming.
- `20-api.md` — public API stability / docstrings.
- `30-testing.md` — test expectations.
- `40-perf.md` — cache-performance bar.
- `50-safety.md` — cache poisoning, eviction correctness.

```

- [ ] **Step 3: Verify markdownlint is clean on both files**

Run: `pre-commit run markdownlint-cli2 --files skills/ac-review/SKILL.md skills/ac-review/criteria/README.md`
Expected: PASS. Fix and re-run on any finding.

- [ ] **Step 4: Commit**

```bash
git add skills/ac-review/SKILL.md skills/ac-review/criteria/README.md
git commit -m "docs(skills): add ac-review skill + weekly evolution protocol

Signed-off-by: $(git config user.name) <$(git config user.email)>"
```

---

## Task 6: Author the criteria content files and template

Strong starting templates (not empty stubs) so `ac-review` is useful on day one.

**Files:**

- Create: `skills/ac-review/criteria/00-meta.md`
- Create: `skills/ac-review/criteria/10-style.md`
- Create: `skills/ac-review/criteria/20-api.md`
- Create: `skills/ac-review/criteria/30-testing.md`
- Create: `skills/ac-review/criteria/40-perf.md`
- Create: `skills/ac-review/criteria/50-safety.md`
- Create: `skills/ac-review/criteria/_template.md`

- [ ] **Step 1: Write `00-meta.md`**

```markdown
# 00 — Meta (always-true repo rules)

## TRIGGER

Always applies to every diff that touches the repo.

## CRITERION M1: DCO sign-off present

- **Severity:** blocker
- **Check:** Every commit in the PR carries a `Signed-off-by: Name <email>` trailer. CI's `dco` job enforces this; do not bypass it.
- **Fix:** `git commit --amend --signoff` (or rebase with `--signoff`) on every offending commit.

## CRITERION M2: pre-commit clean

- **Severity:** blocker
- **Check:** `pre-commit run --all-files` exits 0. Covers ruff check, ruff format, typos, markdownlint.
- **Fix:** Run `pre-commit run --all-files` locally, fix every reported issue, re-run until clean.

## CRITERION M3: PR template fields filled

- **Severity:** warning
- **Check:** The PR description uses `.github/PULL_REQUEST_TEMPLATE.md` and its required sections are filled (not left as placeholder text).
- **Fix:** Edit the PR description to complete every required section.

## CRITERION M4: no secrets or tokens

- **Severity:** blocker
- **Check:** The diff contains no API keys, tokens, or private endpoint URLs.
- **Fix:** Remove the secret; rotate it if it was pushed; use environment variables / a secrets manager.
```

- [ ] **Step 2: Write `10-style.md`**

```markdown
# 10 — Style

## TRIGGER

Applies to any diff that adds or modifies `*.py` files under `src/` or `tests/`.

## CRITERION S1: formatting passes ruff format

- **Severity:** blocker
- **Check:** `ruff format --check <changed files>` reports no changes needed. (Covered by pre-commit; verify explicitly for style-only feedback.)
- **Fix:** `ruff format <changed files>`.

## CRITERION S2: imports sorted and first-party known

- **Severity:** blocker
- **Check:** `ruff check --select I <changed files>` is clean, and `agentcache` is treated as first-party (configured in `pyproject.toml` `[tool.ruff.lint.isort]`).
- **Fix:** `ruff check --select I --fix <changed files>`.

## CRITERION S3: naming follows snake_case for functions/variables

- **Severity:** warning
- **Check:** Module-level functions and variables use `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE`.
- **Fix:** Rename the offending symbol and update all call sites.

## CRITERION S4: module-level docstring on every new module

- **Severity:** nit
- **Check:** Every new `.py` file begins with a one-line module docstring describing its responsibility.
- **Fix:** Add the docstring as the first statement in the module.
```

- [ ] **Step 3: Write `20-api.md`**

```markdown
# 20 — Public API

## TRIGGER

Applies when the diff modifies `src/agentcache/__init__.py`, anything exported
via `__all__`, or any symbol reachable as `agentcache.<name>`.

## CRITERION A1: public functions/classes have docstrings

- **Severity:** blocker
- **Check:** Every symbol exported from the package top level has a docstring with at least a one-line summary.
- **Fix:** Add a docstring summarising what the symbol does and how to call it.

## CRITERION A2: public API changes are documented in the PR

- **Severity:** warning
- **Check:** If the diff adds, removes, or changes the signature of a public symbol, the PR description calls it out under a "Breaking changes" / "API changes" note.
- **Fix:** Add the note to the PR description.

## CRITERION A3: no unintended private-symbol leakage

- **Severity:** warning
- **Check:** Symbols intended to be private are prefixed with `_` and are not in `__all__`.
- **Fix:** Prefix the symbol with `_` and remove it from `__all__` if present.
```

- [ ] **Step 4: Write `30-testing.md`**

```markdown
# 30 — Testing

## TRIGGER

Applies to any diff that adds or modifies `*.py` under `src/` or `tests/`.

## CRITERION T1: cache mutations have a test

- **Severity:** blocker
- **Check:** Any new/changed code path that writes to, evicts from, or invalidates the cache has at least one test exercising that behaviour.
- **Fix:** Add a test under `tests/` that drives the mutation and asserts the observable effect.

## CRITERION T2: tests do not depend on ordering

- **Severity:** warning
- **Check:** Tests pass when run in any order (`pytest -p no:randomly` is not required to make them green). No test relies on another test having run first.
- **Fix:** Make the test self-contained: set up its own state in a fixture.

## CRITERION T3: smoke test imports succeed

- **Severity:** blocker
- **Check:** For newly added public symbols, a smoke test asserting the import succeeds exists (per `ac-bootstrap` step 5).
- **Fix:** Add the import smoke test.
```

- [ ] **Step 5: Write `40-perf.md`**

```markdown
# 40 — Performance

## TRIGGER

Applies when the diff changes cache behaviour (hit/miss path, eviction,
keying) or anything under `benchmarks/`.

## CRITERION P1: no unmeasured perf claim

- **Severity:** blocker
- **Check:** Any claim in the PR description that the change "improves" / "does not regress" performance is backed by a `benchmarks/results/*.json` run compared against `benchmarks/baseline.json` (per `ac-benchmark`).
- **Fix:** Run `ac-benchmark`, attach the result file, and report the measured deltas.

## CRITERION P2: no obvious O(n^2) in the hot path

- **Severity:** warning
- **Check:** The cache lookup / insert / evict path does not introduce an O(n^2) (or worse) loop over requests or entries.
- **Fix:** Restructure to use a dict/set/indexed structure for the lookup.

## CRITERION P3: baseline only updated with a measured run

- **Severity:** blocker
- **Check:** If `benchmarks/baseline.json` is modified, the PR includes the result file that justifies the new baseline.
- **Fix:** Attach the measured run; if none justifies it, revert the baseline change.
```

- [ ] **Step 6: Write `50-safety.md`**

```markdown
# 50 — Safety / correctness

## TRIGGER

Applies when the diff touches eviction policy, cache key construction, or any
path that decides whether cached data is returned to a request.

## CRITERION F1: no cross-request cache poisoning

- **Severity:** blocker
- **Check:** The cache key includes every element that affects the response (prompt, model, generation params, tenant/owner where relevant). Two requests that should get different responses cannot collide to the same key.
- **Fix:** Add the missing element(s) to the key; add a regression test that asserts distinct responses are not served from the same entry.

## CRITERION F2: eviction does not corrupt in-flight reads

- **Severity:** blocker
- **Check:** Eviction cannot free an entry that a concurrent reader is still consuming.
- **Fix:** Add refcounting / read-lock / generation check so eviction skips entries in active use; add a concurrency test.

## CRITERION F3: invalidation covers all derived entries

- **Severity:** warning
- **Check:** When an entry is invalidated, any entry that was derived from it (e.g. a longer prefix built on a now-stale prefix) is also invalidated.
- **Fix:** Walk the derivation chain on invalidation; add a test covering the derived-entry case.
```

- [ ] **Step 7: Write `_template.md`**

```markdown
# _template — copy this to start a new category file

Copy this file to `<NN>-<topic>.md` (next free number) and edit. Delete this
header comment when done.

## TRIGGER

Describe which diffs this category applies to.

## CRITERION <id>: <one-line summary>

- **Severity:** blocker | warning | nit
- **Check:** <how to evaluate it against a diff>
- **Fix:** <what the author should do if it fails>
```

- [ ] **Step 8: Verify markdownlint is clean on all seven files**

Run:

```bash
pre-commit run markdownlint-cli2 --files \
  skills/ac-review/criteria/00-meta.md \
  skills/ac-review/criteria/10-style.md \
  skills/ac-review/criteria/20-api.md \
  skills/ac-review/criteria/30-testing.md \
  skills/ac-review/criteria/40-perf.md \
  skills/ac-review/criteria/50-safety.md \
  skills/ac-review/criteria/_template.md
```

Expected: PASS. Fix and re-run on any finding.

- [ ] **Step 9: Commit**

```bash
git add skills/ac-review/criteria/
git commit -m "docs(skills): seed ac-review criteria (meta, style, api, testing, perf, safety) + template

Signed-off-by: $(git config user.name) <$(git config user.email)>"
```

---

## Task 7: Full-tree verification gate

**Files:** none modified — verification only.

- [ ] **Step 1: Run the entire pre-commit suite on the skills tree**

Run: `pre-commit run --all-files`
Expected: all hooks PASS (ruff-check, ruff-format, typos, markdownlint-cli2, signoff-commit, suggestion). If any hook reports findings, fix the flagged lines in the offending file and re-run until clean.

- [ ] **Step 2: Sanity-check frontmatter contract**

For each of the four `SKILL.md` files, confirm the first lines match:

```
---
name: <skill-name>
description: <one sentence>
---
```

Run a quick grep to confirm all four have the contract:

```bash
grep -L "^name: ac-" skills/*/SKILL.md
```

Expected: no output (every SKILL.md matches). If a file is listed, fix its frontmatter.

- [ ] **Step 3: Confirm the tree matches the planned structure**

Run:

```bash
find skills -type f | sort
```

Expected output (exactly):

```
skills/README.md
skills/ac-benchmark/SKILL.md
skills/ac-bootstrap/SKILL.md
skills/ac-integrate/SKILL.md
skills/ac-review/SKILL.md
skills/ac-review/criteria/00-meta.md
skills/ac-review/criteria/10-style.md
skills/ac-review/criteria/20-api.md
skills/ac-review/criteria/30-testing.md
skills/ac-review/criteria/40-perf.md
skills/ac-review/criteria/50-safety.md
skills/ac-review/criteria/README.md
skills/ac-review/criteria/_template.md
```

If anything is missing or misplaced, fix it and commit the fix.

- [ ] **Step 4: If any fixups were needed in steps 1–3, commit them**

```bash
git add -A
git commit -m "docs(skills): fixup verification findings

Signed-off-by: $(git config user.name) <$(git config user.email)>"
```

(If nothing changed, skip this step.)

---

## Notes for the executor

- **Commit message sign-off:** the repo's `commit-msg` hook auto-appends
  `Signed-off-by:` if missing, so the `$(git config ...)` expansion in the plan
  is belt-and-braces. Either form satisfies DCO.
- **pre-commit must be installed:** `pre-commit install` should already be done;
  if not, run it once so hooks fire on commit.
- **No runtime code is written in this plan.** The `src/agentcache/` and
  `benchmarks/` snippets live *inside the skill text* as instructions to a
  future agent, not as files this plan creates. Do not create them now.
- **markdownlint is strict** (CI runs `markdownlint-cli2 --fix` via pre-commit).
  If a step's "Expected: PASS" fails, the fix is almost always: blank line
  around lists/headings, or remove a trailing space.
