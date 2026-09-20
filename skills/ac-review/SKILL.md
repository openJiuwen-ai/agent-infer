---
name: ac-review
description: Use when reviewing a diff or PR against AgentInfer conventions; supports quick local precheck and full reviewer-readiness checks against evolving criteria.
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# ac-review

Review a local diff, PR draft, or PR evidence package against AgentInfer's review
criteria. The review **process** is stable and lives here. The **criteria**
evolve and live in `criteria/*.md`, refreshed weekly.

## WHEN TO INVOKE

- After `ac-bootstrap`, `ac-benchmark`, `ac-integrate`, or `ac-design` produces a diff or design.
- Before opening a PR (`quick` mode).
- Before marking a PR ready for review (`full` mode).
- The user asks to "review this change", "check this PR", "precheck", or
  "what conventions does this break".

Do NOT invoke for: writing new criteria (that is the weekly refresh), reviewing
non-AgentInfer code, posting to GitHub, or editing files.

## MODES

### Quick mode

Use before opening a PR. Quick mode can run from local state only.

Inputs:

- Local branch diff.
- Changed file list.
- Recent commit message or branch name.
- Optional PR draft text if already available.

Checks:

- RFC or issue link is expected when behavior, design, API, benchmark, or workflow changes.
- PR purpose, if drafted, matches the diff.
- Test Plan, if drafted, includes exact commands or justified skips.
- Test Result, if drafted, includes output/artifact paths or justified skips.
- Diff has no obvious unrelated changes.
- Relevant pre-commit, tests, smoke checks, or benchmark checks are identified.

Do not fail solely because no PR exists yet. Report PR-only gaps separately:

```text
Missing because PR not opened yet:
- PR RFC link
- PR Test Result
```

### Full mode

Use before marking a PR ready for review. Full mode requires reviewer-facing
PR text plus any evidence needed for the changed area.

Resolve PR input in this order:

1. Explicit PR URL or PR number.
2. Copied PR body or local PR draft text.
3. Branch-name lookup via GitHub CLI, if available.

If PR lookup fails, fall back to quick mode and report that PR metadata was not
available. Do not treat an unresolved branch-name lookup as proof that no PR
exists.

For benchmark or test-utility work, also require an explicit validation/artifact
manifest. The manifest supplements PR text; it is not a substitute for PR text.

For benchmark or test-utility work, the evidence should include:

```md
## Validation Evidence

- Commit:
- Command:
- Config:
- Host / env:
- Output path:
- Raw logs/results:
- Parsed summary:
- Full benchmark/E2E run: yes/no
- If no, fallback smoke/regression:
```

## STEPS

1. **Select mode and gather inputs**

   Default to quick mode if no PR body, PR URL, PR number, or resolved PR metadata
   is available. Use full mode only when reviewer-facing PR text is present. For
   benchmark or test-utility work, full mode also requires a validation/artifact
   manifest.

2. **Gather all change surfaces**

   Start with repository state and local changes:

   ```bash
   git status --short --branch
   git diff --name-only
   git diff
   git diff --cached --name-only
   git diff --cached
   ```

   Determine the comparison base from the requested PR target or PR metadata. If
   neither is available, inspect configured remotes and choose the canonical main
   ref (`upstream/main` when present, otherwise `origin/main`). Review the committed
   branch range as well:

   ```bash
   git diff --name-only <base-ref>...HEAD
   git diff <base-ref>...HEAD
   ```

   Combine committed, staged, unstaged, and untracked file lists when deciding
   which criteria trigger. Read relevant untracked files directly because they are
   absent from `git diff`.

3. **Read every file in `criteria/`** in numeric-prefix order. For each file,
   read its `## TRIGGER` block. If the trigger applies to this diff or evidence
   package, apply every `## CRITERION`; otherwise skip it.

4. **For each criterion**, record one of:

   - **PASS** — criterion satisfied.
   - **FAIL** — criterion violated; cite `file:line`, evidence, and criterion id.
   - **NA** — criterion does not apply to this diff.
   - **MISSING** — full-mode evidence is required but was not provided.

5. **Report findings** in a terminal-only readiness report:

   ```text
   Review mode: quick | full
   Result: PASS | WARN | FAIL

   Blockers:
   - ...

   Warnings:
   - ...

   Evidence checked:
   - ...

   Missing evidence:
   - ...

   Suggested next step:
   - ...
   ```

6. **Propose new criteria**. If you found yourself repeating a check that is not
   in any `criteria/*.md`, or the PR establishes a new convention, draft a
   `## CRITERION` block and point the author at `criteria/README.md` to add it
   in the next weekly refresh.

## DON'T

- Don't invent criteria on the fly and apply them as blockers; unwritten rules
  are suggestions, recorded only as proposals in step 6.
- Don't auto-fix violations you weren't asked to fix.
- Don't skip reading a `## TRIGGER` block.
- Don't post to GitHub, open PRs, or edit files from this skill.
- Don't treat missing E2E benchmark CI as proof of readiness; require PR evidence
  or an explicit not-run explanation with fallback smoke/regression checks.

## AFTER

Hand the findings back to the author. Blockers must be resolved before merge;
proposed new criteria go into the next weekly refresh (see `criteria/README.md`).
