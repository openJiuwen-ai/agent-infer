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
- The user asks to "review this change", "check this PR", or "what
  conventions does this break".
- As the final pre-merge validation step.

Do NOT invoke for: writing new criteria (that is the weekly refresh), or
reviewing non-AgentCache code.

## STEPS

1. Gather the diff and the list of changed files:

   ```bash
   git diff --name-only origin/main...HEAD
   git diff origin/main...HEAD
   ```

2. Read every file in `criteria/` (in numeric-prefix order: `00-`, `10-`,
   ...). For each file, read its `## TRIGGER` block. If the trigger applies
   to this diff, apply every `## CRITERION` in that file; otherwise skip it.
3. For each criterion, check it against the diff and record one of:

   - **PASS** — criterion satisfied.
   - **FAIL** — criterion violated; cite `file:line` and the criterion id.
   - **NA** — criterion does not apply to this diff.

4. Report findings grouped by severity:

   - **blocker** — must fix before merge.
   - **warning** — should fix; surface to the author.
   - **nit** — optional; mention but don't block.

   Do not auto-fix beyond what a criterion explicitly authorizes.
5. Propose new criteria. If you found yourself repeating a check that is not
   in any `criteria/*.md`, or the PR establishes a new convention, draft a
   `## CRITERION` block and point the author at `criteria/README.md` to add
   it in the next weekly refresh. This is how the skill evolves.

## DON'T

- Don't invent criteria on the fly and apply them as blockers — unwritten
  rules are suggestions, recorded only as proposals in step 5.
- Don't auto-fix violations you weren't asked to fix.
- Don't skip reading `criteria/README.md`'s trigger rules.

## AFTER

Hand the findings back to the author. Blockers must be resolved before merge;
proposed new criteria go into the next weekly refresh (see
`criteria/README.md`).
