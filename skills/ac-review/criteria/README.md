# `ac-review` criteria — evolution protocol

The review **process** lives in `../SKILL.md` and is stable. The actual checks
live in this directory, one category per file, and evolve as the repo develops.

## File conventions

- One category per file, named `<NN>-<topic>.md` where `NN` is a zero-padded
  number. The number sets evaluation order and makes gaps visible.
- Every file starts with a `## TRIGGER` block describing which diffs it
  applies to, followed by one or more `## CRITERION` blocks.
- A `## CRITERION` block has the shape:

  ````markdown
  ## CRITERION <id>: <one-line summary>

  - **Severity:** blocker | warning | nit
  - **Check:** <how to evaluate it against a diff>
  - **Fix:** <what the author should do if it fails>
  ````

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
