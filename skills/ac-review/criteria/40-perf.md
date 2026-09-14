# 40 — Performance

## TRIGGER

Applies when the diff changes cache behavior, scheduling or admission decisions,
request handling, or anything under `agentinfer/agentbench/` or
`tests/agentbench/`.

## CRITERION P1: no unmeasured perf claim

- **Severity:** blocker
- **Check:** Any claim in the PR description that the change "improves" /
  "does not regress" performance is backed by a benchmark run compared against
  baseline (per `ac-benchmark`).
- **Fix:** Run `ac-benchmark`, attach the result file, and report the
  measured deltas.

## CRITERION P2: no obvious O(n^2) in the hot path

- **Severity:** warning
- **Check:** The cache lookup / insert / evict path does not introduce an
  O(n^2) (or worse) loop over requests or entries.
- **Fix:** Restructure to use a dict/set/indexed structure for the lookup.

## CRITERION P3: baseline only updated with a measured run

- **Severity:** blocker
- **Check:** If the baseline is modified, the PR includes the result file that
  justifies the new baseline.
- **Fix:** Attach the measured run; if none justifies it, revert the
  baseline change.
