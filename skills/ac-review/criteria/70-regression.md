# 70 — Performance regression and validation evidence

## TRIGGER

Applies when a diff can plausibly change measured cache or serving performance:
cache behavior; scheduling, admission, queue ordering, or progress-TTL decisions;
serving hot-path request adaptation; benchmark/test utility behavior; benchmark
orchestration or metrics; or configs/helpers that affect measured runs.
Documentation-only changes, refactors with no hot-path behavior change, and
request-handling fixes outside measured behavior do not trigger a full benchmark
unless the PR makes a performance claim.

Before applying path-specific checks, inventory current repo paths with read-only
search. `70-regression.md` owns command, artifact, benchmark, and fallback-smoke
evidence. PR/RFC linkage and reviewability belong in `60-workflow.md`.

## CRITERION PR1: validation evidence manifest available

- **Severity:** blocker
- **Check:** Benchmark, performance, correctness, or test-utility claims include
  reviewer-facing evidence with commit, command, config, host/env, output path,
  raw logs/results, parsed summary, whether full benchmark/E2E ran, and fallback
  smoke/regression evidence when full validation did not run.
- **Fix:** Add a validation evidence block to the PR body or handoff:

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

## CRITERION PR2: proportional and fair comparison available

- **Severity:** blocker
- **Check:** Performance claims and performance-sensitive behavior changes include
  baseline and candidate evidence proportional to scope. Directional smoke tests
  may support development feedback; release or merge claims require the intended
  mode/shape matrix, explicit omissions, and matching cold runs with the same
  commit, model, ordered task selection, agent profile, concurrency, timeouts,
  tensor parallelism, vLLM options, and hardware.
- **Fix:** Declare the claim and experiment scope, then run comparable validation
  or remove the claim until evidence exists.

## CRITERION PR3: cold and reproducible evidence available

- **Severity:** blocker
- **Check:** For release or merge performance evidence, vLLM is fully restarted
  between arms, cold state is confirmed from source artifacts, and manifests,
  summaries, request traces, source-control evidence, and service logs are
  retained or unavailable evidence is disclosed. Release decisions use repeated
  cold runs.
- **Fix:** Rerun missing or warm arms, retain source artifacts, and compare all
  repeated run directories instead of manually transcribing results.

## CRITERION PR4: no correctness regression

- **Severity:** blocker
- **Check:** A performance-sensitive behavior change includes comparable
  SWE-bench-compatible correctness evaluation for baseline and candidate, and
  the candidate resolved count does not decrease unless the reviewer explicitly
  accepts the blocker and follow-up. `completed` is execution status, never a
  correctness result.
- **Fix:** Evaluate patches against FAIL_TO_PASS and PASS_TO_PASS tests, report
  resolved/unresolved separately, and investigate any resolved-count regression.

## CRITERION PR5: benchmark claims are artifact-backed

- **Severity:** blocker
- **Check:** Benchmark/E2E claims cite raw artifacts/logs and parsed summaries.
  Reports cover metrics relevant to the claim, such as end-to-end wall time,
  prefix hit rate, TTFT, task execution status, evidence availability, source
  health, and correctness. Tables and plots are derived from artifacts.
- **Fix:** Attach or cite artifact paths and parse them directly for summaries.

## CRITERION PR6: exact reproduction command is present

- **Severity:** warning
- **Check:** The PR includes exact commands, config paths, commit or branch,
  service endpoint/port when relevant, host/env, and output paths needed for a
  reviewer to reproduce or inspect validation.
- **Fix:** Add the exact command and environment details, or explain why they are
  unavailable.

## CRITERION PR7: full benchmark not run has fallback evidence

- **Severity:** warning
- **Check:** If full benchmark/E2E validation was not run, the PR explains why and
  includes a smaller reviewer-runnable smoke/regression substitute.
- **Fix:** Add the not-run reason and fallback command/result.

## CRITERION PR8: test utilities have smoke paths and readable failures

- **Severity:** warning
- **Check:** New or changed test utilities provide a reviewer-runnable smoke path
  and failures that distinguish code regression, benchmark/data issue,
  environment/host issue, and missing dependency/config issue where practical.
- **Fix:** Add a smoke command and improve failure messages or result summaries.

## CRITERION PR9: measurable TTFT regression checked when relevant

- **Severity:** warning
- **Check:** If the change touches scheduling, admission, queue ordering, or
  progress-TTL factors, cold/new-session TTFT is reported against baseline or
  explicitly deferred.
- **Fix:** Add TTFT metrics to validation evidence or document why they are
  deferred.
