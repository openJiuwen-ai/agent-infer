---
name: agentbench-investigate-regression
description: Use when an AgentBench daily, baseline/candidate pair, or dashboard metric appears slower or regressed; validates finalized source artifacts, separates workload amplification from serving cost, localizes task/session tails, and produces an evidence-ranked diagnosis before any rerun or release claim.
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
---

# agentbench-investigate-regression

Investigate an apparent AgentBench regression from source artifacts. A longer run
is not automatically a serving regression: first separate how much work the agent
did from how quickly the backend served that work.

## WHEN TO INVOKE

- A daily benchmark wall time, throughput, latency, completion count, or cache
  metric looks worse.
- Someone asks why one baseline/candidate pair disagrees with nearby runs.
- A performance claim needs task/session-level evidence before a rerun or PR
  decision.

Do NOT invoke to launch a benchmark, update a baseline, publish Kanban data, or
review an implementation diff without benchmark artifacts. Use `ac-benchmark`
to run a new measurement and `ac-review` to review a PR.

## INPUTS

Core evidence for pair and task analysis:

- finalized baseline and candidate run directories for the same agent;
- the expected shape/mode matrix and the questioned metric;
- `manifest.json`, `summary.json`, `sessions.csv`, `task_index.json`, and
  `tasks/*/result.json` in each run.

Use additional evidence only when the diagnostic question needs it:

- `requests.jsonl` for request-level attribution or trajectory expansion;
- `agents.csv` and `distribution_samples.csv` for actor and distribution detail;
- service/launcher logs for backend lifecycle, queue, error, or launch claims;
- nearby comparable pairs for variance context;
- launcher/config and commit history for change attribution;
- agent transcripts for suspected tool loops or post-completion behavior.

Missing optional evidence narrows the supported claims; it does not invalidate
unrelated workload or task-level facts. List every expected arm before analysis
and state omitted or unavailable arms explicitly. Never silently substitute CC
for JiuwenSwarm or vice versa.

## WORKFLOW

### 1. Freeze the target

Record exact run IDs, paths, host, commit, model, agent/profile, task selection
and order, concurrency, task timeout, TP, vLLM options, scheduler/policy, and
service-log paths. Preserve historical artifacts unchanged.

### 2. Validate core evidence

Fail closed when run identity, finalization, or task identity is inconsistent.
Verify:

- manifest and summary both identify the directory and report `completed`;
- task identities agree across task index, session CSV, and per-task results;
- the evidence needed for the specific claim is available and healthy;
- cold status is evidence-based. If it cannot be confirmed, retain the warning.

Use `docs/en/reference/run-artifacts.md` and
`design/module/benchmarking/artifacts-and-evaluation.md` as metric and
evidence contracts. Prefer repository-native
`vllm bench serve --agentinfer compare` for the summary-level comparison. Use
optional evidence only for the claim it supports; for example, missing actor
diagnostics must not erase task-duration evidence.

### 3. Check comparability

Compare same-agent pairs first. Require equality for task identities/order,
commit, model, profile, task count, concurrency, timeout, and material benchmark
configuration. Check TP, hardware, endpoint, serve flags, environment, and CLI
provenance when the captured evidence exposes them; otherwise record the gap
rather than manufacturing equality. A dirty checkout is a warning even when both
arms report the same dirty state.

A single pair has low statistical power. Nearby runs are variance context, not
identical repetitions unless every relevant field and cold-state contract
matches.

### 4. Decompose workload

Measure candidate-versus-baseline changes in:

- completed/failed tasks and patches;
- request count and successful/failed requests;
- input/output tokens;
- per-session agent count and trajectory length;
- termination reasons and outcome transitions.

Large request/token differences mean the agent executed a different amount of
work. Do not attribute raw wall-time differences directly to the scheduler.

### 5. Decompose serving cost

Independently compare:

- request, input-token, and output-token throughput;
- latency and TTFT mean/p50/p95/p99;
- vLLM queue, prefill, decode, and inference means;
- service restarts, OOM, tracebacks, request failures, and scheduler warnings.

A lower decode/inference mean contradicts a uniform backend slowdown even when
run wall time increases. Higher queue time or p99 still establishes a real tail
problem and must not be hidden by a better mean.

### 6. Localize task/session tails

Join `sessions.csv` by `instance_id`, with `task_index.json` and per-task
`result.json` authoritative for order/outcome. Report:

- candidate-slower versus candidate-faster task counts;
- median paired task-duration delta;
- top duration, request, and token deltas;
- contribution of top stragglers to positive duration deltas;
- completion/termination transitions;
- whether the run wall time is controlled by one or a few late tasks.

For repeatable pair/task tables, optionally use the packaged helper
`analyze_benchmark_regression.py` in this skill directory. It is a convenience,
not a required pipeline: inspect source artifacts directly when the question
exceeds its summary and task-level scope.

### 7. Treat cache metrics as supporting evidence

Keep API cache rates distinct from vLLM prefix/prompt-token rates. Preserve known
counter-accounting discontinuities and compare query/input-token shapes when
relevant. A higher cache hit rate alone does not prove causality or an end-to-end
performance gain.

### 8. Inspect deeper evidence only as needed

Use the smallest source that can answer the remaining question:

- `requests.jsonl` for request ownership, timing, failures, and token growth;
- `agents.csv` for actor-level attribution;
- `distribution_samples.csv` for distribution-level drill-down;
- task transcripts for tool/agent trajectory mechanisms;
- service and launcher logs for backend lifecycle and launch configuration;
- code and commit history for a candidate implementation cause.

Missing optional evidence narrows the supported claims; it does not invalidate
facts established by healthy core artifacts. Only after artifact decomposition,
inspect changes in launch commands, config, agent/runtime versions,
scheduler/middleware/controller, vLLM, and host state. Link each claim to a
source line, commit, or log. Do not invent a cause from a coincident commit.

### 9. Classify and recommend

Classify each claim:

- `PROVEN`: directly computed or observed in authoritative evidence;
- `SUPPORTED`: multiple facts support it, but causal isolation is incomplete;
- `UNKNOWN`: evidence cannot decide it.

Use an overall verdict such as `WORKLOAD_AMPLIFICATION_WITH_TAIL`,
`SERVING_REGRESSION`, `MIXED`, or `INCONCLUSIVE`.

For causal follow-up, prefer repeated cold pairs and controlled replay of the
dominant task outliers. Hold commit, task order, model, agent/profile,
concurrency, timeout, TP, hardware, and service state fixed.

## OUTPUT

Write an append-only investigation report plus a machine-readable JSON evidence
companion. Include:

1. executive verdict;
2. exact source paths and comparability gates;
3. workload decomposition;
4. serving decomposition;
5. task/session outliers and termination transitions;
6. nearby-run context;
7. log/config/code evidence;
8. `PROVEN` / `SUPPORTED` / `UNKNOWN` claims;
9. interpretation boundaries and the minimum next experiment.

Correctness is external: task completion and patch presence are not SWE-bench
resolution.

## AFTER

The investigation itself is read-only and ends at the report. Any code, config,
or harness change motivated by the findings is a separate task: implement it
without this skill, then validate it with `ac-review` before opening the PR.

## SAFETY AND DON'T

- Read-only by default. Do not launch jobs, stop services, clean remote state,
  rewrite artifacts, update Kanban, or publish comments without separate
  authorization.
- Do not compare different agents as a baseline/candidate pair.
- Do not call a wall-time increase a scheduler regression before normalizing
  workload and localizing tails.
- Do not erase cold-start, missing-evidence, reset, or low-power warnings.
- Do not derive final causal claims only from dashboard summaries when source
  artifacts are available.
- Do not create one-off scripts for a single daily or outlier. Add durable
  automation only when it is generic, parameterized, and repeatedly useful.
