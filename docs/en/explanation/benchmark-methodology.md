# Benchmark Methodology

The AgentInfer benchmark answers one constrained question: under the same agent workload and deployment conditions,
how does a candidate scheduler change runtime, request, and Prefix Cache metrics relative to a pinned vLLM baseline?
It does not equate agent-process completion with task correctness.

## Experiment Arms

```text
baseline:  Claude Code -> Request Proxy -> vLLM AsyncScheduler
candidate: Claude Code -> Request Proxy -> vLLM AgentCacheAsyncSchedulerBridge
```

Both arms use the same Request Proxy, BenchKit runner, dataset selection, and agent profile. The scheduling difference
exists only in vLLM startup configuration so request observation or agent-driving differences are not mistaken for a
scheduler improvement.

## Evidence Layers

Collectors capture external raw evidence and availability. Metrics turn source-owned facts into normalized values.
Artifacts validate and serialize task and run contracts. Comparison reads only finalized summaries. These layers do
not duplicate one another's formulas.

Request traces, service metrics, task outcomes, patches, environment, and source-control state remain authoritative.
Missing values must be unavailable or not applicable, never a successful numeric zero.

## Correctness Boundary

`completed` means the agent execution flow completed, not that its patch is correct. SWE-bench correctness is separate
post-processing:

```text
model.patch
  -> check out the task repository at base_commit
  -> apply the patch
  -> run FAIL_TO_PASS and PASS_TO_PASS tests
  -> record resolved or unresolved
```

A release claim requires both runtime and cache comparison evidence and external correctness evidence.

## Cold-Start Requirement

Command order alone does not prove cold state. Each arm must use a fresh service process, with service-start evidence
or `evidence/vllm_metrics_start.prom` showing that cumulative metrics and Prefix Cache state were not inherited. The
report must warn when evidence is insufficient.

Stop vLLM completely after the baseline before starting the candidate. Result directories, tmux sessions, and
lifecycle sockets must not be reused between arms.

## Fair Comparison

Each paired experiment must keep these conditions identical:

- AgentInfer commit, model, and hardware.
- Task set, task order, and agent profile.
- Concurrency and per-task and whole-run timeouts.
- Tensor parallelism, vLLM arguments, and service endpoint.
- Request Proxy and evidence-collection configuration.

One baseline and candidate pair supports a directional smoke test. Release decisions should use repeated paired cold
runs and retain service logs, raw metrics, and task artifacts for both arms.

See [Run a benchmark](../how-to/run-benchmark.md) for the procedure and
[Run artifacts](../reference/run-artifacts.md) for evidence files.

## Multiple agents and replay boundaries

BenchKit supports Claude Code, JiuwenSwarm, and DSH, but paired runs must use the same runtime and profile. The
request proxy forwards transparently and prefers canonical identity metadata; the native DSH session header is a
fallback and cannot establish subagent roles by itself.

Trace Replay reconstructs request workloads and dependencies, not original tool semantics or patch correctness.
Comparisons must also fix the trace, sample seed, and synthetic prefix budgets.
