# AgentInfer Benchmark Quickstart

This guide runs the same Claude Code workload against two cold vLLM deployments:

```text
baseline:  Claude Code -> Request Proxy -> vLLM AsyncScheduler
candidate: Claude Code -> Request Proxy -> vLLM AgentCacheAsyncSchedulerBridge
```

BenchKit records runtime and cache evidence. SWE-bench correctness evaluation is a separate post-processing step.

## Requirements

Install Python 3.10+, vLLM 0.23.0, Claude Code, tmux, Git, and an AgentInfer runtime build that provides
`AgentCacheAsyncSchedulerBridge`, `AgentCacheLifecycleMiddleware`, and the progress-TTL controller in the same virtual
environment. The model must support Anthropic `POST /v1/messages`.

```bash
source /path/to/vllm/.venv/bin/activate
python -m pip install -e /path/to/AgentInfer
```

Explicit `vllm bench serve --agentinfer` commands go to BenchKit; ordinary `vllm` commands continue to upstream vLLM.

## Prepare SWE-bench inputs

```bash
vllm bench serve --agentinfer prepare swebench \
  --output-dir agentinfer/agentbench/data/swebench
```

This creates `instances.jsonl`, `task-lists/default.txt`, and `manifest.json`.

## Run the baseline

Start a fresh upstream vLLM server:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 2 \
  --enable-prompt-tokens-details \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000 \
  --scheduler-cls vllm.v1.core.sched.async_scheduler.AsyncScheduler
```

In another terminal, run BenchKit:

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --task-num 64 \
  --max-concurrency 32 \
  --agent-executable claude \
  --agent-profile plan-subagent \
  --result-dir results/vllm/run1
```

The YAML config supplies values not overridden on the command line, including:

```yaml
backend:
  base_url: http://127.0.0.1:8000
  model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
  endpoint: /v1/messages
```

Use `plan-subagent` only as a non-root user or with an approved Claude Code permission setup because this profile uses
`bypassPermissions`.

Stop the baseline vLLM process completely before starting the candidate.

## Run the candidate

Choose an unused lifecycle socket:

```bash
export LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle.sock
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET="$LIFECYCLE_SOCKET"
```

Start a fresh vLLM server with the AgentInfer scheduler bridge:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 2 \
  --enable-prompt-tokens-details \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000 \
  --scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware \
  --additional-config \
  '{"agentcache":{"controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller"}}'
```

Optional runtime diagnostics use the `additional_config.agentcache.observability` namespace:

```json
{
  "observability": {
    "enabled": true
  }
}
```

| Parameter | Default | Purpose |
| --- | --- | --- |
| `enabled` | `false` | Emit decision-event logs and periodic Progress-TTL state diagnostics. |
| `log_interval_seconds` | `5.0` | Set the minimum interval between periodic diagnostics without changing the scheduling interval. |

AgentInfer reuses vLLM's existing handlers, so do not set `VLLM_LOGGING_CONFIG_PATH`.

Native Uvicorn startup and access logs remain enabled unless `--disable-uvicorn-access-log` is passed.

In another terminal, run the same BenchKit workload:

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --task-num 64 \
  --max-concurrency 32 \
  --agent-executable claude \
  --agent-profile plan-subagent \
  --result-dir results/agentinfer/run1
```

Stop the candidate vLLM process after the run.

## Compare the runs

```bash
vllm bench serve --agentinfer compare \
  --baseline results/vllm/run1 \
  --candidate results/agentinfer/run1
```

For repeated trials, pass multiple run directories per side:

```bash
vllm bench serve --agentinfer compare \
  --baseline results/vllm/run1 results/vllm/run2 results/vllm/run3 \
  --candidate results/agentinfer/run1 results/agentinfer/run2 results/agentinfer/run3
```

Add `--json` for machine-readable output.

## Combine run summaries

Export one or more completed runs as a CSV for side-by-side comparison:

```bash
vllm bench serve --agentinfer summarize \
  results/agentinfer/run1 results/agentinfer/run2 results/agentinfer/run3
```

The command writes `combined-summary.csv` by default. Use `--output <csv>` to choose another path.

## CLI reference

```text
vllm bench serve --agentinfer prepare swebench --output-dir <data-dir>
vllm bench serve --agentinfer run --config <yaml> [overrides]
vllm bench serve --agentinfer summarize <run-dir...> [--output <csv>]
vllm bench serve --agentinfer compare --baseline <run-dir...> --candidate <run-dir...>
```

Configuration precedence is defaults, then YAML, then CLI. YAML paths resolve relative to the YAML file; CLI paths
resolve relative to the current directory. Unknown fields are rejected.

## YAML config guide

The Scheduler baseline and candidate use the same config because their difference is in how vLLM is started. The config
keeps Router disabled.

```yaml
experiment:
  task_num: null
  result_dir: ../results
  max_concurrency: 1
  task_timeout_seconds: 3600
  run_timeout_seconds: null
  patch_flush_seconds: 15
  prompt_delivery_timeout_seconds: 30

dataset:
  name: swebench_verified
  index_path: ../data/swebench/instances.jsonl
  selection_path: ../data/swebench/task-lists/default.txt
  cache_dir: ../data/swebench/repo-cache

agent:
  type: claude
  profile: plan-subagent
  executable: claude

backend:
  type: vllm
  base_url: http://127.0.0.1:8000
  model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
  endpoint: /v1/messages
  api_key_env: null

request_proxy:
  listen_url: http://127.0.0.1:0
  request_timeout_seconds: 3600
  startup_timeout_seconds: 120
  shutdown_timeout_seconds: 30

router:
  enabled: false
  base_url: null
  control_timeout_seconds: 10
```

Important fields:

| Field | Meaning |
| --- | --- |
| `experiment.task_num` | First N selected tasks. `null` means the full selection. |
| `experiment.result_dir` | Output directory; it must not already exist. |
| `experiment.max_concurrency` | Number of concurrent agent sessions. |
| `dataset.index_path` | SWE-bench task metadata in JSONL format. |
| `dataset.selection_path` | Ordered, newline-separated task IDs. |
| `dataset.cache_dir` | Shared Git repository cache. |
| `agent.profile` | `single` or `plan-subagent`. |
| `backend.base_url` | Direct vLLM API base URL. |
| `backend.model` | Model name sent through Claude Code requests. |
| `request_proxy.listen_url` | Loopback address for the transparent Request Proxy. |
| `router.enabled` | Keep `false` for Scheduler-only comparisons. |

## Output artifacts

A run writes artifacts under:

```text
<result_dir>/
  manifest.json
  summary.json
  requests.jsonl
  evidence/
    environment.json
    source_control.json
    vllm_metrics_start.prom
    vllm_metrics_end.prom
  tasks/<instance_id>/
    result.json
    model.patch
    transcript.jsonl
    claude-settings.json
    claude-launch-command.txt
    terminal-*.log
  workspaces/<instance_id>/
```

Evidence that could not be collected is recorded as unavailable in `manifest.json` or `summary.json`; an optional file
may therefore be absent.

Key artifacts:

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Run configuration, lifecycle status, and evidence availability. |
| `summary.json` | Normalized task, request, vLLM, correctness, and source-health metrics. |
| `requests.jsonl` | One immutable row per proxied LLM request. |
| `evidence/vllm_metrics_*.prom` | Raw vLLM metrics at the start and end of the run. |
| `tasks/<instance_id>/result.json` | Agent outcome, termination reason, topology, and patch status. |
| `tasks/<instance_id>/model.patch` | Agent-produced patch, when present. |

## Correctness evaluation

`completed` means the agent execution completed; it is not a SWE-bench correctness result. Correctness requires a
separate SWE-bench-compatible evaluation:

```text
model.patch
  -> checkout the task repository at base_commit
  -> apply the patch
  -> run FAIL_TO_PASS and PASS_TO_PASS tests
  -> record resolved or unresolved
```

## Fair comparison guidance

Keep the commit, model, task list and order, agent profile, concurrency, timeouts, tensor parallelism, vLLM options, and
hardware identical. Fully stop vLLM between arms so cumulative metrics and Prefix Cache state reset. Retain both service
logs.

One baseline and one candidate run are enough for a directional smoke test. Use repeated cold runs for release
decisions. Compare reports a warning when cold state cannot be confirmed from `evidence/vllm_metrics_start.prom`.
