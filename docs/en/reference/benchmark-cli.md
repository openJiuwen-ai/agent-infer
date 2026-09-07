# Benchmark CLI

The `vllm` entry point installed by AgentInfer invokes BenchKit only when the arguments contain
`bench serve --agentinfer`. Every command below uses this prefix:

```text
vllm bench serve --agentinfer
```

Other `vllm` commands are delegated unchanged to the upstream vLLM CLI.

## `prepare`

Prepare benchmark dataset inputs.

```text
vllm bench serve --agentinfer prepare swebench --output-dir PATH
```

| Argument | Required | Description |
| --- | --- | --- |
| `dataset` | Yes | Dataset type; currently only `swebench`. |
| `--output-dir PATH` | Yes | Directory for the index, task list, and manifest. Relative paths use the current directory. |

## `run`

Load a strictly validated YAML configuration and run one benchmark.

```text
vllm bench serve --agentinfer run --config PATH [OVERRIDES]
```

| Argument | Type or value | Configuration field |
| --- | --- | --- |
| `--config PATH` | File path, required | Configuration file |
| `--task-num INT` | `>= 1` | `experiment.task_num` |
| `--result-dir PATH` | Directory path | `experiment.result_dir` |
| `--max-concurrency INT` | `>= 1` | `experiment.max_concurrency` |
| `--timeout INT` | Seconds, `>= 1` | `experiment.task_timeout_seconds` |
| `--dataset-name swebench_verified` | Fixed value | `dataset.name` |
| `--index-path PATH` | File path | `dataset.index_path` |
| `--selection-path PATH` | File path | `dataset.selection_path` |
| `--agent-type claude` | Fixed value | `agent.type` |
| `--agent-profile {single,plan-subagent}` | Enum | `agent.profile` |
| `--agent-executable PATH` | Path or command name | `agent.executable` |
| `--base-url URL` | HTTP URL | `backend.base_url` |
| `--model NAME` | String | `backend.model` |
| `--endpoint /v1/messages` | Fixed value | `backend.endpoint` |
| `--enabled` / `--no-enabled` | Boolean | `router.enabled` |
| `--router-url URL` | HTTP URL | `router.base_url` |

Precedence is built-in defaults, YAML, then explicit CLI overrides. Relative YAML paths resolve from the YAML file;
CLI paths resolve from the current directory. Unknown YAML fields are rejected. Router mode requires both `--enabled`
and `--router-url`.

## `summarize`

Combine `summary.json` from one or more completed runs into CSV.

```text
vllm bench serve --agentinfer summarize RUN_DIR [RUN_DIR ...] [--output CSV]
```

| Argument | Default | Description |
| --- | --- | --- |
| `RUN_DIR` | None; one or more | Completed result directories. |
| `--output CSV` | `combined-summary.csv` | Output CSV path. |

## `compare`

Compare a set of baseline runs with a set of candidate runs.

```text
vllm bench serve --agentinfer compare --baseline DIR [DIR ...] --candidate DIR [DIR ...] [OPTIONS]
```

| Argument | Default | Description |
| --- | --- | --- |
| `--baseline DIR [DIR ...]` | None; required | One or more finalized baseline directories. |
| `--candidate DIR [DIR ...]` | None; required | One or more finalized candidate directories. |
| `--confidence FLOAT` | `0.95` | Confidence level used by the statistical report. |
| `--json` | `false` | Emit machine-readable JSON instead of the text report. |

See [Run a benchmark](../how-to/run-benchmark.md) for the procedure and
[Benchmark configuration](benchmark-config.md) for YAML fields.
