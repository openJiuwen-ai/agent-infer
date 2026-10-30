# Benchmark CLI

The `vllm` entry point installed by AgentInfer invokes BenchKit only when the arguments contain
`bench serve --agentinfer`. Every command below uses this prefix:

```text
vllm bench serve --agentinfer
```

Other `vllm` commands are delegated unchanged to the upstream vLLM CLI.

For post-run correctness, use `evaluate --result-path RUN_DIR --config YAML`
and `compare-correctness --baseline JSON --candidate JSON [--max-regressions N]`.
See [Evaluate SWE-bench](../how-to/evaluate-swebench.md) for configuration,
artifacts and exit codes.

## `prepare`

Prepare benchmark dataset inputs.

```text
vllm bench serve --agentinfer prepare swebench --output-dir PATH
```

| Argument | Required | Description |
| --- | --- | --- |
| `dataset` | Yes | Dataset type; currently only `swebench`. |
| `--output-dir PATH` | No | Directory for the index, task list, and manifest. Relative paths use the current directory. |

## `run`

Load a strictly validated YAML configuration and run one benchmark.

```text
vllm bench serve --agentinfer run --config PATH [OVERRIDES]
```

| Argument | Type or value | Configuration field |
| --- | --- | --- |
| `--config PATH` | Optional file path | Configuration file |
| `--task-num INT` | `>= 1` | `experiment.task_num` |
| `--result-dir PATH` | Directory path | `experiment.result_dir` |
| `--max-concurrency INT` | `>= 1` | `experiment.max_concurrency` |
| `--timeout INT` | Seconds, `>= 1` | `experiment.task_timeout_seconds` |
| `--dataset-name swebench_verified` | Fixed value | `dataset.name` |
| `--index-path PATH` | File path | `dataset.index_path` |
| `--selection-path PATH` | File path | `dataset.selection_path` |
| `--agent-type {claude,jiuwenswarm,dsh}` | Enum | `agent.type` |
| `--agent-profile PROFILE` | Enum | `agent.profile` |
| `--agent-executable PATH` | Path or command name | `agent.executable` |
| `--base-url URL` | HTTP URL | `backend.base_url` |
| `--model NAME` | String | `backend.model` |
| `--endpoint PATH` | API path | `backend.endpoint` |

Precedence is built-in defaults, YAML, then explicit CLI overrides. Relative YAML paths resolve from the YAML file;
CLI paths resolve from the current directory. Unknown YAML fields are rejected. For Router runs, use `--base-url`
for the transparent upstream and `--metrics-url` for vLLM metrics.

## `summarize`

Combine `summary.json` from one or more completed runs into CSV.

```text
vllm bench serve --agentinfer summarize RUN_DIR [RUN_DIR ...] [--output-csv CSV]
```

| Argument | Default | Description |
| --- | --- | --- |
| `RUN_DIR` | None; one or more | Completed result directories. |
| `--output-csv CSV` | `combined-summary.csv` | Output CSV path. |

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

Comparison accepts finalized runs with status `completed` or `failed` without additional options. Each run must
have a finish timestamp, valid summary and manifest schemas, and matching run identity and lifecycle status.
Running runs and missing summaries are still rejected.

When any included run failed, the report is diagnostic only: it shows status, available request coverage, errors,
and descriptive metrics. JSON marks `metadata.diagnostic_only=true` and records every run under
`metadata.baseline_runs` and `metadata.candidate_runs`. Confidence intervals and significance are `null` because
the executed request populations may differ. Missing planned counts remain unknown. Available Replay node errors
are read from `replay-execution.json`; the text report shows up to five per run, and JSON retains all of them.
The source artifacts are never modified.

See [Run a benchmark](../how-to/run-benchmark.md) for the procedure and
[Benchmark configuration](benchmark-config.md) for YAML fields.

## Additional options and Replay

`run` accepts an omitted `--config` and uses model defaults. `prepare --output-dir` defaults to `data/swebench`.
`run --metrics-url URL` selects the complete vLLM metrics endpoint. `summarize --output-figs-dir PATH` writes
distribution plots.

```bash
vllm bench serve --agentinfer replay --config agentinfer/agentbench/configs/replay_benchmark.yaml
vllm bench serve --agentinfer replay \
  --trace-type tracelab \
  --task-num 8 --max-concurrency 4 \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME
vllm bench serve --agentinfer replay --help
```

`replay` CLI overrides are derived from the configuration schema and map one-to-one to YAML fields:

| Argument | Type or value | Configuration field |
| --- | --- | --- |
| `--task-num INT` | `>= 1` | `experiment.task_num` |
| `--result-dir PATH` | Directory path | `experiment.result_dir` |
| `--max-concurrency INT` | `>= 1` | `experiment.max_concurrency` |
| `--task-timeout-seconds INT` | Seconds, `>= 1` | `experiment.task_timeout_seconds` |
| `--run-timeout-seconds INT` | Seconds, `>= 1`; omit for no total timeout | `experiment.run_timeout_seconds` |
| `--base-url URL` | HTTP URL | `backend.base_url` |
| `--metrics-url URL` | HTTP URL | `backend.metrics_url` |
| `--tokenizer-base-url URL` | HTTP URL; falls back to `backend.base_url` when omitted | `backend.tokenizer_base_url` |
| `--model NAME` | String | `backend.model` |
| `--endpoint PATH` | `/v1/chat/completions`, `/v1/completions`, or `/v1/messages` | `backend.endpoint` |
| `--api-key-env NAME` | Environment variable name | `backend.api_key_env` |
| `--trace-type NAME` | `agentinfer`, `inferact_codex_swebenchpro`, `agentX`, or `tracelab` | `replay.trace_type` |
| `--trace-path PATH` | File path; omitted uses the default dataset | `replay.trace_path` |
| `--trace-same-agent-gap-scale FLOAT` | `>= 0` | `replay.trace_same_agent_gap_scale` |
| `--trace-same-agent-gap-offset-seconds FLOAT` | Signed seconds | `replay.trace_same_agent_gap_offset_seconds` |
| `--interval-mode NAME` | `trace` or `lognormal` | `replay.interval_mode` |
| `--sample-seed INT` | Integer | `replay.sample_seed` |
| `--prompt-transport NAME` | `auto`, `chat_text`, or `completion_token_ids` | `replay.prompt_transport` |
| `--context-adjustment-mode NAME` | `strict` or `adaptive` | `replay.context_adjustment_mode` |
| `--context-micro-trim-max-tokens INT` | `>= 0` | `replay.context_micro_trim_max_tokens` |
| `--context-micro-trim-max-ratio FLOAT` | `0` to `1` | `replay.context_micro_trim_max_ratio` |
| `--prompt-calibration-tolerance-tokens INT` | `0` to `8` | `replay.prompt_calibration_tolerance_tokens` |
| `--request-timeout-seconds INT` | Seconds, `>= 1` | `replay.request_timeout_seconds` |

Structured fields such as `backend.chat_template_kwargs` and `replay.interval_lognormal` are not yet
CLI-overridable; set them in YAML. With `--config`, all other Replay overrides are optional. Without it, Replay
selects a packaged template from
`--trace-type` and requires `--trace-type`, `--task-num`, `--base-url`, and `--model`; `--max-concurrency` defaults to
`1`. A null `trace_path` uses the default AgentInfer, Inferact, or TraceLab dataset. Inferact and TraceLab download
pinned revisions and cache up to the first `task_num` complete records or sessions. An explicit `--trace-path`
overrides that source. See
[Trace Replay](../how-to/run-trace-replay.md).
