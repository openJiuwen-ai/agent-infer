# 基准命令行参考

AgentInfer 安装的 `vllm` 入口仅在参数中出现 `bench serve --agentinfer` 时调用 BenchKit。以下命令均使用前缀：

```text
vllm bench serve --agentinfer
```

未列出的 `vllm` 命令会原样委托给上游 vLLM CLI。

## `prepare`

准备基准数据集输入。

```text
vllm bench serve --agentinfer prepare swebench --output-dir PATH
```

| 参数 | 必需 | 说明 |
| --- | --- | --- |
| `dataset` | 是 | 数据集类型；当前仅支持 `swebench`。 |
| `--output-dir PATH` | 否 | 写入索引、任务列表和清单的目录。相对路径以当前工作目录为基准。 |

## `run`

加载严格校验的 YAML 配置并运行一次基准。

```text
vllm bench serve --agentinfer run --config PATH [OVERRIDES]
```

| 参数 | 类型或取值 | 映射字段 |
| --- | --- | --- |
| `--config PATH` | 文件路径，可选 | 配置文件 |
| `--task-num INT` | `>= 1` | `experiment.task_num` |
| `--result-dir PATH` | 目录路径 | `experiment.result_dir` |
| `--max-concurrency INT` | `>= 1` | `experiment.max_concurrency` |
| `--timeout INT` | 秒，`>= 1` | `experiment.task_timeout_seconds` |
| `--dataset-name swebench_verified` | 固定值 | `dataset.name` |
| `--index-path PATH` | 文件路径 | `dataset.index_path` |
| `--selection-path PATH` | 文件路径 | `dataset.selection_path` |
| `--agent-type {claude,jiuwenswarm,dsh}` | 枚举 | `agent.type` |
| `--agent-profile PROFILE` | 枚举 | `agent.profile` |
| `--agent-executable PATH` | 路径或命令名 | `agent.executable` |
| `--base-url URL` | HTTP URL | `backend.base_url` |
| `--model NAME` | 字符串 | `backend.model` |
| `--endpoint PATH` | API 路径 | `backend.endpoint` |

配置优先级为内置默认值、YAML、显式 CLI 覆盖。YAML 中的相对路径以 YAML 文件目录为基准；CLI 路径以当前
工作目录为基准。未知 YAML 字段会被拒绝。Router 使用 `--base-url` 指向透明转发端点，`--metrics-url` 指向 vLLM 指标端点。

## `summarize`

将一个或多个已完成运行的 `summary.json` 合并为 CSV。

```text
vllm bench serve --agentinfer summarize RUN_DIR [RUN_DIR ...] [--output-csv CSV]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `RUN_DIR` | 无，至少一个 | 已完成的结果目录。 |
| `--output-csv CSV` | `combined-summary.csv` | 输出 CSV 路径。 |

## `compare`

比较一组基线运行和一组候选运行。

```text
vllm bench serve --agentinfer compare --baseline DIR [DIR ...] --candidate DIR [DIR ...] [OPTIONS]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--baseline DIR [DIR ...]` | 无，必需 | 一个或多个已完成的基线目录。 |
| `--candidate DIR [DIR ...]` | 无，必需 | 一个或多个已完成的候选目录。 |
| `--confidence FLOAT` | `0.95` | 统计报告使用的置信水平。 |
| `--json` | `false` | 输出机器可读 JSON，而非文本报告。 |

比较默认支持状态为 `completed` 或 `failed` 的已结束运行，无需额外参数。
仍要求结束时间、有效的 summary 和 manifest 格式，以及一致的运行标识和生命周期状态。
尚在运行或缺少 summary 的目录仍不能比较。

只要包含失败运行，报告就仅供诊断：展示状态、可用的请求覆盖信息、错误和描述性指标。
JSON 中 `metadata.diagnostic_only=true`，每次运行的诊断信息分别位于
`metadata.baseline_runs` 和 `metadata.candidate_runs`。由于实际执行的请求集合可能不同，
置信区间和显著性字段为 `null`；缺失的计划请求数保持未知。Replay 节点错误从
`replay-execution.json` 读取，文本报告每次运行最多展示五条，JSON 保留全部条目。
比较过程不会修改原始产物。

运行步骤见[运行基准测试](../how-to/run-benchmark.md)，配置字段见[基准配置参考](benchmark-config.md)。

## 其他参数与 Replay

`run` 可省略 `--config`，此时使用模型默认值。`prepare --output-dir` 默认是 `data/swebench`。
`run --metrics-url URL` 指定完整的 vLLM 指标端点。`summarize --output-figs-dir PATH` 输出分布图。

```bash
vllm bench serve --agentinfer replay --config agentinfer/agentbench/configs/replay_agentinfer.yaml --task-num 8
vllm bench serve --agentinfer replay \
  --trace-type tracelab \
  --task-num 8 --max-concurrency 4 \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME
vllm bench serve --agentinfer replay --help
```

`replay` 的 CLI 覆盖参数由配置 schema 派生，与 YAML 字段一一对应：

| 参数 | 类型或取值 | 映射字段 |
| --- | --- | --- |
| `--task-num INT` | `>= 1` | `experiment.task_num` |
| `--result-dir PATH` | 目录路径 | `experiment.result_dir` |
| `--max-concurrency INT` | `>= 1` | `experiment.max_concurrency` |
| `--task-timeout-seconds INT` | 秒，`>= 1` | `experiment.task_timeout_seconds` |
| `--run-timeout-seconds INT` | 秒，`>= 1`；省略表示不设总超时 | `experiment.run_timeout_seconds` |
| `--base-url URL` | HTTP URL | `backend.base_url` |
| `--metrics-url URL` | HTTP URL | `backend.metrics_url` |
| `--tokenizer-base-url URL` | HTTP URL；省略时回退到 `backend.base_url` | `backend.tokenizer_base_url` |
| `--model NAME` | 字符串 | `backend.model` |
| `--endpoint PATH` | `/v1/chat/completions`、`/v1/completions` 或 `/v1/messages` | `backend.endpoint` |
| `--api-key-env NAME` | 环境变量名 | `backend.api_key_env` |
| `--trace-type NAME` | `agentinfer`、`inferact_codex_swebenchpro`、`agentX` 或 `tracelab` | `replay.trace_type` |
| `--trace-path PATH` | 文件路径；省略时使用对应默认数据集 | `replay.trace_path` |
| `--trace-same-agent-gap-scale FLOAT` | `>= 0` | `replay.trace_same_agent_gap_scale` |
| `--trace-same-agent-gap-offset-seconds FLOAT` | 有符号秒 | `replay.trace_same_agent_gap_offset_seconds` |
| `--interval-mode NAME` | `trace` 或 `lognormal` | `replay.interval_mode` |
| `--sample-seed INT` | 整数 | `replay.sample_seed` |
| `--prompt-transport NAME` | `auto`、`chat_text` 或 `completion_token_ids` | `replay.prompt_transport` |
| `--context-adjustment-mode NAME` | `strict` 或 `adaptive` | `replay.context_adjustment_mode` |
| `--context-micro-trim-max-tokens INT` | `>= 0` | `replay.context_micro_trim_max_tokens` |
| `--context-micro-trim-max-ratio FLOAT` | `0` 到 `1` | `replay.context_micro_trim_max_ratio` |
| `--prompt-calibration-tolerance-tokens INT` | `0` 到 `8` | `replay.prompt_calibration_tolerance_tokens` |
| `--request-timeout-seconds INT` | 秒，`>= 1` | `replay.request_timeout_seconds` |

`backend.chat_template_kwargs`、`replay.interval_lognormal` 等结构化字段暂不支持 CLI 覆盖，
请在 YAML 中配置。提供 `--config` 时，其他 Replay 覆盖参数均为可选。未提供时，Replay 根据 `--trace-type`
选择内置模板，
并要求显式提供 `--trace-type`、`--task-num`、`--base-url` 和 `--model`；`--max-concurrency` 默认为 `1`。
AgentInfer、Inferact 或 TraceLab 内置模板的 `trace_path` 为 `null` 时会使用对应默认数据集；Inferact 和
TraceLab 下载固定版本，并最多缓存前 `task_num` 个完整记录或 Session。显式 `--trace-path` 会覆盖默认
数据源。详见
[回放指南](../how-to/run-trace-replay.md)。
