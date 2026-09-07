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
| `--output-dir PATH` | 是 | 写入索引、任务列表和清单的目录。相对路径以当前工作目录为基准。 |

## `run`

加载严格校验的 YAML 配置并运行一次基准。

```text
vllm bench serve --agentinfer run --config PATH [OVERRIDES]
```

| 参数 | 类型或取值 | 映射字段 |
| --- | --- | --- |
| `--config PATH` | 文件路径，必需 | 配置文件 |
| `--task-num INT` | `>= 1` | `experiment.task_num` |
| `--result-dir PATH` | 目录路径 | `experiment.result_dir` |
| `--max-concurrency INT` | `>= 1` | `experiment.max_concurrency` |
| `--timeout INT` | 秒，`>= 1` | `experiment.task_timeout_seconds` |
| `--dataset-name swebench_verified` | 固定值 | `dataset.name` |
| `--index-path PATH` | 文件路径 | `dataset.index_path` |
| `--selection-path PATH` | 文件路径 | `dataset.selection_path` |
| `--agent-type claude` | 固定值 | `agent.type` |
| `--agent-profile {single,plan-subagent}` | 枚举 | `agent.profile` |
| `--agent-executable PATH` | 路径或命令名 | `agent.executable` |
| `--base-url URL` | HTTP URL | `backend.base_url` |
| `--model NAME` | 字符串 | `backend.model` |
| `--endpoint /v1/messages` | 固定值 | `backend.endpoint` |
| `--enabled` / `--no-enabled` | 布尔值 | `router.enabled` |
| `--router-url URL` | HTTP URL | `router.base_url` |

配置优先级为内置默认值、YAML、显式 CLI 覆盖。YAML 中的相对路径以 YAML 文件目录为基准；CLI 路径以当前
工作目录为基准。未知 YAML 字段会被拒绝。启用 Router 时必须同时传入 `--enabled` 和 `--router-url`。

## `summarize`

将一个或多个已完成运行的 `summary.json` 合并为 CSV。

```text
vllm bench serve --agentinfer summarize RUN_DIR [RUN_DIR ...] [--output CSV]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `RUN_DIR` | 无，至少一个 | 已完成的结果目录。 |
| `--output CSV` | `combined-summary.csv` | 输出 CSV 路径。 |

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

运行步骤见[运行基准测试](../how-to/run-benchmark.md)，配置字段见[基准配置参考](benchmark-config.md)。
