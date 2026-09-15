# 运行 Trace Replay

Replay 从请求 trace 恢复会话、Agent、请求依赖、间隔和 token 目标，向后端发送请求并记录性能证据。
它不启动原始 Agent，不执行工具或 SWE-bench 正确性测试。合成提示词回放不代表原始文本或性能完全等价。

## 准备后端

安装项目后，启动支持目标模型的 vLLM 服务。后端需要提供 `/tokenize`、`/detokenize`、`/metrics` 和配置的推理端点。
通过 Router 访问时，在 YAML 中设置 `backend.tokenizer_base_url` 为分词服务地址，
`backend.metrics_url` 为完整的 vLLM 指标 URL。使用 `/v1/messages` 时，启动 vLLM 时加载
`agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware`，以传递身份和采样参数。
具体服务参数见[接入 vLLM](integrate-vllm.md)。

## 回放内置样例

在仓库根目录执行，使用实际后端模型名替换 `MODEL_NAME`：

```bash
vllm bench serve --agentinfer replay \
  --config agentinfer/agentbench/configs/replay_benchmark.yaml \
  --trace-path tests/agentbench/replay/claude_trace_8session_requests.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME --task-num 1 \
  --result-dir results/replay-smoke
```

结果目录必须不存在。CLI 路径相对于当前目录，YAML 路径相对于配置文件目录。
默认配置包含用于样例的合成前缀预算；换用其他录制源时需重新标定，后端上下文长度须容纳请求目标。

## 回放 TraceLab

输入每行表示一次 LLM round，包含 `provider`、`session_id`、`round_index`、
`input_tokens_total`、`prefix_tokens`、`newly_append_tokens`、`output_tokens`、`timing_events` 和 `tools`。
Converter 校验 token 拆分，按 Provider、Session 分组并按 round 排序，生成 IR v3。
工具元数据只用于来源审计，不会实际执行。

```bash
vllm bench serve --agentinfer replay \
  --config agentinfer/agentbench/configs/replay_benchmark.yaml \
  --trace-type tracelab --trace-path /path/to/round_trace.jsonl \
  --prompt-shape tracelab_synthetic --endpoint /v1/chat/completions \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME \
  --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-tracelab-8x4
```

保留完整输入/输出时，配置 `max_input_tokens: null`、`max_output_tokens: null`；
`prompt_calibration_tolerance_tokens` 必须为 `0`。支持 `trace` 和 `lognormal` 间隔模式。
`task_num` 表示采样后的 Runtime Session 数，`max_concurrency` 限制同时运行的 Session 数。
重复采样使用不同的 Runtime Session ID 和私有合成内容。

每轮发送单条合成 `user` 消息。模板边界由后端 tokenizer 探测；Plan 使用 `input_after` 记录
输入继承，使用 `send_after` 记录发送依赖，不拼接 Backend 实时 Assistant 输出。
Session 内逐轮构造并发送，后继以前驱实际完成时刻加 interval 为释放时间。
首轮无输入前驱，计划复用为 0；后续轮按源缓存目标及模板、输入长度边界裁剪复用长度。
Execution 严格检查完整 Prompt token 数和相邻请求的最长公共前缀（LCP）。
缺少 SSE `[DONE]`、usage 缺失或实际输入/输出 token 数与计划不同，均记为请求失败。
源 `prefix_tokens` 是 Provider 观测，不要求等于 vLLM 缓存命中数；缓存 usage 缺失表示不可用。

TraceLab IR v3 使用 `requests.jsonl` 和 `manifest.json`，不生成空文本 sidecar。
显式分析复用 `unified_trace_ir.py`，原 Analyzer 和 Inferact IR v2 路径保持兼容。
token recipe profile 路径使用 Plan schema v2 / Planner v10，其他路径保留 v1 / v9。

## 输入与配置

旧配置中的 `claude_code_minimal_v1`、`trace_record`、`token_recipe` 应分别迁移为
`agentinfer_synthetic`、`inferact_synthetic`、`tracelab_synthetic`，不提供旧名称别名。
名称参与 workload fingerprint，迁移后不应按相同 workload ID 比较。
IR 的 `prompt_source.kind=token_recipe`、`token_recipe_profile` 和 `token_recipe_lcp_*` 指标名保持不变。
Inferact 仍保留源 Human 文本及实时 Assistant 历史；超出校准容差记录到
`trace-record-validation.json`，不会重写文本或因此终止 Session。

| 配置 | 含义 |
| --- | --- |
| `replay.trace_type: agentinfer` | 输入为 AgentInfer `requests.jsonl`；使用默认 `prompt_shape: agentinfer_synthetic`。 |
| `replay.trace_type: inferact_codex_swebenchpro` | 输入为 Inferact 原始 JSON；要求 `prompt_shape: inferact_synthetic`、`interval_mode: lognormal` 和 `/v1/chat/completions`。 |
| `replay.trace_type: tracelab` | 归一化、未压缩的 JSONL；要求 `prompt_shape: tracelab_synthetic`、零校准容差和 `/v1/chat/completions`。 |
| `replay.trace_type: agentX` | `prompt_shape: agentX_synthetic` 为预留入口，执行时抛出 `NotImplementedError`。 |
| `replay.interval_mode` | `trace` 保留历史间隔，`lognormal` 按配置的分布生成间隔。 |
| `replay.sample_seed` | 可重复的会话抽样和后端采样种子。 |
| `replay.max_input_tokens` / `max_output_tokens` | 可选 token 目标上限；`null` 保留 trace 目标。 |
| `replay.context_adjustment_mode` | `strict` 拒绝非追加上下文；`adaptive` 审计裁剪和上下文重置。 |
| `replay.request_timeout_seconds` | 单请求超时。 |

完整字段、默认值和约束见[Replay 配置模型](../../../agentinfer/agentbench/replay/config.py)与
[示例 YAML](../../../agentinfer/agentbench/configs/replay_benchmark.yaml)。运行
`vllm bench serve --agentinfer replay --help` 查看支持的 CLI 覆盖参数。

## 检查和比较结果

检查 `manifest.json` 的状态和 `summary.json`，并保留 `requests.jsonl`、`replay-source-analysis.json`、
`replay-plan.json`、`replay-execution.json` 和 `evidence/`。失败时检查 `replay-error.json`；
中途失败时部分产物可能不存在。Inferact 转换产物保存在结果目录的 `convert_result/` 下。

```bash
vllm bench serve --agentinfer compare \
  --baseline results/replay-baseline --candidate results/replay-candidate
```

公平比较要求使用相同 trace、抽样种子、前缀预算、并发和部署参数。每次冷启动比较前重启服务，
并检查起始 Prefix Cache 指标。Replay 不提供任务正确性结论，详见[基准方法](../explanation/benchmark-methodology.md)。
