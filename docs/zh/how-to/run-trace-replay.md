# 运行 Trace Replay

Replay 从请求 trace 恢复会话、Agent、请求依赖、间隔和 token 目标，向后端发送请求并记录性能证据。
它不启动原始 Agent，不执行工具或 SWE-bench 正确性测试。合成提示词回放不代表原始文本或性能完全等价。

## 准备后端

安装项目后，启动支持目标模型的 vLLM 服务。后端需要提供 `/tokenize`、`/detokenize`、`/metrics` 和配置的推理端点。
通过 Router 访问时，在 YAML 中设置 `backend.tokenizer_base_url` 为分词服务地址，
`backend.metrics_url` 为完整的 vLLM 指标 URL。使用 `/v1/messages` 时，启动 vLLM 时加载
`agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware`，以传递身份和采样参数。
具体服务参数见[接入 vLLM](integrate-vllm.md)。

Inferact 转换会从 `/v1/models` 和可选的 `/tokenizer_info` 发现后端 tokenizer。若同机存在对应
tokenizer 文件，并且本地与后端的原始文本和聊天模板 token ID 探针完全一致，转换和回放校准会使用
本地 tokenizer；仅在加性探测通过且消息数为已探测的 1、3、5、9 时采用增量计数，其他长度及
运行期校准始终计算完整聊天模板。无法发现或验证
本地 tokenizer 时自动回退到 `/tokenize`。vLLM 需使用 `--enable-tokenizer-info-endpoint` 才会提供
`/tokenizer_info`；服务端覆盖 chat template 时建议启用该端点。

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

## 输入与配置

| 配置 | 含义 |
| --- | --- |
| `replay.trace_type: agentinfer` | 输入为 AgentInfer `requests.jsonl`；使用合成提示词。 |
| `replay.trace_type: inferact_codex_swebenchpro` | 输入为 Inferact 原始 JSON；要求 `prompt_shape: trace_record`、`interval_mode: lognormal` 和 `/v1/chat/completions`。 |
| `replay.trace_type: agentX` 或 `tracelab` | 预留值，执行时抛出 `NotImplementedError`。 |
| `replay.interval_mode` | `trace` 保留历史间隔，`lognormal` 按配置的分布生成间隔。 |
| `replay.sample_seed` | 可重复的会话抽样和后端采样种子。 |
| `replay.max_input_tokens` / `max_output_tokens` | 可选 token 目标上限；`null` 保留 trace 目标。 |
| `replay.context_adjustment_mode` | `strict` 拒绝非追加上下文；`adaptive` 审计裁剪和上下文重置。 |
| `replay.request_timeout_seconds` | 单请求超时。 |

完整字段、默认值和约束见[Replay 配置模型](../../../agentinfer/agentbench/replay/config.py)与
[示例 YAML](../../../agentinfer/agentbench/configs/replay_benchmark.yaml)。运行
`vllm bench serve --agentinfer replay --help` 查看支持的 CLI 覆盖参数。

### 保留实时回答并对齐输入长度

Inferact `trace_record` 模式默认冻结已发送的历史消息（包括旧 filler），并原样保留实时 assistant
正文和独立的 `reasoning_content`。最新 user 消息不足目标长度时追加确定性 filler；超出目标时仅
裁剪该条源文本的尾部，再重新分词修正边界误差。裁剪按字符边界进行，不删除历史消息。
`context_adjustment_mode` 的历史裁剪和重置规则不适用于这一校准路径。

Inferact 默认且始终要求每个成功请求的输入 token 数精确匹配计划，无需增加模式或容差配置。
非零 `prompt_calibration_tolerance_tokens` 会在 Inferact 配置加载时报错。请删除旧配置中的非零设置，
或显式设为零；合成提示词仍支持配置容差。

若最新 user 没有任何前缀能满足目标上限、有限次后缀修复无法精确命中目标，或 token ID 校验发现校准改变了
原始/空 user 两种模板共有的前缀，请求会在发送前失败。推理响应缺少 `usage.prompt_tokens` 或
实际输入与目标不一致时，该请求也标记失败，依赖它的后续请求跳过。不会通过缩减历史来强行对齐。

`replay-execution.json` 中每轮校准记录包含 `trimmed_current_user_tokens`、
`trimmed_current_user_characters`、`preserved_prefix_tokens` 和 `backend_input_residual_tokens`。
当前源文本的裁剪不会计入 `trimmed_filler_tokens`。
`trace-record-validation.json` 的 `input_length_comparable` 只有在全部计划请求成功且后端 usage
均精确匹配计划时才为真；这是输入长度检查，不能保证 Prefix Cache 命中一致或真实任务语义不受裁剪影响。

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
