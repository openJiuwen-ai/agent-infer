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

## 回放 TraceLab

输入每行表示一次 LLM round，包含 `provider`、`session_id`、`round_index`、
`input_tokens_total`、`prefix_tokens`、`newly_append_tokens`、`output_tokens`、`timing_events` 和 `tools`。
Converter 原样保留源 token 统计，按 Provider、Session 分组并按 round 排序，生成 token 配方 IR。
输入/输出目标直接使用记录的总量，前缀与新增 token 数作为来源证据保留，不校验两者之和。
工具元数据只用于来源审计，不会实际执行。

快速启动时可省略 `--config`；Replay 根据 `--trace-type` 选择内置 TraceLab 模板：

```bash
vllm bench serve --agentinfer replay \
  --trace-type tracelab \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME \
  --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-tracelab-8x4
```

未提供 `--config` 时，必须显式提供 `--trace-type`、`--task-num`、`--max-concurrency`、`--base-url` 和
`--model`。当 TraceLab 的 `trace_path: null` 时，Replay 自动下载固定版本 `v0.0.2` 的
`UW-SyFI/TraceLab` 数据集，并按源文件中首次出现的顺序提取前 `task_num` 个完整 Session；同一 Session
的全部 round 都会保留。生成的未压缩 JSONL 缓存于
`~/.cache/agentinfer/datasets/tracelab/v0.0.2/first-<task_num>-sessions.jsonl`（设置
`XDG_CACHE_HOME` 时使用该缓存根目录），后续运行直接复用。自动下载模式要求 `task_num` 非空。
显式提供 `--trace-path` 时始终使用用户文件，不触发下载。
提供 `--config` 时，其他 CLI 覆盖参数均为可选；显式覆盖参数优先于 YAML。
YAML 路径相对于配置文件，
CLI 路径相对于当前目录解析。

| `--trace-type` | 内置模板 |
| --- | --- |
| `agentinfer` | `replay_agentinfer.yaml` |
| `inferact_codex_swebenchpro` | `replay_inferact.yaml` |
| `tracelab` | `replay_tracelab.yaml` |
| `agentX` | `replay_agentX.yaml` |

回放直接使用 trace 中完整的输入和输出 token 目标值。
`prompt_calibration_tolerance_tokens` 必须为 `0`。支持 `trace` 和 `lognormal` 间隔模式。
`task_num` 表示采样后的 Runtime Session 数，`max_concurrency` 限制同时运行的 Session 数。
重复采样使用不同的 Runtime Session ID 和私有合成内容。

首轮发送一条合成 `user` 消息。续接轮通过 `context_after` 保留前一轮消息，追加 Backend 实时
Assistant 回答，再增加新的合成 `user` 消息；`send_after` 仍控制前驱完成后加 interval 的释放时间。
TraceLab 复用现有图执行器，要求上下文前驱响应成功；请求失败或 Prompt 构造失败时，后续上下文依赖请求会跳过。

后端 tokenizer 对包含 Chat Template 的完整对话计数，校准通过调整合成 user 文本满足输入目标，
不会截断实时 Assistant 回答。strict 模式下历史上下文超出目标会失败；adaptive 模式允许有限裁剪历史
合成 user 文本，但 TraceLab 不会为了满足目标重置或静默丢弃 Assistant 历史。无法校准时请求失败。
缺少 SSE `[DONE]`、usage 缺失或实际输入/输出 token 数与计划不同，同样记为请求失败。

源 `prefix_tokens` 保留为审计证据，不再作为精确 LCP 目标：保留实时对话历史可能产生不同于源 trace 的前缀。
不再生成固定前缀模板 profile 和精确 LCP 指标。缓存 usage 缺失仍表示不可用，首轮/续接轮缓存统计依据 `context_after` 分组。

TraceLab IR 使用 `requests.jsonl` 和 `manifest.json`，不生成文本 sidecar。
显式分析复用 `unified_trace_ir.py`，原 Analyzer 和 Inferact 文本 IR 路径保持兼容。
旧 TraceLab IR 和计划需要重新生成：`context_after` 替代 `input_after`。
转换器、Planner、采样器和时间模型标识不再附带版本后缀；数据格式校验仍保留 `schema_version`。
标识和哈希材料变化会改变 workload fingerprint、Runtime ID 和 seed。
历史前缀复制模式的运行不能视为相同工作负载。

## 输入与配置

提示词形态由 `replay.trace_type` 自动确定。请删除旧 YAML 中的 `replay.prompt_shape` 和命令中的
`--prompt-shape`；显式配置提示词形态会报错。
序列化配置移除此字段后，workload fingerprint 和 Runtime ID 会变化，迁移后不应按相同 workload ID 比较。
IR 的 `prompt_source.kind=token_recipe` 保持不变，token 目标用于构造合成 user 轮次并续接实时 Assistant 上下文。
Inferact 仍保留源 Human 文本及实时 Assistant 历史；超出校准容差记录到
`trace-record-validation.json`，不会重写文本或因此终止 Session。

| 配置 | 含义 |
| --- | --- |
| `replay.trace_type: agentinfer` | 输入为 AgentInfer `requests.jsonl`；自动选择 `agentinfer_synthetic`。 |
| `replay.trace_type: inferact_codex_swebenchpro` | 输入为 Inferact 原始 JSON；自动选择 `inferact_synthetic`，要求 `interval_mode: lognormal`、零校准容差和 `/v1/chat/completions`。 |
| `replay.trace_type: tracelab` | 归一化、未压缩的 JSONL；`trace_path: null` 时自动下载并提取前 `task_num` 个完整 Session。自动选择 `tracelab_synthetic`，要求零校准容差和 `/v1/chat/completions`。 |
| `replay.trace_type: agentX` | 输入为嵌套 Session JSONL；使用完整 hash 块配方和 `/v1/completions` token ID 请求。选中包含零输出请求的 Session 时，规划阶段明确报错。 |
| `replay.interval_mode` | `trace` 保留历史间隔，`lognormal` 按配置的分布生成间隔。 |
| `replay.sample_seed` | 可重复的会话抽样和后端采样种子。 |
| `replay.context_adjustment_mode` | `strict` 拒绝非追加上下文；`adaptive` 审计裁剪和上下文重置。 |
| `replay.request_timeout_seconds` | 单请求超时。 |

完整字段、默认值和约束见[Replay 配置模型](../../../agentinfer/agentbench/replay/config.py)与
[示例 YAML](../../../agentinfer/agentbench/configs/replay_benchmark.yaml)。运行
`vllm bench serve --agentinfer replay --help` 查看支持的 CLI 覆盖参数。

### 回放 AgentX Hash 快照

设置 `replay.trace_type: agentX`、本地 `replay.trace_path` 和
`backend.endpoint: /v1/completions`。源文件每行是一个 Session；转换器展开主请求与子 agent 组内的请求，
保留它们共享的会话时间轴。每条请求的 `hash_ids` 表示完整输入。实时输出只用于本轮测量，不追加到下一轮。
所有 token ID 使用配置的 Backend 模型；来源模型名称保留在分析产物中，并用于隔离同一 Runtime Session
内的块。重复采样会生成独立的 Runtime Session 和块内容。

转换器用 `t`、`api_time` 推断开始前最近完成的调度前驱。该关系只是时间推断，不代表已恢复父子因果；
子 agent 身份仍会传递，`blocks_parent` 设为 false。转换结果保留零输出请求；若采样选中含此类请求的
Session，规划阶段明确失败。后端上下文窗口必须容纳选中请求的输入与输出目标。
`replay.max_inflight_requests` 限制每个 Runtime Session 同时在途的 AgentX 请求数。

### 保留实时回答并对齐输入长度

Inferact `inferact_synthetic` 模式默认冻结已发送的历史消息（包括旧 filler），并原样保留实时 assistant
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
