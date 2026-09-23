# 运行 Trace Replay

Replay 从请求 trace 恢复会话、Agent、请求依赖、间隔和 token 目标，向后端发送请求并记录性能证据。
它不启动原始 Agent，不执行工具或 SWE-bench 正确性测试。合成提示词回放不代表原始文本或性能完全等价。

## 准备后端

安装项目后，启动支持目标模型的 vLLM 服务。回放需要配置的推理端点；使用远程 tokenizer 时还需要
`/tokenize` 和 `/detokenize`。`/metrics` 用于采集 vLLM 前后快照，无法获取时对应证据会标记为不可用，
不会单独导致 Replay 失败。
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
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-smoke
```

结果目录必须不存在。CLI 路径相对于当前目录，YAML 路径相对于配置文件目录。
默认配置包含用于样例的合成前缀预算；换用其他录制源时需重新标定，后端上下文长度须容纳请求目标。

## 回放 TraceLab

最快的运行方式是省略 `--config`，让 Replay 选择内置 TraceLab 模板并自动准备数据集：

```bash
vllm bench serve --agentinfer replay \
  --trace-type tracelab \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME \
  --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-tracelab-8x4
```

未提供 `--config` 时，必须显式提供 `--trace-type`、`--task-num`、`--base-url` 和 `--model`；
`--max-concurrency` 未提供时默认为 `1`，`--result-dir` 未提供时使用模板值 `replay-results`。
`task_num` 是所有 Replay 类型的必填正整数，省略、设为 `null`、`0` 或负数都会在下载或运行前报错。

除尚未实现的 `agentX` 外，`trace_path: null` 会按 `trace_type` 使用默认数据集：

- `agentinfer` 使用随 Python 包发布的 `agentinfer/agentbench/data/agentinfer_trace_requests.jsonl`；该文件包含
  8 个完整 Session，运行时由 Planner 按 `task_num` 选择或循环复用 Session。
- `inferact_codex_swebenchpro` 下载固定版本的 `Inferact/codex_swebenchpro_traces`，取前
  `min(task_num, 数据集记录数)` 个完整顶层 JSON 对象生成本地 JSON 数组。
- `tracelab` 下载固定版本的 TraceLab，并取前 `min(task_num, 数据集 Session 数)` 个完整 Session 及其全部 round。

显式 `trace_path` 始终优先，路径无效时不会退回默认数据集。`agentX` 仍为预留入口，执行时抛出
`NotImplementedError`。

### 自动下载和选择 Session

当 `trace_type: tracelab` 且 `trace_path: null` 时，`run_replay()` 在创建结果目录之前执行以下操作：

1. 通过 Hugging Face Hub 下载 `UW-SyFI/TraceLab` 的固定版本 `v0.0.2`，源文件为
   `data/v0.0.2/syfi_coding_trace.jsonl.gz`。
2. 以 `(provider, session_id)` 作为 Session 身份，按它们在源文件中首次出现的顺序选择前
   `task_num` 个 Session。
3. 扫描完整压缩文件，把这些 Session 的全部 round 原样写入一个未压缩 JSONL。即使同一 Session 的
   round 在源文件中不连续，也不会被截断。
4. 使用临时文件和原子替换生成缓存；JSON、Provider/Session 身份无效或数据集不含 Session 时删除临时文件
   并终止。若源 Session 少于 `task_num`，保留全部源 Session，Planner 再循环取样到目标数量。

生成的 JSONL 缓存于
`~/.cache/agentinfer/datasets/tracelab/v0.0.2/first-<task_num>-sessions.jsonl`（设置
`XDG_CACHE_HOME` 时改用该缓存根目录）。非空缓存文件会被直接复用。Hugging Face 自身还会保留原始
`.jsonl.gz` 下载缓存。下载或解压失败发生在结果目录创建之前，
因此不会留下一个伪装成运行结果的目录。

显式设置 `trace_path` 时始终使用用户文件，不触发自动下载。TraceLab Converter 只接受存在的、未压缩的
`.jsonl` 文件；每行必须表示一次 LLM round，并包含 `provider`、`session_id`、`round_index`、
`input_tokens_total`、`prefix_tokens`、`newly_append_tokens`、`output_tokens`、`timing_events` 和 `tools`。

提供 `--config` 时，CLI 覆盖参数均为可选，显式参数优先于 YAML。YAML 中的相对路径以配置文件目录为
基准；CLI 路径以当前工作目录为基准。

| `--trace-type` | 内置模板 |
| --- | --- |
| `agentinfer` | `replay_agentinfer.yaml` |
| `inferact_codex_swebenchpro` | `replay_inferact.yaml` |
| `tracelab` | `replay_tracelab.yaml` |
| `agentX` | `replay_agentX.yaml`（预留；执行时抛出 `NotImplementedError`） |

### 从源数据到执行计划

TraceLab 的处理顺序如下：

1. Converter 严格校验每一行，以 Provider 和 Session 分组，再按 `round_index` 排序。重复的
   `(provider, session_id, round_index)`、非法 token 数、工具结构、时间戳或事件结构都会使转换失败；
   缺少可用的输入/输出时间事件则记录为 timing 不可用。
2. Converter 把每个 round 写成 token 配方 IR。首轮的 `context_after` 和 `send_after` 为空；后续轮次
   两者都指向前一 round。转换产物位于结果目录的 `convert_result/requests.jsonl` 和
   `convert_result/manifest.json`，TraceLab 不生成文本 sidecar。
3. Planner 使用 `sample_seed` 对可回放 Session 做确定性的哈希乱序，并生成 Runtime Session、请求 ID、
   后端采样 seed、依赖和发送间隔。相同源数据和配置产生相同计划身份。
4. Executor 最多并行运行 `max_concurrency` 个 Runtime Session；同一 Session 内的节点按
   `send_after` 和 `context_after` 依赖释放。

`task_num` 是最终 Runtime Session 数：

- 自动下载时先截取前 `task_num` 个源 Session，再由 Planner 对这些 Session 做确定性乱序；正常情况下
  每个源 Session 恰好回放一次。
- 数据源的可用 Session 少于 `task_num` 时，Sampler 按新的确定性乱序周期重复取样。重复实例拥有
  不同的 Runtime Session ID、请求 ID 和私有合成文本。

### 时间间隔

TraceLab 从 `timing_events` 推导每轮的输入就绪时间和输出结束时间。`interval_mode: trace` 使用前一轮
输出结束到下一轮输入就绪之间的间隔，再应用
`max(0, source_gap * trace_same_agent_gap_scale + trace_same_agent_gap_offset_seconds)`。如果需要间隔的后续
round 无法得到有效 gap（包括缺少可用时间事件或出现负的源间隔），规划会失败。

`interval_mode: lognormal` 不使用历史间隔，而是根据配置的 `p50_seconds`、`p95_seconds` 和
`p99_seconds` 拟合一个 Lognormal 分布，并按 `sample_seed`、Runtime Session 和请求身份确定性采样。
三个锚点必须满足 `p50 < p95 <= p99`，且能够在允许误差内由同一个 Lognormal 分布拟合。

### Prompt 构造、请求和失败传播

回放直接使用源 trace 的 `input_tokens_total` 和 `output_tokens` 作为每轮精确目标。
`prefix_tokens`、`newly_append_tokens`、工具数量和错误数量只作为来源审计信息；源工具不会执行，
`prefix_tokens` 也不是运行时精确 LCP 目标。

首轮从一条空的合成 `user` 消息开始。续接轮复制已经发送的消息，追加后端刚返回的实时
`assistant` 正文和独立的 `reasoning_content`，再追加新的合成 `user` 消息。后端 tokenizer 对完整
Chat Template 计数，校准器只向合成 user 文本加入确定性 filler，直到达到计划输入 token 数。

`prompt_calibration_tolerance_tokens` 对 TraceLab 必须为 `0`：

- `strict` 模式下，只要已有上下文超过下一轮目标就失败。
- `adaptive` 模式仅在超量不超过
  `max(context_micro_trim_max_tokens, ceil(target * context_micro_trim_max_ratio))` 时，允许裁剪历史合成
  user 文本；不会裁剪实时 Assistant 内容，也不会重置上下文。
- filler 修复仍无法精确达到目标时，该节点在发送前失败。

向 `/v1/chat/completions` 发送请求时，Replay 同时设置 `max_tokens` 和 `min_tokens` 为计划输出长度，
设置 `ignore_eos: true`，启用流式 usage，并传递确定性的采样 seed。一次 TraceLab 请求只有在 HTTP 成功、
收到 SSE `[DONE]`，且后端 usage 中的输入和输出 token 数都与计划完全相等时才算成功。

`send_after` 只要求前驱结束，因此时间依赖的后继即使在前驱失败后仍可继续；`context_after` 必须获得
成功的实时回答。上下文前驱失败、请求失败或 Prompt 构造失败时，依赖该上下文的后续节点标记为
`skipped_dependency_failed`，同一 Session 中不相关的节点仍可运行。单 Session 受
`task_timeout_seconds` 限制，整个运行可选地受 `run_timeout_seconds` 限制，每个 HTTP 请求受
`request_timeout_seconds` 限制。

缓存 usage 缺失不会单独终止运行，但对应缓存指标记为不可用。首轮和续接轮的缓存统计按
`context_after` 分组。保留实时回答后，实际前缀可能不同于源 trace，因此不同运行的比较不能把源
`prefix_tokens` 当作精确命中目标。

## 输入与配置

提示词形态由 `replay.trace_type` 自动确定，配置模型拒绝未知字段。旧配置中的 `replay.prompt_shape` 和
命令中的 `--prompt-shape` 已不再支持。自动下载完成后，Runner 使用包含实际本地 `trace_path` 的配置副本，
因此 `manifest.json` 和 `replay-plan.json` 记录的是最终缓存路径；调用方传入的原配置对象不会被修改。

| 配置 | 含义 |
| --- | --- |
| `replay.trace_type: agentinfer` | 输入为 AgentInfer `requests.jsonl`；`trace_path: null` 时使用包内 8-Session 数据集。自动选择 `agentinfer_synthetic`。 |
| `replay.trace_type: inferact_codex_swebenchpro` | 输入为 Inferact 原始 JSON；`trace_path: null` 时下载固定版本并缓存前 `task_num` 条完整记录。自动选择 `inferact_synthetic`，要求 `interval_mode: lognormal`、零校准容差和 `/v1/chat/completions`。 |
| `replay.trace_type: tracelab` | 归一化、未压缩的 JSONL；`trace_path: null` 时自动下载并提取前 `task_num` 个完整 Session。自动选择 `tracelab_synthetic`，要求零校准容差和 `/v1/chat/completions`。 |
| `replay.trace_type: agentX` | 自动选择 `agentX_synthetic`；为预留入口，执行时抛出 `NotImplementedError`。 |
| `replay.interval_mode` | `trace` 保留历史间隔，`lognormal` 按配置的分布生成间隔。 |
| `replay.sample_seed` | 可重复的会话抽样、间隔抽样和后端采样 seed。 |
| `replay.context_adjustment_mode` | `strict` 拒绝超过 token 目标的上下文；`adaptive` 允许有限裁剪。TraceLab 始终保持追加上下文，不执行 reset。 |
| `experiment.task_num` | 必填正整数；Runtime Session 数，也是默认数据源最多截取的完整源 Session/记录数。 |
| `experiment.max_concurrency` | 同时运行的 Runtime Session 上限，默认 `1`。 |
| `experiment.task_timeout_seconds` | 单个 Runtime Session 的超时。 |
| `experiment.run_timeout_seconds` | 整次 Replay 的可选超时；`null` 表示不设置。 |
| `replay.request_timeout_seconds` | 单请求超时。 |

完整字段、默认值和约束见[Replay 配置模型](../../../agentinfer/agentbench/replay/config.py)与
[示例 YAML](../../../agentinfer/agentbench/configs/replay_benchmark.yaml)。运行
`vllm bench serve --agentinfer replay --help` 查看支持的 CLI 覆盖参数。

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
`replay-plan.json`、`replay-execution.json` 和 `evidence/`。单个请求或 Session 失败会记录在
`replay-execution.json` 和请求 trace 中，Runner 仍会汇总其他任务并正常完成结果目录。只有运行级异常
（例如转换失败、全局超时或资源关闭失败）才写入 `replay-error.json` 并把 manifest 标记为失败，此时
部分后续产物可能不存在。

TraceLab 和 Inferact 的转换产物都保存在结果目录的 `convert_result/` 下；Inferact 路径还会生成
`trace-record-validation.json`。

```bash
vllm bench serve --agentinfer compare \
  --baseline results/replay-baseline --candidate results/replay-candidate
```

公平比较要求使用相同 trace、抽样种子、前缀预算、并发和部署参数。每次冷启动比较前重启服务，
并检查起始 Prefix Cache 指标。Replay 不提供任务正确性结论，详见[基准方法](../explanation/benchmark-methodology.md)。
