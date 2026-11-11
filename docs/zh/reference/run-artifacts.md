# 运行产物参考

BenchKit 将每次运行写入一个全新的结果目录。原始请求事实、服务采集、任务结果、补丁和环境证据是权威数据；
`summary.json` 是这些事实的标准化汇总。

## 目录结构

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

无法采集的证据会在 `manifest.json` 或 `summary.json` 中标记为 unavailable 或 not applicable。对应的可选文件
可能不存在；缺少证据不会被静默转换为数值零或成功状态。

`summary.json` 的 `vllm` 块在指标快照缺失时，`reason` 会说明采集失败的具体原因（例如
`metrics_capture_failed: <采集器错误>`，常见于 `metrics_url` 配置错误或指标服务不可达）。没有采集器错误时，
`reason` 为 `missing_metrics_snapshot`。

## 顶层文件

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 解析后的运行配置、CLI 元数据、生命周期状态和证据可用性。 |
| `summary.json` | 标准化的任务、请求、vLLM、正确性和源码健康指标。 |
| `requests.jsonl` | 每个代理请求一行的不可变请求事实。 |

`compare` 只读取已完成且兼容的摘要。未完成、错误配对或模式不兼容的基线和候选摘要会被拒绝。

## `evidence/`

| 文件 | 内容 |
| --- | --- |
| `environment.json` | 主机、Python 和运行环境证据。 |
| `source_control.json` | 分支、提交和工作区状态。 |
| `vllm_metrics_start.prom` | 工作负载开始前采集的原始 vLLM Prometheus 指标。 |
| `vllm_metrics_end.prom` | 工作负载结束后采集的原始 vLLM Prometheus 指标。 |

冷启动结论必须由服务启动证据或起始指标支持。无法证明冷状态时，比较报告会发出警告，不应将该运行用于
发布门禁结论。

## `tasks/<instance_id>/`

| 文件 | 内容 |
| --- | --- |
| `result.json` | Agent 结果、终止原因、拓扑和补丁状态。 |
| `model.patch` | Agent 生成的补丁；未生成补丁时可能不存在。 |
| `transcript.jsonl` | 标准化的 Agent 对话事件。 |
| `claude-settings.json` | 本次任务使用的非敏感 Claude Code 设置。 |
| `claude-launch-command.txt` | 已脱敏的启动命令。 |
| `terminal-*.log` | 周期性终端采集和最终输出。 |

凭据通过启动环境传入，不得出现在设置、命令或日志产物中。发现敏感值时，不应共享该结果目录，并应立即
轮换相关凭据。

## 工作区与正确性

`workspaces/<instance_id>/` 保存任务检出和 Agent 工作目录。`completed` 仅表示 Agent 执行契约完成；
`model.patch` 是否解决 SWE-bench 任务，需要在任务的 `base_commit` 上应用补丁并执行 `FAIL_TO_PASS` 和
`PASS_TO_PASS` 测试后确定。

评估边界和公平比较要求见[基准方法](../explanation/benchmark-methodology.md)。

## 多运行时与 Replay 产物

运行目录还包括 `task_index.json`，记录任务与 Agent 会话的对应关系和失败汇总。Claude 专属的设置和终端文件只适用于 Claude；JiuwenSwarm 和 DSH 使用各自隔离目录与执行证据。

Replay 额外写出 `replay-source-analysis.json`、`replay-plan.json` 和 `replay-execution.json`；失败时写出
`replay-error.json`。这些文件记录规划、执行和审计事实，不代表任务正确性。详见[回放指南](../how-to/run-trace-replay.md)。

`replay-plan.json` 同时也是可加载的产物：`ReplayPlan.load(path)`（或 `ReplayPlan.from_dict` 加
`validate_loaded_plan`）能重建并结构化校验已写出的计划，因此它可作为自包含输入用于检视，或在后续的 runner
路径中恢复执行而无需重新规划。加载只校验产物本身（引用闭包、覆盖 `send_after` 与 `context_after` 的完整等待图
无环、目标合法性、`context_mode` 配对），不重新派生 `workload_fingerprint`/`plan_namespace`——二者内嵌构建期
workload 配置。`to_dict()` 无条件写出的每个 key 都必须存在，即使其值为 null：否则丢失的 `send_after` 或
`context_after` 会被读成"无依赖"，静默改变执行顺序。`interval_model.mode` 还必须与参数一致（`lognormal` 需要
`mu` 与 `sigma`，`trace` 不允许携带），使模式与参数不匹配在加载阶段失败，而不是拖到第一次采样间隔时。
这条存在性规则到 `interval_model` 为止：其拟合字段（`anchors`、`anchor_predictions` 等）全部按可选读取，
产物丢了这些值仍能加载，对应位置为 null。

除逐字段的存在性与类型，加载还会校验所有构建路径都成立的跨字段事实：`execution_ready` 与
`execution_unavailable_reason` 成对设置；`prompt_calibration.status` 必须是 runner 会写出的状态之一；
`replayable_source_sessions` 不超过 `source_sessions`；任务身份在全计划内唯一（`task_id`、`task_index`、
`runtime_session_id`、`(source_session_id, sample_ordinal)` 这次抽样，以及每个请求的 `runtime_request_id`），
两个 session id 均非空。重复抽样尤其要拒绝：它不增加覆盖，任务计数与 fingerprint 却按增加计。请求节点还必须
携带真实的 `prompt_kind` 与 `context_mode`：`"none"` 是 timing 节点的中性取值，出现在请求上会渲染出与计划描述
不符的 prompt，或指向执行器根本不会校验的上下文前驱。见 [Python API](../reference/python-api.md)。

节点序列化出的每个字段都能被解析回来，因此 `from_dict(to_dict())` 保持节点不变。`timing_dependency` 节点在其
请求类字段中携带 `prompt_kind` 以及 actor 身份（`actor_id`、`actor_role`、`parent_actor_id`），使加载后的计划仍能
说明该等待节点属于哪个 agent；其中 prompt kind 必须是中性的 `none`，若产物在此处填入请求类取值，加载会拒绝而非
静默改写。timing 节点上出现的其余请求专属字段（`prompt_recipe_key`、`prompt_ref`、`context_after`、`context_mode`、
各 token 目标等）则会被拒绝：timing 节点不发起请求，没有承载这些字段的位置，接受它们会加载出内容与所得计划不一致
的产物。`source_status` 与 `prompt_recipe_key` 是规划与校验阶段的输入，不会被携带。由于 `workload_fingerprint` 基于序列化后的任务计算，节点序列化规则变化会改变新建计划的 fingerprint；
更早版本写出的计划保留其构建时的 fingerprint，加载过程不会重新计算。

[`replay-plan.example.json`](replay-plan.example.json) 是一份最小且结构合法的产物：单会话、两个存在依赖关系的
请求、以及一个 timing 节点。它可被加载并原样重新序列化，可作为检视工具的起点，也可在测试中篡改单个字段以校验
fail-closed 约定。

DSH 0.2 额外保存 `evidence/dsh-runtime.json`（版本、能力、安装锁哈希），以及任务级的
`dsh-effective-config.yaml`、`dsh-lineage.json`、`dsh-run-status.json` 和 `dsh-processes.jsonl`。
`dsh-sessions/raw/index.json` 建立原始与解码日志的索引；`transcript.jsonl` 按 stdout 的主会话 ID 精确选择。
`dsh-launch-command.json` 不包含经 stdin 发送的 prompt。超时、取消时尽力保留日志和补丁；外层 cancelled
结果的 `has_patch` 仍可能为 false，实际导出状态以 `dsh-run-status.json` 为准。身份校验不会改写 `requests.jsonl`。

流式请求的 SDK 在收到 OpenAI `[DONE]` 或 Anthropic `message_stop` 后关闭 HTTP 连接，记录为成功；
结束标记之前的取消仍记录为错误。

### DSH 执行控制证据

`dsh-lineage.json` 中的 `plan_recoveries` 记录计划恢复提醒次数，最多为 1。
触发执行保护时，`failure` 保存 `reason`、`session_id` 和 `message`；
`dsh-run-status.json` 保留相同诊断，`result.json.termination_reason` 给出最终分类。
执行超时或外部取消仍保留自身的最终分类，原始会话日志和已生成的补丁继续按现有收尾流程采集。

| 分类 | 含义 |
| --- | --- |
| `agent_step_limit` | 主 Agent 超出 200 步执行预算。 |
| `subagent_step_limit` | 子 Agent 超出 100 步探索预算。 |
| `subagent_read_loop` | 最近 8 次工具调用中，相同读取出现第 4 次。 |
| `subagent_failed` | 子 Agent 以非 `completed` 的状态结束。 |
| `plan_not_approved` | 计划协议未完成，包括一次恢复提醒后仍未获批。 |
| `validation_error_loop` | 最近 8 次工具结果中，同类参数错误或计划模式关闭后的退出调用错误达到 4 次。 |
| `max_tokens` | 模型或 DSH 回合报告长度截断。 |

会话收集遍历任务独立 `DSH_HOME` 中的全部会话，以 session ID 和父子关系校验身份；目录名称和日志修改时间不参与主会话选择。
