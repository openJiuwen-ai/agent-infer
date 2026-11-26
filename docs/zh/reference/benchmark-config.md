# 基准配置参考

BenchKit 使用 Pydantic 严格校验 YAML。所有章节都有默认值，未知章节或字段会导致配置加载失败。配置优先级为
内置默认值、YAML、显式命令行覆盖。

## 完整示例

```yaml
experiment:
  task_num: null
  result_dir: results
  max_concurrency: 1
  task_timeout_seconds: 3600
  run_timeout_seconds: null
  patch_flush_seconds: 15
  cleanup_timeout_seconds: 10.0
  prompt_delivery_timeout_seconds: 30

dataset:
  name: swebench_verified
  index_path: data/swebench/instances.jsonl
  selection_path: data/swebench/task-lists/default.txt
  cache_dir: data/swebench/repo-cache

agent:
  type: claude
  profile: single
  executable: claude
  tmux_startup_seconds: 2.0
  terminal_capture_interval_seconds: 30

backend:
  type: vllm
  base_url: http://127.0.0.1:8000
  model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
  endpoint: /v1/messages
  api_key_env: null

request_proxy:
  listen_url: http://127.0.0.1:0
  request_timeout_seconds: 3600
  startup_timeout_seconds: 120.0
  shutdown_timeout_seconds: 30.0
```

## `experiment`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `task_num` | `null` | 选中列表中的前 N 个任务；整数必须 `>= 1`，`null` 表示全部。 |
| `result_dir` | `results` | 新运行的输出目录；目录不得已存在。 |
| `max_concurrency` | `1` | 并发 Agent 会话数，必须 `>= 1`。 |
| `task_timeout_seconds` | `3600` | 单任务超时，必须 `>= 1`；CLI 参数为 `--timeout`。 |
| `run_timeout_seconds` | `null` | 整次运行超时；整数必须 `>= 1`，`null` 表示不另设总超时。 |
| `patch_flush_seconds` | `15` | 周期性写出补丁的间隔，必须 `>= 0`。 |
| `cleanup_timeout_seconds` | `10.0` | DSH 退出后日志收集、写入等待及补丁导出的总预算，必须为有限正数，仅支持 YAML 配置。 |
| `prompt_delivery_timeout_seconds` | `30` | 等待提示词送达 Agent 的超时，必须 `>= 1`。 |

DSH 将清理预算耗尽记录为 `cleanup_timeout`，保留已有的任务超时或取消原因，并将清理详情写入 `dsh-run-status.json`。日志较大或存储较慢时可增大预算；任务成功仍要求会话证据通过校验。

## `dataset`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `name` | `swebench_verified` | 当前唯一支持的数据集名称。 |
| `index_path` | `data/swebench/instances.jsonl` | SWE-bench JSONL 元数据。 |
| `selection_path` | `data/swebench/task-lists/default.txt` | 按行排列、保持顺序的任务 ID。 |
| `cache_dir` | `data/swebench/repo-cache` | 多任务共享的 Git 仓库缓存目录。 |

## `agent`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `type` | `claude` | `claude`、`jiuwenswarm`、`dsh` 或 `zcode`。 |
| `profile` | `single` | Claude/DSH/ZCode：`single` 或 `plan-subagent`；JiuwenSwarm：`agent.code.normal` 或 `team.code.normal`（兼容 `code.normal`、`code.team`）。 |
| `executable` | `claude` | Agent CLI 命令名或路径。 |
| `tmux_startup_seconds` | `2.0` | tmux 会话启动等待时间，必须 `>= 0`。 |
| `terminal_capture_interval_seconds` | `30` | 终端证据采集间隔，必须 `>= 1`。 |
| `context_window` | `262144` | DSH/ZCode 模型上下文容量；须与实际后端一致，仅支持 YAML 配置。 |
| `max_tokens` | `8192` | DSH/ZCode 输出容量；须与实际后端一致，仅支持 YAML 配置。 |
| `max_subagents` | `1` | ZCode plan-subagent 统计每任务接受的子 Agent 启动次数，失败也计入。JiuwenSwarm 统计每任务成功启动的子 Agent 次数，包含队员和后代。 |
| `max_concurrent_subagents` | `1` | ZCode plan-subagent 和 JiuwenSwarm 每任务子 Agent 并发上限。JiuwenSwarm 只统计正在执行模型或工具工作的子 Agent。不得超过 `max_subagents`。 |

Claude 的 `plan-subagent` 使用 `bypassPermissions`。仅应在非 root 用户或经过批准的 Claude Code 权限环境中启用。

## `backend`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `type` | `vllm` | 当前唯一支持的后端类型。 |
| `base_url` | `http://127.0.0.1:8000` | vLLM API 基础 URL。 |
| `model` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | 请求中发送的服务模型名。 |
| `endpoint` | `/v1/messages` | Claude 使用 `/v1/messages`；JiuwenSwarm/DSH/ZCode 使用 `/v1/chat/completions`。 |
| `api_key_env` | `null` | 可选的 API Key 环境变量名；配置中不得写入密钥值。 |

Claude Code 或 JiuwenSwarm 配置了 `backend.api_key_env` 时，BenchKit 读取指定的环境变量。
模型可用性检查与 Request Proxy 都将该值作为 Bearer 密钥使用，代理向后端发送请求时以此替换客户端的认证请求头。
环境变量缺失或为空时，运行在启动任务前报错。DSH 和 ZCode 不支持此字段。

## `request_proxy`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `listen_url` | `http://127.0.0.1:0` | 必须是 HTTP loopback URL；端口 `0` 表示自动选择。 |
| `request_timeout_seconds` | `3600` | 转发请求超时，必须 `>= 1`。 |
| `startup_timeout_seconds` | `120.0` | 代理启动超时，必须 `>= 0`。 |
| `shutdown_timeout_seconds` | `30.0` | 等待在途请求结束的超时，必须 `> 0`。 |

## 路径解析

YAML 中的 `result_dir`、数据集路径和包含目录的 `agent.executable` 相对于 YAML 文件所在目录解析。仅包含命令名
的 `agent.executable` 由 `PATH` 查找。CLI 覆盖中的路径相对于调用命令时的当前目录解析。

可覆盖字段及参数名见[基准命令行参考](benchmark-cli.md)。

## 透明 Router 与 Replay

通过 Router 运行时，将 `backend.base_url` 设为 Router URL，将可选的 `backend.metrics_url` 设为完整的 vLLM `/metrics` URL；默认在
`base_url` 后追加 `/metrics`。不再接受旧 `router` 章节或会话注册控制协议。

Replay 使用独立的 `experiment`、`backend`、`replay` 配置，见[回放指南](../how-to/run-trace-replay.md)。

### Router 工作流预测

以下 Replay 可选字段通过 YAML 配置。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `backend.router_metrics_url` | 未设置 | 完整的 Router 指标 URL，独立于 vLLM 指标端点。 |
| `replay.workflow_profile_id` | 未设置 | 随请求的 agentic context 透传的工作流 Profile ID。 |
| `replay.profile_version` | 未设置 | 随请求的 agentic context 透传的工作流 Profile 版本。 |
| `replay.profile_role_source` | `none` | 用于匹配 Profile 的角色来源。`request_purpose` 取原始追踪请求用途，`actor_id` 取执行者 ID，`agent_role` 取执行者角色。 |

配置 `backend.router_metrics_url` 后，Replay 会尝试在测量窗口前后分别采集
`router_metrics_start.prom` 和 `router_metrics_end.prom`。运行摘要中的
`router.applicable: true` 表示已配置 Router 指标采集。聚合结果是否可用请查看
`router.prediction.available`，不可用时可查看 `router.prediction.reason`。

预测指标使用标注为 `target="all"` 的 Router 汇总序列。Router 全局的 `profile_keys` 单独报告。
预留误差按 `p95` 和 `p99` 分开，路径为
`router.prediction.output_length.reservation_error[quantile][direction]`。

Replay 写入的 `request_purpose` 优先使用源 trace 中的请求用途，缺失时回退到 `prompt_kind`。
即使未配置 Router 预测选项，此行为也会生效。
