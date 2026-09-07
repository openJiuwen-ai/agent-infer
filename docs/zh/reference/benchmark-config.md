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
  prompt_delivery_timeout_seconds: 30

dataset:
  name: swebench_verified
  index_path: ../data/swebench/instances.jsonl
  selection_path: ../data/swebench/task-lists/default.txt
  cache_dir: repo-cache

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

router:
  enabled: false
  base_url: null
  control_timeout_seconds: 10.0
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
| `prompt_delivery_timeout_seconds` | `30` | 等待提示词送达 Agent 的超时，必须 `>= 1`。 |

## `dataset`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `name` | `swebench_verified` | 当前唯一支持的数据集名称。 |
| `index_path` | `../data/swebench/instances.jsonl` | SWE-bench JSONL 元数据。 |
| `selection_path` | `../data/swebench/task-lists/default.txt` | 按行排列、保持顺序的任务 ID。 |
| `cache_dir` | `repo-cache` | 多任务共享的 Git 仓库缓存目录。 |

## `agent`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `type` | `claude` | 当前唯一支持的 Agent 运行时。 |
| `profile` | `single` | `single` 或 `plan-subagent`。 |
| `executable` | `claude` | Claude Code 命令名或路径。 |
| `tmux_startup_seconds` | `2.0` | tmux 会话启动等待时间，必须 `>= 0`。 |
| `terminal_capture_interval_seconds` | `30` | 终端证据采集间隔，必须 `>= 1`。 |

`plan-subagent` 使用 `bypassPermissions`。仅应在非 root 用户或经过批准的 Claude Code 权限环境中启用。

## `backend`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `type` | `vllm` | 当前唯一支持的后端类型。 |
| `base_url` | `http://127.0.0.1:8000` | vLLM API 基础 URL。 |
| `model` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | Claude Code 请求中发送的模型名。 |
| `endpoint` | `/v1/messages` | 当前仅支持 Anthropic Messages 端点。 |
| `api_key_env` | `null` | 可选的 API Key 环境变量名；配置中不得写入密钥值。 |

## `request_proxy`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `listen_url` | `http://127.0.0.1:0` | 必须是 HTTP loopback URL；端口 `0` 表示自动选择。 |
| `request_timeout_seconds` | `3600` | 转发请求超时，必须 `>= 1`。 |
| `startup_timeout_seconds` | `120.0` | 代理启动超时，必须 `>= 0`。 |
| `shutdown_timeout_seconds` | `30.0` | 等待在途请求结束的超时，必须 `> 0`。 |

## `router`

| 字段 | 默认值 | 约束和含义 |
| --- | --- | --- |
| `enabled` | `false` | 是否启用 Router 控制流程。 |
| `base_url` | `null` | Router 控制 API URL。 |
| `control_timeout_seconds` | `10.0` | Router 注册和清理请求超时，必须 `> 0`。 |

`enabled` 和 `base_url` 必须同时启用或同时停用。仅比较 vLLM 调度器时保持 Router 关闭。

## 路径解析

YAML 中的 `result_dir`、数据集路径和包含目录的 `agent.executable` 相对于 YAML 文件所在目录解析。仅包含命令名
的 `agent.executable` 由 `PATH` 查找。CLI 覆盖中的路径相对于调用命令时的当前目录解析。

可覆盖字段及参数名见[基准命令行参考](benchmark-cli.md)。
