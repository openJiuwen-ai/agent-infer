# Benchmark Configuration

BenchKit validates YAML strictly with Pydantic. Every section has defaults, and unknown sections or fields cause
configuration loading to fail. Precedence is built-in defaults, YAML, then explicit command-line overrides.

## Complete Example

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

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `task_num` | `null` | First N selected tasks; an integer must be `>= 1`, and `null` means all tasks. |
| `result_dir` | `results` | Output directory for a new run; it must not already exist. |
| `max_concurrency` | `1` | Concurrent agent sessions; must be `>= 1`. |
| `task_timeout_seconds` | `3600` | Per-task timeout; must be `>= 1`; CLI flag is `--timeout`. |
| `run_timeout_seconds` | `null` | Whole-run timeout; an integer must be `>= 1`, and `null` adds no run timeout. |
| `patch_flush_seconds` | `15` | Periodic patch flush interval; must be `>= 0`. |
| `prompt_delivery_timeout_seconds` | `30` | Timeout while waiting for prompt delivery; must be `>= 1`. |

## `dataset`

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `name` | `swebench_verified` | The only currently supported dataset name. |
| `index_path` | `../data/swebench/instances.jsonl` | SWE-bench metadata in JSONL format. |
| `selection_path` | `../data/swebench/task-lists/default.txt` | Ordered task IDs, one per line. |
| `cache_dir` | `repo-cache` | Shared Git repository cache for tasks. |

## `agent`

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `type` | `claude` | The only currently supported agent runtime. |
| `profile` | `single` | Either `single` or `plan-subagent`. |
| `executable` | `claude` | Claude Code command name or path. |
| `tmux_startup_seconds` | `2.0` | tmux session startup wait; must be `>= 0`. |
| `terminal_capture_interval_seconds` | `30` | Terminal evidence capture interval; must be `>= 1`. |

The `plan-subagent` profile uses `bypassPermissions`. Enable it only as a non-root user or with an approved Claude Code
permission setup.

## `backend`

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `type` | `vllm` | The only currently supported backend type. |
| `base_url` | `http://127.0.0.1:8000` | vLLM API base URL. |
| `model` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | Model name sent in Claude Code requests. |
| `endpoint` | `/v1/messages` | Only the Anthropic Messages endpoint is supported. |
| `api_key_env` | `null` | Optional API-key environment variable name; never put the secret value in YAML. |

## `request_proxy`

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `listen_url` | `http://127.0.0.1:0` | Must be an HTTP loopback URL; port `0` selects a free port. |
| `request_timeout_seconds` | `3600` | Forwarded request timeout; must be `>= 1`. |
| `startup_timeout_seconds` | `120.0` | Proxy startup timeout; must be `>= 0`. |
| `shutdown_timeout_seconds` | `30.0` | Bounded in-flight shutdown wait; must be `> 0`. |

## `router`

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `enabled` | `false` | Whether Router control is enabled. |
| `base_url` | `null` | Router control API URL. |
| `control_timeout_seconds` | `10.0` | Router registration and cleanup timeout; must be `> 0`. |

`enabled` and `base_url` must be enabled or disabled together. Keep Router disabled for scheduler-only comparisons.

## Path Resolution

YAML `result_dir`, dataset paths, and an `agent.executable` containing directories resolve relative to the YAML file.
A bare executable name is found through `PATH`. Paths supplied as CLI overrides resolve from the invocation directory.

See [Benchmark CLI](benchmark-cli.md) for override fields and flag names.
