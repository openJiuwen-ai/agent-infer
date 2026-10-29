# 运行基准测试

本指南使用同一批 Claude Code 工作负载，对比冷启动的上游 vLLM 异步调度器和 AgentInfer
Progress-TTL Agent 感知调度器。BenchKit 通过相同的 Request Proxy 记录两个实验组，避免观测路径差异。

## 准备环境

需要以下环境：

- Linux、Python 3.10 或更高版本，以及 vLLM 0.23.0。
- 可运行目标模型的 CUDA GPU；默认脚本使用两个张量并行进程。
- Claude Code、tmux 和 Git，并已完成 Claude Code 身份验证。
- 支持 Anthropic `POST /v1/messages` 的模型。

在仓库根目录创建环境并安装 AgentInfer：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "vllm==0.23.0"
python -m pip install -e .
```

## 准备 SWE-bench 数据

```bash
vllm bench serve --agentinfer prepare swebench \
  --output-dir agentinfer/agentbench/data/swebench
```

命令会生成 `instances.jsonl`、`task-lists/default.txt` 和 `manifest.json`。

## 运行一次冷启动对比

仓库脚本会依次启动基线服务、运行任务、彻底停止服务，再启动候选服务并输出比较报告：

```bash
REPO=$PWD \
VENV=$PWD/.venv \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
TASK_NUM=1 \
CONCURRENCY=1 \
bash tests/agentbench/run-scheduler-e2e-compare.sh
```

模型适合单卡时可添加 `TENSOR_PARALLEL_SIZE=1`。需要向 vLLM 追加部署参数时，设置 `VLLM_EXTRA_ARGS`。

脚本执行以下流程：

1. 使用 `vllm.v1.core.sched.async_scheduler.AsyncScheduler` 和 `--async-scheduling` 启动基线。
2. 运行 BenchKit，并将请求直接发送至基线 vLLM。
3. 停止基线进程，清除进程内累计指标和 Prefix Cache 状态。
4. 使用 `AgentCacheAsyncSchedulerBridge`、Progress-TTL 控制器和生命周期中间件启动候选服务。
5. 使用相同任务、模型、并发度和配置运行候选组，然后比较两个已完成的结果目录。

如已有目标模型或虚拟环境位于其他路径，可将 `MODEL` 和 `VENV` 设为绝对路径。脚本会拒绝复用同名 tmux
会话或已存在的生命周期套接字，避免污染实验。

## 扩大实验规模

确认单任务烟雾测试完成后，提高任务数和并发度：

```bash
REPO=$PWD \
VENV=$PWD/.venv \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
TASK_NUM=64 \
CONCURRENCY=32 \
bash tests/agentbench/run-scheduler-e2e-compare.sh
```

发布决策应执行多次冷启动实验。每次实验使用新的结果目录，并保持提交、任务及顺序、模型、Agent profile、
超时、张量并行参数、vLLM 参数和硬件一致。

## 重新比较或汇总

比较一个或多个已完成实验：

```bash
vllm bench serve --agentinfer compare \
  --baseline results/vllm/run1 results/vllm/run2 \
  --candidate results/agentinfer/run1 results/agentinfer/run2
```

将运行摘要导出为 CSV：

```bash
vllm bench serve --agentinfer summarize \
  results/agentinfer/run1 results/agentinfer/run2 \
  --output-csv combined-summary.csv
```

命令参数见[基准命令行参考](../reference/benchmark-cli.md)，YAML 字段见
[基准配置参考](../reference/benchmark-config.md)，输出文件见[运行产物参考](../reference/run-artifacts.md)。
`completed` 只表示 Agent 进程完成，不表示补丁正确；发布结论前请阅读
[基准方法](../explanation/benchmark-methodology.md)。

## 选择 Agent 与透明 Router

默认配置位于 `agentinfer/agentbench/configs/swebench_vllm.yaml`。Claude Code 使用 `single` 或 `plan-subagent`，需要 Claude CLI 和
tmux。JiuwenSwarm 使用 `code.normal`，需要 JiuwenSwarm CLI；DSH 使用 `single` 或 `plan-subagent`，需要 DeepSeek Harness
CLI。切换运行时时同时指定类型、profile、可执行文件和匹配端点：

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --agent-type jiuwenswarm --agent-profile code.normal \
  --agent-executable jiuwenswarm --endpoint /v1/chat/completions \
  --task-num 1 --result-dir results/jiuwenswarm-smoke
```

DSH live run 固定支持 **0.2.1-rc.2**，Node 要求为 `^22.19.0 || >=24`。
先将受支持的 Node 加入 `PATH`；激活 Python 虚拟环境不会自动切换 Node。
即使 `--agent-executable` 使用绝对路径，DSH 和运行时预检仍会通过 `PATH` 查找 `node`。
在仓库根目录安装固定的外部运行时，并检查版本：

```bash
node --version
npm install --prefix tests/agentbench/agents/dsh/runtime
DSH_EXECUTABLE="$PWD/tests/agentbench/agents/dsh/runtime/node_modules/.bin/dsh"
"$DSH_EXECUTABLE" --version
```

先准备上述 SWE-bench 数据和 vLLM 服务，并在 `swebench_vllm.yaml` 中设置数据集路径、后端地址和模型名。
将下面字段合并到现有 YAML，确保 `agent:` 是未被注释的顶层配置：

```yaml
agent:
  type: dsh
  profile: plan-subagent
  context_window: 262144
  max_tokens: 8192
backend:
  endpoint: /v1/chat/completions
```

示例的上下文和输出容量须按实际后端调整；可用 `/v1/models` 返回的 `max_model_len` 核对上下文上限。
模型服务支持较长上下文时，过小的 `context_window` 会限制客户端的可用输出预算。
字段含义见[基准配置参考](../reference/benchmark-config.md)。

以下示例执行 2 个任务，任务并发为 1；子 Agent 可在同一任务内并行探索：

```bash
RESULT_DIR="results/dsh-plan-subagent-$(date +%Y%m%d-%H%M%S)"
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --base-url http://127.0.0.1:8000 \
  --agent-type dsh --agent-profile plan-subagent --endpoint /v1/chat/completions \
  --agent-executable "$DSH_EXECUTABLE" \
  --task-num 2 --max-concurrency 1 --result-dir "$RESULT_DIR"
```

结果目录必须尚不存在，由 BenchKit 创建。使用全局安装时，将 `--agent-executable` 改为实际 DSH 路径。
使用 `--agent-profile single` 可运行单 Agent 模式。`plan-subagent` 强制初始计划模式、子 Agent 只读工具和
计划审批；主 Agent 必须等待所有子 Agent 正常完成，再提交计划并获得审批。只有明确的 plan-review 问题会自动批准。

DSH 通过 pi-ai Chat Completions 请求 Request Proxy，prompt 经 stdin 发送，运行时版本和 npm lockfile 哈希
记录在 `evidence/dsh-runtime.json`。当前 DSH run 不支持非空的 `backend.api_key_env`。

DSH 默认收尾预算为 10 秒，可通过 YAML 的 `experiment.cleanup_timeout_seconds` 调整。
例如，日志较大或存储较慢时可设置：

```yaml
experiment:
  cleanup_timeout_seconds: 60.0
```

日志收集、补丁刷新等待和补丁导出共用该预算；刷新等待不超过 `patch_flush_seconds` 和预算的一半。
预算耗尽会记录为 `cleanup_timeout`，已有的执行超时或取消原因会保留。
清理详情见 `tasks/<instance_id>/dsh-run-status.json`，任务成功仍要求会话证据通过校验。

通过 Router 运行时使用 `swebench_agentinfer.yaml`，设置 `backend.base_url` 和直接指向 vLLM 的 `backend.metrics_url`。Router
仅透明转发，不需要旧注册/清理接口。

已有请求 trace 可使用[Trace Replay](run-trace-replay.md)。

## 本地烟雾验证

执行 `python -m pytest tests/agentbench/test_benchmark_smoke.py -q`，无需模型服务即可验证
生产 CLI、配置、工作区、分派、请求代理及结果汇总。测试使用本地运行时和 loopback 后端，
不提供真实模型质量或性能结论。调度对比脚本使用目标仓库现有 controller factory 支持的字段。

## DSH 并发运行记录（2026-10-29）

本节记录 `test-dsh-20261029-102951` 的最终结果，作为执行链路的观测实例。
运行时间为 2026-10-29 10:29:52 至 11:30:06（Asia/Shanghai）。
使用 SWE-bench Verified 的 16 个 Astropy 实例，任务并发为 8，单任务超时为 3600 秒，未另设整次运行超时。
代理为 DSH `0.2.1-rc.2` 的 `plan-subagent`，Node `24.22.1`，Python `3.12.13`；
模型为 `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`，上下文配置为 `262144`，输出上限为 `8192` tokens。

### 结果摘要

| 指标 | 本次结果 |
| --- | --- |
| 任务执行完成 / 失败 | 14 / 2；执行完成率 87.5% |
| 导出补丁 | 14 个任务 |
| 整次运行耗时 | 3614.25 秒，约 60 分 14 秒 |
| 单任务耗时 | 平均 739.97 秒；P50 538.78 秒 |
| 模型请求 | 1,972 次；成功 1,971 次，失败 1 次 |
| 请求耗时（Request Proxy） | 平均 8.333 秒；P50 3.897 秒；P95 34.876 秒 |
| TTFT（Request Proxy） | 平均 1.201 秒；P50 0.844 秒；P95 2.887 秒 |
| 输入 / 输出 tokens | 122,046,278 / 295,020 |
| 已完成请求的缓存命中 tokens | 119,865,936 |
| 前缀缓存查询 token 命中率 | 98.22%，按 vLLM 起止计数器差值计算 |
| 运行收尾 | 正常完成；1,972 条请求事实全部写入，无待写入记录或收尾异常 |

`manifest.status=completed` 表示整批流程已收尾；各任务状态以 `task_index.json` 为准。
本轮没有正确性评测产物，`correctness.available=false`，因此 87.5% 仅为代理执行完成率。
补丁正确性仍需在匹配的 SWE-bench 实例环境中评测。

### 两个失败任务

| 实例 | 最终分类 | 原因与证据 |
| --- | --- | --- |
| `astropy__astropy-13033` | `timeout` | 一个只读子代理持续重复读取同一文件，导致 DSH 无法结束；最终触发 3600 秒单任务超时，无补丁。 |
| `astropy__astropy-13579` | `harness_error` | 4 个子代理均正常结束，但主代理未完成计划审批即退出，触发 `plan-subagent completed without an approved plan` 校验，无补丁。 |

`13033` 的异常子代理为 `55c296a1-ec3f-492f-b021-4ccbc0fe892d`。
其持久化日志记录了 939 次 `read`，末尾仍在交替读取 `astropy/timeseries/core.py` 的相同两段内容。
该子代理关联 953 次模型请求，其中 952 次成功，最后一次在超时收尾时被取消，记录为 `request cancelled`。
该失败请求的 HTTP 状态虽然为 200，但流式响应未完整结束，仍按失败请求统计。
长尾任务显著影响整批耗时和聚合指标，不能据本轮数据推断正常工作负载的并发扩展收益。

### 参数与证据范围

同参数重跑时，先按前文准备受支持的 Node、DSH、数据集和后端，并将 YAML 中的上下文设为实际模型容量。
使用新结果目录，例如：

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --base-url http://127.0.0.1:8000 \
  --agent-executable "$DSH_EXECUTABLE" \
  --task-num 16 --max-concurrency 8 --timeout 3600 \
  --result-dir "results/dsh-16-c8-$(date +%Y%m%d-%H%M%S)"
```

此命令用于重现参数，不表示上述循环问题已修复。本节未重新执行任务，也未对两份失败结果做成功回填。
收尾时采集的工作区快照为 `b74bd42b221804f3ba6b5fabe7fb6b559665250e`，`dirty=true`；
该证据未保存本地修改差异，不能据此证明某个干净提交已完整通过本轮验收。

原始产物位于本机结果目录 `/home/yangjianjuan/AgentInfer/test-dsh-20261029-102951`，未纳入代码仓库。
以下路径均相对于该结果目录；跨机器评审需另行提供原始产物：

- `manifest.json`、`summary.json`、`task_index.json`：运行配置、汇总指标和 16 个任务的最终状态。
- `requests.jsonl`、`evidence/vllm_metrics_start.prom`、`evidence/vllm_metrics_end.prom`：请求事实与计数器差值来源。
- `evidence/dsh-runtime.json`、`evidence/source_control.json`：运行时预检和工作区快照。
- `tasks/<instance_id>/` 下的 `result.json`、`dsh-run-status.json`、`dsh-lineage.json`：逐任务状态与计划审批记录。
- `tasks/astropy__astropy-13033/dsh-sessions/session-55c296a1-ec3f-492f-b021-4ccbc0fe892d.jsonl`：循环子代理日志。
