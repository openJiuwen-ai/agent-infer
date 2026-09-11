# 运行基准测试

本指南使用同一批 Claude Code 工作负载，对比冷启动的上游 vLLM 异步调度器和 AgentInfer
Progress-TTL 调度桥。BenchKit 通过相同的 Request Proxy 记录两个实验组，避免观测路径差异。

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

DSH 对应参数为 `--agent-type dsh --agent-profile single --agent-executable dsh --endpoint /v1/chat/completions`。
通过 Router 运行时使用 `swebench_agentinfer.yaml`，设置 `backend.base_url` 和直接指向 vLLM 的 `backend.metrics_url`。Router
仅透明转发，不需要旧注册/清理接口。

已有请求 trace 可使用[Trace Replay](run-trace-replay.md)。

## 本地烟雾验证

执行 `python -m pytest tests/agentbench/test_benchmark_smoke.py -q`，无需模型服务即可验证
生产 CLI、配置、工作区、分派、请求代理及结果汇总。测试使用本地运行时和 loopback 后端，
不提供真实模型质量或性能结论。调度对比脚本使用目标仓库现有 controller factory 支持的字段。
