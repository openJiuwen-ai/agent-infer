# 用 AgentBench 运行 Workspace-Bench Full

在 Linux 仓库根目录执行以下命令。准备阶段需要能访问 GitHub、Hugging Face，并安装 Git 和 unzip。
两个语言版本的工作区压缩包合计约 38 GB，请同时为解压后的五种角色目录、标准副本及并发任务预留磁盘。
英文和中文使用各自的原始文件名。

```bash
DATA=agentinfer/agentbench/data/workspacebench
vllm bench serve --agentinfer prepare workspacebench --output-dir "$DATA"
```

已有指定提交的官方仓库时，给 prepare 添加 `--upstream-checkout /path/to/Workspace-Bench`。
下载中断后可用相同参数继续。数据目录的 `manifest.json` 记录任务、工作区与官方评测代码的版本。
每种语言各有 388 个实例，`task-lists/selected.txt` 默认选中 2、1 两个题号的英文和中文版本。
全量运行将 `--selection-path` 指向 `task-lists/all.txt`。

## 在宿主机运行 JiuwenSwarm

和 SWE-bench、Terminal-Bench 一样，先用 `prepare` 准备可复用的输入。Workspace-Bench 按语言和角色保存完整的标准目录。
每次 `run` 为各题复制一份可写工作目录，任务结束后收集交付物和轨迹，并清理副本。
文件系统支持写时复制时会自动使用，否则复制文件内容。同角色的下一题仍从标准目录开始。

激活安装了 AgentInfer、WorkSwarm 0.2.8 和 OpenJiuwen 0.1.20 的 Python 环境，并确保
`jiuwenswarm`、`jiuwenbox-server` 和 `bwrap` 可用。根据题目在宿主机准备 LibreOffice 和文档处理库。
Agent 工具继续通过 JiuwenBox 沙箱访问任务目录。

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/workspacebench_jiuwenswarm.yaml \
  --workspacebench-execution host \
  --base-url http://127.0.0.1:8000 --model YOUR_MODEL \
  --result-dir results/wsb-jiuwen-host
```

默认从本地索引同级的 `upstream/` 加载固定版本的官方 runner。若 prepare 使用了外部 checkout，
给 run 加上 `--workspacebench-checkout /path/to/Workspace-Bench`。需要评分时再准备下节的裁判镜像。

## 在容器中运行

使用 prepare 下载的官方仓库（或原先提供的外部 checkout）构建镜像。

```bash
WSB_UPSTREAM="$DATA/upstream"
docker build -t workspace-bench:local -f "$WSB_UPSTREAM/evaluation/docker/Dockerfile" "$WSB_UPSTREAM"
docker build -t agentinfer/workspacebench:3fbd0f1 -f docker/workspacebench.Dockerfile .
```

将模型地址和模型名换成实际部署值。命令使用 WorkSwarm 0.2.8、OpenJiuwen 0.1.20 和 `agent.code.normal`。
每题有独立容器、角色工作区副本和会话 ID，模型请求经过本机 AgentBench 代理。
Linux Docker 需要支持 host 网络和非特权用户命名空间，JiuwenBox 会在容器内使用 bubblewrap。
Jiuwen 任务容器使用非特权用户、丢弃全部 capability，并放宽外层 Docker 的 seccomp、AppArmor 和系统路径限制，以便内层沙箱建立独立的进程和挂载空间。

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/workspacebench_jiuwenswarm.yaml \
  --base-url http://127.0.0.1:8000 --model YOUR_MODEL \
  --result-dir results/wsb-jiuwen
```

同一配置可以通过下表中的参数切换运行时。`--agent-executable` 对 Jiuwen 和 ZCode 表示**镜像内**命令。
ZCode 需要在上述镜像上安装仓库支持的 Node 版 0.17.1 启动脚本及构建物，并用 `AGENTBENCH_WORKSPACEBENCH_IMAGE` 指定该镜像。
Claude Code 和 DSH 使用官方 Workspace-Bench SDK 适配。
任务容器对角色标准目录只读，只有本题的副本可写。[ZCode 构建要求](run-benchmark.md#zcode)

| Agent | `--agent-type` | `--agent-profile` | `--endpoint` |
| --- | --- | --- | --- |
| JiuwenSwarm | `jiuwenswarm` | `agent.code.normal` | `/v1/chat/completions` |
| Claude Code | `claude` | `single` | `/v1/messages` |
| DSH | `dsh` | `single` | `/v1/chat/completions` |
| ZCode | `zcode` | `single` | `/v1/chat/completions` |

## 独立评分与重评

配置受信任的 Anthropic 兼容裁判服务。裁判可读取题目输入、模型交付物和操作记录。通过环境设置 `JUDGE_BASE_URL`、`JUDGE_MODEL` 和 `JUDGE_API_KEY`，然后执行：

```bash
vllm bench serve --agentinfer evaluate --result-path results/wsb-jiuwen
```

也可以给 `run` 添加 `--with-correctness`。两种方式均在性能采集结束后评分。
自定义配置使用 `evaluate --config` 或 `run --evaluation-config`，字段与官方 judge YAML 对齐，包括 `baseUrl`、`model`、`apiKey`，可使用
`${ENV_VAR}` 引用。可选的 `model_name` 用作结果文件名，`timeout_seconds` 控制每题裁判时限。

| 产物 | 用途 |
| --- | --- |
| `requests.jsonl`、`summary.json` | 被测模型的请求和运行性能 |
| `workspacebench.json` | 本次选题、数据来源、执行方式及对应环境 |
| `tasks/<实例>/official/<题号>/output/` | 实际交付的文件和目录 |
| 同目录的 `agent.json` 和 `raw/` | 官方执行轨迹及原生 Agent 日志 |
| `evaluations/<评测ID>/correctness.json` | 分语言评分项通过率、覆盖率和逐题错误 |
| 各评测实例下的 `rubrics_judge--*.json` | 官方逐项裁判证据 |

每次重评创建独立评测目录，复用原始交付物。请保留准备好的原始输入数据。容器运行默认使用原镜像评分，宿主机运行可通过 `AGENTBENCH_WORKSPACEBENCH_IMAGE` 指定裁判镜像。
评分项通过率使用有效评分项作分母，覆盖率使用选中任务数作分母。裁判报错或缺少评分项时，命令返回非零并保留已评分任务的结果。
[官方评测代码](https://github.com/OpenDataBox/Workspace-Bench/tree/3fbd0f1a136720fece86786545983e26642c3db2/evaluation)
