# 运行基准测试

本指南介绍 Claude Code、JiuwenSwarm、DSH 和 ZCode 的安装与基准运行。
BenchKit 通过 Request Proxy 记录模型请求，并保存任务结果、日志和代码补丁。
本指南同时介绍 SWE-bench Verified 精度评测，包括运行后自动评分。

## 准备环境

需要以下环境：

- Linux、Python 3.10 或更高版本，以及 vLLM 0.23.0。
- 可运行目标模型的 CUDA GPU。
- Git，以及下文所选 Agent 的命令行和运行依赖。
- 支持工具调用的模型服务：Claude Code 使用 `/v1/messages`，其余三种使用 `/v1/chat/completions`。

Claude 的 `plan-subagent` 模式必须使用非 root 账号运行（`id -u` 不为 `0`）。该模式使用
`--permission-mode bypassPermissions`，以 root/sudo 身份启动会被 Claude 拒绝，并提示
`--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons`。
请切换到有代码、数据读取权限及工作区、结果目录写入权限的普通账号。使用 `--with-correctness` 时，
同一账号还必须能访问 Docker，可先运行 `docker info` 验证；否则生成结束后，在有 Docker 权限的环境单独运行 `evaluate`。

Claude 启动时，框架会在工作区信任提示中选择 `Yes, I trust this folder`，确认选中后再继续，
兼容默认选中 `No, exit` 的界面。无法识别选项时不自动确认。
终端显示 `esc to interrupt` 时视为仍在执行，不因同时显示输入框而触发空闲终止。
计划审批提示兼容终端换行与大小写变化，包括 `Yes, and switch to BYPASS PERMISSIONS` 选项。
同一审批提示在确认后持续显示 5 秒时，最多重试一次，避免界面尚未就绪时丢失按键导致停滞。
文件编辑等已高亮 Yes 的确认框也采用相同的受限重试；提示消失后重置，未选中 Yes 时不重试确认。

在仓库根目录创建环境并安装 AgentInfer：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "vllm==0.23.0"
python -m pip install -e .
```

## 安装 Agent

按需安装一种即可。以下命令面向 Linux Bash；安装 Agent 不会安装目标任务所需的测试依赖。
AgentInfer 的 Python 环境与 Agent 自身的环境可以分开，运行时用 `--agent-executable` 指定完整路径。

| Agent | 本仓适配要求 | 额外依赖 |
| --- | --- | --- |
| Claude Code | 未锁定 CLI 版本；记录 `claude --version` 并先做单任务验证 | tmux、bubblewrap、socat |
| JiuwenSwarm | `code.normal`；按 CLI 能力校验，未锁定包版本 | 独立 Python 环境、jiuwenbox-server、bubblewrap |
| DSH | 固定 `0.2.1-rc.2`；其他版本预检不通过 | Node `^22.19.0` 或 `>=24`，可用的本地沙箱 |
| ZCode | 固定 CLI `0.17.1`，使用下文指定源码构建 | Node `24.14.0`、pnpm `10.33.2`，评测用外部沙箱或容器 |

### 系统依赖与 Node

Ubuntu/Debian 可安装以下系统包；其他发行版使用对应包管理器。`tmux`、`socat` 用于 Claude，
`bubblewrap` 提供 `bwrap` 命令；编译工具用于安装原生依赖。

```bash
sudo apt-get update
sudo apt-get install -y git curl tmux bubblewrap socat build-essential pkg-config python3-dev python3-venv
```

DSH 和 ZCode 可共用 Node `24.14.0`。以下使用 [nvm 官方安装方式](https://github.com/nvm-sh/nvm#install--update-script)，
已有 nvm 时跳过第一行：

```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.41.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"
nvm install 24.14.0
nvm use 24.14.0
node --version
npm --version
```

运行基准的终端也要使用这个 Node。激活 Python 虚拟环境不会切换 Node；
DSH 即使使用绝对路径启动，预检仍会通过 `PATH` 查找 `node`。

### Claude Code

使用 [Claude 官方原生安装器](https://code.claude.com/docs/en/setup#install-a-specific-version)，不需要 Node：

```bash
curl -fsSL https://claude.ai/install.sh | bash -s stable
export PATH="$HOME/.local/bin:$PATH"
CLAUDE_EXECUTABLE="$(command -v claude)"
"$CLAUDE_EXECUTABLE" --version
tmux -V
bwrap --version
socat -V
```

`stable` 用于首次安装；复现实验时，将它替换为已验证的具体 CLI 版本，并保存版本输出。
本仓预检不锁定 Claude 版本，安装成功不代表任意新版本都已通过评测。
使用本地无鉴权 vLLM 时，适配器会设置测试用占位凭据，无需登录 Anthropic；
如接入需要鉴权的服务，应提供对应凭据。`plan-subagent` 使用 `bypassPermissions`，
应在非 root 用户下运行。

### JiuwenSwarm

官方 Python 包名是 [`workswarm`](https://pypi.org/project/workswarm/0.2.8/)。
以下安装示例固定为 `0.2.8`，该发布包要求 Python `>=3.11,<3.14`，这里使用已安装的 Python 3.12：

```bash
JIUWEN_VENV="$HOME/.venvs/agentbench-jiuwenswarm"
python3.12 -m venv "$JIUWEN_VENV"
"$JIUWEN_VENV/bin/python" -m pip install "workswarm==0.2.8"
"$JIUWEN_VENV/bin/python" -m pip show workswarm
JIUWEN_EXECUTABLE="$JIUWEN_VENV/bin/jiuwenswarm"
"$JIUWEN_EXECUTABLE" --help
"$JIUWEN_EXECUTABLE" chat --help
test -x "$JIUWEN_VENV/bin/jiuwenswarm-init"
test -x "$JIUWEN_VENV/bin/jiuwenswarm-start"
"$JIUWEN_VENV/bin/jiuwenbox-server" --help
```

包名和入口说明见 [JiuwenSwarm 官方仓库](https://github.com/openJiuwen-ai/jiuwenswarm)。
`chat --help` 必须包含 `--jsonl`、`--mode`、`--session`、`--cwd`、`--project-dir`、`--gateway-url` 和 `--timeout`。
缺少命令或参数时，需选择包含这些能力的版本；本节未完成 `0.2.8` 的真实任务验证。

运行时将 `--agent-executable` 设为 `"$JIUWEN_EXECUTABLE"`。AgentBench 会为每项任务调用初始化程序、
生成独立配置并启动服务，无需预先手动运行 `jiuwenswarm-init` 或 `jiuwenswarm-start`。

### DSH

先按上文准备 Node，再安装仓库固定支持的
[`@deepseek-ai/dsh@0.2.1-rc.2`](https://www.npmjs.com/package/@deepseek-ai/dsh/v/0.2.1-rc.2)：

```bash
node --version
npm install --global @deepseek-ai/dsh@0.2.1-rc.2
DSH_EXECUTABLE="$(command -v dsh)"
"$DSH_EXECUTABLE" --version
"$DSH_EXECUTABLE" --profile headless --help
```

版本应为 `0.2.1-rc.2`，headless 帮助应包含 JSON 输出、stdin 和会话参数。
AgentBench 还会在启动前验证实际沙箱能力。日志解压所需的 Python `zstandard` 已在 AgentInfer 依赖中声明，
应安装在运行基准的 Python 环境中。

### ZCode 源码构建与启动脚本

使用上文的 Node `24.14.0`，按 [ZCode 对应源码的构建说明][zcode-build] 安装：

```bash
node --version
npm install --global pnpm@10.33.2
pnpm --version
ZCODE_SOURCE="$HOME/.local/src/agentbench-zcode"
mkdir -p "$HOME/.local/src"
git clone https://github.com/zai-org/ZCode.git "$ZCODE_SOURCE"
git -C "$ZCODE_SOURCE" checkout --detach a9fe401eede6e6e5e01be9048679c206fe6f02d9
(
  cd "$ZCODE_SOURCE"
  pnpm install --frozen-lockfile
  pnpm --filter '@zcode/cli...' build
)
```

安装过程需要下载依赖和运行资源，不要跳过安装脚本。构建后保留整个源码目录及相邻资源，
不要只复制 `zcode.cjs`。下面生成启动脚本，把 Node 和构建物的绝对路径固定下来：

```bash
ZCODE_NODE="$(command -v node)"
ZCODE_CLI="$ZCODE_SOURCE/apps/zcode-cli/packages/cli/dist/zcode.cjs"
ZCODE_EXECUTABLE="$HOME/.local/bin/agentbench-zcode"
mkdir -p "$HOME/.local/bin"
printf '#!/bin/bash\nexec %q %q "$@"\n' "$ZCODE_NODE" "$ZCODE_CLI" > "$ZCODE_EXECUTABLE"
chmod +x "$ZCODE_EXECUTABLE"
"$ZCODE_EXECUTABLE" --version
git -C "$ZCODE_SOURCE" rev-parse HEAD
sha256sum "$ZCODE_CLI"
```

版本输出必须为 `0.17.1`，不能用仓库根目录的版本 `3.17.2` 代替 CLI 版本。
运行时将 `--agent-executable` 设为 `"$ZCODE_EXECUTABLE"`；启动脚本保留 `NODE_OPTIONS` 并转发全部参数，
以支持适配器加载长提示词。该脚本仅负责启动，正式评测还需外部沙箱或容器。
版本校验和运行限制见下文 [ZCode](#zcode)。

[zcode-build]: https://github.com/zai-org/ZCode/blob/a9fe401eede6e6e5e01be9048679c206fe6f02d9/apps/zcode-cli/docs/usage/01-quickstart.md

## 准备 SWE-bench 数据

```bash
vllm bench serve --agentinfer prepare swebench \
  --output-dir agentinfer/agentbench/data/swebench
```

命令会生成 `instances.jsonl`、`task-lists/default.txt` 和 `manifest.json`。

## 验证 SWE-bench Verified 任务精度

使用官方 SWE-bench **4.1.0** 评测已有补丁，仅选择 outcome=completed 的任务。
整批运行可以包含失败任务，评测不重新调用模型。5.x 要求不同的数据集字段。
评测编排面向 Linux，使用 POSIX 进程组执行有界清理；Windows 请使用 WSL2 等 Linux 环境。

配置文件可选。省略时使用运行 AgentInfer 的当前 Python 环境，该环境需要安装 swebench==4.1.0：

```bash
vllm bench serve --agentinfer evaluate --result-path results/run-a
```

默认并发为 2，单实例超时 1800 秒，整体超时 7200 秒。
使用独立解释器或自定义参数时，再通过 --config 指定配置文件。

### 准备与执行

如需独立评测环境，安装固定版本，确保 Docker 可访问，并保存配置 evaluation.yaml：

```bash
uv venv --python 3.12 /path/to/swebench-venv
uv pip install --python /path/to/swebench-venv/bin/python 'swebench==4.1.0'
```

```yaml
harness_python: /path/to/swebench-venv/bin/python
expected_harness_version: "4.1.0"
max_workers: 2
timeout_seconds: 1800
overall_timeout_seconds: 7200
cleanup_timeout_seconds: 30
```

```bash
vllm bench serve --agentinfer evaluate \
  --result-path results/run-a \
  --config evaluation.yaml \
  --output-dir results/run-a/evaluations/eval-001
```

输入需要已结束的 manifest/summary、任务索引、逐任务 result/prompt 和 manifest 引用的
本地 Verified 数据集。核对任务 ID、仓库和 base commit；无成功任务时不生成成绩。
解释器相对路径以配置文件目录为基准，未知字段和不支持的版本会被拒绝。
预检查先确认已安装的 SWE-bench 版本，匹配后才导入评分模块并采集环境信息；版本不匹配时不启动评分。

省略 --output-dir 时，在原 run 的 evaluations/ 中创建唯一目录；已有目录不覆盖。
原始补丁、manifest 和性能 summary 不修改。首次使用会下载官方 Docker 镜像。
结束时批量清理本次评测遗留的容器；清理失败或超时会记录到评测 manifest，并使评测标记为 failed。

### 运行后自动评测

在 `run` 中添加 `--with-correctness`，可在一次命令中完成任务运行与精度评测：

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --result-dir results/run-a \
  --with-correctness
```

使用独立评测环境时，再添加 `--evaluation-config evaluation.yaml`。
`--config` 是运行配置，`--evaluation-config` 是上述评测配置；后者省略时使用当前 Python 和默认评测参数。
单独传入 `--evaluation-config` 而未开启 `--with-correctness` 会返回参数错误 2。
评测配置在模型运行前加载并校验；Docker 和已安装的 SWE-bench 等评分环境检查仍在评测阶段进行。

运行完成性能统计、产物写入和资源清理后，自动评测实际运行目录中的 completed 任务。
结果保存至该目录的 `evaluations/` 下，终端打印结果路径及精度摘要。
评分不计入运行阶段的耗时，评测超时和并发使用评测配置，原始性能 summary 和补丁保持不变。

未开启开关时沿用原有运行方式。运行编排异常或中断时，跳过自动评测并保留原始失败；
完整收尾的有效产物仍可事后使用独立 `evaluate` 评分。
部分任务失败但整批编排正常结束时，只评 completed 子集；没有 completed 任务时返回 1，不生成成绩。
正常评分中的 unresolved 或 empty_patch 返回 0，这不表示所有补丁均正确；评分错误和中断使用下述退出码。

### 结果

每个入选任务在 task_results.jsonl 中有一条记录。correctness.json 的 counts 仅包含
total、resolved、unresolved、empty_patch、error 五项。唯一汇总精度指标为
resolved_rate = resolved / total，error 不从分母删除。
该分数表示成功完成任务子集的解决率，不代表整个 Verified 数据集。

没有补丁文件时，只有 result 与 DSH 导出状态共同证明 no_changes 才计为空补丁；
否则计为 error。失败任务即使有补丁也不入选。F2P/P2P 明细保留在 raw/ 下官方逐题报告中。
实际命令、日志、版本、镜像身份和文件哈希全部保留。

退出码：0 表示评测无错误，unresolved 是正常结果；1 表示输入、执行或评测错误；
2 表示配置错误；130 表示中断。确定任务集合后的错误仍保存汇总。

### 对比

```bash
vllm bench serve --agentinfer compare-correctness \
  --baseline results/baseline/evaluations/eval-001/correctness.json \
  --candidate results/candidate/evaluations/eval-001/correctness.json
```

核对文件哈希、任务集合、题目内容、已记录的生成配置/提示词、评测环境和镜像身份。
范围不同或证据不完整时返回 insufficient_evidence，不静默取交集。
来源产物未记录的模型 checkpoint revision 无法被独立验证。

输出逐题退化和改善列表，即使总分持平也能发现退化。任何任务退化都会导致比较失败，改善不能抵消退化。
退出码为 0（通过）、1（发现任务退化）
或 2（证据不足）。该门禁只适用于固定任务集，不是模型总体能力的统计结论。

参考：[官方指南](https://www.swebench.com/SWE-bench/guides/evaluation/)、
[benchmark 方法](../explanation/benchmark-methodology.md)。

## 选择 Agent 与透明 Router

默认配置位于 `agentinfer/agentbench/configs/swebench_vllm.yaml`。Claude Code 使用 `single` 或 `plan-subagent`，需要 Claude CLI 和
tmux。JiuwenSwarm 使用 `code.normal`；DSH 和 ZCode 使用 `single` 或 `plan-subagent`。
按[安装 Agent](#安装-agent)准备对应命令行后，切换运行时时同时指定类型、profile、可执行文件和匹配端点：

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --agent-type jiuwenswarm --agent-profile code.normal \
  --agent-executable "$JIUWEN_EXECUTABLE" --endpoint /v1/chat/completions \
  --task-num 1 --result-dir results/jiuwenswarm-run
```

DSH 的安装和版本要求见上文 [DSH](#dsh)。以下使用安装时保存的 `DSH_EXECUTABLE`。

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

通过 `--task-num` 和 `--max-concurrency` 调整任务数和任务并发，每次使用新的结果目录。
DSH 当前仅支持无鉴权模型服务，`backend.api_key_env` 应为空。
运行时版本记录在 `evidence/dsh-runtime.json`，步骤上限和循环终止原因见
[运行产物参考](../reference/run-artifacts.md)。

DSH 默认收尾预算为 10 秒，可通过 YAML 的 `experiment.cleanup_timeout_seconds` 调整。
例如，日志较大或存储较慢时可设置：

```yaml
experiment:
  cleanup_timeout_seconds: 60.0
```

日志收集、补丁刷新等待和补丁导出共用该预算。预算耗尽会记录为 `cleanup_timeout`，
详情见 `tasks/<instance_id>/dsh-run-status.json`。

通过 Router 运行时使用 `swebench_agentinfer.yaml`，设置 `backend.base_url` 和直接指向 vLLM 的 `backend.metrics_url`。Router
仅透明转发，不需要旧注册/清理接口。

已有请求 trace 可使用[Trace Replay](run-trace-replay.md)。

## ZCode

按上文[源码构建与启动脚本](#zcode-源码构建与启动脚本)完成安装。
本接入对应 [RFC #71](https://github.com/openJiuwen-ai/agent-infer/issues/71)，在 Linux 上支持
`single` 和 `plan-subagent`；原生 SEA 二进制不支持此接入方式。

预检要求 CLI 版本精确匹配 `0.17.1`。其他版本会报 `Expected ZCode CLI 0.17.1, got ...`，
任务不会开始执行。预检只校验版本和所需能力，**不验证构建物来自哪个 Git 提交**；
`launcher_sha256` 是启动脚本摘要，复现实验还需保存 `zcode.cjs` 摘要和源码版本。
预检使用临时工作目录、独立 HOME 和 ZCode 状态路径。

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --agent-type zcode --agent-profile single \
  --agent-executable "$ZCODE_EXECUTABLE" \
  --endpoint /v1/chat/completions --task-num 1 --result-dir results/zcode-single
```

配置中的 `context_window`、`max_tokens` 须匹配后端模型。当前仅支持无鉴权本地服务，
`backend.api_key_env` 应为空。每项任务使用独立的配置、状态目录和日志。

`single` 禁止委派；`plan-subagent` 默认委派一次只读 Explore，批准计划后由主 Agent
修改代码。Explore 只能使用 Read、Glob、Grep，最多执行 8 轮工具调用，再返回报告。
两种模式均关闭后台任务。
权限模式不提供操作系统隔离，正式评测须使用外部沙箱或容器。

可在 YAML 的 `agent` 下设置两个正整数上限，或使用 CLI 的 `--max-subagents` 和
`--max-concurrent-subagents` 覆盖。例如：

```yaml
agent:
  type: zcode
  profile: plan-subagent
  max_subagents: 4
  max_concurrent_subagents: 2
```

上述配置允许每题最多接受四次探索委派、最多同时运行两个；默认均为 1，不要求用满预算。
并发上限不得超过累计启动上限。
子 Agent 可以在同一次响应中并行启动，也可分批启动。满额时在执行前拒绝新调用，不排队；
失败的委派仍消耗累计预算。已有成功报告后可继续探索或提交计划，但必须等全部子 Agent 完成且无失败才能批准。
计划批准后禁止继续委派。非默认上限仅支持 ZCode 的 `plan-subagent`，`single` 始终禁用委派。
`--max-concurrency` 仍控制任务并发，与每任务子 Agent 并发分别计数。
`zcode-lineage.json` 记录限额、累计启动数、峰值并发及各次委派状态；超时或取消时保留未完成委派记录。

任务目录包含原始 `transcript.jsonl`、`zcode.stderr.log`、`zcode-hooks.jsonl`、`zcode-lineage.json`、
`zcode-config.json` 和 `zcode-run-status.json`。代码修改仍导出为 `model.patch`；当前会话自动生成的计划文件
不会混入补丁。`completed` 表示执行完成，代码是否正确仍需独立评分。

排查失败时，先查看 `zcode-run-status.json` 中的 `native_error`、`last_model_error` 和
`model_log_warnings`，分别了解原生错误、模型调用错误和日志解析问题，再结合原始日志。
有解析告警时，错误计数为 0 不能说明没有模型错误。

## 比较或汇总结果

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
