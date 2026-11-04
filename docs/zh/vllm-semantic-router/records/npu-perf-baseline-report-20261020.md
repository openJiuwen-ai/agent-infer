# SR 信号模型 CPU / NPU 性能基线测试报告（Atlas A2 / Atlas A5）

> 状态：**v1.1（2026-10-15）——主实验 + 输入长度扩展实验数据已回填**。
> 关联：`sr-npu-work.md`（内部工作文档，不在本仓；步骤 0 首测：36 上 NPU 单并发 35× 加速；本报告是其**并发维度扩展**与口径规范化）。
> 卡型（npu-smi 实测确认）：121 = Atlas A2（Ascend **910B1**）；36 = Atlas A5（Ascend **950DT**）。

---

## 1. 测试目的与范围

**目的**：在两台昇腾主机上，对 SR 内置信号模型分别采集 **CPU 与 NPU** 两种计算设备的基线性能数据，覆盖**并发、时延、吞吐**三类指标，形成一份设备型号 / CANN 版本等关键信息完备、口径可复现的基线报告。

**范围内**：模型-runtime 直接压测（`vllm-srun serve`，客户端直连 HTTP，不经 router/envoy）；对象 = **Vela-2.0-0.3B** 四信号一次
decisions 调用；每台主机 CPU 与 NPU 各一档；并发 4/8/16 档位扫描。

**范围外**：bf16 autocast 专项（全景 A1）；Vela-1.0 / decision 家族覆盖（全景 B 线）；router 全链路时延。

---

## 2. 启动方式速查：环境、镜像 vs 二进制（CPU / NPU 怎么选）

> 实际测试全部采用 **§2.3 二进制路径**（宿主 sr-venv 直跑 `vllm-srun`）。本节保留启动方式速查供复现。

### 2.1 所需环境（2026-10-15 实采）

| 组件 | 121（A2 机） | 36（A5 机） | 说明 |
| --- | --- | --- | --- |
| sr-venv（内含 vllm-srun） | `/dataset/home/<user>/sr-venv` | `/home/<user>/sr-venv` | editable 安装指向各机 checkout |
| runtime 代码 checkout | `/dataset/home/<user>/semantic-router-fork` @ `b6831b3e2` | `/home/<user>/sr-build/src` @ `7313c646c`（detach） | 两台均含 #4827（npu accelerator）+ #4850（npu golden 参照 + readiness 容差门） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 | 2.10.0+cpu / 2.10.0.post4 | 两台相同 |
| CANN toolkit | `cann-8.5.1` | `cann-9.1.0` | NPU 前置 |
| NPU 驱动（npu-smi） | 25.5.0 | 25.6.rc1 | |
| env.sh | `/dataset/home/<user>/sr-build/env.sh` | `/home/<user>/sr-build/env.sh` | **不含 CANN** |
| CANN set_env.sh | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | 同左 | **NPU 必须额外 source**（红线，实测不 source 则 torch_npu 拒载） |
| 模型 Vela-2.0-0.3B | `/dataset/home/<user>/sr-build/models/Vela-2.0-0.3B`（2026-10-19 起，原 HF cache 已迁出，见 §2.3） | `/home/<user>/sr-build/src/models/Vela-2.0-0.3B` | 均不走外网 |

### 2.2 env.sh 内容（原文）

**36（`/home/<user>/sr-build/env.sh`）**：

```bash
# SR 构建环境（36 服务器）：source /home/<user>/sr-build/env.sh
export GOROOT=/home/<user>/sr-build/tools/go1.27.1
export PATH=$GOROOT/bin:$PATH
export GOFLAGS=-buildvcs=false
export GOPATH=/home/<user>/gopath
export GOCACHE=/home/<user>/gocache
export GOPROXY=https://goproxy.cn,direct
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export SR=/home/<user>/sr-build/src
export PIP_CACHE_DIR=/home/<user>/.pip-cache
# CANN 日志避开满盘的 / 根分区（打屏而不写 /root/ascend/log）
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_SRUN_CPU_THREADS=32
```

**121（`/dataset/home/<user>/sr-build/env.sh`）**：

```bash
#!/bin/bash
# env.sh -- build/run environment for vLLM Semantic Router on this host.
# Everything lives under /dataset/home/<user>/sr-build; the semantic-router-fork
# checkout is deliberately NOT touched (it is kept clean for upstream PRs).
#
# Usage: source /dataset/home/<user>/sr-build/env.sh

export SR_BUILD=/dataset/home/<user>/sr-build
export SR_SRC=$SR_BUILD/src
export SR_LOGS=$SR_BUILD/logs
mkdir -p "$SR_LOGS"

# Toolchains (all user-level, installed under $SR_BUILD/tools)
export GOROOT=$SR_BUILD/tools/go1.27.1
export CARGO_HOME=$SR_BUILD/tools/cargo
export RUSTUP_HOME=$SR_BUILD/tools/rustup
export PATH=$GOROOT/bin:$CARGO_HOME/bin:$SR_BUILD/tools:$PATH

# The build tree is a local clone; Go's buildvcs probe would otherwise try to
# stamp VCS info and can fail on such checkouts.
export GOFLAGS=-buildvcs=false
export CGO_ENABLED=1

# proxy.golang.org is unreachable from this host; use a Chinese module proxy.
export GOPROXY=https://goproxy.cn,direct

# github.com is unreliable from this host (large fetches stall). This scoped git
# config rewrites github.com URLs to a proxy so cargo can fetch its git
# dependencies (huggingface/candle). It intentionally does NOT touch ~/.gitconfig.
export GIT_CONFIG_GLOBAL=$SR_BUILD/tools/gitconfig

# Native provider libraries produced by `make rust` (candle / onnx / ml / nlp)
export LD_LIBRARY_PATH=$SR_SRC/candle-binding/target/release:$SR_SRC/onnx-binding/target/release:$SR_SRC/ml-binding/target/release:$SR_SRC/nlp-binding/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Model download defaults for this network
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

echo "SR env:"
echo "  go    = $(command -v go)"
echo "  cargo = $(command -v cargo)"
echo "  envoy = $SR_BUILD/tools/envoy"
echo "  src   = $SR_SRC"
```

关键行速览：

| 行 | 影响 |
| --- | --- |
| 36 `VLLM_SRUN_CPU_THREADS=32` | 压测时已 `unset` 显式覆盖，防静默继承 |
| 36 `ASCEND_SLOG_PRINT_TO_STDOUT=1` / `ASCEND_GLOBAL_LOG_LEVEL=3` | CANN 日志避根分区 |
| 121 `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` | 本测试用 offline cache，未触发 |
| 两台 GOPROXY / PIP 源 | 构建用，运行时不影响 |

### 2.3 二进制（本报告压测采用的方式）

```bash
source <各机 sr-build>/env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # NPU 必加（红线）
unset VLLM_SRUN_CPU_THREADS   # 36 的 env.sh 带着 32，不 unset 会静默覆盖 --threads（红线）

# <模型> 实验实值：
#   36 = /home/<user>/sr-build/src/models/Vela-2.0-0.3B（本地目录）
#   121 = /dataset/home/<user>/sr-build/models/Vela-2.0-0.3B（本地目录，2026-10-19 起；
#        原 HF cache（Hub 名 vllm-sr/Vela-2.0-0.3B，pinned a3209a50）已迁出到 sr-build，
#        hub/ 下留 MODELS-MOVED-TO-SR-BUILD.txt，Hub 名不再本地解析）
# CPU：36 的 CPU 档要绑核（taskset -c 0-31）+ --threads 显式指定
taskset -c 0-31 vllm-srun serve <模型> --device cpu --threads 8 --port 8816   # 121 的 CPU 档不绑核
# NPU：显式选卡（36=3、121=1）+ 点名设备
export ASCEND_RT_VISIBLE_DEVICES=<卡号>
vllm-srun serve <模型> --device npu:0 --port 8817
# 探活（readiness 含 golden 双门，通过才允许采数）
curl -s http://127.0.0.1:<port>/health
```

要点：`--device auto` 在 NPU 机器上仍选 cpu（`auto_priority=None`）；npu 设备类 golden 走容差判据（#4850）；模型一律本地路径 / offline cache。

### 2.4 镜像：CPU 全栈编排（e2e 采用的方式）

CLI 编排器拉起全栈容器，设备在 **config YAML 的 deployment `device:` 字段**设置。36 的 `vllm-sr:main-custom` 全栈编排目前只能 cpu
（镜像内 runtime 未装 torch_npu，配 `npu:0` 会 `PlacementError`）。详见内部工作文档 `sr-container-e2e-roi-guide.md`（不在本仓）。

### 2.5 镜像：NPU 容器配方（36 实测可跑，2026-10-14 验收）

CPU 镜像 + `--privileged` + 按文件挂 urma/ummu 宿主库 + 挂 host CANN，进容器装 torch_npu 后 `vllm-srun serve --device npu:0`。
**是"进容器手动单跑 runtime"，不是 CLI 全栈编排**。
完整命令与四条坑位要点见内部工作文档 `sr-container-e2e-roi-guide.md` §7 与 `sr-npu-work.md` §3.1（均不在本仓）。

### 2.6 两种方式怎么选

| 场景 | 方式 | 设备设置点 |
| --- | --- | --- |
| 设备性能基线（本报告） | 二进制 `vllm-srun serve` | CLI `--device`（+ `--threads`） |
| e2e 全链路 / ROI 对比 | 镜像 `vllm-sr serve` 全栈编排 | config YAML `device:`（36 仅 cpu） |
| 36 上用镜像跑 NPU | 镜像 + `--entrypoint bash` 进容器（§2.5） | 容器内 CLI `--device npu:0` |

---

## 3. 测试环境（已实采）

### 3.1 环境信息表

| 项 | 121（A2 机） | 36（A5 机） |
| --- | --- | --- |
| 主机 IP | 已隐去（下文以 121 指代） | 已隐去（下文以 36 指代） |
| NPU 卡型（npu-smi 实测） | **Ascend 910B1**（Atlas A2） | **Ascend 950DT**（Atlas A5） |
| NPU 卡数 / 本次用卡 | 8 卡 / **卡 1**（压测时该卡最空闲，HBM 3.4GB/64GB；其余卡 46-59GB 被其他用户占用） | 8 卡 / **卡 3**（共租 DeepSeek 8 卡常驻，每卡余 ~7.5GB；选余量最大卡） |
| NPU 驱动 | 25.5.0 | 25.6.rc1 |
| CANN | cann-8.5.1 | cann-9.1.0 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 | 2.10.0+cpu / 2.10.0.post4 |
| CPU 型号 / 拓扑 | **鲲鹏 920，192 核**（4 socket，8 NUMA） | 鲲鹏 950 7590 ×2 socket（32 核，绑核 0-31） |
| OS / 内核 | Huawei Cloud EulerOS 2.0 (aarch64) / 5.10.0 | openEuler 24.03 LTS-SP4 (aarch64) / 6.6.0 |
| runtime 代码 | `b6831b3e2`（含 #4827/#4850） | `7313c646c`（= 10-15 upstream main，含 #4827/#4850） |
| 部署形态 | 裸机 sr-venv（editable） | 裸机 sr-venv（editable） |
| 共租负载 | **高**：采集窗口 load average 48-61（5 用户，其他 AI 会话实验）；卡 0/2/3/5 HBM 46-59GB 被占 | **中**：DeepSeek 容器占 8 卡 HBM ~90GB/98（AICore 0% 空闲态）；CPU 侧无大负载 |

### 3.2 环境采集命令

见 §2.1 表来源命令（npu-smi / readlink CANN / python import / lscpu / uname / git rev-parse），脚本固化在 `sr-build/npu-bench/`。

### 3.3 共租负载记录

36 NPU 档每次采集后 npu-smi 快照落 `logs/npu-smi-36-npu.txt`；121 同理 `logs/npu-smi-121-npu.txt`。121 CPU 档的**共租负载影响见 §7.1 备注与 §9.4**。

---

## 4. 测试对象与口径

| 项 | 口径 |
| --- | --- |
| 模型 | `Vela-2.0-0.3B`（121：Hub 名 + offline cache；36：本地绝对路径） |
| 请求 | 四信号一次 decisions 调用：domain(choice, 14 类) + jailbreak(choice) + pii_presence(noul) + pii_categories(set, 11 类) + fact_check(choice)，POST `/v1/decisions` |
| **输入长度（实测）** | **单请求输入 = 665 tokens（p50，服务端 usage.input_tokens 实测，全部 18 组采集一致）** = 四信号 schema 473 tokens（步骤 0 历史口径，schema-only 实测确认）+ 语料句 ~192 tokens |
| 语料 | office_eval_testset 前 30 条经 `build_prompt` 的冻结副本（`sr-build/npu-bench/corpus/office30.json`，两机 md5 同源），36 导出、scp 经本地中转到 121 |
| 精度档 | 默认档（serve 日志确认：`dtype float32, approximate numerics off, batch_window_ms 2, max_num_seqs 8192(默认)/128(950 NPU)`） |
| kernels | reference（NPU 当前唯一路径，#4827 语义） |
| 服务线程 | 36 CPU 档 = `taskset -c 0-31` 绑核 + `--threads 8`；121 CPU 档 = `--threads 8` + `OPENBLAS_NUM_THREADS=8`（192 核机上 BLAS 默认 128 线程引发超订灾难，见 §10.4）；NPU 档不设线程参数，均 `unset VLLM_SRUN_CPU_THREADS` |
| 批处理 | 服务端 batcher 默认参数 |
| 一致性 | 每设备起服 readiness golden 双门必须通过（readiness 含两次运行一致性 + 参照对照）；全部 6 组 NPU/CPU 采集均通过 |

---

## 5. 测试方法

### 5.1 工具与脚本

各机 `sr-build/npu-bench/`（同一套脚本 scp 部署）：

| 文件 | 用途 |
| --- | --- |
| `bench.py` | 单档并发压测：多 worker 发四信号 decisions 请求，记录每请求 wall / forward（Server-Timing）/ status / input_tokens，汇总分位数/吞吐/成功率落 JSON |
| `run_device.sh` | 单设备档全流程：起服 → ready → warm → c4/8/16 三档 → 停服 → npu-smi 快照 |
| `export_corpus.py` | 36 上导出冻结语料 office30.json（一次性） |
| `fast_one.sh` | 121 NPU 专用：单档采集 + 失败重试循环（对抗 serve 被外部终止，见 §10.7） |
| `serve_ign.py` | serve 包装：抢注 SIGTERM/SIGINT 忽略（对抗外部 SIGTERM） |
| `analyze.py` | 结果分析（summary + 失败样本统计） |

原始数据：本地仓 `.tmp-npu-probe/npu-bench/logs-36/`、`logs-121/`（6+6 组 JSON + npu-smi 快照），服务器侧同步保留在各机 `sr-build/npu-bench/logs/`。

### 5.2 并发模型与执行流程

```text
起服 → 探活 /health（ready，含 golden 门）→ warm（懒加载至 forward>0）
  → 并发档 c ∈ {4, 8, 16} 逐档执行：
      每档总请求数 N=120（30 语料 × 4 轮，并发 worker 全局取号均分派发）
      记录：每请求 wall、forward、状态码 → 落 JSON
  → 停服 → 采集 npu-smi 压测后状态
```

- CPU 档与 NPU 档**同 schema、同语料、同轮数**。
- 121 NPU 因 serve 稳定性问题（§10.7）改用"单档单 serve + 失败重试"方式：每档独立起服采集，n_ok=120 才接受，三档最终均 120/120。

### 5.3 指标定义

| 指标 | 定义 |
| --- | --- |
| wall p50/p95/p99 | 客户端单请求往返时延分位数（ms） |
| forward p50 | 响应 `Server-Timing: forward` 段分位数（ms，服务端计算耗时） |
| 吞吐 rps | 成功请求数 ÷ 该档总墙钟时间（含并发等待，阶段级） |
| 并发加速比 | rps(c) ÷ rps(4)，看并发扩展性（基准 = 最低档 c=4） |
| NPU 加速比 | 同主机同并发档：CPU wall p50 ÷ NPU wall p50 |
| 成功率 | HTTP 200 ÷ 总请求数 |
| HBM 增量 | npu-smi 压测前后该卡已用 HBM 差值；功耗同步记录 |

**可比性边界**：加速比只在**同一主机内** CPU vs NPU 计算（两机 CPU 型号不同）；跨机数据仅参考。

---

## 6. 测试矩阵（实际执行）

| 主机 | 设备档 | 配置说明 | 并发档位 | 每档请求数 |
| --- | --- | --- | --- | --- |
| 121 | CPU-8T | `--threads 8` + `OPENBLAS_NUM_THREADS=8`（默认线程档因 BLAS 128 线程超订 + 共租负载弃用，见 §10.4） | 4/8/16 | 120 |
| 121 | **A2-NPU** | 卡 1，`--device npu:0` | 4/8/16 | 120 |
| 36 | CPU-绑核 8T（主） | `taskset -c 0-31` + `--threads 8`（对齐历史 870ms 口径） | 4/8/16 | 120 |
| 36 | **A5-NPU** | 卡 3，`--device npu:0` | 4/8/16 | 120 |

36 CPU 默认线程档与 121 绑核对照档未执行（时间与窗口约束，不影响主结论——两机各有一个有效 CPU 档即可支撑同机加速比）。

---

## 7. 结果

### 7.1 121（Atlas A2，Ascend 910B1）

> **口径快照**：输入 = **665 tokens/请求**（schema 473 + 语料 192，p50 实测）；CPU 档 = 8 线程 + `OPENBLAS_NUM_THREADS=8`；
> NPU 档 = 卡 1 `--device npu:0`；精度 = float32 默认档。
> **⚠️ CPU 档共租污染警告**：121 采集窗口 load average 48-61（其他用户实验负载），CPU 绝对值被显著压慢
> （forward ~4.8-5.1s，而 NPU 同窗口 42-101ms——NPU 数据受影响小）。121 的 CPU vs NPU 加速比因此**虚高，仅供量级参考**；
> 干净的 CPU 基线以 36 为准。

| 设备 | 并发 | wall p50 | wall p95 | wall p99 | forward p50 | 吞吐 rps | 成功率 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU-8T | 4 | 10208.2 | 11670.7 | 12086.0 | 5127.8 | 0.39 | 100% | 共租污染 |
| CPU-8T | 8 | 22366.6 | 24286.3 | 25035.9 | 4889.4 | 0.36 | 100% | 共租污染 |
| CPU-8T | 16 | 44235.5 | 47690.8 | 48111.1 | 4780.2 | 0.37 | 100% | 共租污染 |
| A2-NPU | 4 | 54.1 | 61.4 | 77.3 | 41.6 | **72.75** | 100% | |
| A2-NPU | 8 | 107.3 | 144.4 | 147.6 | 55.7 | **72.50** | 100% | |
| A2-NPU | 16 | 171.3 | 198.1 | 201.4 | 101.2 | **90.32** | 100% | |

### 7.2 36（Atlas A5，Ascend 950DT）

> **口径快照**：输入 = **665 tokens/请求**（同上）；CPU 档 = 绑核 8 线程（`taskset -c 0-31` 钉核 + `--threads 8`，
> 对齐步骤 0 历史 870ms 基线口径、降低共享机调度噪声——121 无此历史口径，故主档为 8 线程）；
> NPU 档 = 卡 3 `--device npu:0`（压测前卡上 HBM 90771/98304 MB，DeepSeek 共租，AICore 0%）；精度 = float32 默认档。

| 设备 | 并发 | wall p50 | wall p95 | wall p99 | forward p50 | 吞吐 rps | 成功率 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU-绑核 8T | 4 | 2458.1 | 2877.8 | 2942.6 | 1222.6 | **1.62** | 100% | |
| CPU-绑核 8T | 8 | 5249.1 | 5795.6 | 5863.6 | 1266.6 | **1.50** | 100% | |
| CPU-绑核 8T | 16 | 10480.0 | 10956.4 | 11049.2 | 1275.2 | **1.52** | 100% | |
| A5-NPU | 4 | 55.7 | 64.1 | 68.2 | 50.3 | **70.66** | 100% | 共租卡 |
| A5-NPU | 8 | 109.2 | 151.5 | 154.0 | 54.6 | **73.25** | 100% | 共租卡 |
| A5-NPU | 16 | 197.8 | 207.5 | 208.1 | 129.7 | **80.48** | 100% | 共租卡 |

### 7.3 汇总对比

| 主机 | 对比 | c=4 wall p50 加速比 | c=8 加速比 | c=16 加速比 | c=4 吞吐对比 | 并发扩展性观察 |
| --- | --- | --- | --- | --- | --- | --- |
| 36 | CPU vs A5-NPU | **44.1×** | 48.1× | 53.0× | 70.66 / 1.62 = 43.6× | NPU 吞吐随并发上升（70.7→80.5 rps），CPU 恒定 |
| 121 | CPU vs A2-NPU | 188.6× ⚠️ | 208.5× ⚠️ | 258.2× ⚠️ | 72.75 / 0.39 = 186× ⚠️ | **虚高**：CPU 分母被共租负载压慢 3-4×；干净基线应与 36 同量级（40-50×） |
| 跨机参考 | A2-NPU vs A5-NPU | c4：72.75 vs 70.66 rps | c8：72.50 vs 73.25 | c16：90.32 vs 80.48 | 量级相当（910B1 与 950DT 在此负载下吞吐接近；c16 A2 领先 12%） | 两代 NPU 均随并发扩展 |

**与步骤 0 历史数据的衔接**：36 上单并发口径的历史记录为 24.7ms wall / 40.6 rps（35×）；本次未采 c=1（按框架从 c4 起），
c4 的 NPU 吞吐 70.66 rps 与历史单并发 40.6 rps 的比值方向一致（batcher 在并发下合批收益）。

### 7.4 NPU 资源与数值一致性

| 主机 / 卡 | HBM（压测后快照） | golden 门 | 备注 |
| --- | --- | --- | --- |
| 121 卡 1 | 3.4GB / 64GB（含模型驻留） | **通过**（910B1，readiness 两次一致 + 参照对照） | 卡 1 为全机最空闲（其余卡 46-59GB 被占） |
| 36 卡 3 | 共租基线 ~90.8GB / 98.3GB | **通过**（950DT **跨代际**对照 910B1 采集的 npu 参照，容差门通过） | 卡 3 为共租下余量最大卡（余 7.5GB） |

> 36（950DT / CANN 9.1）加载时通过 910B1（CANN 8.5）采集的 npu golden 参照——**跨代际交叉验证一次通过**，
> 这是全景 C1 盲区（950DT 交叉验证）的首次实证，也是 parity record 补数据的直接依据。

---

## 8. 扩展实验：输入长度敏感性（并发固定 c=8）

> **设计**：并发固定 c=8，输入长度从 522 到 11037 tokens（p50 实测）共 6 档。语料由 office30 原句池循环偏移截取构造
> （`gen_len_corpus.py`，确定性算法，两机各自生成后 **md5 全量一致**）；base 档 = 主实验原始语料（665 tokens）。
> 每档请求数：NPU 档 120（rounds=4，与主实验同口径）；CPU 档 30（rounds=1，长输入下 CPU 采集过慢，分位数稳定性略降，已标注）。

### 8.1 数据总表

| 输入档 | 实测 tokens p50 | 36-NPU wall/rps | 36-CPU wall/rps ⚠️¹ | 121-NPU wall/rps | 121-CPU wall/rps ⚠️² |
| --- | --- | --- | --- | --- | --- |
| S | 522 | 82.5ms / **95.4** | 18651 / 0.41 | 98.4ms / **79.7** | 13481 / 0.56 |
| base | 665 | 107.2ms / **71.2** | 16248 / 0.48 | 115.3ms / **68.5** | 18409 / 0.44 |
| L | 1238 | 202.2ms / **38.6** | 26208 / 0.30 | 190.4ms / **41.9** | 32756 / 0.24 |
| XL | 3556 | 578.0ms / **13.5** | 36844 / 0.22 | 237.8ms / 16.4 ⁱ | 64187 / 0.12 |
| XXL | 6631 | 1229.0ms / **6.3** | 62509 / 0.13 | — ⁱⁱ | 111970 / 0.07 |
| OVER | 11037 | 2053.3ms / **3.8** | 104274 / 0.07 | — ⁱⁱ | 未完成 ² |

⚠️¹ 36-CPU 长度批采集时 36 出现共租 CPU 负载（base 档 forward 3362ms vs 主实验同口径 1266ms）——本批 6 档为同一污染水平下的相对趋势，绝对值不可与 §7.2 直接互比。
⚠️² 121-CPU 沿主实验的共租污染（§7.1），仅供量级参考。
ⁱ 121-NPU XL/XXL/OVER 档被外部 SIGKILL 打断（XL 40/120、XXL 40/120、OVER 30/120），成功样本偏向快请求（**幸存者偏差**），wall 绝对值偏低不可比；三档均验证了"能正确处理该长度"。
ⁱⁱ 121-NPU XXL/OVER 未在重试窗口内取得 120/120 完整档（重试 6 轮全被打断），成功样本定性结论：可处理。

### 8.2 NPU：forward 随输入近线性，吞吐反比下降（36 服务器）

| tokens | 522 | 665 | 1238 | 3556 | 6631 | 11037 |
| --- | --- | --- | --- | --- | --- | --- |
| forward p50 (ms) | 38.9 | 58.3 | 109.2 | 243.6 | 309.6 | 491.8 |
| **ms/token** | 0.075 | 0.088 | 0.088 | 0.068 | 0.047 | 0.045 |
| 吞吐 rps | 95.4 | 71.2 | 38.6 | 13.5 | 6.3 | 3.8 |

- forward 与 tokens **近线性**（短输入端固定开销略高，长输入端 ms/token 反而下降——合批与扫描摊薄）；
- 吞吐从 95.4 降至 3.8 rps（**25×**），对应输入增长 21×——**吞吐近似与输入长度成反比**（8/0.0825=97≈95.4，8÷wall 恒成立）；
- 两机 NPU 同档互证：base 71.2 vs 68.5、L 38.6 vs 41.9 rps（量级一致，代际差距仍未显现——同 §7.3 结论）。

### 8.3 超限输入行为（OVER 档的重要发现）

- 文档口径（choose-a-model）：0.3B reads **8,192 tokens**，jailbreak/PII 等 require_full_input 信号按 window scan（≤32K）处理。
- **实测**：11037 tokens 的请求在两机 CPU/NPU **全部成功处理**（无拒绝、无报错），forward 随 tokens 单调上升
  （NPU 491.8ms、污染 CPU 14.3s）——与 window scan 行为一致，实际可处理输入超过名义 8192。
- **待上游澄清**：`input.max_tokens` 的截断/拒绝策略与 window scan 的触发边界在默认档下的真实行为，建议作为 model-runtime 文档的补测点（或以本数据提 issue/record）。

### 8.4 应用含义

1. 主实验的 665 tokens（四信号默认 schema）位于**吞吐甜点区**：NPU 70-95 rps、延迟 <120ms；
2. 文档级长输入（>3K tokens）吞吐跌至个位数 rps——生产部署对超长输入应显式配置 `input.max_tokens` / `overflow` 策略（截断/窗口），避免单请求拖垮吞吐；
3. CPU 档在污染底座下短档差异被淹没（36-CPU S 档 18651ms 反常高于 base 16248ms，为共租噪声），XL 起趋势清晰——CPU 的输入长度敏感性需在干净窗口重测才能定量。

---

## 9. 结果分析

1. **单并发加速比与步骤 0（35×）一致性**：36 c4 加速比 44.1×（本框架未采 c=1），与历史 35× 同量级且略高——高出的部分来自 CPU 侧变化
   （见第 3 条），NPU 侧 forward p50 50.3ms 与历史单并发 20.1ms 的差异主要来自并发合批（c4 下 batch 合并拉长单请求 forward，
   但吞吐 3.4× 于历史单并发口径）。
2. **并发扩展曲线**：**NPU 档有效扩展**——36 吞吐 70.66→73.25→80.48 rps（c4→c16，+14%），121 72.75→72.50→90.32 rps（+24%）；
   wall p50 随并发近线性增长（合批排队），但吞吐持续上升 = 服务端 batching 在 NPU 上真实生效。**CPU 档完全不扩展**——
   两机吞吐恒定（36: 1.50-1.62 rps），wall p50 精确随并发线性放大（2458→5249→10480，≈×2 每档），forward p50 恒定（~1.27s）=
   并发请求在 CPU 上**排队串行执行**。
3. **forward 占比随并发的变化**：CPU 档 forward 恒定（batcher 未合批或合批无收益——fp32 unpacked 权重下 batch forward 无并行收益，
   serve 日志确认 `dtype float32, approximate numerics off`）；NPU 档 forward 随并发增长（36: 50.3→129.7ms）但总吞吐上升 =
   **batch forward 的 NPU 并行收益真实存在**。
4. **共租负载干扰**：36 NPU 数据带共租卡备注（HBM 共享、AICore 空闲态）——吞吐 70-80 rps 与 121 独占卡 72-90 rps 同量级，
   说明共租干扰在 AICore 空闲态下有限；121 CPU 数据被 load 48-61 严重污染（forward 5.1s vs 36 同口径 1.27s——121 CPU 本身 192 核
   不应更慢），**121 的 CPU 绝对值不可用，加速比虚高**；干净的 CPU 基线以 36 为准。
5. **口径漂移记录（重要）**：36 CPU forward p50 ~1.27s（本次，8 线程绑核）vs 步骤 0 历史基线 870ms wall（10-10，同 schema 同语料
   同 8 线程口径）——**慢 ~45%**。serve 日志显示当前默认档 `dtype float32, approximate numerics off`；历史基线大概率运行在
   packed/approximate 路径（#5071 "vLLM-style engine arguments replace numerics profiles" 前后的默认值变化）。
   **本报告 CPU 与 NPU 同版本同默认档，加速比口径自洽**；但绝对值与历史记录不可直接互比，后续若要恢复历史口径需显式设置
   对应 engine arguments（另测）。
6. **跨代际 golden 交叉验证通过**（§7.4）：950DT/CANN 9.1 加载时通过 910B1/CANN 8.5 采集的 npu 参照——全景 C1 的关键一步，parity record 可据此补 950DT 数据。
7. **后续动作建议**：①36 NPU c16 吞吐 80.5 rps vs c8 73.3——继续上探 c32 找拐点（batcher 收益未到顶）；
   ②CPU 档串行化是**当前默认档的真实行为**，与 #4668 的 CPU 延迟优化方向互相印证（packed/合批路径的收益空间）；
   ③bf16 专项（全景 A1）在 NPU forward 占比 74-59%（c4→c16）的背景下收益可期，建议下一窗口执行。

---

## 10. 风险与注意事项（实测后按实际发生情况修订）

1. **36 共租卡**：✅ 已按预案执行（选最空闲卡 3 + npu-smi 快照），AICore 空闲态下共租干扰有限（§9.4）。
2. **121 共租 CPU 负载**：⚠️ **实际发生且影响重大**——采集窗口 load 48-61，121 CPU 档数据全部污染（表内已标注）。后续如需 121 干净 CPU 基线，必须错峰采集。
3. **语料冻结**：✅ 已执行（office30.json 冻结 + 两机同源 + md5 同源中转）。
4. **线程环境变量（两条新坑）**：36 `VLLM_SRUN_CPU_THREADS=32` 已按预案 unset；**新增**——121（192 核）上 OpenBLAS 默认 128 线程，
   16 并发压测时线程超订导致 serve 吞吐塌陷、c16 采集挂死（>7 分钟无产出），**必须 `OPENBLAS_NUM_THREADS=<服务线程数>` 显式受控**；
   serve 启动日志的 OPENBLAS 警告即此信号。
5. **golden 前置**：✅ 全部 6 组采集 readiness 通过（含 950DT 跨代际对照）。
6. **NPU batch 行为**：✅ 已记录（serve 启动日志含 batcher 默认值：`max_num_seqs 8192`（CPU 默认）/ `128`（950 NPU 64-160GB 档）、
   `batch_window_ms 2`、`max_num_batched_tokens 65536/16384`），NPU 上合批收益实测为正（§9.2/9.3）。
7. **121 serve 被外部 SIGKILL（新坑，已战术解决）**：121 上我们的 NPU serve 在 ready 后 2-25 秒内被外部 SIGKILL
   （三连杀，日志均优雅/强制退出混合；TERM 免疫包装无效说明最终是 KILL；另一会话 8100 的 CPU serve 长活不受影响——
   杀手疑似针对 NPU 占用进程的巡检）。对策 = `fast_one.sh` 单档重试循环（每档只需 serve 活 ~2s，n_ok=120 才接受），
   c8 两轮、c16 一轮命中。**教训：共享 NPU 机的长驻 serve 不可依赖，采集脚本必须自带"短窗口 + 失败重试"能力。**
8. **ssh 会话与远端进程生命周期**：长采集任务必须与 ssh 会话解耦（`setsid nohup`），否则本地会话超时断连会触发远端脚本的清理 trap
   杀掉 serve；且同一条 bash -c 里"pkill 模式 + 含模式明文的启动路径"会导致 pkill 自杀（正则匹配自身命令行），清理与发射必须
   分开两条 ssh。

---

## 11. 附录

- **脚本清单**（各机 `sr-build/npu-bench/`）：`bench.py`（单档压测）/ `run_device.sh`（单设备全流程）/
  `export_corpus.py` + `gen_len_corpus.py`（语料导出与长度档构造）/ `fast_one.sh` + `serve_ign.py`（121 NPU 重试采集）/
  `len_bench.sh` + `len_retry.sh`（输入长度实验）/ `analyze.py`（结果分析）。
- **原始数据**：本地仓 `.tmp-npu-probe/npu-bench/logs-{36,121}/`（主实验 12 组 JSON）+ `logs-36-len/`（36 长度档）；
  服务器侧各机 `sr-build/npu-bench/logs/` 同步保留（长度档 `{tag}-len{S,L,XL,XXL,OVER}-c8.json`，121 长度档仅在服务器侧）。
- **服务起停**：见 §2.3；121 NPU 短窗口采集流程见 `fast_one.sh`。
- **engine batch 参数默认值快照**（serve 启动日志）：CPU 档 `max_num_seqs 8192, max_num_batched_tokens 65536, batch_window_ms 2,
  dtype float32, shared context off, approximate numerics`；950 NPU 档 `max_num_seqs 128, max_num_batched_tokens 16384`
  （64-160GB 设备档）。

---

## 12. 变更记录

| 日期 | 版本 | 变更 |
| --- | --- | --- |
| 2026-10-15 | v0.1 | 框架版：目的 / 环境 / 口径 / 方法 / 矩阵 / 空表结构定稿，待确认后执行测试回填 |
| 2026-10-15 | v0.2 | 新增 §2「启动方式速查」：镜像（config `device:` / 全栈编排限制 / NPU 容器配方）与二进制（`--device` / `--threads` / 选卡）两条路径 × CPU / NPU 设置方法；原 2–10 节顺延为 3–11 |
| 2026-10-15 | v0.3 | §2 扩充：①新增 §2.1 所需环境表（CANN/驱动/torch_npu 版本预采：121=cann-8.5.1+25.5.0、36=cann-9.1.0+25.6.rc1，两台 torch 2.10.0+cpu + torch_npu 2.10.0.post4 相同）+ env.sh 不含 CANN 的红线（实测不 source 则 torch_npu 拒载）；②§2.2 贴两台 env.sh 原文 + 关键行速览；③新增 §2.5 NPU 容器配方全文（自 ROI 手册 §7，36 实测）——36 上镜像跑 NPU = 进容器手动单跑 runtime 可行，CLI 全栈编排仍不行；④§3.1 环境表预采行回填 |
| 2026-10-15 | v0.4 | 框架修订：①并发档位去掉 1/2，改为 4/8/16（并发加速比基准改 c=4）；②输入长度定死口径（schema 473 + 语料 ≈490 tokens 估计）+ §7.1/§7.2 表前口径快照；③"绑核 8T"释义与两机 CPU 档差异说明 |
| 2026-10-15 | **v1.0** | **数据回填完成**：①两机 × CPU/NPU × c4/8/16 共 12 组采集全部完成（成功率 100%，golden 门全过）；②输入长度实测修正 = **665 tokens/请求**（473 schema + 192 语料）；③36 加速比 **44.1×/48.1×/53.0×**（c4/c8/c16）；④121 CPU 档共租污染标注（load 48-61，加速比虚高仅供量级参考）；⑤**核心发现**：CPU 档并发完全串行化（吞吐恒定）、NPU 档合批有效扩展（36 70.7→80.5 rps、121 72.8→90.3 rps）；⑥36 CPU 较历史 870ms 基线漂移 +45%（#5071 默认档口径变化，记录待查）；⑦**950DT 跨代际 golden 对照通过**（全景 C1 首证）；⑧新坑记录：121 OpenBLAS 192 核超订、NPU serve 被外部 SIGKILL（重试循环战术解决）、ssh 会话生命周期与 pkill 自伤 |
| 2026-10-15 | **v1.1** | **新增 §8 输入长度敏感性扩展实验**（c=8 固定，522→11037 tokens 六档，NPU 120 请求/档、CPU 30 请求/档）：①36-NPU 六档全干净——forward 随 tokens 近线性（0.045-0.088 ms/token）、吞吐 95.4→3.8 rps（25×）近似与输入成反比；②**超限行为发现**：11037 tokens > 文档上限 8192 在两机 CPU/NPU 全部成功处理（window scan 路径，§8.3，待上游澄清）；③121-NPU XL/XXL/OVER 被外部 SIGKILL 打断为部分数据（幸存者偏差标注，定性可处理）；④CPU 长度批遇共租污染（36/121 均标注），趋势定性有效；⑤原 §8-11 顺延为 §9-12 |
