# AgentInfer 全家桶 — CI 质量看护与版本发布总指南

> **作者与定位**：本指南面向 **对 CI/CD / Jenkins / Workflow 零基础** 的同事，目标是回答以下四个问题：
>
> 1. **这是在干嘛** —— 用大白话解释 CI / CD / Pipeline / Jenkins / Workflow 这些词，它们为什么存在、分别做什么。
> 2. **我们要做成什么样** —— 结合 SR (Semantic Router)、VR (vLLM Router)、AgentInfer 三仓三组件的架构，给出**质量看护分层 (L1\~L5)** 与**版本发布**的设计。
> 3. **我该怎么做** —— 给一份可落地的**分阶段实施路线图**，从第 0 步 (搭 Jenkins) 开始，到 L1 → L2 → L3 → 正式发布。
> 4. **我要申请什么** —— 列一张清晰的**资源申请清单**（服务器、账号、权限、工具、仓库配置、外部服务）。
>
> **参考体系**：分层设计、测试目录规范、标签体系均参考 vllm-omni 项目的 CI 测试体系文档
> （`docs/contributing/ci/test_system_overview.md` 与 `test_writing_guide.md`）。
>
> **版本状态**：v0.1 DRAFT（逐步完善）

***

## 第一部分 · 概念科普：CI/CD、Jenkins、Pipeline 是什么

> 如果一个词完全不知道，就按顺序读。如果知道某个词，可以跳到对应小节。

### 1.1 最顶层的两个词：CI 与 CD

| 缩写 | 英文全称 | 中文 | 一句话解释 |
| ------ | -------------------------------- | --------- | ------------------------------------------------------- |
| **CI** | Continuous Integration | 持续集成 | 你每次把代码提交 / 合并到主干的那一刻，机器立刻帮你跑一遍编译、lint、测试，**不让坏代码混进主干**。 |
| **CD** | Continuous Delivery / Deployment | 持续交付 / 部署 | 在 CI 通过的基础上，**自动把软件包、镜像、二进制打出来，甚至自动推送到测试环境 / 生产环境**。 |

类比成"做包子"：

```text
CI（持续集成）
   │ 面粉(代码)→ 称重(代码规范检查) → 发面(编译) → 包馅(组装) → 尝一口(单元测试)
   │   每一步不合格 → 立刻丢掉这坨面团，通知你重做
   ▼
CD（持续交付/部署）
   │ 蒸包子(构建镜像 / 打包 wheel) → 装盘(归档产物) → 上架(推送镜像仓库/发布)
   │   有的团队把 CD 分成两步：
   │     Continuous Delivery：把包子做好端上柜台（准备好可随时发版，但需要人点确认）
   │     Continuous Deployment：包子直接自动送到顾客桌上（全自动发布）
```

#### 我们为什么需要 CI/CD

| 没有 CI/CD（手工时代） | 有了 CI/CD（自动化时代） |
| ---------------------------------------------- | ------------------------------------------- |
| 改了一行代码，不敢知道会不会编译失败，要等同事说"你代码编不过" | push 之后 5 分钟内收到结果，不行立刻改 |
| 每个合入 PR 都要专人登录构建机执行 `make build && make test` | 流水线自己跑，24/7 不间断 |
| 做一次版本发布：手工改版本号、跑测试、打 tar、传镜像、写 Release 页面，半天一天 | 打一个 `git tag v0.2.0`，其余全自动，1 小时出成品 |
| "我机器上能跑啊，怎么你那边不行" —— 环境不一致 | 所有构建/测试在**同一套标准环境**（Docker 容器）里跑，结果可复现 |
| 每个人打出来的镜像不一定一样（缺依赖/版本飘） | 每次都用同一份 Dockerfile，tag 锁定代码 commit，100% 可追溯 |

### 1.2 Pipeline / Workflow / Stage / Step / Job：流水线结构

这几个词描述的是 **CI 的结构层次**，所有主流工具（GitHub Actions、Jenkins、Buildkite、GitLab CI）都用这套概念，只是名字略有差异：

```text
Workflow / Pipeline（工作流/流水线）
    └── 一台 Jenkins 任务 / 一个 yml 文件 / 一个流程定义
    │
    ├── Stage / Group（阶段 / 分组）
    │     例如："Fast Checks" → "Build" → "Test" → "Publish"
    │     阶段之间通常串行：Build 过了才能进 Test
    │
    │     └── Job（任务）
    │           每个 Job 在一台独立的机器 / 容器里执行
    │           同一阶段内的多个 Job 可以并行
    │           例："Build x86" 与 "Build arm64" 并行
    │
    │           └── Step（步骤）
    │                 一条 shell 命令 / 一个 action
    │                 例：checkout 代码 → pip install → pytest
```

各 CI 工具对应术语表：

| 概念通用名 | Jenkins | GitHub Actions | Buildkite |
| ------- | ------------------------------------ | ---------------------------------- | -------------------------------- |
| 工作流/流水线 | Pipeline (Jenkinsfile) | Workflow (yml) | Pipeline (yml) |
| 阶段/分组 | `stage()` | `jobs.<name>` (组合在一个 job 里的 steps) | `group` / `key` |
| 任务 | 一般 stages 就够用；多 agent 并行用 `parallel` | `jobs` (每个 job 在一台 runner 上跑) | `steps.label` (每个 label 是一个 job) |
| 执行机 | Agent / Node (Executor) | Runner | Agent |
| 步骤 | `sh / script / checkout` | `uses:` / `run:` | plugins/docker / command |

#### 我们自己已有的示例

1. VR (vLLM-router) 使用 **Buildkite**：见上游仓库的
   [pipeline.yml](https://github.com/vllm-project/router/blob/main/.buildkite/pipeline.yml)，
   结构是 `group: Fast Checks` → `wait` → `group: Build` → `wait` → `group: Tests` → GPU Job → Docker Build Job。
2. SR (Semantic Router) 使用 **GitHub Actions**：见上游仓库的
   [ci-changes.yml](https://github.com/vllm-project/semantic-router/blob/main/.github/workflows/ci-changes.yml)
   （变更识别）与
   [release.yml](https://github.com/vllm-project/semantic-router/blob/main/.github/workflows/release.yml)
   （发布校验）。
3. AgentInfer 初步用 **GitHub Actions**：见 [.github/workflows/ci.yml](../.github/workflows/ci.yml)，
   只跑了 pre-commit / clippy / DCO。
4. Jenkins 初版方案（内部资料，未随本仓发布）：结构是 `Jenkinsfile`（在业务仓根目录） + `jenkins-config/config-*.yaml`（配置） + 共享构建代码仓。

这说明**三件事**：

* 工具本身不关键，关键是**流水线结构**和**看护设计**（用哪个 CI 工具只是语法差别）。

* 你手头的 SR/VR 上有"**GitHub Actions + Buildkite**"两种现成范式，Jenkins 是未来要走的第三套平台。

* 你文档里写的 "Jenkinsfile 随业务代码 + 配置分散" 是业内最佳实践，方向完全正确。

### 1.3 Jenkins 是什么（30 秒版 + 组件图）

**一句话**：Jenkins 是一个**公司自己部署在自己机房里的、可视化的流水线调度中心**，它负责：

1. 监听 Git 事件（push、tag、PR、定时）
2. 把流水线任务派给\*\*执行机（Agent）\*\*跑
3. 收集结果，展示界面，发邮件 / 企业微信通知

类比：Jenkins = "食堂点菜+派单系统"。厨师（执行机）散落在各角落，食客（开发者）在菜单（Pipeline 配置）上下单，Jenkins 自动派给合适的厨师。

#### Jenkins 核心组件

```text
              开发者
                 │  push / PR / tag
                 ▼
          ┌─────────────┐
          │  Git 服务器  │  (Gitee / GitLab / GitHub)
          └──────┬──────┘
                 │ Webhook（有代码事件就 POST 个 HTTP 通知）
                 ▼
          ┌──────────────────────────────┐
          │  Jenkins Controller          │  ← 主节点（只做调度，不做构建）
          │  (Controler / 以前叫 Master) │     • 8C/16G 够用
          │                               │     • 装 Jenkins 本体程序
          │                               │     • 存所有流水线配置
          │    任务队列 + 派单逻辑        │     • 80/443 端口暴露 Web UI
          └────────┬─────────────────────┘
                   │ 通过 SSH / WebSocket 连接
           ┌───────┴─────────┬──────────────┐
           ▼                 ▼              ▼
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │ Agent #1   │   │ Agent #2   │   │ Agent #N   │
    │ CPU x86 机  │   │ GPU L4/A100│   │ 鲲鹏 arm64  │
    │ (编译+UT)  │   │ (L2~L5测试) │   │ (arm64构建) │
    └────────────┘   └────────────┘   └────────────┘
         │                  │                │
         ▼                  ▼                ▼
    构建产物 / 测试报告 → 归档回来（Artifacts）  ← Jenkins Controller 拉走
         │
         ▼
  推送到镜像仓库 / 制品仓库 (Harbor / Nexus / 本地 S3 / NFS)
```

**关键设计原则（重要，别踩坑）**：

1. **Controller 不干活，干活的是 Agent**。不要把编译和测试任务放在 Controller 节点上跑，不然它会崩。
2. **Agent 按"硬件能力"分标签**：`c-compile`、`gpu-l4`、`gpu-a100`、`arm64-build`……
   每条流水线写"我需要 `label gpu-l4 && cards_2`"，Jenkins 自动派给符合条件的 Agent。
3. **推荐用"Docker Agent"模式**：Agent 本身只装 Docker，每个 Job 在一个干净的 Docker 容器里跑
   （镜像里放 Rust/Go/Python 工具链）。好处：环境 100% 一致，用完就删，Agent 永远干净。
4. **Multibranch Pipeline 项目类型**：一个仓库建一个 Multibranch Pipeline 项目，Jenkins 自动扫描每个分支/PR 的 Jenkinsfile，不用手工建 N 个任务。

### 1.4 Runner / Agent / Executor：干活的机器

| 名字 | 在哪里出现 | 含义 |
| ---------------------- | -------------- | --------------------------------------------------------- |
| **Self-Hosted Runner** | GitHub Actions | 在自己机器上装个程序，GitHub 把任务派给它（对应 VR 的 arm64 鲲鹏 runner）。 |
| **Agent / Node** | Jenkins | Jenkins 上注册的"干活机"，Controller 通过 SSH/JNLP 连上它。 |
| **Executor** | Jenkins | 在同一个 Agent 上并行跑几条任务的"槽位"。一台 16C Agent 一般设置 4\~8 Executor。 |

### 1.5 Artifact / 制品 / 产物 / Release：它们有什么区别

| 词 | 含义 | 例子 |
| ------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Artifact / 产物** | 流水线跑完一次生成的文件，临时存档 30 天左右，用于跨 Job 传递或快速调试 | `coverage.xml`、`target/release/vllm-router`、未版本号的镜像 tar |
| **制品 / Release Artifact** | 正式版本发布时对外交付的东西，带版本号永久存档 | `vllm-router-v0.1.15-linux-amd64` 二进制、`vllm-router:v0.1.15-amd64` 镜像、`vllm_sr-0.3.0-cp310-*.whl` |
| **Release / 发布** | GitHub / Harbor / PyPI 上的一个页面 + 一套版本化制品集合（对应 SR 的 `release.yml`）。 | SR 的 `ghcr.io/.../semantic-router/vllm-sr:v0.3.0` 镜像 + PyPI `vllm-sr==0.3.0` + Helm chart + GitHub Release 页 |

### 1.6 质量看护 (Quality Gate) 是什么

**一句话**：每一层 CI 都像一道"关卡"（Gate），失败了就不让代码进入下一关。

例如 Omni 的 L1 是 PR 的硬门槛：

```text
开发者提交 PR
    │
    └──▶ L1 单元测试 + Lint 自动执行
             │  失败 ❌ → PR 标红，禁止合并
             │  成功 ✅ → 可以进入 Review
             ▼
        Reviewer 人工审核 → Merge 合入 main
             │
             ▼
        main 分支变更 → 自动触发 L3 (合并后集成测试 + GPU E2E)
```

我们后面会把 3 个组件都套上这个模式。

***

## 第二部分 · 三仓整体架构：三组件分别构建、分别看护、分别发布

### 2.1 代码与交付物全景（你描述的现状，加上我建议的完善版）

```text
                            AgentInfer 总仓（未来的 Monorepo 或 三个独立仓）
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │                                                                                │
   │  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌────────────────┐ │
   │  │ agentinfer/（Agent 侧） │  │ agentrouter/（VR 代码） │  │ semantic-...   │ │
   │  │  Python 包             │  │  Rust bin + Python API  │  │  (SR 代码，    │ │
   │  │  包装 vllm / vllm-ascend│  │  高性能路由转发         │  │   未来导入)    │ │
   │  │  提供 vllm serve 入口   │  │  Cargo.toml:0.1.15      │  │  Go+Rust+Python│ │
   │  └────────────┬────────────┘  └────────────┬────────────┘  └───────┬────────┘ │
   │               │                            │                       │          │
   │               ▼                            ▼                       ▼          │
   │  ┌──────────────────────┐   ┌──────────────────────────┐  ┌──────────────┐    │
   │  │ 构建形态（建议）      │   │ 构建形态（确定）           │  │ 构建形态（确 │    │
   │  │                      │   │                          │  │  定）          │    │
   │  │ • Docker 镜像 ★      │   │ • Docker 镜像            │  │ • Docker 镜像 │    │
   │  │   (FROM vllm:tag 包  │   │   (musl 全静态/动态)      │  │   (3+3 镜像：  │    │
   │  │    一层 agentinfer)   │   │ • Python wheel           │  │  extproc,     │    │
   │  │ • 备选：wheel 包      │   │ • Rust 二进制(optional)  │  │  vllm-sr,     │    │
   │  │   (pip 进 vllm 镜像)  │   │ • GitHub / Harbor Release│  │  dashboard,   │    │
   │  │                      │   │                          │  │  operator ... │    │
   │  └────────────┬─────────┘   └────────────┬─────────────┘  └──────┬───────┘    │
   │               │                          │                        │            │
   │               ▼                          ▼                        ▼            │
   │  ┌────────────────────────────────────────────────────────────────────────┐    │
   │  │           统一 Jenkins CI 平台（你同事帮你部署）                          │    │
   │  │   • 每个组件 → 1 个 Multibranch Pipeline 项目 → 对应 1 份 Jenkinsfile     │    │
   │  │   • 按路径变化触发：改 SR 不触发 VR 的 GPU 测试 (diff-aware)              │    │
   │  │   • Agent 标签：cpu-build / gpu-l4 / gpu-a100 / arm64-build             │    │
   │  │   • 制品推送 → 内网 Harbor（镜像）+ 私有 PyPI/文件服务器（wheel）          │    │
   │  └────────────────────────────────────────────────────────────────────────┘    │
   └────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 agentinfer 该打包成镜像还是 wheel？结论：优先镜像，辅助 wheel

你这个判断非常到位，我完全赞同，理由补充如下：

| 维度 | 镜像方案（推荐） | 纯 wheel 方案（备选） |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **vllm 是什么** | vllm 本身就是**以镜像为主要部署形态**的（vllmproject/vllm:0.22.1），有 CUDA / NPU 等多平台预置环境 | 目标用户要自己解决 CUDA / NPU 驱动、PyTorch、vllm 本体安装，依赖链地狱 |
| **agentinfer 做什么** | 在 vllm 基础上打一层（`vllm serve` 入口、调度器 patch）→ 最自然的做法就是 `FROM vllm:0.22.1-cuda12`，`pip install agentinfer-0.1.0.whl` | 需要用户 `pip install agentinfer` 后能正确 patch 掉已安装的 vllm，耦合度高、容易出版本冲突 |
| **下游使用** | `docker run ... ghcr.io/.../agentinfer:v0.1.0` 即用 | `pip install agentinfer` 但要求用户装了正确版本的 vllm |
| **NPU 版本（vllm-ascend）** | 分别打两份镜像：`agentinfer:v0.1.0-cuda`、`agentinfer:v0.1.0-ascend`（Base 不同） | wheel 本身架构无关，但安装时依然要对应 cuda / ascend 版 vllm |
| **CI 方便性** | 构建 Dockerfile 里先 pip install 再 smoke test，天然一体 | 要单独打 wheel，再用另外一张镜像装进去做 smoke test |

**建议的正式交付物：**

```text
agentinfer 交付物
├── 镜像（主）
│     ├── agentinfer:v0.1.0-cuda-amd64      (FROM vllm/vllm:0.22.1-cuda12.1)
│     ├── agentinfer:v0.1.0-ascend-arm64    (FROM vllm-ascend:<对应版本>-npu)
│     └── agentinfer:v0.1.0-cuda-arm64      (可选，看是否有 CUDA arm64 机器)
├── wheel（辅 / 给想自己装的人 / 或测试环境 pip install 用）
│     └── agentinfer-0.1.0-py3-none-any.whl  （Python 包，纯 py，无 arch 依赖）
└── GitHub / Harbor Release 页面
      └── 镜像拉取命令 + CHANGELOG + wheel 下载
```

### 2.3 三个组件的发布流（独立触发）

**一条原则：改哪个组件就触发哪个组件的发布**，不互相绑定版本号。每个组件独立维护自己的 `Cargo.toml` / `pyproject.toml` 版本号和独立的 tag。

```text
┌──────────────────── 语义路由器 SR ────────────────────┐
│  tag v0.3.1 → Jenkins SR-release Pipeline            │
│    ① Cross-Surface Version Check (pyproject/Cargo/go  │
│      module 版本都要 == tag)                         │
│    ② 构建全部镜像 (extproc / vllm-sr / dashboard /    │
│      operator / sim ...) + helm chart                │
│    ③ 推送到 Harbor                                    │
│    ④ 构建 wheel (vllm-sr) + 推私有 PyPI              │
│    ⑤ 生成 Release 页（所有产物列表 + 拉取命令）        │
└──────────────────────────────────────────────────────┘

┌──────────────────── vLLM Router VR ───────────────────┐
│  tag v0.1.16 → Jenkins VR-release Pipeline           │
│    ① 检查 Cargo.toml + pyproject.toml 版本 == tag    │
│    ② amd64 + arm64 并行构建 Docker 镜像 + binary     │
│    ③ 构建 Python wheel (manylinux)                    │
│    ④ Smoke test binary + wheel install               │
│    ⑤ 推 Harbor + 私有 PyPI                            │
│    ⑥ 生成 Release 页                                  │
└──────────────────────────────────────────────────────┘

┌──────────────────── AgentInfer ───────────────────────┐
│  tag v0.2.0 → Jenkins AgentInfer-release Pipeline    │
│    ① 检查 pyproject.toml 版本 == tag                 │
│    ② 构建 CUDA 镜像 (amd64) + NPU 镜像 (arm64/amd64) │
│    ③ 构建 wheel (pip install 验证)                    │
│    ④ Smoke test: 启动 vllm serve 10s 不挂、/health   │
│    ⑤ 推 Harbor + 私有 PyPI                            │
│    ⑥ 生成 Release 页                                  │
└──────────────────────────────────────────────────────┘
```

### 2.4 组件独立发布但在集成场景上互相依赖怎么处理

SR/VR/AgentInfer 部署在同一个推理集群上时，确实存在 A 组件升级了要验证 B 组件能不能一起工作。答案是**在 L3 / L4 集成测试环节**
   用 docker-compose 或者 K8s kind 把它们一起拉起来做全链路 E2E，**不是在发布阶段绑定版本号**。发布阶段各自独立。

***

## 第三部分 · 质量看护分层设计（L1 → L5）

直接对 vllm-omni 的体系做 **适配化裁剪**（因为 omni 是大模型多模态项目，我们是 LLM 路由 + 推理增强，略简单一些）。

### 3.1 分层总表

> 注意：执行频率 = 触发时机（对应 Gate 等级）；表格中"硬件"列如果写了 GPU，表示对应卡型号。

| 层级 | 中文名 | 范围 & 重点 | 执行频率 | 耗时上限 | 硬件 | 失败后的动作 |
| ---------- | ------------------ | ----------------------------------------------- | -------------------------------------------- | -------- | -------------------- | ----------------------------------- |
| **Common** | 通用规范 | PR Checklist、CI Failure Handbook、代码格式/规范 | 每次 PR 提交流程 | / | / | 提交流程无法走完（PR 模板） |
| **L1** | 单元测试 & 快速检查 | 组件级单测（不启大模型、不启 GPU）+ 代码格式 + 静态检查 + 编译通过 | **每次 PR 创建/更新**（Pre-merge Gate） | < 15 min | CPU | **❌ PR 禁止合并** |
| **L2** | 基础 E2E & GPU 单测 | 关键功能的小模型启动冒烟、主要接口调用（非空即可，不做精度）、实例启动相关的 GPU 单测 | **PR 打 ready 标签后**（Pre-merge Gate，可选用于关键 PR） | < 30 min | GPU（L4 级） | **❌ PR 禁止合并**（或作为可选 gate） |
| **L3** | 核心集成测试 + 精度 & 关键性能 | 合并到 main 之后：全量关键功能 + 真实权重模型 + 精度/相似度 + 关键性能基线对比 | **每次 PR Merge 后**（Post-merge） | < 1 h | GPU（L4/H100/NPU A2+） | 邮件/企业微信群报警（主干红了），**下一 PR 合入前必须先恢复** |
| **L4** | 全量功能 + 全性能 + 文档 | 夜间跑：所有模型、所有功能路径、性能基准 (benchmark)、文档示例测试、扩展模型覆盖 | **每日夜间 2:00**（Nightly） | < 3 h | GPU（按需分配多卡） | 次日晨会 review，输出失败清单 |
| **L5** | 稳定性 + 可靠性 + 覆盖率 | 长时间运行稳定性（如连续跑 24h）、故障注入（断连/重启/卡）、性能压力极端场景 | **每周一次** + **版本发布前 72h** | 1\~7 天 | GPU（真机集群） | 版本 Release 前必须全部绿 |

### 3.2 三个组件各自在每一层跑什么（映射表）

> ⚠️ SR 现在还没整进 AgentInfer 仓，但是我们按"未来会搬进来"来设计。当前先在 semantic-router 自己的仓加 Jenkinsfile。

#### Common 规范

| 规范 | 内容 | 位置 |
| ------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| PR Checklist | 改动涉及的模块、是否加了测试、是否同步更新文档、版本号是否改了 | `.github/PULL_REQUEST_TEMPLATE.md`（三个仓各一份或 AgentInfer 一份） |
| CI Failure Handbook | 常见错误速查（pytest 报错怎么看、构建失败常见原因）、调试用环境变量 | `docs/zh/how-to/ci-failure-handbook.md`（新建，未来逐步填充） |
| DCO 签名 | 每条 commit 必须 `Signed-off-by:` | AgentInfer 已有，[ci.yml](../.github/workflows/ci.yml) |

#### L1（PR 必过 Gate · CPU · < 15 min）

| 组件 | 检查项 | 对应命令 / 技术 |
| ------------------------ | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **AgentInfer** | pre-commit (ruff / markdownlint / typos / .editorconfig) | `pre-commit run --all-files`，见 [.pre-commit-config.yaml](../.pre-commit-config.yaml) |
| | 代码能打包（`pip install -e .` 成功） | `pip install -e .` |
| | Python 单元测试（agentcache / agentbench / tests） | `pytest tests/agentcache/core/test_agent_scheduler.py -v`（已有） |
| **VR (agentrouter/)** | rustfmt + clippy（AgentInfer 已有此 job） | `cargo fmt --check`、`cargo clippy -D warnings`，见 [ci.yml](../.github/workflows/ci.yml) |
| | VR Python 侧格式检查 | `ruff check py_src/ py_test/` 或 agentrouter 下的 pre-commit |
| | VR Rust 单元测试 | `cargo test --lib --bins`（对应 Buildkite pipeline） |
| | VR Python 单元测试（py\_test/unit，不含 e2e/integration） | `pytest py_test/unit/ -v` |
| **SR (semantic-router)** | pre-commit / golangci-lint / ruff / rustfmt + clippy | SR 自己的 `.pre-commit-config.yaml` + `make agent-lint` |
| | Go 单元测试（非 e2e） | `go test ./src/semantic-router/...` |
| | Rust binding UT（candle-binding / ml-binding 等如有） | `cargo test --lib` |
| | Python UT（vllm-sr / fleet-sim / bench） | `pytest src/fleet-sim/tests/ -v` |

**推荐 L1 执行机器**：一台 x86 CPU Agent（16C 32G 足够），装 Docker，所有 L1 步骤在统一的 `build:ci-l1` 镜像里跑（镜像提前装好 Python/Rust/Go+protobuf 工具链）。

#### L2（PR Ready Gate · GPU · < 30 min）

| 组件 | 检查项 | 对应命令 / 技术 |
| -------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **AgentInfer** | 进程内 E2E（tiny 模型 + `LLM(...)`） | 见 [tests/README.md](../tests/README.md)：`pytest tests/agentcache/core/test_agent_scheduler_e2e.py` |
| | serve 子进程 E2E（`vllm serve` 真实启动 1.3GB 模型，测 OpenAI API） | `pytest tests/agentcache/core/test_agent_scheduler_serve_e2e.py` |
| **VR** | Python 集成测试（`py_test/integration/`，包含 mock worker） | `pytest py_test/integration/ -v` |
| | Rust 集成测试（`tests/*.rs`，mock vllm 服务器） | `cargo test --test '*'` |
| | P/D 分离基础逻辑（如不需要真 GPU 可降级到 L1） | `pytest py_test/integration/test_pd_routing.py` |
| **SR** | Go E2E core smoke (envoy+router+mock vllm，用 kind 或 docker-compose 拉起) | `make agent-ci-gate`（SR 已有 `make agent-ci-gate`） |
| | vllm-sr CLI smoke（`vllm-sr serve` 拉起来能启 dashboard+envoy） | SR 已有 `vllm-sr-cli` 容器 + test |
| | 核心模型选择分类用例（keyword / domain / ml） | `pytest <基础分类 smoke 路径>` |

**硬件**：至少 1 张 L4 或同等级 GPU（24G 显存，能跑 1.3B 模型就行）。昇腾 NPU 环境单独拉一条 L2 队列（`npu-l2`），用相同测试代码，走 pytest marker `-m npu`。

#### L3（Main 分支合入后 · 真实权重 · < 1 h）

| 组件 | 检查项 | 对应命令 / 技术 |
| -------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| **AgentInfer** | 真实权重 serve E2E（如 Qwen2-7B，测 /v1/chat/completions） | `pytest tests/e2e/serve_e2e_real_model.py`（**未来新建**） |
| | AgentBench 核心基准（SWE-bench 轻量子集） | `python tests/e2e/run_benchmark.py --scenario smoke`（已有） |
| | 与 VR + mock vllm 的串联测试（SR 网关 → VR → AgentInfer 推理 → 返回） | docker-compose 三组件联合 smoke（**未来新建**） |
| **VR** | E2E 常规路由（py\_test/e2e/test\_regular\_router.py，真实小权重 vllm worker） | `pytest py_test/e2e/test_regular_router.py` |
| | Embedding 路由 | `pytest py_test/e2e/test_e2e_embeddings.py` |
| | P/D 分离 e2e（pd\_disagg\_vllm，4 GPU 真卡跑 LM-Eval） | VR Buildkite 已有 `pd-disagg-test` job，迁移即可 |
| **SR** | 全量 Go E2E (kind k8s + istio + helm + real 模型) | SR `make e2e-test`，已有 e2e/testcases/\*.go |
| | 信号引擎：Keyword / DomainClassify / PII / RBAC / Entropy 等 | SR 已有 testcases |
| | Dashboard 登录 + mcp 核心流程 | SR e2e `dashboard_auth.go` / `mcp_common.go` |

**硬件**：L4 × 2 或 A100 × 1 至少；4× 卡独立队列跑 P/D 分离用例。

#### L4（夜间全量 · < 3 h）

| 组件 | 检查项 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **AgentInfer** | 更多模型的 serve E2E（不同架构、不同 size） |
| | Benchmark：吞吐量、首 token 延迟、decode 延迟与 baseline 对比（阈值 2%\~5% 报警） |
| | AgentBench 全量（非 smoke） |
| **VR** | 全模型 E2E 扩展（`*_expansion.py`） |
| | 全策略 benchmark（cache\_aware / consistent\_hash / power\_of\_two 等吞吐和延迟） |
| | 文档示例测试 |
| **SR** | 全 profile e2e（istio / aibrix / llm-d / routing-strategies / production-stack / response-api-redis / authz-rbac 等 20+ profile） |
| | Perftest（SR `perf/` 目录下 perf.yaml + thresholds.yaml 基线对比） |
| | 全部 models accuracy 指标 |

#### L5（每周 / 发版前 · 1\~7 天级别）

| 组件 | 检查项 |
| -------------- | ------------------------------------------------------------------------------- |
| **AgentInfer** | 72h 长稳：`vllm serve` 长跑 + 持续压测，OOM / 内存泄漏 / 句柄泄漏检测 |
| | 故障恢复测试：推理过程 Kill worker，能否自动恢复、请求是否正确重试 |
| **VR** | 长稳：2w QPS 持续路由 24h；动态增删 worker、熔断器、重试 |
| | 覆盖率：`cargo tarpaulin` / `pytest --cov` 报告（目标 Rust ≥ 80%，Python ≥ 70%） |
| **SR** | K8s operator 滚动升级 + 回滚、多副本容灾 |
| | 信号引擎长稳：百万样本分类的精度不漂（模型文件没被误改） |
| | 安全扫描：Trivy（镜像漏洞）、CodeQL（代码扫描，SR 已有 `.github/workflows/codeql-analysis.yml` 可移植） |

### 3.3 Diff-Aware（按改动路径跳过不相关 job）

直接借鉴 omni 的 L2/L3 diff-aware 设计 + SR 现成的 `ci-changes.yml`。

**核心思想**：一个 PR 只改了 SR 的 `website/docs/*.md`，没必要跑 GPU 测试；只改了 VR 的 README，没必要跑 AgentInfer E2E。

Jenkins 侧实现方式（推荐）：

* 在每个组件的 Jenkinsfile 开头用 `changeset` 或 pathsFilter 脚本把路径映射成一组布尔 flag：

  ```text
  changed_agentinfer_code = changelog 里有 agentinfer/ tests/ pyproject.toml
  changed_vr_code         = changelog 里有 agentrouter/
  changed_sr_code         = changelog 里有 semantic-router/ (或独立仓自然就是全量)
  changed_docs_only       = 只改了 docs/** .md .mdx
  ```

* 然后每个 stage 加 `when { expression changed_xxx_code }` 决定跑不跑。

### 3.4 pytest markers 标签体系（建议照抄 Omni 再删没用的）

在 AgentInfer 根 `pyproject.toml` 里加一个统一的 markers 定义（SR / VR / AI 三组件各一份也可以，marker 名保持一致即可）：

```toml
[tool.pytest.ini_options]
markers = [
    # 分层（对应 L1~L5）
    "l1: 单元测试/快速检查（L1 门控）",
    "l2: 基础 E2E 和 GPU UT（L2 门控，PR Ready 后）",
    "l3: 核心集成+精度+关键性能（L3，merge 后）",
    "l4: 全功能+全性能+文档（L4，Nightly）",
    "l5: 稳定性/可靠性/覆盖率（L5，Weekly/Pre-release）",

    # 硬件（和 Omni 一致，方便未来扩展）
    "cpu: 纯 CPU 可跑",
    "gpu: 需要 GPU（含 CUDA / ROCm / NPU）",
    "cuda: NVIDIA CUDA GPU",
    "npu: 昇腾 NPU",
    "rocm: AMD ROCm GPU",
    "L4: 需要 L4 及以上卡",
    "A100: 需要 A100 及以上卡",
    "A2: 需要昇腾 A2 NPU",
    "A3: 需要昇腾 A3 NPU",
    "cards_1 / cards_2 / cards_4 / cards_8: 需要 N 张卡",

    # 功能域
    "slow: 慢速测试，PR 默认跳过",
    "benchmark: 性能基准测试（不做 pass/fail，只出指标）",
    "local_model: 需要本地模型文件（非 HF 公开下载）",
]
```

然后本地 / Jenkins 调用：

```bash
# L1：只跑 L1 标记 + CPU
pytest tests/ agentrouter/py_test/unit/ -m "l1 and cpu" -v

# L2：跑 L2 + L1（没打 L1 标记的小模型 smoke 默认算 L2），需要 L4 单卡
pytest tests/ -m "(l1 or l2) and L4 and cards_1" -v

# L3：合并后 nightly
pytest tests/ -m "l3 and (L4 or A100)" -v

# 只跑性能基准，失败不做 fail，出报告
pytest tests/benchmarks/ -m benchmark -v --benchmark-json=report.json
```

***

## 第四部分 · 实施路线图：从零到正式发布（分 8 阶段，可执行）

> **里程碑节奏建议**：**第 0 阶段 + 阶段 1 是你说的"先从部署 Jenkins 和实现 L1 开始"**，按顺序来。

```text
阶段 0  搭平台 ──── 阶段 1  L1 跑通 ──── 阶段 2  L2 跑通 ──── 阶段 3  L3 跑通
  (Jenkins/仓)    (三仓 PR Gate 绿)    (GPU UT/基础 E2E)  (Main 集成+真实权重)
                                                        │
                                    ┌───────────────────┘
                                    ▼
                              阶段 4  夜间 L4 ──── 阶段 5  L5 长稳
                              (全量/性能/文档)  (稳/可靠/安全扫描)
                                                        │
                                    ┌───────────────────┘
                                    ▼
                              阶段 6  正式发布流 ── 阶段 7  文档 + 培训
                              (Tag → 全自动化交付)  (全员使用)
```

### 阶段 0 · 基础设施搭建（1\~2 周，和同事一起推进）

**目标**：Jenkins 可用，三个仓都能触发流水线，至少跑一个 Hello World。

**任务清单**：

| # | 任务 | 负责人/协助 | 产出 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| 0.1 | 申请 Jenkins Controller 服务器 + Agent 机器（详见第五部分资源清单） | 你申请 / 同事协助部署 | 机器清单 + 账号 |
| 0.2 | Jenkins Controller 部署（nginx 反代 + HTTPS + LDAP/企业微信登录） | 同事部署 Jenkins 本体，你可以先看 / 跟着学 | 能访问 Jenkins Web UI，登录 |
| 0.3 | Jenkins 基础配置：① 安装常用插件（Git、Pipeline、Docker、Multibranch、Build Authorization、Config File Provider、Credentials、Workspace Cleanup、Role-based Authorization）② 设置全局工具（JDK、Git、Docker、Maven/Go 可选）③ 配置 Cloud（Docker 云 或 Kubernetes 云，如果以后要 K8s 动态 Agent） | 同事或你自己 | 全局配置页 OK |
| 0.4 | Agent 注册：**至少 3 台 Agent**——① CPU 构建机（16C 32G，L1 + build）② GPU 机（L4/A100，L2/L3）③ arm64 机（鲲鹏，VR/AI-NPU 镜像） | 同事带你注册，标签：`cpu-build`、`gpu-l4-1card`、`gpu-a100-4card`、`arm64-build`、`npu-a2-1card`…… | Jenkins → Build Executor Status 能看到 Agent 在线 |
| 0.5 | Credentials 配置：① Git 拉代码用的 SSH Key 或 Token ② Harbor 镜像推送账号 ③ 私有 PyPI 推送 Token ④ HF\_TOKEN（下权重用）⑤ 企业微信群机器人 webhook（发通知） | 你申请 + 配置进 Jenkins Credentials，ID 起好名（`git-read-token`、`harbor-push`、`pypi-publish`、`hf-token`、`wx-robot-ci`） | Credentials 列表里都能看到 |
| 0.6 | 为三个仓创建 **Multibranch Pipeline** 项目：① AgentInfer ② semantic-router（独立仓先跑着，等搬进来后再迁）③ vllm-router（VR，独立仓先跑） | 你自己建，每个项目配置：Branch Sources = 仓库 URL + Credential；Build Configuration = by Jenkinsfile（从 SCM） | 三个 Multibranch 项目出现，可手动点 "Scan Repository Now" |
| 0.7 | 三个仓各写一份最简 Jenkinsfile（只 echo 文本），验证：push 代码 → Jenkins 自动扫描 → 流水线跑起来 → 企业微信通知成功/失败 | 你写 | Hello World 流水线全链路绿 |

### 阶段 1 · L1 质量门落地（1 周）

**目标**：三个仓的 PR，不改对就不让合并（L1 Gate 绿 + DCO + 格式检查）。

**任务清单**：

| # | 任务 | 产出 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| 1.1 | 写/完善三个仓 Jenkinsfile 的 L1 stage，按 3.2 的 L1 表逐一项落下去：• agentinfer：pre-commit → pip install -e → pytest UT• VR：rustfmt → clippy → cargo UT → pytest py\_test/unit• SR：pre-commit → gofmt → go vet → go build → go UT | 三份 Jenkinsfile 都有 L1 stage |
| 1.2 | 构建**三仓共用的 L1 CI 基础 Docker 镜像**（`internal/ci-base:l1-ubuntu22.04`，含 Python3.10/3.12、Rust 1.95.0、Go 1.22+、protobuf、libsdl、pre-commit 环境），避免每次 CI 从头装 | Dockerfile + Harbor 已推送镜像 |
| 1.3 | 在 Git 平台（Gitee/GitLab）上配 **Branch Protection**：main 分支**禁止直接 push**，**必须通过 PR**；"Required status checks" 选上 Jenkins L1 job 的状态名 | PR 页面能看到 L1 的 check mark，不绿不能点 Merge |
| 1.4 | 配置 **PR 模板**（PULL\_REQUEST\_TEMPLATE.md），把 Checklist 写清楚："我是否加了对应 L1/L2 测试 / 是否同步改了文档 / 是否改了版本号" | PR 提交即显示模板 |
| 1.5 | 提交 3 个实验 PR（各一个），故意写错格式 / 写错语法，验证 L1 会红，禁止合并 | L1 拦截生效 |
| 1.6 | 写 CI Failure Handbook 第一版：常见错误示例 + 定位方法 + 怎么手动本地跑等价命令 | `docs/zh/how-to/ci-failure-handbook.md` v0.1 |

### 阶段 2 · L2（PR Ready Gate · GPU UT + 基础 E2E）落地（1\~2 周）

**目标**：关键 PR 在合入之前已经在 GPU 机器上跑过基础 E2E，保证 main 不会"今天编译过了明天启动挂"。

| # | 任务 | 产出 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 2.1 | GPU Agent 验证：手动在 GPU Agent 机上跑一遍 AgentInfer / VR / SR 的 L2 命令（参考 3.2 的 L2 表），确认环境能通、驱动正常、CUDA/NPU 版本对齐 | 本地能 100% 跑通 L2 命令 |
| 2.2 | Jenkinsfile 加 L2 stage：`agent { label 'gpu-l4-1card' }`，用 docker agent 拉起测试镜像。L2 默认不是 PR 强制 gate，**可以做成"可选 Gate"**（PR 评论 `/run-l2` 或 PR 打上 `ready-for-test` 标签触发） | Jenkinsfile 有 L2 stage，评论触发能跑 |
| 2.3 | pytest marker 体系（3.4 小节定义）加到各仓 pyproject.toml，测试文件全部补好 marker decorator | `pytest -m "l2 and gpu"` 可以筛出 L2 |
| 2.4 | 做 2 次"故意搞破坏"验证：故意改个会导致 serve 启动失败的参数提交 PR，L1 绿但 L2 红；故意 mock worker 返回格式错，VR L2 红 | L2 能抓出启动类错误 |
| 2.5 | 小优化：Diff-Aware（3.3）先做初版——只改 docs 的 PR 直接跳过 L2 | 改文档 PR 不再占 GPU |

### 阶段 3 · L3（Main 分支合并后 · 核心集成 + 真实权重 + 关键性能）（1\~2 周）

| # | 任务 | 产出 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 3.1 | 补 AgentInfer 侧真实模型 E2E：写 `tests/e2e/serve_qwen2_7b.py`，用 Qwen2-7B 测 chat/completions、streaming、多轮（真实权重，本地 HF cache） | 真实权重测试用例 |
| 3.2 | 补 AgentInfer × VR × mock-vllm 三组件 docker-compose 联合测试：模拟真实部署流量路径（HTTP → VR 路由 → AgentInfer 推理 → 返回） | `tests/e2e/compose_stack_smoke/`（docker-compose.yml + test.py） |
| 3.3 | VR 已有 `cargo test --test '*'` 和 `py_test/e2e/test_regular_router.py`，把这些串进 L3 stage；P/D 分离测试（4 卡）放到独立的 `gpu-a100-4card` 队列，用 Buildkite 原逻辑套进 Jenkins | VR L3 全绿 |
| 3.4 | SR 已有完整 Go E2E (kind) 体系和 20+ profile，把它迁到 Jenkins GPU Agent：`make e2e-test PROFILE=core` 作为 L3 基础 profile 跑；其余全量放 L4 | SR L3 能跑完 core profile 绿 |
| 3.5 | **性能基线 v0.1**：各组件先跑一遍"基准 commit"保存基线 JSON（AgentInfer: 首 token 延迟 / decode 速度；VR: QPS/平均延迟；SR: 分类吞吐），后续每次 L3 对比，劣化超过阈值（默认 5%）则红 | `tests/benchmarks/baselines/v0.1/*.json` + 阈值配置 |
| 3.6 | 通知：L3 只要红就发企业微信 @ 对应 committer + 维护者群 | 机器人 webhook 配置 OK |

### 阶段 4 · L4（夜间全量 · 功能 + 文档 + 性能）（1 周）

| # | 任务 | 产出 |
| --- | -------------------------------------------------------------------------------------------------- | ------------------------- |
| 4.1 | Jenkins 加 "Nightly Trigger"（Cron `H 2 * * *`，每天凌晨 2 点），触发三仓各 L4 Pipeline | 每天自动跑 |
| 4.2 | 补全 AgentInfer L4 扩展模型（Qwen2-14B、Llama-3-8B 等，和业务侧对齐"关键模型清单"） | L4 `*_expansion.py` 用例 |
| 4.3 | Benchmark 报告接入：HTML 报告归档到 Jenkins artifacts，劣化自动进第二天晨会清单 | Build Summary 里能点开 report |
| 4.4 | 文档示例测试：把 docs 里的命令（docker run / curl 测 API）抽成 script 跑一遍，和 omni `tests/examples/online_serving` 同理 | `tests/examples/` 目录 |
| 4.5 | 每日 L4 结果 summary 发群 | 群内每天早上 9 点能看到 "昨夜 N 个失败" |

### 阶段 5 · L5（长稳 · 可靠 · 安全 · 覆盖率 · Release 前专项）（2\~4 周，可与 6/7 并行）

| # | 任务 | 产出 |
| --- | -------------------------------------------------------------------------------------------- | ------------------------------------- |
| 5.1 | 长稳脚本 + 容灾脚本（K8s kind + 故障注入，kill pod、断网、OOM 模拟） | `tests/stability/` 目录 |
| 5.2 | 覆盖率报告：Rust `cargo tarpaulin`、Python `pytest --cov` + 阈值（Rust ≥80%，Python ≥70%），低于阈值红 | Coveralls / 内网 HTML 报告页 |
| 5.3 | 安全扫描集成：① Trivy 镜像漏洞扫描（VR 和 SR 自己的 CI 已有 trivy.yaml）→ 迁移到 Jenkins ② CodeQL 代码扫描（或用 SonarQube） | 每次发布前必跑，Critical 漏洞必须修 |
| 5.4 | 发布前 72h L5 Block：tag 打的那一刻，自动触发"全量 L5 Block"，不绿不让继续 release pipeline | release Jenkinsfile 前序加 L5 Block gate |

### 阶段 6 · 正式发布流水线（Tag → 全自动化交付）（1 周）

| # | 任务 | 产出 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 6.1 | 三仓各一份 Release Jenkinsfile（或用同一份多分支，trigger by tag）：按 2.3 节"三个组件发布流"逐一步骤实现 | Release Pipeline 可以手动 dry-run |
| 6.2 | 版本号一致性检查脚本（和 SR `check_version_contract.py` 一样思路），tag ↔ pyproject ↔ Cargo.toml ↔ go.mod 比对 | 不匹配直接 fail |
| 6.3 | 多架构构建：`docker buildx build --platform linux/amd64,linux/arm64`，或两台 agent 各 build 一台后 docker manifest create（参考 VR release-pipeline.yml 202+ 行的 manifest 写法） | Harbor 里同一个版本同时存在 amd64/arm64 manifest |
| 6.4 | Smoke test release 产物：二进制跑 `--version`、wheel pip 安装 import 成功、镜像 `docker run ... /health` 返回 200 | Release 不跑过 Smoke 不推送 |
| 6.5 | 推送：镜像 push Harbor、wheel push 私有 PyPI、Release Note 写 Gitee/GitHub Release 页 | Tag v0.1.x 的时候全流程一遍绿 |
| 6.6 | 打三个 "内测 tag"（如 v0.0.1-test），把 6.1\~6.5 全部实际跑一遍，排查坑 | 至少 1 次完整端到端 Release 演练成功 |

### 阶段 7 · 文档沉淀 + 团队培训（可与 4/5/6 并行，持续进行）

| # | 任务 |
| --- | ---------------------------------------------------------------------------------- |
| 7.1 | 完善本文档（`design/ci-cd-quality-gate-setup-guide.md`）后续章节、常见坑位、FAQ |
| 7.2 | 写 `docs/zh/how-to/run-ci-locally.md`：教开发者本地一条命令跑 L1，不依赖 Jenkins |
| 7.3 | 写 `docs/zh/how-to/release.md`：一步一步教维护者怎么打 tag 触发发布 |
| 7.4 | 给团队做一次 30min 分享：CI 是啥、L1\~L5 流程、怎么看 Jenkins 失败日志、怎么本地复现 |

***

## 第五部分 · 资源申请清单（你需要去要什么）

> 分**必须**、**强烈建议**、**可选**三档。优先申请"必须"的，其他的以后加。
> 备注：以下数量假设三仓并行开发，并发度约 5 条流水线（3 L1 + 2 GPU）。如果团队小（< 8 人开发）可以减半。

### 5.1 服务器

| 分类 | 数量 | 配置（建议） | 用途 | 优先级 |
| -------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------- |
| **Jenkins Controller** | 1 台 | 8C / 16G / 200G SSD，Linux（Ubuntu 22.04 或 openEuler 24.03），不需要 GPU | Jenkins 主节点：调度 + Web UI + Artifacts 存储（短期） | **必须** |
| **CPU 构建 Agent (x86)** | 2 台（或 1 台 32C，开 8 executor） | 16C / 32G / 500G SSD，Ubuntu 22.04，装 Docker | L1 编译 + 测试 + 打非 GPU 镜像 | **必须** |
| **GPU Agent (单卡 L4 或同级)** | 1\~2 台 | 16C / 64G / 1T SSD，L4 × 1（24G 显存），装对应 CUDA 驱动 + Docker + nvidia-container-toolkit | L2 基础 E2E + AgentInfer 真实模型 L3 单模型 | **必须** |
| **GPU Agent (多卡 A100 或同级)** | 1 台 | 32C / 256G / 2T NVMe，A100 × 4（或按需求 2 卡也行） | VR P/D 分离 L3 测试、全量 L4 测试、多卡用例 | **强烈建议** |
| **鲲鹏 arm64 CPU Agent** | 1 台 | 16C / 32G / 500G SSD，openEuler 24.03，装 Docker | VR / AgentInfer NPU / ascend arm64 镜像构建 + L1 arm64 UT | **必须**（如果有 NPU / arm64 发布需求） |
| **昇腾 NPU Agent** | 1 台 | 16C / 64G / 1T SSD，A2 或 A3 NPU 至少 1 张，openEuler 24.03 + 对应 CANN + 驱动 | AgentInfer NPU 版本 L2/L3（`agentinfer:ascend` 镜像用） | **强烈建议**（若需要支持 NPU） |
| **Harbor / 制品仓库存储** | 1 台（或复用存储服务） | 16C / 32G / 10T+ HDD（镜像膨胀快），Ubuntu/openEuler | Harbor 私有镜像仓库 + Helm chart OCI 仓库 | **必须** |
| **Nexus / 文件服务器**（可选，如 Harbor 不托管 Python 制品） | 1 台（或用公司已有） | 8C / 16G / 2T SSD | 私有 PyPI（wheel）+ 二进制下载站 | **强烈建议**（否则 wheel 和二进制放 Harbor OCI 也行） |
| **长稳测试专用 GPU 集群**（L5 用） | 1 套独立 K8s（可以和测试环境共享） | 至少 2 节点 GPU + 1 master | L5 长稳 / 可靠性 / 故障注入测试。不建议和 L2/L3 共用 Agent，会把长稳测试打断。 | **可选**，阶段 5 前到位即可 |

### 5.2 域名 / 网络 / 端口

| 资源 | 需求 | 用途 |
| ------------- | -------------------------------------------------------------------------------------------- | ----------------------------- |
| Jenkins 域名 | 1 个，如 `jenkins.xxx-internal.com`，HTTPS 证书 | 团队访问 Web UI + Git Webhook 回调用 |
| Harbor 域名 | 1 个，如 `harbor.xxx-internal.com`，HTTPS | `docker pull/push` |
| Nexus 域名（如需要） | 1 个，如 `nexus.xxx-internal.com` | `pip install --index-url` |
| 出网 | Agent 机器必须能访问：① Git 仓、② PyPI/CRATES.io 镜像源、③ HuggingFace（或 HuggingFace 内网镜像站）、④ Harbor/Nexus | 构建必须 |
| 入网 | Git 平台（Gitee/GitLab）必须能 POST Webhook 到 Jenkins `80/443` | PR/push 能触发流水线 |

### 5.3 账号 / Token / 权限

| 资源 | 用途 | 申请到了之后存到 Jenkins Credentials |
| ------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------ |
| Git 仓**只读** SSH Key 或 Personal Access Token | Jenkins 拉代码 | ID: `git-read-token` (username/password 或 secret file) |
| Git 仓**读写** PAT（可选） | Release Pipeline 写 Release Notes / 更新 GitHub/Gitee Release Page | ID: `git-write-token` |
| Harbor **管理员或项目管理员**账号 | push 镜像、创建项目、创建 robot | 账号名 + 密码 |
| Harbor robot 账号（建议单独建） | 流水线 push/pull 镜像（不挂真人账号） | ID: `harbor-robot` |
| 私有 PyPI / Nexus PyPI publisher 账号 token | twine upload wheel | ID: `pypi-publish-token` |
| **HuggingFace Token** (HF\_TOKEN，建议申请 team 级别的) | 下载 Llama/Qwen 等 gated 模型权重；L3/L4 E2E 要用 | ID: `hf-token`，类型 Secret Text |
| 企业微信群机器人 **Webhook** URL（2 个：一个日常通知、一个 L3+Release 紧急通知） | CI 红绿推送群消息 | ID: `wx-ci-notify`、`wx-ci-urgent` |
| Jenkins 管理员账号（给你一个） | 你自己配置流水线、加 Agent、装插件 | —— |
| Jenkins 普通开发者账号（按人头） | 团队成员看日志、重跑流水线 | —— |
| LDAP / SSO 集成（可选） | 免密登录 Jenkins（不用每个人建账号） | —— |

### 5.4 工具链 / 软件授权

| 工具 | 用途 | 备注 |
| --------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| Docker Engine / Docker Buildx | 所有 Agent 必须装，用于 Docker Agent 模式 + 镜像构建 | 开源免费，直接 apt/yum 装 |
| nvidia-container-toolkit | GPU Agent 必须装，才能 `--runtime=nvidia` 把 GPU 挂进容器 | NVIDIA 免费 |
| CANN Toolkit + NPU Docker 运行时 | NPU Agent 必须 | 昇腾官方 |
| Harbor 企业版 / 开源版 | 私有镜像仓库 | 开源版免费，够用 |
| Nexus Repository OSS（或 Harbor 兼用） | 私有 PyPI | 开源免费 |
| SonarQube（可选，阶段 5 引入） | 代码质量静态分析 + 安全扫描 | 开源免费 CE 版 |
| Trivy（阶段 5） | 镜像漏洞扫描 | 开源免费，直接容器里跑 |
| CodeQL（可选） | 代码语义级漏洞扫描 | 开源免费，GitHub Actions 内置；如果用 Jenkins 也能跑 CLI |
| 企业微信机器人（如果还没有群） | 消息通知 | 免费，群设置里直接建 |

### 5.5 其他需要你主动确认/推动的事

1. **代码放在哪个平台**：现在是本地 code 目录 + 快捷方式。CI 必须要"一个可被 Jenkins Clone 的 Git 服务器"
   ——是 Gitee / GitLab（企业内部部署的）/ GitHub（公有）？确认清楚，Webhook 才能配。
2. **分支策略**：用 Git Flow、Trunk-Based（main + feature branch）还是其它？
   推荐 **Trunk-Based + short-lived feature branch + squash merge**，对 CI 最友好。
3. **版本号规范**：三组件都统一 SemVer 吗？tag 命名 `v{组件名}-{版本}`？还是三个独立 tag（`sr-v0.3.1` / `vr-v0.1.16` / `ai-v0.2.0`）？建议独立 tag，见 2.3 节。
4. **HuggingFace 模型镜像**：L3/L4 需要反复下权重，强烈建议在公司内搭一个 **HF 镜像站（或本地权重盘/NFS）**，不然每次下几十 GB 会把 CI 拖爆。
5. **谁来审核失败**：L3 红了谁先看、响应 SLA 是多久、要不要 Oncall 排班。一般你作为 owner 先顶，后面和团队一起定。

***

## 第六部分 · 从今天就能动手做的 3 件事

不想被这么多内容压到？先做最小 3 步，跑起来再补全：

```text
□ 1. 和同事约 30 分钟对齐：
     ├─ Jenkins Controller 预计什么时候装好？
     ├─ 我能先自己在一台测试 VM 上装 Jenkins 练手吗？
     └─ 我们的 Git 仓在什么平台？有没有 Harbor / 私有 PyPI？

□ 2. 自己本地写三份 Hello-World Jenkinsfile
     ├─ AgentInfer/Jenkinsfile
     ├─ semantic-router/Jenkinsfile（独立仓）
     └─ vllm-router/Jenkinsfile（独立仓）
     结构：只做 checkout → echo "L1 start" → sh "cat pyproject.toml" 之类。
     现在本地写好了，等 Jenkins 一部署好直接 push 就能跑。

□ 3. 同步去申请资源清单里"必须"的 5 项：
     ├─ Jenkins Controller 服务器
     ├─ CPU Agent × 2
     ├─ GPU Agent (L4) × 1
     ├─ arm64 Agent（如果有 NPU）
     └─ Git token + Harbor 账号 + 企业微信群机器人
```

***

## 附录 · 术语对照表（中英快速查）

| 英文 | 常见中文译名 | 本指南里怎么用 |
| ------------------------------------- | ------------- | ---------------------------------------- |
| CI / Continuous Integration | 持续集成 | 见 1.1 |
| CD / Continuous Delivery / Deployment | 持续交付 / 部署 | 见 1.1 |
| Pipeline / Workflow | 流水线 / 工作流 | Jenkins 里一个"项目"或 GitHub Actions 的 yml 文件 |
| Stage | 阶段 | Build → Test → Publish 这种大段 |
| Step | 步骤 | 每条 shell 命令 / action |
| Job / Task | 任务 | 一台 Agent 上执行的一组 steps |
| Agent / Runner / Node | 执行机 / 跑者 / 节点 | 真正跑命令的机器 |
| Controller / Master | 主节点 / 控制节点 | Jenkins 调度中心 |
| Artifact / 产物 | 工件 / 产物 | 流水线跑完产出的文件 |
| Release | 发布 / 发行 | 带版本号的正式交付物集合 |
| Quality Gate / Gate | 质量门 / 关卡 | 不通过就禁止进入下一阶段（比如 L1 红了不能合并） |
| Pre-commit | 提交前钩子 | 本地 git commit 时自动触发的格式化/检查脚本 |
| Webhook | 网络钩子 | Git 平台有新事件时 POST 到 Jenkins 的 HTTP 回调 |
| Smoke Test | 冒烟测试 | 让程序跑 `--version` 或 `/health`，不死就行的最低限度验证 |
| Diff-Aware | 按变化路径感知 | 只改了 docs，就跳过 CPU/GPU 测试 |
| Multibranch Pipeline | 多分支流水线 | Jenkins 一种项目类型，自动扫描每个分支的 Jenkinsfile |
| DCO / Developer Certificate of Origin | 开发者来源证书 | commit 末尾的 `Signed-off-by:` 签名 |

***

> 本指南 v0.1 是 DRAFT，后续每个阶段做完请你回来同步更新这里的计划/产出/踩坑。有看不懂的段落直接圈出来问我，我继续补充示例和图解。
