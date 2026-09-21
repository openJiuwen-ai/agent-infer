# vLLM Router — Linux 编译与部署手册

**文档版本**：v1.0　｜　**对应代码**：VR tag `v0.1.15` (<https://github.com/vllm-project/router/releases/tag/v0.1.15>)

**适用读者**：负责在 Linux 服务器（x86 Ubuntu 22.04/24.04 或 arm64 openEuler 内核 6.6 / glibc ≤ 2.38）上构建、交付并部署自编译 VR 的工程师。

**交付形态**：**Docker 镜像**（唯一部署形态，目标服务器零 Python/Go/Rust 工具链依赖）+ 配置模板 + 本手册。

**两种架构同时支持**：本手册所有构建命令均提供 amd64 与 arm64 双路径。本文档描述从源码编译 vLLM Router，以 v0.1.15版本为例，制作**Docker 镜像**，并交付部署到目标机器的完整流程。

---

## 0. vLLM Router 是什么

vLLM Router 是 vLLM 官方生态中的**高性能轻量级请求路由器（负载均衡器）**：它自己不做推理，而是站在一组 vLLM 推理服务（worker）前面，接收客户端的所有 LLM 请求（OpenAI
兼容接口），再按调度策略把请求分发给最合适的 worker。类比理解：vLLM worker 是干活的"厨师"，Router 是看单分菜的"传菜员"。

### 主要特性

| 特性 | 说明 |
| --- | --- |
| **多算法负载均衡** | cache_aware（默认，按 KV 缓存前缀匹配度调度，减少重复计算）、round_robin、random、power_of_two、consistent_hash、rendezvous_hash |
| **PD 分离（Prefill-Decode Disaggregation）** | 预填充与解码阶段部署在不同节点时的专用路由（`--vllm-pd-disaggregation`），配合 ZMQ/K8s 服务发现 |
| **K8s 服务发现** | worker 以 Pod 形式动态注册/摘除（`--service-discovery` + `--selector`），无需重启 router |
| **健康检查与容错** | 后台周期探测 worker `/health`；熔断器（Circuit Breaker）、请求重试可配置 |
| **企业特性** | Prometheus 指标（默认 29000 端口）、OpenTelemetry 分布式追踪、API Key 校验、请求体大小限制 |
| **多后端类型** | `--backend vllm`（默认，多 worker 负载均衡）/ `openai`（对接外部 OpenAI 兼容 API，单上游代理）/ `trtllm`、`anthropic`（v0.1.15 尚未实现，仅保留参数） |

### 后端接入说明（重要）

一个 router 实例**只能选一种后端模式**（由 `--backend` 决定），不能把 vLLM worker 和外部 API 混在一个实例里负载均衡。两种模式的差异：

| | `--backend vllm`（默认） | `--backend openai` |
| --- | --- | --- |
| 上游 | 自建 vLLM 服务，**可多个** | 外部 OpenAI 兼容 API，**只能 1 个**（源码校验 `worker_urls.len() == 1`） |
| 行为 | 真负载均衡，按策略分发 | 单上游代理（统一入口/鉴权/指标），不分发 |
| `--worker-urls` 填法 | 各 worker 地址，如 `http://10.1.1.11:8000 http://10.1.1.12:8000` | 上游 base URL，如 `https://api.deepseek.com/v1` 或 `https://api.openai.com/v1` |
| 上游鉴权 | 可用 `--api-key` 配置 router→worker 的密钥 | **客户端请求里的 `Authorization` 头原样透传**给上游，密钥由客户端携带 |

两种模式的具体配置与测试命令见 **第 10 节**。

---

## 1. 目标环境与产物

| 目标环境 | CPU 架构 | 部署镜像架构 | 说明 |
| --- | --- | --- | --- |
| Ubuntu 22.04 / 24.04 | x86_64 | `linux/amd64` | |
| openEuler 24.03 LTS SP4（内核 6.6，glibc 2.38） | arm64（鲲鹏） | `linux/arm64` | 昇腾 NPU 机器 |

**产物清单：**

| 产物 | 名称 | 说明 |
| --- | --- | --- |
| amd64 镜像 | `vllm-router:v0.1.15-amd64` | 在 x86 构建机上构建，**部署用** |
| arm64 镜像 | `vllm-router:v0.1.15-arm64` | 在鲲鹏机器上构建，**部署用** |
| amd64 二进制 | `vllm-router-v0.1.15-linux-amd64` | musl 全静态二进制，从镜像提取（见 7.3），与镜像内完全同源，**一并提交** |
| arm64 二进制 | `vllm-router-v0.1.15-linux-arm64` | 同上，鲲鹏机器产出，**一并提交** |
| 离线镜像包 | `vllm-router_v0.1.15_amd64.tar` / `vllm-router_v0.1.15_arm64.tar` | `docker save` 导出，目标机器 `docker load` 导入 |

> 镜像名前缀：若后续确定了内部镜像仓库（SWR/Harbor 等），在所有命令前加 `仓库地址/命名空间/`，例如
> `swr.cn-southwest-2.myhuaweicloud.com/yuanrong-dev/vllm-router:latest`。

**两个关键结论（先建立认知，后面所有步骤都基于此）：**

1. **Router 是纯 CPU 的 HTTP 流量转发器**，不直接访问 NPU。镜像里**不需要** CANN/昇腾驱动，运行时**不需要**挂载 `/dev/davinci*` 设备。昇腾相关的东西属于它身后的
vLLM（vllm-ascend）后端，与本镜像无关。
2. **二进制采用 musl 全静态链接**：单个文件、不依赖 glibc 和任何 `.so`。因此关于glibc版本的约束自动满足，同一架构下任何 Linux 发行版（Ubuntu/openEuler/其他）裸机也能直接跑这个二进制。

---

## 2. 总体流程

```text
┌─────────────────┐      ┌──────────────────┐
│  x86 Linux 构建机 │      │ 鲲鹏(arm64) 构建机 │
│  (可连外网)       │      │  (可连外网)        │
└───────┬─────────┘      └────────┬─────────┘
        │ docker build             │ docker build
        │ (rust:1-alpine 容器内     │ (完全相同的命令，
        │  musl 静态编译)           │  宿主架构决定产物)
        ▼                          ▼
 vllm-router:v0.1.15-amd64   vllm-router:v0.1.15-arm64
        │                          │
        ├─ 提取二进制 (7.3) ────────┤ ──► 交付物 2: vllm-router-v0.1.15-linux-<arch>
        │ docker save ──► tar 包 ──┤ ──► 交付物 1: 镜像 tar 包
        ▼                          ▼
┌──────────────────────────────────────────┐
│  目标机器（可完全离线）                      │
│  docker load -i xxx.tar                   │
│  docker run / docker compose up           │
└──────────────────────────────────────────┘
```

要点：

- **两个架构各自在"同架构机器"上原生构建**，不需要 docker buildx、不需要 QEMU 模拟（QEMU 模拟 arm64 编译 Rust 非常慢且偶有诡异错误）。
- 所有编译都发生在 `rust:1-alpine` 容器内部，**构建机上只需要装 Docker**，不需要安装 Rust 工具链、Python、cmake 等任何依赖。
- 交付默认按**离线场景**设计（`docker save` 拷贝），目标机器不需要连外网；若两台机器都能访问同一内部仓库，改用 `docker push/pull` 即可。

---

## 3. 前置条件

| 机器 | 要求 |
| --- | --- |
| x86 构建机 | Linux + Docker ≥ 20.10；能访问 Docker Hub（拉 `rust:1-alpine`、`alpine:3.22`）和 crates.io |
| 鲲鹏构建机 | arm64 Linux（openEuler 即可）+ Docker ≥ 20.10；网络要求同上 |
| 目标机器 | Docker ≥ 20.10（alpine 镜像对 Docker 版本要求很低）；**无需外网** |

openEuler 安装 Docker：

```bash
dnf install -y docker
systemctl enable --now docker
```

网络不通畅时（可选）：为 Docker 配置 registry mirror 拉取基础镜像，为 Rust 配置 rsproxy 镜像源（见 FAQ 第 7 条）。

---

## 4. 准备源码

```bash
git clone --depth 1 --branch v0.1.15 https://github.com/vllm-project/router.git
cd router
```

然后在 `router` 目录内准备以下改动（相对上游 v0.1.15 的**全部改动**，详见第 5、6 节）：

- `Dockerfile.static` —— 静态构建用的 Dockerfile，完整内容见第 6 节，直接在 `router` 目录下创建
- `.dockerignore` —— 缩小构建上下文，完整内容见下
- `Cargo.toml` 中的一行依赖改动（hf-hub 的 feature 调整），见第 5 节

`.dockerignore` 完整内容：

```text
.git
.github
target
dist
docs
examples
tests
*.md
```

---

## 5. 源码改动说明

上游 v0.1.15 的依赖树里有两个"非纯 Rust"组件，直接影响静态链接：

| 依赖 | 问题 | 处理 |
| --- | --- | --- |
| `hf-hub 0.4.3`（拉取 tokenizer 用） | 默认 feature 开启 `native-tls`，引入 **OpenSSL**（这是官方 Dockerfile 装 `libssl-dev` 的原因） | 关闭默认 feature，换纯 Rust 的 `rustls`（见下） |
| `zmq`（PD 分离模式的服务发现） | 默认链接系统 libzmq C 库 | **无需改动**：`zmq-sys 0.12` 检测不到系统 libzmq 时会自动用内置的 `zeromq-src` 源码编译并静态链入，构建容器里装好 `cmake/make/g++` 即可 |
| `reqwest 0.13`（HTTP 客户端） | — | **无需改动**：0.13 版默认已用 rustls，无 OpenSSL |

`Cargo.toml` 中唯一的改动（1 行）：

```diff
- hf-hub = { version = "0.4.3", features = ["tokio"] }
+ hf-hub = { version = "0.4.3", default-features = false, features = ["tokio", "rustls-tls"] }
```

- 代码中只用了 `hf_hub::api::tokio::ApiBuilder`（异步 API，见 `src/tokenizer/hub.rs`），关闭同步的 `ureq` 客户端不影响任何功能；
- TLS 能力由 OpenSSL 换成 rustls（纯 Rust 实现，直接编译进二进制），HTTPS 功能完全一致。

---

## 6. Dockerfile.static

（在 `router` 目录下按以下完整内容创建 `Dockerfile.static`）

```dockerfile
# syntax=docker/dockerfile:1
#
# vLLM Router v0.1.15 - musl 全静态链接构建
# 产物: 单个静态二进制 /usr/local/bin/vllm-router（无任何 .so / glibc 依赖）
# 用法: 在对应架构的机器上直接 `docker build`（x86 机产 amd64，鲲鹏机产 arm64，
#       无需 buildx/QEMU 交叉模拟）

# ---------- 阶段 1: 编译 ----------
# rust:1-alpine 的宿主编译目标本身就是 *-unknown-linux-musl，
# 默认开启 crt-static，因此直接编译即得到全静态二进制
FROM rust:1-alpine AS builder

# musl-dev : C 标准库头文件与静态档案
# cmake make g++ : zmq-sys 检测不到系统 libzmq 时，自动用内置的
#                  zeromq-src 从源码编译 libzmq（C++ 项目）并静态链入
# python3 : pyo3 构建脚本仅需要解释器探测（extension-module，不链接 libpython）
RUN apk add --no-cache musl-dev cmake make g++ python3

# crates.io 国内镜像（rsproxy），避免国内直连 crates.io 时
# HTTP/2 流被重置（"Stream error in the HTTP/2 framing layer"）
RUN mkdir -p $CARGO_HOME && \
    printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n\n' > $CARGO_HOME/config.toml && \
    printf '[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n' >> $CARGO_HOME/config.toml

WORKDIR /app

# 先拷贝依赖清单，利用 Docker 层缓存
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# 只构建 vllm-router 二进制（跳过 cdylib/Python 扩展与测试），release 优化
RUN cargo build --release --bin vllm-router && \
    strip target/release/vllm-router

# ---------- 阶段 2: 运行镜像 ----------
FROM alpine:3.22

# ca-certificates : 出站 HTTPS（如从 HF 拉取 tokenizer）需要系统根证书
# tzdata          : 日志使用本地时区
RUN apk add --no-cache ca-certificates tzdata

COPY --from=builder /app/target/release/vllm-router /usr/local/bin/vllm-router

# 30000: 默认服务端口（--port）
# 29000: 默认 Prometheus 指标端口（--prometheus-port）
EXPOSE 30000 29000

ENTRYPOINT ["/usr/local/bin/vllm-router"]
CMD ["--host", "0.0.0.0", "--port", "30000"]
```

设计要点：

- **构建容器 = alpine（musl 系统）**：`rust:1-alpine` 里 `cargo build` 的默认编译目标就是 `*-unknown-linux-musl` 且 `crt-static`
开启，不需要任何交叉编译配置。
- **两阶段构建**：编译工具链、 crates.io 依赖缓存全部留在 builder 层，运行镜像只有二进制 + CA 证书，最终约 25~35MB。
- **`--bin vllm-router`**：只编译 Rust 二进制，跳过 Python 扩展（cdylib）与全部测试代码，又快又干净。
- 运行镜像用 `alpine:3.22` 而非 `scratch`：保留 busybox shell 和 `wget`，方便进容器排障、写健康检查，同时提供系统 CA 证书路径。

---

## 7. 构建命令

> 如需在单台 Windows 电脑上同时构建 amd64 + arm64（buildx + QEMU，无需 Linux 构建机），参见 [deploy-vllm-router-on-windows.md](deploy-vllm-router-on-windows.md)。

### 7.1 x86_64 镜像（在 x86 构建机上，源码目录内执行）

```bash
docker build -f Dockerfile.static -t vllm-router:v0.1.15-amd64 .
```

### 7.2 arm64 镜像（在鲲鹏机器上，命令完全相同）

```bash
docker build -f Dockerfile.static -t vllm-router:v0.1.15-arm64 .
```

宿主机是什么架构，产物就是什么架构——鲲鹏机上 `uname -m` 为 `aarch64`，产出的自然是 arm64 镜像，无需任何额外参数。

首次构建需从 crates.io 下载依赖并完整编译（release 含 LTO，全量约 10~20 分钟，取决于机器性能）。首次构建后 `Cargo.lock` 可能被 cargo 自动微调（feature 变化的正常结果），建议保留。

### 7.3 二进制文件构建与提取（可选）

二进制在 7.1/7.2 的镜像构建过程中已经编译完成（位于镜像内 `/usr/local/bin/vllm-router`）。**不需要单独再编译一次**——直接从构建产物中提取，保证提交的二进制与镜像内的二进制**完全同源一致**。

> 为什么不在构建机宿主机上直接 `cargo build`：构建机无需安装 Rust 工具链（第 3 节），且镜像/二进制一次编译两用，杜绝"提交的二进制和镜像版本不一致"的问题。

#### 方式 A：从镜像提取（推荐，任何 Docker 版本可用）

```bash
# x86 构建机上（amd64）
docker create --name vr-bin vllm-router:v0.1.15-amd64
docker cp vr-bin:/usr/local/bin/vllm-router ./vllm-router-v0.1.15-linux-amd64
docker rm vr-bin

# 鲲鹏构建机上（arm64）
docker create --name vr-bin vllm-router:v0.1.15-arm64
docker cp vr-bin:/usr/local/bin/vllm-router ./vllm-router-v0.1.15-linux-arm64
docker rm vr-bin
```

#### 方式 B：BuildKit 直出二进制（可选，不产生中间镜像）

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.static \
  --target builder --output type=local,dest=./bin .
# 产物路径（保留容器内目录结构）：
#   ./bin/app/target/release/vllm-router
mv ./bin/app/target/release/vllm-router ./vllm-router-v0.1.15-linux-amd64
rm -rf ./bin
```

**生成校验和（与二进制一并提交，用于目标机器验证完整性）：**

```bash
sha256sum vllm-router-v0.1.15-linux-amd64 > vllm-router-v0.1.15-linux-amd64.sha256
# arm64 侧同理
sha256sum vllm-router-v0.1.15-linux-arm64 > vllm-router-v0.1.15-linux-arm64.sha256
```

**二进制验证（构建机上执行，三项都过才算合格）：**

```bash
# 1. 确认是全静态（预期输出包含 "statically linked"）
file vllm-router-v0.1.15-linux-amd64

# 2. 确认无动态库依赖（预期输出 "not a dynamic executable"）
ldd vllm-router-v0.1.15-linux-amd64

# 3. 确认可执行
./vllm-router-v0.1.15-linux-amd64 --version
```

**裸机直接运行二进制（可选，应急/调试场景）：**

```bash
chmod +x vllm-router-v0.1.15-linux-arm64
./vllm-router-v0.1.15-linux-arm64 \
    --host 0.0.0.0 --port 30000 \
    --worker-urls http://192.168.1.11:8000 http://192.168.1.12:8000
```

静态二进制在 Ubuntu/openEuler 上裸跑无任何库依赖，唯一要求是系统装有 CA 证书包（`ca-certificates`，两个发行版默认都已安装，用于出站 HTTPS）。

### 7.4 镜像验证（强烈建议在构建机上完成）

```bash
# 1. 版本与帮助正常
docker run --rm vllm-router:v0.1.15-amd64 --version
docker run --rm vllm-router:v0.1.15-amd64 --help

# 2. 冒烟测试：--service-discovery + --selector 让空 worker 列表通过校验，服务立即监听
#    （非 K8s 环境下服务发现启动失败仅记 warn 日志，不影响服务，见 server.rs:1000；
#     selector 为 K8s label 选择器格式 key=value，冒烟时用任意占位值即可）
docker run -d --name vr-smoke -p 30000:30000 vllm-router:v0.1.15-amd64 \
    --host 0.0.0.0 --port 30000 --service-discovery --selector app=vllm
sleep 2
curl -s -w "\n%{http_code}\n" http://127.0.0.1:30000/liveness   # 预期 200 "OK"
docker rm -f vr-smoke
```

> **三个坑**：
>
> 1. **冒烟测试不要用假地址 `--worker-urls http://127.0.0.1:1`**。router 启动时会先等所有列出的
>    worker 健康才开始监听端口（最长 `--worker-startup-timeout-secs`，默认 600s，见
>    `src/routers/http/router.rs:60`），假 worker 会让端口长时间不开，curl 报 connection refused（`000`）。
> 2. **启动校验是连环的**：不带 `--worker-urls` 报 `Regular mode requires --worker-urls`；
>    带 `--service-discovery` 但没有 `--selector` 报 `requires a non-empty selector`。
>    所以冒烟测试的最小参数集是 `--service-discovery --selector app=vllm`。
> 3. **端点语义不同**：
>    - `/liveness`：进程活着就返回 200，与后端无关 —— 冒烟测试/容器健康检查用这个；
>    - `/health`：**所有** worker 都健康才 200，否则 503 `Unhealthy servers: [...]` —— 部署后检查整体集群状态用这个；
>    - `/readiness`：有健康后端才 200。
>    本测试 0 个 worker：`/liveness` 200，`/health` 503 —— 都是预期行为。

arm64 侧把镜像 tag 换成 `v0.1.15-arm64` 重复一遍即可。

---

## 8. 离线交付（默认方式）

每个架构的完整交付物 = **镜像 tar 包 + 静态二进制 + sha256 校验文件**（部署用镜像，二进制作为裸机应急/调试备用，一并提交）：

| 架构 | 交付文件（3 个） |
| --- | --- |
| amd64 | `vllm-router_v0.1.15_amd64.tar`、`vllm-router-v0.1.15-linux-amd64`、`vllm-router-v0.1.15-linux-amd64.sha256` |
| arm64 | `vllm-router_v0.1.15_arm64.tar`、`vllm-router-v0.1.15-linux-arm64`、`vllm-router-v0.1.15-linux-arm64.sha256` |

镜像导出（构建机上）：

```bash
# amd64（x86 构建机上）
docker save -o vllm-router_v0.1.15_amd64.tar vllm-router:v0.1.15-amd64

# arm64（鲲鹏机上）
docker save -o vllm-router_v0.1.15_arm64.tar vllm-router:v0.1.15-arm64

# 可选：压缩传输（load 时可直接读 .gz 无需解压，取决于 docker 版本；保险起见传输后解压）
gzip -k vllm-router_v0.1.15_arm64.tar
```

二进制文件（7.3 节产物）随镜像 tar 包一起通过 scp / U 盘拷贝到目标机器，然后：

```bash
docker load -i vllm-router_v0.1.15_arm64.tar
docker images | grep vllm-router

# 可选：校验二进制交付物完整性
sha256sum -c vllm-router-v0.1.15-linux-arm64.sha256
```

若内部镜像仓库已确定（替换 `<registry>` 为实际地址）：

```bash
docker tag vllm-router:v0.1.15-arm64 <registry>/vllm-router:v0.1.15-arm64
docker push <registry>/vllm-router:v0.1.15-arm64
# 目标机器: docker pull <registry>/vllm-router:v0.1.15-arm64
```

---

## 9. 部署运行

### 9.1 docker run（常规模式）

```bash
docker run -d --name vllm-router \
  --restart unless-stopped \
  -p 30000:30000 -p 29000:29000 \
  --add-host=host.docker.internal:host-gateway \
  vllm-router:v0.1.15-amd64 \
  --host 0.0.0.0 --port 30000 \
  --worker-urls http://host.docker.internal:8000 \
  --policy cache_aware \
  --prometheus-host 0.0.0.0
```

- `--worker-urls`：vLLM 后端地址列表（后端跑在别的容器/机器上，填**从 router 容器可达**的地址）。**同机部署时不要写 `127.0.0.1`**——容器内的 127.0.0.1
是容器自己，不是宿主机；router 启动时会因等不到 worker 健康而一直不监听端口（表现为 curl 连接被重置，600s 后超时）。同机后端的三种写法：

  ```bash
  # 写法1（推荐）：显式把宿主机暴露给容器
  docker run -d --name vllm-router ... \
    --add-host=host.docker.internal:host-gateway \
    vllm-router:v0.1.15-amd64 \
    --worker-urls http://host.docker.internal:8000 ...
  
  # 写法2：直接填宿主机局域网 IP
  --worker-urls http://192.168.x.x:8000
  
  # 写法3：host 网络模式（此时 127.0.0.1 真的指宿主机；不能再用 -p 映射）
  docker run -d --name vllm-router --network host ... --worker-urls http://127.0.0.1:8000 ...
  ```

- `--policy`：`round_robin` / `cache_aware`（默认，推荐）/ `power_of_two` / `consistent_hash` / `rendezvous_hash` / `random`。
- `--prometheus-host 0.0.0.0`：让指标端口可被外部抓取（Prometheus 访问 `http://<机器>:29000/metrics`）。
- 验证：`curl http://<机器>:30000/health`、`curl http://<机器>:30000/get_server_info`（查看已注册后端与配置）、`curl
http://<机器>:30000/list_workers`（后端列表）。

### 9.2 docker compose（推荐）

#### (1) 安装 docker compose 插件（目标机器，一次性操作）

```bash
# Ubuntu 22.04 / 24.04
apt install -y docker-compose-plugin

# openEuler 24.03（若 dnf 源中无此包，用下面的离线方式）
dnf install -y docker-compose-plugin

# 离线安装（目标机器不通外网时，在构建机下载后拷过去）：
#   https://github.com/docker/compose/releases 下载对应架构文件
#   x86: docker-compose-linux-x86_64    arm64: docker-compose-linux-aarch64
mkdir -p /usr/local/lib/docker/cli-plugins
install -m 755 docker-compose-linux-aarch64 /usr/local/lib/docker/cli-plugins/docker-compose

# 验证（两种方式都应输出版本号）
docker compose version
```

> 旧版独立 `docker-compose`（带横杠）同样兼容本文的 yml 文件，把下文所有 `docker compose` 命令换成 `docker-compose` 即可。

#### (2) 创建部署目录和 docker-compose.yml

```bash
mkdir -p /opt/vllm-router
cd /opt/vllm-router
vi docker-compose.yml      # 粘贴以下内容并保存
```

```yaml
services:
  vllm-router:
    image: vllm-router:v0.1.15-arm64      # x86 机器上改为 :v0.1.15-amd64
    container_name: vllm-router
    restart: unless-stopped
    ports:
      - "30000:30000"
      - "29000:29000"
    command: >
      --host 0.0.0.0
      --port 30000
      --worker-urls http://192.168.1.11:8000 http://192.168.1.12:8000
      --policy cache_aware
      --prometheus-host 0.0.0.0
    healthcheck:
      # 用 /liveness（进程存活，恒 200）而非 /health（所有 worker 健康才 200）。
      # 否则任一后端故障时，路由器本身会被误标为 unhealthy。
      test: ["CMD-SHELL", "wget -qO- http://127.0.0.1:30000/liveness || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"
```

注意：

- `image` 必须与 `docker load` 后的镜像 tag 完全一致（arm64 机器 `-arm64`，x86 机器 `-amd64`）；
- `--worker-urls` 改成实际后端地址（不要写 127.0.0.1，见 9.1 的说明）。

**两个端口的分工**：

| 端口 | 参数 | 用途 | 谁访问 |
| --- | --- | --- | --- |
| 30000 | `--port`（默认 30000） | 业务端口：推理 API（`/v1/chat/completions` 等）+ 健康/就绪 + 集群管理（`/list_workers` 等） | 业务程序、运维 |
| 29000 | `--prometheus-port`（默认 29000） | 指标端口：仅 `GET /metrics`，输出 Prometheus 格式运行指标 | Prometheus/Grafana 等监控系统 |

注意 metrics 服务**默认只绑 `127.0.0.1`**，容器内不监听外部流量——所以 command 里必须带
`--prometheus-host 0.0.0.0`，否则端口映射了也不通（且不报错，只表现为连接拒绝）。没有监控系统时该端口可不映射。

#### (3) 启动与日常操作（都在 /opt/vllm-router 目录下执行）

```bash
docker compose up -d       # 启动（后台运行）
docker compose ps          # 查看运行状态（STATUS 显示 healthy 表示健康检查通过）
docker compose logs -f     # 跟踪日志（Ctrl+C 退出，不影响服务）
docker compose restart     # 重启
docker compose down        # 停止并删除容器（镜像保留，随时可再 up）
```

#### (4) 修改配置（更换后端列表 / 策略 / 端口等）

编辑 `docker-compose.yml`（主要是 `command` 段）后重新执行：

```bash
docker compose up -d       # compose 检测到配置变化，自动重建容器
```

#### (5) 验证

```bash
curl -s -w "\n%{http_code}\n" http://127.0.0.1:30000/liveness   # 200 "OK"：路由器进程正常
curl -s http://127.0.0.1:30000/health                            # 所有后端健康时 200；有后端掉线时 503 + Unhealthy servers 列表（用于排查是哪个后端）
curl http://127.0.0.1:30000/list_workers                         # 查看已注册后端
```

### 9.3 PD 分离模式（可选）

Prefill-Decode 分离部署时使用服务发现模式（worker 启动后自动向 router 注册，不需要 `--worker-urls`）：

```bash
docker run -d --name vllm-router \
  --restart unless-stopped \
  -p 30000:30000 -p 30001:30001 -p 29000:29000 \
  vllm-router:v0.1.15-amd64 \
  --host 0.0.0.0 --port 30000 \
  --vllm-pd-disaggregation \
  --vllm-discovery-address 0.0.0.0:30001 \
  --policy consistent_hash \
  --prometheus-host 0.0.0.0
```

**关于 30001 端口**：PD 模式专用（普通模式不需要、不映射）。它跑的是 ZMQ 协议而非 HTTP，作用是 worker **注册通道**——PD 模式下 router 不用 `--worker-urls` 写死后端，而是由各
prefill/decode 后端启动时主动连到 `router_ip:30001` 登记自己的地址（源码 `src/main.rs:117-120`）。因此后端 vLLM 需在其 `--kv-transfer-config` 中配置
`router_ip:30001` 指向本注册口，具体见 vllm-ascend PD 文档。

### 9.4 Kubernetes

仓库自带 K8s 模板：`scripts/k8s/`（含 Deployment/Service/Helm），将其中镜像替换为本文档的镜像名即可。

---

## 10. 后端接入配置与测试

两种后端都支持，但**一个 router 实例只能选一种**（`--backend` 参数决定）。源码依据：`src/main.rs:391-395`（按 backend 切换
RoutingMode）、`src/config/validation.rs:108-121`（OpenAI 模式强制恰好 1 个 URL）、`src/main.rs:679-689`（trtllm/anthropic 未实现警告）。

### 10.1 模式一：vLLM 自建后端（默认，多 worker 负载均衡）

**启动配置**（docker run 形式，compose 同理改 command 段）：

```bash
docker run -d --name vllm-router \
  --restart unless-stopped \
  -p 30000:30000 -p 30001:30001 -p 29000:29000 \
  vllm-router:v0.1.15-amd64 \
  --host 0.0.0.0 --port 30000 \
  --vllm-pd-disaggregation \
  --vllm-discovery-address 0.0.0.0:30001 \
  --policy consistent_hash \
  --prometheus-host 0.0.0.0
```

前提：各 worker 是**已启动的 vLLM 服务**（vllm-ascend 也行，只要暴露 OpenAI 兼容接口和 `/health`）。注意启动时 router 会阻塞等**所有** worker 健康才监听端口（默认
600s 超时，`--worker-startup-timeout-secs` 可调），所以先起 worker 再起 router。

**测试请求**（对着 router 发，而不是对 worker 发）：

```bash
# 1. 确认 worker 都注册且健康
curl http://127.0.0.1:30000/list_workers
curl http://127.0.0.1:30000/health          # 200 = 全部健康

# 2. 模型列表（透传自 worker）
curl http://127.0.0.1:30000/v1/models

# 3. 对话请求（标准 OpenAI 格式，model 填 vLLM 启动时的 --served-model-name）
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct-FP8",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32
  }'

# 4. 流式测试
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "讲个笑话"}], "max_tokens": 64, "stream": true}'

# 5. 验证负载均衡：连发 4 次普通补全，看 router 日志里目标 worker 是否轮换
for i in 1 2 3 4; do
  curl -s http://127.0.0.1:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen2.5-7B-Instruct", "prompt": "hello", "max_tokens": 4}' > /dev/null
done
docker logs vllm-router 2>&1 | grep -i "routed\|selected" | tail -8
```

### 10.2 模式二：外部 OpenAI 兼容 API（单上游代理）

适用于 DeepSeek / 智谱 / 通义 / OpenAI 等任何 OpenAI 兼容的服务商 API。

**启动配置**：

```bash
docker run -d --name vllm-router \
  --restart unless-stopped \
  -p 30010:30000 -p 29010:29000 \
  vllm-router:v0.1.15-amd64 \
  --host 0.0.0.0 --port 30000 \
  --backend openai \
  --worker-urls https://yibuapi.com
```

三条规则（都是源码硬约束）：

1. `--worker-urls` **只能给 1 个**上游 base URL（到 `/v1` 为止，不带末尾斜杠），给多个直接启动报错；
2. **API Key 不配在 router 上**：router 把客户端请求的 `Authorization` 头原样透传上游（`openai_router.rs:274-279`），密钥由每次请求携带；
3. 此模式不做负载均衡，价值是统一入口、熔断、指标。

**此模式下哪些特性失效/保留**：负载均衡算法、PD 分离、动态增删后端、K8s 服务发现全部失效（单上游无从调度，`add_worker` 直接报错）。仍生效的：熔断器、健康检查（探上游
`/v1/models`）、Prometheus 指标、OTel 追踪、SSE 流式转发、**参数清洗**（转发前自动删掉 `top_k`/`min_p`/`regex` 等十几个 vLLM 专有字段，避免外部 API 返回 400，见
`openai_router.rs:247-269`）。定位：连 vLLM 用的是调度能力，连外部 API 用的是网关能力。

**测试请求**（把正常直连服务商的命令里的域名换成 router 地址即可）：

```bash
# 先把服务商的 API Key 导入环境变量（不要把真实 Key 写进任何文档/代码）
export OPENAI_API_KEY="sk-xxxx你的真实Key"

# 1. 模型列表（携带服务商 API Key）
curl http://127.0.0.1:30010/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 2. 对话请求
curl http://127.0.0.1:30010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32
  }'
```

### 10.3 两种后端能否混用？

**不能在同一个实例里混。** `--backend` 是全局开关，vLLM 模式按多 worker 负载均衡设计，OpenAI 模式按单上游代理设计，源码层面互斥。如果业务上确实需要"自建 vLLM + 外部
API"同时服务，**在同一台服务器上起两个 router 实例**（多实例是唯一正解，不是权宜之计）：

```bash
# 实例A：自建 vLLM 后端，占宿主机 30000
docker run -d --name vllm-router \
  --restart unless-stopped \
  -p 30000:30000 -p 30001:30001 -p 29000:29000 \
  vllm-router:v0.1.15-amd64 \
  --host 0.0.0.0 --port 30000 \
  --vllm-pd-disaggregation \
  --vllm-discovery-address 0.0.0.0:30001 \
  --policy consistent_hash \
  --prometheus-host 0.0.0.0

# 实例B：外部 API 代理，占宿主机 30010（容器内仍是 30000，互不干扰）
docker run -d --name router-external \
  --restart unless-stopped \
  -p 30010:30000 -p 29010:29000 \
  vllm-router:v0.1.15-amd64 \
  --host 0.0.0.0 --port 30000 \
  --backend openai \
  --worker-urls https://yibuapi.com \
  --prometheus-host 0.0.0.0
```

多实例只有两个约束：**容器名不重复**、**宿主机端口不重复**（`-p` 左侧；容器内部端口可以相同，每个容器网络栈独立）。镜像共用一份不重复占磁盘，每实例仅一个静态二进制进程（内存几十 MB、空闲 CPU 近零、不碰
NPU）。compose 用法则在同一个 yml 里写两个 service（名字不同、ports 不同）。

客户端按需选入口：自建集群走 `:30000`，外部 API 走 `:30010`（携带服务商 Key）；需要统一入口时在前面再加一层 Nginx/网关按路径或模型名分流。

---

## 11. 常用运维命令

| 操作 | 命令 |
| --- | --- |
| 进程存活（恒 200） | `curl http://<host>:30000/liveness` |
| 集群健康（所有后端健康才 200，否则 503 并列出病后端） | `curl http://<host>:30000/health` |
| 就绪状态（有健康后端才 200） | `curl http://<host>:30000/readiness` |
| 后端/路由信息 | `curl http://<host>:30000/get_server_info` |
| 后端列表 | `curl http://<host>:30000/list_workers` |
| 各后端负载 | `curl http://<host>:30000/get_loads` |
| 动态摘除后端 | `curl -X POST "http://<host>:30000/remove_worker?url=http://192.168.1.11:8000"` |
| 动态添加后端 | `curl -X POST "http://<host>:30000/add_worker?url=http://192.168.1.13:8000"` |
| Prometheus 指标 | `curl http://<host>:29000/metrics` |
| 查看日志 | `docker logs -f vllm-router` |
| 进入容器排障 | `docker exec -it vllm-router sh` |
| 更新后端列表 | 重启容器并修改 `--worker-urls`（静态列表模式） |

---

## 12. FAQ

**1. 为什么不用担心目标机器的 glibc 版本？**
二进制是 musl 全静态（`file` 输出 `statically linked`），不引用任何动态库；容器运行时镜像自带全部依赖。Ubuntu 与 openEuler 的差异（glibc
2.35/2.38/2.39）完全不影响。裸机直接执行该二进制同样可行。

**2. 昇腾机器上部署 router 需要什么特殊配置？**
不需要。Router 只做 HTTP 转发，不碰 NPU。不需要 `--device /dev/davinci*`、不需要挂载 CANN、不需要驱动版本匹配。昇腾相关的环境只影响后端 vLLM 容器。

**3. 镜像里为什么没有 Python？**
v0.1.15 的功能全部由 Rust 二进制提供，Python launcher（`pip install vllm-router`）只是另一种启动方式。去掉后镜像从 ~1GB 降到 ~30MB，功能无损。

**4. 构建报错找不到 python？**
Dockerfile.static 已在构建容器安装 `python3`（pyo3 构建脚本探测用）。若自行修改过 Dockerfile，确保保留 `python3` 包。

**5. zmq/libzmq 是怎么处理的？**
构建容器里故意**不装**系统 zeromq，`zmq-sys` 检测不到系统库时会自动用内置源码编译并静态链入，这正是我们要的效果（不需要目标机器有任何 libzmq）。

**6. 构建很慢怎么办？**
全量编译（含 libzmq、ring 等本地 C/C++ 编译）一次性成本约 10~20 分钟。重复构建时 `Cargo.toml`/`Cargo.lock` 未变则依赖层命中缓存，只重编译 `src/`。如需进一步加速，可在
`docker build` 命令加 BuildKit 缓存挂载：

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.static \
  --mount type=cache,target=/app/target ... # 或使用 buildx --build-arg
```

（简单场景可忽略，仅改源码时的增量构建已足够快。）

**7. 构建机访问 crates.io / Docker Hub 慢或不通？**

- Docker Hub：为 Docker daemon 配置国内 registry mirror（`/etc/docker/daemon.json` 的 `registry-mirrors`）。
- crates.io（典型报错：`Stream error in the HTTP/2 framing layer` / `stream ... reset by server`，国内直连 crates.io 被 HTTP/2
重置）：**Dockerfile.static 已内置 rsproxy 国内镜像配置**（`[source.crates-io] replace-with` 那段），无需再手动处理。若镜像源不可用想换回官方源，删除 Dockerfile
中写 `$CARGO_HOME/config.toml` 的那段 RUN 即可；备选镜像：`sparse+https://mirrors.ustc.edu.cn/crates.io-index/`。
- Rust 工具链下载慢（仅当自行改动 Dockerfile 安装 rustup 时才涉及）：

  ```dockerfile
  ENV RUSTUP_DIST_SERVER=https://rsproxy.cn \
      RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
  ```

**8. 目标机器 docker load 后镜像名对不上？**
`docker save` 保存的是完整镜像名（含 tag），`load` 后名称不变。用 `docker images | grep vllm-router` 确认。

**9. 想固定 Rust 编译器版本以获得可复现构建？**
把 Dockerfile 中 `FROM rust:1-alpine` 固定为具体版本，例如 `rust:1.90-alpine`（以 Docker Hub 实际存在的 tag 为准）。

**10. 端口规划冲突？**
服务端口默认 30000（`--port`）、指标端口 29000（`--prometheus-port`）、PD 注册端口
30001（`--vllm-discovery-address`），均可通过参数修改；宿主机侧映射端口按机房规范调整 `-p` 左侧即可。
