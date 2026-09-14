# vLLM Semantic Router — Linux 编译与部署手册

> **文档版本**：v1.0　｜　**对应代码**：SR tag `v0.3.0` (<https://github.com/vllm-project/semantic-router/releases/tag/v0.3.0>)
>
> **适用读者**：负责在 Linux 服务器（x86 Ubuntu 22.04/24.04 或 arm64 openEuler 内核 6.6 / glibc ≤ 2.38）上构建、交付并部署自编译 SR 的工程师。
>
> **交付形态**：**Docker 镜像**（唯一部署形态，目标服务器零 Python/Go/Rust 工具链依赖）+ 配置模板 + 本手册。
>
> **两种架构同时支持**：本手册所有构建命令均提供 amd64 与 arm64 双路径。

---

## 0. 先看这 3 条

1. **构建产物的架构由 `--platform linux/<arch>` 决定**，不是 `--build-arg TARGETARCH=arm64`。后者只驱动 Dockerfile 内的交叉编译逻辑，**不会改变镜像的
Architecture 元数据**——作者亲测踩过。arm64 镜像一律带 `--platform linux/arm64` 构建。
2. **需要自己构建的镜像只有 3 个**（加上 CLI 编排器共 4 个）：`vllm-sr-router`（改 SR
代码就重建它）、`vllm-sr-dashboard`、`vllm-sr-sim`（可禁用不构建）、`vllm-sr-cli`（一般不改代码，构建一次复用）。其余 6
个容器（Envoy/Redis/Postgres/Jaeger/Prometheus/Grafana）全部使用官方现成镜像，版本号已从源码逐一核实。
3. **CLI 推荐容器化**（`vllm-sr-cli`）：容器化后交付物 100% 是镜像，目标服务器不需要安装 Python（openEuler 自带 Python 常低于 CLI 要求的 3.10）。CLI 工作靠 DooD
机制（挂载宿主机 docker.sock），不是嵌套 Docker。

---

## 1. 交付物全景

`vllm-sr serve` 拉起的 9 个容器组（与 `docker ps` 看到的一一对应）：

```text
                                    ┌──────────────────────────────────────────────┐
                                    │  宿主机 docker engine                          │
 Agent / 业务方                      │                                              │
┌──────────┐  OpenAI 格式            │  ┌─────────────┐    ext_proc(gRPC)           │
│ SDK/curl │──────────── :8888 ─────┼─▶│ ① envoy     │──────────────┐              │
└──────────┘                        │  │ (官方镜像)   │◀─────────────┘              │
                                    │  └─────────────┘       ┌──────────────┐      │
 浏览器                              │                        │ ② router     │      │
┌──────────┐  http://宿主:8700 ─────┼─▶ ┌─────────────┐       │ (自建镜像★)   │      │
│ 管理员    │                        │  │ ③ dashboard │◀──────│ Go+Rust 大脑  │      │
└──────────┘                        │  │ (自建镜像★)  │  API  └──────┬───────┘      │
                                    │  └─────────────┘              │               │
                                    │  ┌─────────────┐              │ ④ sim        │
                                    │  │ ⑤ redis     │◀─ 存储 ──────│ (自建★,可关)  │
                                    │  │ (官方镜像)   │              │               │
                                    │  └─────────────┘              ▼               │
                                    │  ┌─────────────┐      ┌──────────────┐        │
                                    │  │ ⑥ postgres  │      │ 后端模型池     │        │
                                    │  │ (官方镜像)   │      │ vLLM / VR /  │        │
                                    │  └─────────────┘      │ 外部 API      │        │
                                    │  ⑦ jaeger ⑧ prometheus│ (§6 配置)    │        │
                                    │  ⑨ grafana (官方镜像) └──────────────┘        │
                                    └──────────────────────────────────────────────┘
```

**容器互连原理**：CLI 先创建共享 docker bridge 网络（默认名 `vllm-sr-network`），然后把所有容器接入该网络。同一 bridge 网络内可用**容器名作为域名**互相访问（docker 内置
DNS），比如 envoy → `vllm-sr-router-container:50051`，router → `vllm-sr-redis:6379`。不需要手写任何 IP。

### 1.1 镜像清单总表

| # | 容器名（默认） | 来源 | 镜像名:版本 | 用途 | 改代码后要重建 |
| --- | --- | --- | --- | --- | --- |
| ① | vllm-sr-envoy-container | 官方 | `envoyproxy/envoy:v1.34-latest` | 流量入口网关 / ext_proc | 否 |
| ② | vllm-sr-router-container | 自建★ | `vllm-sr-router:v0.3.0-{amd64\|arm64}` | SR 核心（Go + Rust FFI） | **是** |
| ③ | vllm-sr-dashboard-container | 自建★ | `vllm-sr-dashboard:v0.3.0-{amd64\|arm64}` | Web 控制台（TS + Go + Py） | **是** |
| ④ | vllm-sr-sim-container | 自建★（可选） | `vllm-sr-sim:v0.3.0-{amd64\|arm64}` | 车队模拟器（性能实验用） | 视情况 |
| ⑤ | vllm-sr-redis | 官方 | `redis:7-alpine` | response_api / 会话状态 | 否 |
| ⑥ | vllm-sr-postgres | 官方 | `postgres:16-alpine` | 路由审计 / 记忆元数据 | 否 |
| ⑦ | vllm-sr-jaeger | 官方 | `jaegertracing/all-in-one:latest` | 链路追踪 | 否 |
| ⑧ | vllm-sr-prometheus | 官方 | `prom/prometheus:v2.53.0` | 指标采集 | 否 |
| ⑨ | vllm-sr-grafana | 官方 | `grafana/grafana:11.5.1` | 指标可视化 | 否 |
| — | CLI 编排器（无容器常驻） | 自建★ | `vllm-sr-cli:0.3.0-{amd64\|arm64}` | 执行 `vllm-sr serve/stop/status/logs` | 可能需要 |

> 版本号来源：`src/vllm-sr/cli/docker_services.py`、`cli/consts.py` 源码硬编码，与官方 v0.3.0 行为一致。
>
> **milvus（向量库，端口 19530）**：仅当 config 中 `stores.semantic_cache.backend_type: milvus`
> 且连接指向托管容器时才按需启用（语义缓存后端）。默认关闭（性能测试推荐关闭，保证口径纯净）。启用需追加官方镜像 `milvusdb/milvus:v2.3.3`。

### 1.2 名词速查

| 名词 | 一句话解释 |
| --- | --- |
| 观测栈 | Jaeger + Prometheus + Grafana：分别回答“卡在哪一步”“QPS/延迟多少”“画成图给人看” |
| ext_proc | Envoy 的扩展处理器机制：envoy 收到请求后通过 gRPC 调 router（:50051）做路由决策 |
| DooD | Docker-out-of-Docker：容器内挂载宿主机 `/var/run/docker.sock`，CLI 直接操作**宿主机** Docker，不是嵌套容器 |
| 交叉编译 | 在 x86 机器上编译 arm64 产物，比 QEMU 模拟全流程快一到两个数量级 |
| 语义缓存 | 按“语义相似度”而非精确匹配命中缓存（向量库做后端），命中则不请求大模型 |

### 1.3 glibc / 内核兼容性结论

Docker 镜像自带完整用户态库（含 glibc），**镜像内 glibc 与宿主机无关**，宿主机只要满足：内核版本 ≥ glibc 要求的最低值（很低）+ 装有 Docker。

| 自建镜像运行层 | 基础镜像 | 内置 glibc | 与约束（≤2.38，内核 6.6）对照 |
| --- | --- | --- | --- |
| vllm-sr-router | debian:bookworm-slim | 2.36 | ✅ |
| vllm-sr-dashboard | python:3.11-slim-bookworm | 2.36 | ✅ |
| vllm-sr-sim / cli | python:3.x-slim-bookworm | 2.36 | ✅ |
| 官方 6 镜像 | 各自基础镜像 | 皆 ≤ 2.38 | ✅ |

仓库已把 Rust/Go 编译阶段钉在 Debian bullseye（glibc 2.31）以保证产物落在旧基线。**结论：容器化部署完全不用操心 glibc 兼容性。**

---

## 2. 构建环境准备（Linux 本机）

### 2.1 构建机选型（两种策略）

| 策略 | 做法 | 适用 |
| --- | --- | --- |
| **A. x86 一机搞定双架构**（推荐） | 在任意一台有网的 x86 Linux 构建机（Ubuntu 22.04/24.04 均可）上：amd64 原生、arm64 用 `--platform linux/arm64`（router 走交叉编译最快；dashboard/sim/CLI 走 QEMU） | 有 x86 构建机、无鲲鹏机时 |
| **B. 分机构建** | x86 机产 amd64 镜像 + 鲲鹏（arm64）机产 arm64 镜像 | 有现成鲲鹏机、dashboard 构建想省时间 |

本手册按**策略 A（一机双架构）**写，策略 B 对应命令只需要去掉 `--platform` 参数即可。

### 2.2 软件依赖（只需 Docker + 可选 skopeo）

**构建机上不需要安装 Go / Rust / Node / Python 工具链**（所有编译都在 `docker build` 的构建容器内部进行，对应版本的构建镜像由 Dockerfile 自动拉取），只需要 Docker：

```bash
# Ubuntu 22.04/24.04（官方源 + 便捷脚本可选）
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# 重新登录一次 shell，然后验证
docker --version            # 期望 27.x+
docker info --format '{{.OSType}}/{{.Architecture}}'   # 期望 linux/x86_64
df -h /var/lib/docker       # 剩余 ≥ 60GB（构建缓存 + 双架构镜像 ≈ 40~50GB）

# QEMU（arm64 用 --platform 构建需要，docker-ce 自带 binfmt-support 包通常已装好）
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx ls            # 应看到 platforms: 行里包含 linux/arm64
```

openEuler（若将来策略 B 在鲲鹏机上构建）：用 `sudo dnf install -y docker-ce`（加 Docker 官方源）或系统自带 docker，QEMU 在 arm64 本机构建时不需要。

### 2.3 获取源码（v0.3.0 tag）

```bash
cd ~
git clone --depth 1 --branch v0.3.0 https://github.com/vllm-project/semantic-router.git sr-src
cd sr-src
# 校验
git describe --tags     # 期望 v0.3.0

# （可选）若有本地改动，建议基于 tag 开分支
git checkout -b v0.3.0-custom v0.3.0
```

后文所有 `docker build` 命令的工作目录都是**仓库根目录**（`~/sr-src` 或你自己的路径）。

### 2.4 网络（国内/内网环境）

构建阶段需要访问：Docker Hub / ghcr.io（基础镜像）、Go module 代理、crates.io（Rust）、PyPI、npm。内网/国内通常需要走公司代理或自建镜像源。

```bash
# 如果有 HTTP(S) 代理
export http_proxy=http://<proxy-host>:<port> https_proxy=http://<proxy-host>:<port>
# 验证（期望 HTTP/2 200；空行 = 代理没通）
curl -sI https://proxy.golang.org --max-time 5 | head -1
```

让构建容器内部的四类包管理器（go/cargo/pip/npm）也走代理，二选一：

- **方式 1：daemon 级（推荐）**：在 `/etc/docker/daemon.json` 里写入 `"proxies":{"http-proxy":"...","https-proxy":"..."}`，`sudo
systemctl reload docker`。之后镜像拉取 + 构建容器内代理自动生效，构建命令无需附加参数。
- **方式 2：命令级**：每条 `docker build` 命令追加以下参数：

```bash
--build-arg HTTP_PROXY="$http_proxy" --build-arg HTTPS_PROXY="$https_proxy"
```

> `--network=host` 只共享网络栈，不传递环境变量，所以两种方式必须选一种（否则依赖下载会卡在网络层面）。

### 2.5 skopeo（用于官方镜像按架构打包，建议装）

```bash
# Ubuntu
sudo apt install -y skopeo
# openEuler
sudo dnf install -y skopeo

# 有代理就确保 https_proxy 有值（skopeo 读标准代理变量）
skopeo --version
```

---

## 3. 构建 4 个自建镜像（amd64 + arm64）

### 3.1 router 镜像（核心，最重要）

构建原理：多阶段 Dockerfile 一次 `docker build` 内部完成：

```text
阶段1 rust-builder      编译 3 个 Rust 库(candle/ml/nlp binding) → .so
阶段1a/1b ml/nlp        同上，另外两个库
阶段2 go-builder        Rust .so + Go 源码 → CGO 链接 → router 二进制
阶段3 运行时镜像        debian:bookworm-slim + router + 3×.so + python3(轻量)
```

**amd64 构建**：

```bash
cd ~/sr-src
docker build --network=host \
  --platform linux/amd64 \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/vllm-sr/Dockerfile \
  -t vllm-sr-router:v0.3.0-amd64 .
```

**arm64 构建（x86 机上交叉编译）**：

```bash
docker build --network=host \
  --platform linux/arm64 \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/vllm-sr/Dockerfile \
  -t vllm-sr-router:v0.3.0-arm64 .
```

关键点：

- `--platform linux/arm64` 下，重的 Rust/Go 编译阶段仍被钉在 `--platform=$BUILDPLATFORM`（amd64 原生）跑，只有最终运行阶段的 apt/pip 在 QEMU
模拟下运行（小头）。
- `GIT_SSL_NO_VERIFY=1`：公司代理自签证书兜底。
- **不要传 `--build-arg TARGETARCH/BUILDPLATFORM`**：BuildKit 会自动推导，Dockerfile 开头注释明确不要覆盖它们。

**结果核查（必须做）**：

```bash
docker image inspect vllm-sr-router:v0.3.0-amd64 --format '{{.Architecture}}'   # 期望 amd64
docker image inspect vllm-sr-router:v0.3.0-arm64 --format '{{.Architecture}}'   # 期望 arm64
docker images | grep vllm-sr-router
# 尺寸量级：~400-600MB / arch
```

首次构建 30~60 分钟（Rust 依赖是大头）；有层缓存后改 Go 代码几分钟。

### 3.2 dashboard 镜像（amd64 原生；arm64 双方案）

**amd64 构建**：

```bash
docker build --network=host \
  -f dashboard/backend/Dockerfile \
  -t vllm-sr-dashboard:v0.3.0-amd64 .
```

**arm64 构建**：dashboard 的 backend-builder 阶段**没有安装交叉工具链**，所以只有两条可行路径：

| 路径 | 命令 | 耗时 |
| --- | --- | --- |
| **A. x86 + QEMU**（一机双架构时用） | `docker build --network=host --platform linux/arm64 -f dashboard/backend/Dockerfile -t vllm-sr-dashboard:v0.3.0-arm64 .` | 40~90 分钟（wasm + npm×2 + Go CGO + pip torch 全在 QEMU 下） |
| **B. 鲲鹏机原生构建**（有条件时用，最快） | 源码 tar 打包传到鲲鹏机，解压后同命令去掉 `--platform` 即可 | 15~30 分钟 |

路径 A 偶发 npm 段错误崩溃：**直接重跑同一条命令**，Docker 层缓存从断点续上。

> dashboard 的 Dockerfile 没有 `GIT_SSL_NO_VERIFY` 参数，不传。
> 改 dashboard 代码后的重建：同样命令重跑，层缓存跳过未变部分（改前端只重跑 npm build，几分钟）。

### 3.3 sim 镜像（可选，精简交付可以不构建）

纯 Python，无编译环节，双架构都轻松。

```bash
# amd64
docker build --network=host \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/fleet-sim/Dockerfile -t vllm-sr-sim:v0.3.0-amd64 .

# arm64（x86 + QEMU；鲲鹏机原生同命令去掉 --platform）
docker build --network=host \
  --platform linux/arm64 \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/fleet-sim/Dockerfile -t vllm-sr-sim:v0.3.0-arm64 .
```

不构建 sim：部署时传 `-e VLLM_SR_SIM_ENABLED=false`。

### 3.4 CLI 编排器镜像（交付便捷性的关键）

**为什么容器化 CLI**：`vllm-sr serve` 是 Python 包（≥3.10），且需要能执行 `docker` 命令（源码确认：CLI 通过 subprocess 调 docker CLI）。容器化后：

- 目标服务器零 Python 依赖（正好绕开 openEuler 自带 Python < 3.10）
- 交付物统一为镜像
- CLI 一般不改代码，构建一次即可复用

**新建 CLI Dockerfile**（保存为仓库根目录下 `src/vllm-sr/cli/docker/cli.Dockerfile`）：

```dockerfile
# vllm-sr CLI 编排器镜像：pip 安装 CLI + 借用 docker CLI 二进制
# 用途：容器内运行 `vllm-sr serve`，通过挂载宿主机 docker.sock 编排整栈
FROM python:3.11-slim-bookworm

# 只借 docker CLI 可执行文件（不需要完整 docker 引擎）
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

# 安装 vllm-sr CLI（纯 Python 编排器，不含模型文件）
RUN pip install --no-cache-dir vllm-sr==0.3.0

WORKDIR /workspace
ENTRYPOINT ["vllm-sr"]
```

> 如果你改了 CLI 源码（`src/vllm-sr/cli/**/*`），把 `RUN pip install vllm-sr==0.3.0` 换成 `COPY src/vllm-sr /tmp/vllm-sr && pip
> install --no-cache-dir /tmp/vllm-sr` 即可。

**构建（双架构，QEMU 也很快）**：

```bash
# amd64
docker build --network=host \
  -f src/vllm-sr/cli/docker/cli.Dockerfile -t vllm-sr-cli:0.3.0-amd64 .

# arm64
docker build --network=host \
  --platform linux/arm64 \
  -f src/vllm-sr/cli/docker/cli.Dockerfile -t vllm-sr-cli:0.3.0-arm64 .
```

**快速验证**：

```bash
docker run --rm vllm-sr-cli:0.3.0-amd64 --version        # 期望 vllm-sr version: 0.3.0
docker run --rm vllm-sr-cli:0.3.0-amd64 serve --help     # 打印帮助即 OK
```

**一条必须理解的约束（路径一致规矩）**：CLI 启动栈时会把 config.yaml 同目录下的 `.vllm-sr/`、`models/` 等生成物用 `-v` 挂进其它容器。而 `-v` 的**左侧路径由宿主机 Docker
daemon**解析。CLI 容器内的路径如果和宿主机路径不一致，daemon 就找不到文件。所以**CLI 容器挂载工作目录时，容器内路径必须与宿主机完全相同**（如 `-v /opt/sr:/opt/sr`）。§5.3 严格遵守此规矩。

### 3.5 自建镜像检查点

```bash
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.Architecture}}' \
  | grep -E 'vllm-sr-(router|dashboard|sim|cli)' | sort
```

量级参考：

| 镜像 | amd64 | arm64 |
| --- | --- | --- |
| vllm-sr-router:v0.3.0-* | ~400-600MB | ~400-600MB |
| vllm-sr-dashboard:v0.3.0-* | ~2.5-3.5GB | ~2.5-3.5GB（含 torch CPU） |
| vllm-sr-sim:v0.3.0-* | ~300MB | ~300MB |
| vllm-sr-cli:0.3.0-* | ~200MB | ~200MB |

### 3.6 二进制存档（可选）

镜像里已有二进制。若验收要求附二进制存档，从镜像提取：

```bash
arch=amd64    # arm64 同理换 tag
id=$(docker create vllm-sr-router:v0.3.0-$arch)
mkdir -p ~/sr-binaries-$arch
docker cp "$id":/usr/local/bin/router           ~/sr-binaries-$arch/
docker cp "$id":/usr/local/lib/.                ~/sr-binaries-$arch/lib/
docker rm "$id"
tar czf ~/sr-router-binaries-v0.3.0-$arch.tar.gz -C ~/sr-binaries-$arch .
```

> **裸跑不在支持范围**（router 是 CGO 动态链接产物，裸跑需自备 glibc ≥2.36 + OpenSSL 3）。部署一律用镜像。

---

## 4. 官方镜像获取与双架构离线打包

### 4.1 官方镜像清单

| 镜像 | 版本 | 用途 |
| --- | --- | --- |
| envoyproxy/envoy | v1.34-latest | 入口网关 |
| redis | 7-alpine | KV 缓存 / 会话 |
| postgres | 16-alpine | 审计 / 元数据 |
| jaegertracing/all-in-one | latest | 链路追踪 |
| prom/prometheus | v2.53.0 | 指标采集 |
| grafana/grafana | 11.5.1 | 指标可视化 |
| milvusdb/milvus（可选） | v2.3.3 | 语义缓存向量库（config 启用时用） |

### 4.2 按架构导出（推荐 skopeo，两种 Docker 存储模式通吃）

skopeo 绕过本地 Docker，直接把 registry 上**指定架构**的内容抽成 `docker load` 可识别的 tar。架构不存在时它会直接报错，不碰本地镜像。

```bash
mkdir -p ~/sr-official && cd ~/sr-official

IMAGES=(
  "envoyproxy/envoy:v1.34-latest"
  "redis:7-alpine"
  "postgres:16-alpine"
  "jaegertracing/all-in-one:latest"
  "prom/prometheus:v2.53.0"
  "grafana/grafana:11.5.1"
)
# 可选：启用语义缓存则追加
# IMAGES+=("milvusdb/milvus:v2.3.3")

FAILED=0
for img in "${IMAGES[@]}"; do
  name=$(echo "$img" | tr '/:' '__')
  for arch in arm64 amd64; do
    if skopeo copy --override-os linux --override-arch "$arch" \
         "docker://$img" "docker-archive:${name}-${arch}.tar:${img}"; then
      echo "OK  $name-$arch"
    else
      echo "FAIL $img $arch"; FAILED=1; break 2
    fi
  done
done
[ "$FAILED" = 0 ] && ls -lh ~/sr-official   # 期望 12 个 tar（含 milvus 则 14）
```

**验证导出的 tar 架构正确性（交付前强烈建议跑一遍）**：

```bash
for f in *-arm64.tar; do
  echo "$f -> $(skopeo inspect --format '{{.Architecture}}' "docker-archive:$f")"
done
# 每行期望 arm64；amd64 同理
```

> 若 skopeo 版本老不支持对 docker-archive inspect，用 tar+python 兜底：
>
> ```bash
> for f in *-arm64.tar; do
>   cfg=$(tar -xOf "$f" manifest.json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["Config"])')
>   echo "$f -> $(tar -xOf "$f" "$cfg" | python3 -c 'import json,sys; print(json.load(sys.stdin)["architecture"])')"
> done
> ```

### 4.3 全量离线导出（自建 + 官方）

```bash
mkdir -p ~/sr-offline && cd ~/sr-offline

# ----- 自建镜像（amd64 + arm64 各自） -----
for arch in amd64 arm64; do
  docker save vllm-sr-router:v0.3.0-$arch       -o vllm-sr-v0.3.0-router-$arch.tar
  docker save vllm-sr-dashboard:v0.3.0-$arch    -o vllm-sr-v0.3.0-dashboard-$arch.tar
  docker save vllm-sr-cli:0.3.0-$arch           -o vllm-sr-v0.3.0-cli-$arch.tar
  # （可选 sim）
  docker save vllm-sr-sim:v0.3.0-$arch          -o vllm-sr-v0.3.0-sim-$arch.tar 2>/dev/null || true
done
# 官方镜像已由 §4.2 直接导出到 ~/sr-official/，此处无需重复 save
```

### 4.4 推内部镜像仓库的替代路径（Harbor 等）

```bash
REG="registry.example.com/sr"
for img in vllm-sr-router:v0.3.0-amd64 vllm-sr-router:v0.3.0-arm64 \
           vllm-sr-dashboard:v0.3.0-amd64 vllm-sr-dashboard:v0.3.0-arm64 \
           vllm-sr-cli:0.3.0-amd64 vllm-sr-cli:0.3.0-arm64; do
  docker tag "$img" "$REG/$img" && docker push "$REG/$img"
done
# 官方镜像同理 retag + push；部署时所有 --xxx-image 参数写完整仓库地址
```

> 双架构合并成多架构单
> tag（`docker buildx imagetools create -t reg/sr/vllm-sr-router:v0.3.0 amd64 arm64`）也可以，与本手册“按后缀分开管理”二选一即可。分开管理排障更直观。

---

## 5. 部署运行（目标服务器，离线 tar 模式）

### 5.1 目标服务器准备（一次性）

目标机仅需要：**Docker**（≥ 24 建议含 buildx；Ubuntu 用官方源，openEuler 用 `dnf install docker-ce` 或系统 docker）。无需 Python / Go / Rust。

```bash
docker version && docker info --format '{{.Architecture}}'    # x86_64 或 aarch64
systemctl is-active docker                                     # running
```

### 5.2 导入镜像（按目标架构选 tar）

把上一节的 `sr-offline/` + `sr-official/` 两个目录整体拷贝到目标服务器（scp / U 盘 / 内网文件服务器任意）。然后按目标架构导入：

```bash
# ===== 示例 1：x86 + Ubuntu =====
ARCH=amd64
cd ~/sr-offline
for f in router dashboard cli sim; do
  tar="vllm-sr-v0.3.0-$f-$ARCH.tar"
  [ -f "$tar" ] && docker load -i "$tar"
done
cd ~/sr-official
for f in envoyproxy_envoy_v1.34-latest redis_7-alpine postgres_16-alpine \
         jaegertracing_all-in-one-latest prom_prometheus_v2.53.0 grafana_grafana_11.5.1; do
  docker load -i "$f-$ARCH.tar"
done

# ===== 示例 2：arm64 + openEuler（鲲鹏） =====
ARCH=arm64
# 同上，把 ARCH 改一下
```

**抽验架构**（load 后 tar 内的原始 tag 会自动恢复，零 retag）：

```bash
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Architecture}}' | head -20
docker inspect redis:7-alpine --format '{{.Architecture}}'   # 期望与目标机匹配
```

### 5.3 部署目录与 config

```bash
# 统一使用 /opt/AgentInfer/semantic-router（可换，但必须遵守路径一致规矩）
SR_DIR=/opt/AgentInfer/semantic-router
sudo mkdir -p "$SR_DIR" && sudo chown "$USER" "$SR_DIR"
# 把你的 config.yaml 放进来（模板见 §6；先用验证过的版本冒烟）
cp /path/to/config.yaml "$SR_DIR/config.yaml"
```

### 5.4 一键启动（核心命令）

```bash
# ============================================================
# x86 机器示例；arm64 机器把所有 -amd64 换成 -arm64
# ============================================================
ARCH=amd64
SR_DIR=/opt/AgentInfer/semantic-router

docker run --rm -it \
  --name sr-orchestrator \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$SR_DIR:$SR_DIR" \
  -w "$SR_DIR" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e VLLM_SR_SIM_ENABLED=false \
  "vllm-sr-cli:0.3.0-$ARCH" \
  serve \
    --config "$SR_DIR/config.yaml" \
    --router-image "vllm-sr-router:v0.3.0-$ARCH" \
    --envoy-image envoyproxy/envoy:v1.34-latest \
    --dashboard-image "vllm-sr-dashboard:v0.3.0-$ARCH" \
    --image-pull-policy never
```

逐参数解读：

| 参数 | 作用 |
| --- | --- |
| `-v /var/run/docker.sock:...` | DooD：CLI 容器直接操作宿主机 Docker；拉起的 9 个容器都是**宿主机容器**（`docker ps` 可见） |
| `-v $SR_DIR:$SR_DIR` | 路径一致规矩：CLI 写的 `.vllm-sr/envoy.yaml` 等文件在容器与宿主机同路径，daemon 后续 `-v` 挂载能正确解析 |
| `-w $SR_DIR` | CLI 工作目录 |
| `-e OPENAI_API_KEY=...` | CLI 只透传白名单变量（OPENAI_/ANTHROPIC_/HF_*），后端 key 从这里进栈 |
| `-e VLLM_SR_SIM_ENABLED=false` | 禁用 sim（交付了 sim 镜像则去掉此行） |
| `--config ...` | 主配置（**用绝对路径**，与 -v 路径一致） |
| `--router-image / --envoy-image / --dashboard-image` | 三个角色镜像显式指定，不依赖 tag 推导，交付最可控 |
| `--image-pull-policy never` | 镜像已离线导入，禁止 CLI 联网拉取（防内网超时卡死） |

**预期日志序列**：创建 docker 网络 → 启动 router → envoy → dashboard →（redis/postgres/观测栈按 config 决定）→ `Configured listeners: ...
0.0.0.0:8888`。

**冒烟验收**：

```bash
# 1) 容器全在 Running
docker ps --format '{{.Names}}\t{{.Status}}'

# 2) 请求拿到 200 + x-vsr-selected-model 头（路由决策的铁证）
curl -sS -D - -o /dev/null http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"auto","messages":[{"role":"user","content":"你好"}],"max_tokens":32}' \
  | grep -iE '^HTTP|x-vsr-selected'
# 期望两行：HTTP/1.1 200 OK + x-vsr-selected-model: <你的模型逻辑名>
```

### 5.5 常用运维

```bash
# 写进 ~/.bashrc 少打字（把 ARCH / SR_DIR 换成你的）
ARCH=amd64
SR_DIR=/opt/AgentInfer/semantic-router
sr() {
  docker run --rm -i \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$SR_DIR:$SR_DIR" -w "$SR_DIR" \
    "vllm-sr-cli:0.3.0-$ARCH" "$@"
}

sr status                # 查看栈内容器状态
sr logs router           # 路由决策日志（排障第一现场）
sr logs envoy            # 入口网关日志
sr logs dashboard        # 控制台日志
sr stop                  # 停掉整栈并清理容器与网络
```

> `sr-orchestrator` 前台进程的两个提醒：① `Ctrl+C` 只停 CLI，**栈容器不会随之停**，需要 `sr stop`；② 想后台常驻：§5.4 命令里 `--rm -it` 换成 `-d`，后续看编排日志
> `docker logs sr-orchestrator`。

### 5.6 启动失败残局清理（必看）

`serve` 中途失败退出时**不会自动回收**已拉起的容器（如 router 起了、envoy 挂了）。残留容器占名字和端口，重跑 serve 会因容器名冲突继续失败。**先清理再重试**：

```bash
# 首选：CLI 自己的 stop（按固定容器名逐个清理，不读 config，连 -v 挂载都不需要）
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  "vllm-sr-cli:0.3.0-$ARCH" stop

# 兜底：宿主机裸 docker 一条（按名字前缀匹配全部 vllm-sr 容器）
docker rm -f $(docker ps -aq --filter name=vllm-sr) 2>/dev/null
docker network rm vllm-sr-network 2>/dev/null

# 确认
docker ps -a --format '{{.Names}}' | grep vllm-sr || echo "已清空"
```

> ① CLI 按固定容器名（`vllm-sr-router-container`、`vllm-sr-redis`…，源码 `runtime_stack.py` 默认值）识别，所以半启动残局它也能收；② 若设置了
> `VLLM_SR_STACK_NAME`（本手册未用），前缀和网名换成 `<stack名>-`；③ postgres/redis 每次运行产生匿名卷，多次迭代后顺手 `docker volume prune -f`。

### 5.7 部署形态变体

| 形态 | 启动命令差异 | 启动的容器 |
| --- | --- | --- |
| 完整栈（默认，sim 禁用） | 同上 §5.4 | router/envoy/dashboard + redis/postgres（config 驱动）+ jaeger/prometheus/grafana |
| 完整栈 + sim | 去掉 `-e VLLM_SR_SIM_ENABLED=false` | 上述 + sim |
| 精简栈（性能实验推荐） | 加 `--minimal` | router + envoy 仅两个（无 dashboard/观测栈/sim） |

---

## 6. 后端连接配置（config.yaml 怎么写）

源码config\config.yaml提供了较为完整的示例，这里只提供简单示例

SR 只认 `providers.models[].backend_refs`，后端只要是 OpenAI 兼容 API 就行——**自建 vLLM 与 vLLM-Router（VR）只差 `base_url` 指向谁**。

```text
方式①直连 vLLM：  Agent → SR(envoy→router) → vLLM-1(:8000/v1)         (单副本)
方式②经 VR：      Agent → SR(envoy→router) → VR(:8000/v1) → vLLM-1..N (VR 做多副本均衡)
```

### 6.1 方式①：直连自建 vLLM 后端

先确保 vLLM 本身能通（昇腾用 vllm-ascend 的对应启动方式）：

```bash
# 探测
curl -s http://<vllm-host>:8000/v1/models | head
```

SR 侧 config.yaml 关键段：

```yaml
listeners:
  - name: "agent-http"
    address: "0.0.0.0"
    port: 8888
    timeout: "300s"          # 大模型慢，超时给足

providers:
  defaults:
    default_model: "Qwen3-Coder-30B-A3B-Instruct-FP8"   # 兜底模型：小、快
  models:
    - name: "glm-5.3"
      provider_model_id: "glm-5.3"      # 发给后端的真实模型 id
      backend_refs:
        - name: "glm-5.3"
          base_url: "http://IP:8077/v1"
          provider: "openai"
          auth_header: "Authorization"
          auth_prefix: "Bearer"
          api_key_env: "OPENAI_API_KEY" # 从容器环境变量读 key（CLI 会透传）
          weight: 100
    - name: "Qwen3-Coder-30B-A3B-Instruct-FP8"
      provider_model_id: "Qwen3-Coder-30B-A3B-Instruct-FP8"
      backend_refs:
        - name: "qwen-local"
          base_url: "http://IP:8000/v1"
          provider: "openai"
          auth_header: "Authorization"
          auth_prefix: "Bearer"
          api_key_env: "OPENAI_API_KEY"
          weight: 100

routing:
  modelCards:
    - name: "glm-5.3"
      modality: "text"
      description: "GLM 5.3 — 大模型，适合复杂任务"
    - name: "Qwen3-Coder-30B-A3B-Instruct-FP8"
      modality: "text"
      description: "Qwen3-Coder-30B-A3B-Instruct-FP8 — 小模型，快"
  decisions:
    # 规则 1（优先级高）：复杂/代码/数学类关键词 → 大模型
    - name: "complex_to_big"
      description: "Complex/code/math queries -> big model"
      priority: 200
      rules:
        operator: "OR"
        conditions:
          - type: keyword
            name: "complex_keywords"
      modelRefs:
        - model: "glm-5.3"
          use_reasoning: false
    # 规则 2（优先级低，空条件=匹配一切）：默认 → 小快模型
    - name: "default_to_small"
      description: "Default catch-all -> small fast model"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "Qwen3-Coder-30B-A3B-Instruct-FP8"
          use_reasoning: false
  signals:
    keywords:
      - name: "complex_keywords"
        operator: "OR"
        keywords:
          - "algorithm"
          - "math"
          - "prove"
          - "refactor"
          - "architecture"
          - "optimize"
          - "debug"
          - "设计"
          - "算法"
          - "优化"
```

`routing.decisions` / `signals` 段（复杂→大模型、默认→小模型）把 `modelRefs` 换成上面 `models[].name` 即可，与现有手册一致。

### 6.2 方式②：经 vLLM-Router（VR）转发

VR 在 SR 看来就是一个 OpenAI 兼容后端。把 `base_url` 指向 VR，例如：

```yaml
    - name: "Qwen3-Coder-30B-A3B-Instruct-FP8"
      provider_model_id: "qwen"
      backend_refs:
        - name: "vr-a-local-vllm"
          base_url: "http://IP:30000/v1"
          provider: "openai"
          auth_header: "Authorization"
          auth_prefix: "Bearer"
          api_key_env: "OPENAI_API_KEY"
          weight: 100
```

**混合使用完全可行**：同一 config 里不同模型指向不同后端（例：小模型直连本机 vLLM，大模型经 VR 走异构集群）。同一模型也可以配多个 `backend_refs`（`weight` 加权，SR
自带同模型多副本分流）——与 VR 的副本级负载均衡是两层能力：

- **SR = 模型级智能路由**（这个请求**该用哪个模型**）+ 缓存/改写/安全
- **VR = 副本级负载均衡**（这个模型**发给哪个副本**）
- 两者叠加 = “模型选择 × 副本分发”完整两级路由

### 6.3 关于后端 API key

vLLM 启动时加 `--api-key <令牌>` 可以给推理服务加“门禁”（Authorization: Bearer 必须匹配）。**内网自用完全可以不加**。

与 SR 的关系：

- **未启用 `--api-key`**：vLLM 会忽略 Authorization 头。但 SR 的 authz（`fail_open: false`）要求 `api_key_env` 指向的变量**有值**，否则请求会被 SR
自己拒。照常配置 `api_key_env: "OPENAI_API_KEY"`，启动 CLI 容器时传 `-e OPENAI_API_KEY=dummy-internal-key`（随便什么占位字符串）。
- **启用了 `--api-key`**：`OPENAI_API_KEY` 填 vLLM/VR 的真实令牌即可。
- VR 启用鉴权同处理。

---

## 7. 端口占用清单与修改方法

### 7.1 完整端口清单（供多团队协调）

构建阶段不占任何端口；以下为**运行阶段发布到宿主机**的全部端口（默认偏移=0，来源：`runtime_stack.py` / `docker_start.py` / `docker_services.py` 端口映射逐一核实）：

| 宿主机端口 | 所属容器 | 容器内端口 | 用途 | 谁会访问 |
| --- | --- | --- | --- | --- |
| **8888** | envoy | 8888 | **Agent 入口（OpenAI 兼容 API）** | 业务方 / 压测工具 |
| 8700 | dashboard | 8700 | Web 控制台 | 管理员浏览器 |
| 8080 | router | 8080 | router 管理 API | dashboard / 运维 |
| 9190 | router | 9190 | router 指标（/metrics） | prometheus / 运维 |
| 50051 | router | 50051 | gRPC ext_proc（envoy↔router） | envoy（本机容器间） |
| 6379 | redis | 6379 | 缓存 / 会话存储 | 本机容器间 |
| 5432 | postgres | 5432 | 审计 / 元数据存储 | 本机容器间 |
| 16686 | jaeger | 16686 | Jaeger UI（链路追踪） | 管理员浏览器 |
| 4318 | jaeger | 4317 | OTLP 遥测上报 | 本机容器间 |
| 9090 | prometheus | 9090 | 指标采集 | grafana / 管理员 |
| 3000 | grafana | 3000 | 监控图表 | 管理员浏览器 |
| 8810 | sim（若启用） | 8000 | 车队模拟器 API | 本机 |
| 19530 | milvus（仅 config 启用） | 19530 | 语义缓存向量库 | router |

> 注意：50051/6379/5432/4318 这类容器间互访端口**也会发布到宿主机**（CLI 默认行为），虽然外部没人访问，但会占宿主端口，协调时必须算进去。envoy admin 9901 仅容器网络内部，**不占宿主端口**。

按部署形态的实际占用：

| 形态 | 占用端口数 | 端口列表 |
| --- | --- | --- |
| 完整栈（sim 禁用） | 11 | 8888, 8700, 8080, 9190, 50051, 6379, 5432, 16686, 4318, 9090, 3000 |
| 完整栈 + sim | 12 | 上述 + 8810 |
| `--minimal` 精简栈 | 4（+ 存储） | 8888, 8080, 9190, 50051（+ 6379/5432 视 config 存储配置） |

### 7.2 改端口的方法

1. **整体偏移（推荐）**：设 `VLLM_SR_PORT_OFFSET=N`，上表所有宿主机端口统一 +N（**入口 8888 也包含在内**）。协商出专属端口区间后用这个最省事。CLI 容器化部署这样传：

```bash
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$SR_DIR:$SR_DIR" -w "$SR_DIR" \
  -e VLLM_SR_PORT_OFFSET=100 \
  "vllm-sr-cli:0.3.0-$ARCH" serve ...（其余同 §5.4）
# 效果：8888→8988，8700→8800，6379→6479，3000→3100……所有端口一起 +100
```

1. **只改入口**：config.yaml `listeners[].port`（支持多个 listener）。它同样叠加偏移（宿主端口=配置端口+偏移）。改后两个连带动作：① Agent 侧 base_url 同步改；② 若改成
<1024 的特权端口，需 root 权限。
2. **redis/postgres 被占**：CLI 启动前会检测端口占用并打印警告。处理方式仍是整体偏移或与占用方协调。

冲突排查：`ss -lntp | grep -E '8888|8700|3000|9090'`（或 `netstat -lntp`）。

### 7.3 放行防火墙（多团队共用服务器时必做）

8888/8700/3000（grafana）/9090 等对外端口需要系统防火墙允许，并且云服务器还要再加一层安全组：

```bash
# openEuler（firewalld）
sudo firewall-cmd --add-port=8888/tcp --permanent   # Agent 入口
sudo firewall-cmd --add-port=8700/tcp --permanent   # dashboard
sudo firewall-cmd --add-port=3000/tcp --permanent    # grafana（监控）
sudo firewall-cmd --add-port=16686/tcp --permanent   # jaeger（链路追踪）
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports

# Ubuntu（ufw）
sudo ufw allow 8888/tcp
sudo ufw allow 8700/tcp
sudo ufw allow 3000/tcp
sudo ufw allow 16686/tcp
sudo ufw status
```

> 不想动防火墙（或策略严格不好改）可以用 SSH 隧道从本地访问：
>
> ```bash
> # 本地机器执行；一个 -L 一个转发端口，保持 shell 开着
> ssh -L 8700:localhost:8700 -L 3000:localhost:3000 -L 16686:localhost:16686 user@<服务器>
> # 然后本地浏览器访问 http://localhost:8700
> ```

---

## 8. 交付清单

| # | 交付物 | 说明 | 检查方式 |
| --- | --- | --- | --- |
| 1 | `vllm-sr-router:v0.3.0-amd64/arm64` 镜像 tar | 自建，SR 核心 | load 后 `docker image inspect ... '{{.Architecture}}'` |
| 2 | `vllm-sr-dashboard:v0.3.0-amd64/arm64` | 自建，控制台 | 同上 |
| 3 | `vllm-sr-cli:0.3.0-amd64/arm64` | 自建，编排器 | `docker run --rm <img> --version` 输出 0.3.0 |
| 4 | （可选）`vllm-sr-sim:v0.3.0-*` | 自建，未交付则部署时禁用 | — |
| 5 | 官方镜像 6 个（双架构 tar） | envoy/redis/postgres/jaeger/prometheus/grafana，版本见 §1.1 | load 后 `docker images` 对表 |
| 6 | config.yaml 模板 | 外部 API / 直连 vLLM / 经 VR（§6） | CLI 容器内 `vllm-sr validate --config <路径>` |
| 7 | 部署操作文档 | 即本手册 §5–§7 裁剪版 | — |
| 8 | （可选）二进制存档 tar.gz | §3.6 产物 | 解包见 router + lib/*.so |
| 9 | 镜像校验记录 | 每个镜像的 `Architecture` / 大小 / digest 清单 | 构建时导出存档 |

**验收四项（目标机，全过即交付合格）**：

1. §5.4 启动成功无 error，`sr status` 所有容器 Up
2. §5.4 冒烟 curl 拿到 HTTP 200 + `x-vsr-selected-model` 头
3. `http://<服务器>:8700` 或 SSH 隧道访问 dashboard 可打开
4. `sr logs router` 最近 50 行无 ERROR 级别日志

---

## 9. FAQ / 排障速查

| 症状 | 原因与处理 |
| --- | --- |
| CLI 容器报 "Docker daemon is not reachable" | docker.sock 未挂载或宿主机 Docker 没起。检查 `-v /var/run/docker.sock:/var/run/docker.sock` 与 `systemctl status docker` |
| envoy 秒退，日志报挂载路径不存在 | 违反路径一致规矩（CLI 容器内 config 路径 ≠ 宿主机路径）。严格按 §5.4：`-v $SR_DIR:$SR_DIR` 且 `--config` 用容器内绝对路径 |
| `docker load` 后鲲鹏机报 "exec format error" | 架构不匹配的铁证：arm64 机器必须 load `-arm64` 后缀的 tar。`docker image inspect <img> '{{.Architecture}}'` 核对每包 |
| `skopeo copy` 报 "commit v1 image: Error parsing" | skopeo 过老；升级到 1.13+ 或改用同架构宿主机 docker pull/inspect/save |
| arm64 镜像 inspect 显示 Architecture=amd64 | 构建命令缺 `--platform linux/arm64`（`--build-arg TARGETARCH=arm64` 无效）。按 §3.1/3.2 正确命令重建，编译层缓存可复用 |
| router arm64 交叉编译报 OpenSSL/linker 错 | 交叉工具链没装全。确认基于 v0.3.0 tag 重跑；公司代理确保 `GIT_SSL_NO_VERIFY=1` |
| dashboard arm64 构建报 gcc/CGO 错 | 只会发生在"不带 `--platform` 只传 `--build-arg TARGETARCH=arm64`"这条错误路径。正确做法：`--platform linux/arm64` QEMU 构建，或鲲鹏机原生构建 |
| dashboard arm64 QEMU 构建中途 npm 崩溃 | QEMU 已知偶发。直接重跑同一条命令，层缓存从断点续上；频繁崩溃改用鲲鹏机原生构建 |
| 构建卡在依赖下载（Go 模块 / crates.io / pip / npm） | 代理未传给构建容器。§2.4 两种方式二选一：daemon 级 `proxies` 或每条 build 传 `--build-arg HTTP(S)_PROXY=...` |
| 启动卡在 "Pulling image..."（内网） | 忘了 `--image-pull-policy never`，或镜像名拼写与 `docker images` 不一致（注意 -amd64/-arm64 后缀） |
| 请求 401/403 | `OPENAI_API_KEY` 没通过 `-e` 传给 CLI 容器，或后端 `--api-key` 值不匹配（§6.3） |
| 请求 503 upstream connect error | envoy→后端网络不通。目标机直测 `curl -v http://<vllm-host>:8000/v1/models`；容器出网受防火墙限制时检查 iptables/安全组 |
| 本机浏览器连不上远程 dashboard（8888 API 能通） | 8700 没放行（与 8888 是独立放行项）。排查：① `docker ps` dashboard Up？② 服务器本机 `curl http://localhost:8700/healthz` 通？③ openEuler：`sudo firewall-cmd --add-port=8700/tcp --permanent && firewall-cmd --reload`（Ubuntu：`ufw allow 8700/tcp`；云服务器再加安全组）。不想动防火墙就 SSH 隧道：`ssh -L 8700:localhost:8700 user@服务器` |
| 想改 8888 | config.yaml `listeners[].port`，或 `VLLM_SR_PORT_OFFSET` 整体偏移（入口也吃偏移）。§7.2 |
| 什么时候会看到 milvus 容器？我部署时怎么没见到 | milvus 做语义缓存存储，CLI 按需启动：config 的 `stores.semantic_cache.backend_type: milvus` 才拉。本手册默认关闭（性能测试推荐关闭，命中缓存不经过模型，口径失真）。启用：config 打开 + 离线包补 `milvusdb/milvus:v2.3.3` 镜像 + 放行 19530 |
| `sudo apt install` 永远卡在 Connecting | sudo 清空了环境变量（env_reset），apt 拿不到 `http_proxy`。单独写：`/etc/apt/apt.conf.d/95proxy` → 两行 `Acquire::http::Proxy "...";` / `Acquire::https::Proxy "...";` |
| `sr stop` 后再 serve 报容器名冲突 | 残留容器。`docker rm -f $(docker ps -aq --filter name=vllm-sr)` + `docker network rm vllm-sr-network` 清理后重试 |
| CLI 容器内 `vllm-sr status` 看不到容器 | status 也要在挂了 docker.sock 的 CLI 容器里跑（用 §5.5 的 `sr` 函数），且工作目录与 serve 用同一路径 |

---

**执行任何一步与预期不符，排查优先级**：`sr logs router` → 本手册 §9 → 检查防火墙/网络。构建侧问题：看 `docker build` 的**第一个报错**（不是最后一个），通常是根因。
