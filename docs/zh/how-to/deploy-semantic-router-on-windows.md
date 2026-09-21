# vLLM Semantic Router v0.3.0 — 在 Windows（+ WSL2）上编译构建手册

> **文档版本**：v1.0　｜　**对应代码**：SR tag `v0.3.0` (<https://github.com/vllm-project/semantic-router/releases/tag/v0.3.0>)
>
> **适用场景**：你的主力工作机是 Windows，但需要为 Linux 目标服务器（x86 Ubuntu 或 arm64 openEuler）构建 SR 的 Docker 镜像。
>
> **输出物**：
>
> - 自建镜像（4 个 ×
> 双架构）：`vllm-sr-router:v0.3.0-{amd64\|arm64}`、`vllm-sr-dashboard:v0.3.0-*`、`vllm-sr-cli:0.3.0-*`、（可选）`vllm-sr-sim:v0.3.0
> -*`
> - 官方镜像 tar（6 个 × 双架构，或含 milvus 共 7 个）
>
> **部署运行不在本手册范围**：部署请配合同目录的 [deploy-semantic-router.md](deploy-semantic-router.md) 的 **§5–§9**
> （目录规划 / 一键启动 / 后端连接 / 端口 / 交付 / FAQ）。

---

## 0. 必读的 4 个坑（作者亲测）

| # | 坑 | 一句话规避 |
| --- | --- | --- |
| 1 | `--build-arg TARGETARCH=arm64` 不会产出 arm64 镜像 | arm64 镜像必须带 `--platform linux/arm64` |
| 2 | `sudo apt` 不走 shell 里的代理变量 | 单独写 `/etc/apt/apt.conf.d/95proxy` |
| 3 | Docker Desktop 启用 containerd 存储时，`docker inspect` 官方多架构镜像的 Architecture 会是空 | 官方镜像导出一律用 **skopeo**，不用 `docker pull --platform` 那条路 |
| 4 | CLI 容器工作目录挂载必须**路径一致**（容器内路径 == 宿主机路径） | 部署时 `-v /opt/sr:/opt/sr`，不能 `-v /opt/sr:/workspace` 这种 |

---

## 1. 整体流程

```text
步骤 1  环境准备     → Windows + WSL2 + Docker Desktop + 代理（§2）
步骤 2  获取源码     → git clone v0.3.0 tag（§3）
步骤 3  构建自建镜像 → router / dashboard / sim / CLI，amd64 + arm64（§4）
步骤 4  拉官方镜像   → 6 镜像 × 双架构，用 skopeo 导出 tar（§5）
步骤 5  全量打包     → docker save + skopeo 产物归档（§6）
步骤 6  送到目标服务器 → 按 deploy-semantic-router.md §5–§9 部署
```

---

## 2. 环境准备（Windows + WSL2）

### 2.1 软件清单

| 软件 | 版本要求 | 作用 |
| --- | --- | --- |
| Windows | 10 / 11（22H2+） | 宿主机 |
| WSL2 | 默认内核版本 5.15+ | 真正执行构建命令的地方（Ubuntu 22.04/24.04） |
| Docker Desktop | 4.30+，引擎 27.x+ | 提供 Linux Docker daemon 给 WSL2，且自带 QEMU（arm64 构建需要） |
| 代理 | 可选，7890 等端口 | 国内环境下载依赖几乎必需（本文假设你有） |

### 2.2 一步步装 WSL2

```powershell
# ===== Windows PowerShell（管理员）=====

# 1) 启用 WSL + 虚拟机平台（已启用则跳过）
wsl --install                # 会自动安装 WSL2 + Ubuntu 默认版
# 如果需要手动选发行版：
# wsl --list --online
# wsl --install -d Ubuntu-24.04

# 2) 重启后，开始菜单打开 "Ubuntu"，设置用户名和密码
# 3) 确认版本
wsl --status                 # Default Version: 2
wsl -l -v                    # Ubuntu VERSION = 2
```

> **常见坑：Wsl/Service/E_UNEXPECTED**（WSL 起不来）：
>
> 1. 以管理员身份打开 `services.msc` → 找到 "Hyper-V Virtual Machine Management" / "Virtual Machine Management" / "Windows
> Subsystem for Linux" 三个服务，设为自动并启动。
> 2. 控制面板 → 程序 → 启用或关闭 Windows 功能 → 勾上 "Hyper-V"、"适用于 Linux 的 Windows 子系统"、"虚拟机平台" → 重启。
> 3. `wsl --update` + `wsl --shutdown` 再试。

### 2.3 Docker Desktop

1. 官网下载安装 Docker Desktop（默认 Next 到底）。
2. 启动 Docker Desktop → Settings → **General** → 勾上 "Use the WSL 2 based engine"（默认开的）。
3. Settings → **Resources → WSL Integration** → 把你用的 Ubuntu 发行版开关打开 → **Apply & Restart**。
4. （国内）Settings → **Resources → Proxies** → Manual proxy configuration → HTTP 框 `http://127.0.0.1:7890` / HTTPS 框
`http://127.0.0.1:7890` → Apply。

### 2.4 WSL Ubuntu 内验证

```bash
# WSL Ubuntu 终端里跑（开始菜单搜 Ubuntu）
docker --version                       # 27.x+
docker info --format '{{.OSType}}/{{.Architecture}}'   # linux/x86_64
uname -m                               # x86_64
df -h ~                                # 剩余 ≥ 60GB（构建缓存+双架构镜像）

# QEMU（arm64 构建）
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx ls | grep linux/arm64    # 能看到 linux/arm64 就 OK
```

### 2.5 代理（国内环境关键）

```bash
# WSL Ubuntu 终端里
# 1) 拿到宿主机（Windows）在 WSL 虚拟网卡里的 IP（每次 WSL 重启可能变）
export hostip=$(ip route show default | awk '{print $3}')
echo "$hostip"                         # 比如 172.30.128.1

# 2) 设代理（注意：不要加任何反引号/引号，直接裸写）
export http_proxy=http://$hostip:7890 https_proxy=http://$hostip:7890
# 检查（值首尾干净、无反引号）
env | grep -i proxy

# 3) 验证（期望 HTTP/2 200；空行或超时 = 没通，见 §8 FAQ）
curl -sI https://proxy.golang.org --max-time 5 | head -1
```

**`sudo apt` 单独代理（必做）**：sudo 默认清空环境变量，apt 拿不到代理，在直连不通的网络下**会永远卡在 Connecting**：

```bash
sudo tee /etc/apt/apt.conf.d/95proxy <<EOF
Acquire::http::Proxy "http://$hostip:7890";
Acquire::https::Proxy "http://$hostip:7890";
EOF
# 验证：sudo apt update 几秒内能跑出源信息即生效
sudo apt update -qq
```

> hostip 在 WSL 重启后可能变，以后 `sudo apt` 又卡住时先 `ip route show default` 查新 IP，重写 95proxy。

**构建容器内部代理也走得通**：上面 Docker Desktop Proxies 已经配好的话，镜像拉取 + 构建容器内代理全部自动生效；不想配 Docker Desktop，就在每条 `docker build` 命令后追加：

```bash
--build-arg HTTP_PROXY="$http_proxy" --build-arg HTTPS_PROXY="$https_proxy"
```

### 2.6 （可选但推荐）准备工作目录

把代码和构建产物都放在 Windows 下的工作盘（E 盘），便于后期拷贝给同事。WSL 里访问 E 盘是 `/mnt/e/...`（注意大小写敏感）：

```bash
# 示例
export SR_SRC=/mnt/e/Huawei/code/semantic-router
export SR_BUILD=/mnt/e/Huawei/code/AgentInfer/build/semantic-router
mkdir -p "$SR_BUILD"
```

---

## 3. 获取 v0.3.0 源码

```bash
cd "$SR_SRC"
# 方式一：全新 clone（推荐，保证干净，不受本地改动污染）
cd /tmp && git clone --depth 1 --branch v0.3.0 https://github.com/vllm-project/semantic-router.git sr-v0.3.0
cd sr-v0.3.0 && git describe --tags    # 期望 v0.3.0

# 方式二：使用已有仓库（确认切到 v0.3.0 tag；有本地改动就基于 tag 开分支）
cd "$SR_SRC"
git fetch --tags
git checkout v0.3.0
git checkout -b v0.3.0-custom v0.3.0    # 建议开分支，便于后续合入改代码
```

> **重要**：后文所有 `docker build` 命令的当前工作目录必须是**仓库根目录**（含 `candle-binding/`、`dashboard/`、`src/` 等目录）。

### 3.1 源码里各目录对应的产物（改代码后知道重建谁）

| 改这里 | 要重建的镜像 |
| --- | --- |
| `candle-binding/` / `ml-binding/` / `nlp-binding/` / `src/semantic-router/` | `vllm-sr-router` |
| `dashboard/backend/`（Go）或 `dashboard/frontend/`（TS）或 `dashboard/wizmap/` | `vllm-sr-dashboard` |
| `src/fleet-sim/` | `vllm-sr-sim` |
| `src/vllm-sr/cli/`（Python 编排器） | `vllm-sr-cli`（改 Dockerfile：把 pip 安装换成源码安装，见 §4.4 备注） |

---

## 4. 构建 4 个自建镜像（amd64 + arm64）

### 4.1 router 镜像（核心，最重要）

```bash
# ========== amd64（x86 上原生，默认平台） ==========
docker build --network=host \
  --platform linux/amd64 \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/vllm-sr/Dockerfile \
  -t vllm-sr-router:v0.3.0-amd64 .

# ========== arm64（x86 上交叉编译，快） ==========
docker build --network=host \
  --platform linux/arm64 \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/vllm-sr/Dockerfile \
  -t vllm-sr-router:v0.3.0-arm64 .
```

**核查（关键！）**：

```bash
docker image inspect vllm-sr-router:v0.3.0-amd64 --format '{{.Architecture}}'   # amd64
docker image inspect vllm-sr-router:v0.3.0-arm64 --format '{{.Architecture}}'   # arm64
# 量级：~400-600MB / arch
```

**必须理解的坑**：`--build-arg TARGETARCH=arm64` 只驱动 Dockerfile 内部装交叉工具链，**不会改变镜像的目标架构元数据**。不带 `--platform linux/arm64`
的所谓"arm64 构建"，产物永远是 amd64 镜像里装着 arm64 二进制——两头都跑不起来。`make VLLM_SR_TARGETARCH=arm64` 同样中招（make 构建参数没有 `--platform`）。所以
arm64 **一律用上面的 docker build 命令**。

> 首次构建 30~60 分钟（Rust 依赖编译是大头）；有层缓存后改 Go 代码几分钟。`--network=host` 共享宿主机网络（走代理），`GIT_SSL_NO_VERIFY=1` 应对公司代理自签证书。

### 4.2 dashboard 镜像

```bash
# ========== amd64（x86 上原生） ==========
docker build --network=host \
  -f dashboard/backend/Dockerfile \
  -t vllm-sr-dashboard:v0.3.0-amd64 .

# ========== arm64：两条路，二选一 ==========
# 路径 A：QEMU 全模拟（无需鲲鹏机，慢）
docker build --network=host \
  --platform linux/arm64 \
  -f dashboard/backend/Dockerfile \
  -t vllm-sr-dashboard:v0.3.0-arm64 .
# 预期 40~90 分钟；npm 在 QEMU 下偶发段错误崩溃 → 直接重跑同一条命令，层缓存续上

# 路径 B：鲲鹏机原生构建（有条件推荐，最快）
# WSL 里打包源码（排除 .git/node_modules 减体积）
tar --exclude='.git' --exclude='node_modules' -C /tmp -czf sr-v0.3.0.tar.gz sr-v0.3.0
# 用 scp/U盘 传到鲲鹏机后：
# tar xzf sr-v0.3.0.tar.gz && cd sr-v0.3.0
# docker build --network=host -f dashboard/backend/Dockerfile -t vllm-sr-dashboard:v0.3.0-arm64 .
```

> dashboard 的 Dockerfile 没有 `GIT_SSL_NO_VERIFY` 参数（和 router 不同），所以命令不传它。

### 4.3 sim 镜像（可选，精简交付不构建）

sim 是纯 Python（无编译）。交付时直接禁掉最省事（`VLLM_SR_SIM_ENABLED=false`，连镜像都不用构建）。如果需要 sim：

```bash
# amd64
docker build --network=host \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/fleet-sim/Dockerfile -t vllm-sr-sim:v0.3.0-amd64 .

# arm64（QEMU 即可）
docker build --network=host \
  --platform linux/arm64 \
  --build-arg GIT_SSL_NO_VERIFY=1 \
  -f src/fleet-sim/Dockerfile -t vllm-sr-sim:v0.3.0-arm64 .
```

### 4.4 CLI 编排器镜像（新增，交付便捷性关键）

**新建 CLI Dockerfile**，保存为 `src/vllm-sr/cli/docker/cli.Dockerfile`（仓库里原本没有这个文件）：

```dockerfile
# vllm-sr CLI 编排器镜像：pip 安装 CLI + 借用 docker CLI 二进制
# 用途：容器内运行 `vllm-sr serve`，通过挂载宿主机 docker.sock 编排整栈
FROM python:3.11-slim-bookworm

COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
RUN pip install --no-cache-dir vllm-sr==0.3.0

WORKDIR /workspace
ENTRYPOINT ["vllm-sr"]
```

> 如果你改了 CLI 源码（`src/vllm-sr/cli/**/*.py`），把 `RUN pip install vllm-sr==0.3.0` 换成：
>
> ```dockerfile
> COPY src/vllm-sr /tmp/vllm-sr
> RUN pip install --no-cache-dir /tmp/vllm-sr
> ```

**构建（纯 Python 层，QEMU 也很快）**：

```bash
# amd64
docker build --network=host \
  -f src/vllm-sr/cli/docker/cli.Dockerfile -t vllm-sr-cli:0.3.0-amd64 .

# arm64
docker build --network=host \
  --platform linux/arm64 \
  -f src/vllm-sr/cli/docker/cli.Dockerfile -t vllm-sr-cli:0.3.0-arm64 .
```

验证：

```bash
docker run --rm vllm-sr-cli:0.3.0-amd64 --version       # 期望 vllm-sr version: 0.3.0
docker run --rm vllm-sr-cli:0.3.0-amd64 serve --help    # 打印帮助即 OK
```

### 4.5 自建镜像总检查点

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

### 4.6 二进制存档（可选）

镜像里已有二进制。若验收要求附二进制存档，从镜像提取：

```bash
for arch in amd64 arm64; do
  id=$(docker create vllm-sr-router:v0.3.0-$arch)
  d="$SR_BUILD/binaries-$arch"
  mkdir -p "$d/lib"
  docker cp "$id":/usr/local/bin/router "$d/router"
  docker cp "$id":/usr/local/lib/. "$d/lib/"
  docker rm "$id"
  tar czf "$SR_BUILD/sr-router-binaries-v0.3.0-$arch.tar.gz" -C "$d" .
  echo "✅ sr-router-binaries-v0.3.0-$arch.tar.gz"
done
```

---

## 5. 官方镜像按架构导出（用 skopeo，推荐）

官方镜像（envoy/redis/postgres/jaeger/prometheus/grafana）都是**多架构镜像**（一个 tag 内同时有 amd64/arm64 两份内容）。不能用 `docker tag` 打别名（tag
是别名不复制镜像），也不能简单 `docker pull --platform linux/arm64`（Docker Desktop 默认启用 containerd 存储时，拉完后 `docker inspect`
Architecture 为空——无法验证架构，作者亲测踩坑）。

**正确做法：skopeo 从 registry 直接按架构抽取**（绕过本地 Docker，指定架构不存在时直接报错，确定性强）。

```bash
# 1) 装 skopeo + 确认代理
sudo apt install -y skopeo
echo "$https_proxy"                 # 有值才会走代理

# 2) 批量导出 6 镜像 × 2 架构
IMAGES=(
  "envoyproxy/envoy:v1.34-latest"
  "redis:7-alpine"
  "postgres:16-alpine"
  "jaegertracing/all-in-one:latest"
  "prom/prometheus:v2.53.0"
  "grafana/grafana:11.5.1"
)
# 可选：如果启用语义缓存（milvus）则追加
# IMAGES+=("milvusdb/milvus:v2.3.3")

mkdir -p "$SR_BUILD" && cd "$SR_BUILD"

FAILED=0
for img in "${IMAGES[@]}"; do
  name=$(echo "$img" | tr '/:' '__')          # 生成安全文件名（如 redis_7-alpine）
  for arch in arm64 amd64; do
    if skopeo copy --override-os linux --override-arch "$arch" \
         "docker://$img" "docker-archive:${name}-${arch}.tar:${img}"; then
      echo "✅ ${name}-${arch}.tar"
    else
      echo "❌ $img（$arch）失败，停止"; FAILED=1; break 2
    fi
  done
done
[ "$FAILED" = 0 ] && echo "官方镜像 tar 全部导出完成" && ls -lh "$SR_BUILD"/*.tar
```

**验证每个 tar 的架构正确性（交付前必跑）**：

```bash
cd "$SR_BUILD"
echo "=== arm64 tar ==="
for f in *-arm64.tar; do
  arch=$(skopeo inspect --format '{{.Architecture}}' "docker-archive:$f" 2>/dev/null)
  echo "$f -> $arch"
done
echo "=== amd64 tar ==="
for f in *-amd64.tar; do
  arch=$(skopeo inspect --format '{{.Architecture}}' "docker-archive:$f" 2>/dev/null)
  echo "$f -> $arch"
done
# 每行期望对应 arm64 / amd64；两种架构的 tar 大小略有差异是正常的（之前"大小完全一样"是红牌）
```

> skopeo 版本过老不支持 docker-archive inspect 时，用 tar+python 兜底（任何机器都能跑）：
>
> ```bash
> for f in *-arm64.tar; do
>   cfg=$(tar -xOf "$f" manifest.json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["Config"])')
>   echo "$f -> $(tar -xOf "$f" "$cfg" | python3 -c 'import json,sys; print(json.load(sys.stdin)["architecture"])')"
> done
> ```

---

## 6. 全量打包

```bash
mkdir -p "$SR_BUILD/offline" && cd "$SR_BUILD/offline"

# 自建镜像（8 个或 6 个，不含 sim）
for arch in amd64 arm64; do
  docker save vllm-sr-router:v0.3.0-$arch    -o vllm-sr-v0.3.0-router-$arch.tar
  docker save vllm-sr-dashboard:v0.3.0-$arch -o vllm-sr-v0.3.0-dashboard-$arch.tar
  docker save vllm-sr-cli:0.3.0-$arch        -o vllm-sr-v0.3.0-cli-$arch.tar
  docker save vllm-sr-sim:v0.3.0-$arch       -o vllm-sr-v0.3.0-sim-$arch.tar 2>/dev/null || true
done
# 官方镜像 tar 已由 §5 导出到 $SR_BUILD/，拷贝过来形成交付包
cp "$SR_BUILD"/*-amd64.tar "$SR_BUILD"/*-arm64.tar ./
# （你上面那批 tar 名是官方镜像的，这里和自建 tar 会合并到一个目录，共 20 或 22 个 tar）

# 最终校验清单
echo "=== 交付 tar 总数（自建 8 + 官方 12 = 20，含 sim/milvus 则更多）==="
ls -1 | wc -l
ls -lhS
```

### 分发方式

- **离线 tar 传**：整个 `$SR_BUILD/offline/` 目录打 zip 或 rsync / scp / U 盘拷到目标服务器 → 按 `deploy-semantic-router.md` §5.2 导入。
- **推内部镜像仓库（Harbor 等）**：见 `deploy-semantic-router.md` §4.4。

---

## 7. 之后还需要做什么？

本手册只完成了**构建阶段**。真正把 SR 跑起来、接入后端、配置端口、验收冒烟：

```text
看同目录下的 deploy-semantic-router.md
      阅读 §5（部署运行）
           §6（后端连接：直连 vLLM / 经 VR / 外部 API）
           §7（端口占用与修改 / 防火墙放行）
           §8（交付清单与验收四项）
           §9（FAQ / 排障速查）
```

重点再重复一遍部署命令里必须满足的**三条铁律**：

1. `-v /var/run/docker.sock:/var/run/docker.sock`（CLI 操作宿主机 Docker 的通道，**不能丢**）
2. `-v $SR_DIR:$SR_DIR`（工作目录**路径一致**，左侧宿主机路径 == 右侧容器内路径）
3. `--image-pull-policy never`（离线环境，不许联网拉镜像）

---

## 8. FAQ / Windows 特有排障速查

| 症状 | 原因与处理 |
| --- | --- |
| `go` / `rustc` / `npm` command not found（WSL 里没这些） | **正常**。所有编译都在 `docker build` 的构建容器内部跑（自动拉 golang/rust/node 镜像），宿主机只要 Docker 就行 |
| `sudo apt install` 永远卡在 Connecting，但我 shell 代理变量明明有值 | sudo 默认清空环境变量（env_reset），apt 拿不到 `http_proxy`。按 §2.5 单独写 `/etc/apt/apt.conf.d/95proxy`；另检查 `env \| grep proxy` 输出里代理地址有没有被多余反引号污染（export 时误用反引号会导致代理无效） |
| §2.5 curl 验证输出为空（代理没通） | 三步排查：① `echo "$http_proxy"` 输出非空且干净；② Windows 侧确认代理监听 7890（`netstat -ano \| findstr 7890`）；③ 代理软件必须打开"Allow LAN（允许局域网连接）"，否则 WSL 虚拟网卡连不上宿主机的 7890 |
| `skopeo copy` 报 "x509: certificate signed by unknown authority" | 公司代理自签证书。追加参数：`skopeo copy --src-tls-verify=false ...` 或加 `--src-cert-dir /usr/local/share/ca-certificates` 导入公司 CA |
| `docker build` 卡在下载 Go 模块 / cargo 依赖 | 构建容器内部代理没生效。确保 Docker Desktop → Proxies 已配（推荐）；或每条 build 命令加 `--build-arg HTTP_PROXY=... HTTPS_PROXY=...` |
| dashboard arm64 QEMU 构建慢 / npm 段错误崩溃 | QEMU 模拟已知现象。直接重跑同一条命令，层缓存断点续上；频繁崩溃换鲲鹏机原生构建（§4.2 路径 B） |
| 构建时报 "failed to create NAT" 或 Docker daemon 连不上 | WSL 网络状态异常。WSL 内 `wsl --shutdown`（PowerShell）→ 重新开终端 → `docker ps` 验证。Docker Desktop 也要开着（别退出） |
| 在 `/mnt/e/...` 里 `git checkout` 报权限 / 符号链接错 | `/mnt/e/...` 走 9p 文件系统，性能和权限语义都不如 WSL 本地盘。建议 clone 到 WSL 的 `~/sr-src`（本机 ext4）再做构建，最后把 save 出的 tar cp 到 `/mnt/e/...` 分发 |
| PowerShell 里用 docker pull 出的官方镜像转 WSL 里 docker 看不到 | Docker Desktop 已经为 WSL 统一了镜像存储，两边应一致；不一致说明发行版没开 WSL Integration：Docker Desktop → Settings → Resources → WSL Integration → 打开对应发行版 → Apply & Restart |
| arm64 镜像 inspect 显示 Architecture=amd64 | 构建命令缺 `--platform linux/arm64`（`--build-arg TARGETARCH=arm64` / make 的 `VLLM_SR_TARGETARCH=arm64` 都无效）。按 §4.1 正确命令重建 |
| "exec user process caused: exec format error"（在鲲鹏机上跑自建镜像） | 架构不匹配铁证。`docker image inspect <img> '{{.Architecture}}'` 核对；arm64 机器必须用 `-arm64` 后缀镜像 |

---

*本手册完。构建阶段任何一步和预期不符，先看对应命令的**第一个报错**（通常是根因），然后查本节 FAQ。*
