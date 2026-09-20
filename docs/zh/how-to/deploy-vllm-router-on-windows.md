# vLLM Router — 在 Windows（+ WSL2）上编译构建手册

**文档版本**：v1.0　｜　**对应代码**：VR tag `v0.1.15` (<https://github.com/vllm-project/router/releases/tag/v0.1.15>)

**适用场景**：你的主力工作机是 Windows，但需要为 Linux 目标服务器（x86 Ubuntu 或 arm64 openEuler）构建 VR 的 Docker 镜像。

**交付形态**：**Docker 镜像**

**部署运行不在本手册范围**：见主文档 [deploy-vllm-router.md](deploy-vllm-router.md) 第 8、9 节。

---

## 1. 原理与可行性

| 产物 | 构建方式 | 速度 |
| --- | --- | --- |
| `linux/amd64` 镜像/二进制 | Windows 上原生构建（本机就是 x86_64） | 快 |
| `linux/arm64` 镜像/二进制 | buildx + **QEMU 模拟** aarch64 指令集 | 明显慢（模拟执行，通常比原生慢数倍以上），但一次性任务可接受 |

关键点：

- **QEMU binfmt**：Docker Desktop 自带 QEMU，注册了 arm64 指令翻译，因此 x86 机器既能**构建** arm64 镜像，也能**运行** arm64 容器（用来验证产物）。
- 编译发生在 `rust:1-alpine` 容器内部（musl 静态编译，与主文档同一份 `Dockerfile.static`），Windows 只是"宿主"，**不需要在 Windows 上安装任何 Rust 工具链**。
- 产物是标准 Linux 镜像/ELF 二进制，Windows 只是个构建工厂，交付后与构建机无关。

本机条件核对：

| 条件 | 状态 | 备注 |
| --- | --- | --- |
| x64 CPU | x64-based PC | |
| 内存 | 是否≥32 GB | Rust release 编译含 LTO，16GB 以下容易 OOM |
| WSL2 | 未安装 | 第 2.1 节安装 |
| Docker | 未安装 | 第 2.2 节安装 |

---

## 2. 环境安装

### 2.1 启用 WSL2（Docker Desktop 的前置依赖）

以**管理员**身份打开 PowerShell：

```powershell
wsl --install
```

此命令自动启用所需的 Windows 功能并安装默认 Linux 发行版（Ubuntu）。**完成后推荐重启电脑。**

重启后若 Ubuntu 窗口按提示创建**用户名/密码**。

验证：

```powershell
wsl --status        # 默认版本应为 2
```

> 家庭中文版支持 WSL2（需要 Win10 2004 及以上，`winver` 查看）。若 `wsl --install` 报错"不支持"，先执行 `dism.exe /online /enable-feature
> /featurename:VirtualMachinePlatform /all /norestart` 并检查 BIOS 虚拟化（任务管理器→性能→CPU→"虚拟化: 已启用"）。

### 2.2 安装 Docker Desktop

1. 下载安装：<https://www.docker.com/products/docker-desktop/> （Windows 版，amd64）
2. 安装时勾选 **"Use WSL 2 instead of Hyper-V"**（家庭版只能选这个）
3. 安装完成重启，等待 Docker Desktop 左下角显示 **Engine running**（绿色）

资源设置（Docker Desktop → Settings → Resources → WSL Integration / Advanced）：

- WSL2 后端默认可吃满宿主机内存（32GB 无需调整）；若手动限制了内存，确保 ≥ 8GB
- 磁盘预留 ≥ 30GB 可用空间（镜像 + 编译缓存）

验证（PowerShell）：

```powershell
docker version          # Client + Server 都有输出
docker buildx version   # Docker Desktop 自带 buildx
```

### 2.3 确认 arm64 模拟支持（binfmt）

```powershell
# 看 arm64 是否已注册（Docker Desktop 一般自带；输出里应有 enabled: true）
docker run --rm --privileged tonistiigi/binfmt --install arm64
```

此命令幂等，重复执行无害。若拉取该镜像失败，先跳过——多数 Docker Desktop 版本已内置，直接试第 4 节的 arm64 构建，报 `exec format error` 再回来执行本步。

---

## 3. 准备源码

**建议把源码放到纯英文路径**：

```powershell
"E:\your-path\router"
cd E:\your-path\router
```

所需文件（在 `router` 源码目录内准备，与主文档一致）：

- `Dockerfile.static`、`.dockerignore`：按主文档第 4、6 节创建
- `Cargo.toml`：应用主文档第 5 节的 hf-hub 依赖改动
- `Cargo.lock`、`src/`：上游源码自带

---

## 4. 构建镜像（buildx，注意不是 docker build）

**必须用 `docker buildx build`**（传统 `docker build` 不支持跨架构）。在源码目录下执行，两条命令分开跑：

### 4.1 amd64（原生，快）

```powershell
# 先预拉基础镜像（见下方"为什么要预拉取"）
docker pull --platform linux/amd64 alpine:3.22
docker pull --platform linux/amd64 rust:1-alpine

docker buildx build --platform linux/amd64 -f Dockerfile.static -t vllm-router:v0.1.15-amd64 --load .
```

### 4.2 arm64（QEMU 模拟，会慢一些）

```powershell
docker pull --platform linux/arm64 alpine:3.22
docker pull --platform linux/arm64 rust:1-alpine

docker buildx build --platform linux/arm64 -f Dockerfile.static -t vllm-router:v0.1.15-arm64 --load .
```

要点：

- **为什么要先手动预拉基础镜像**：`docker pull` 走 Docker daemon（认 Docker Desktop 的代理配置，能成功）；
  而 buildx 的 `FROM` 解析走内置 BuildKit 引擎，基础镜像不在本地时它要直连 Docker Hub 查 manifest，
  代理不稳的环境下会被重置（报 `failed to fetch oauth token` / `connection forcibly closed`）。
  基础镜像一旦在本地 containerd 存储里（注意 amd64/arm64 是两套下载物，**每个平台都要单独拉**），
  BuildKit 解析 `FROM` 完全不联网，构建直接往下走。这是代理不稳环境下最可靠的工作流；
- `--platform` 指定目标架构；`--load` 把构建好的镜像装入本机 `docker images`（**一次只能 load 一个平台**，所以两条命令分开执行，不要写 `--platform
linux/amd64,linux/arm64`）；
- 联网拉取 `rust:1-alpine` / `alpine:3.22` / crates.io 依赖走的是 Docker Desktop 的网络（即 Windows 本机的网络），与服务器上那个坏代理无关；家庭网络若需代理，在
Docker Desktop → Settings → Resources → **Proxies** 里配 GUI，不要改 daemon.json；
- arm64 构建耗时数倍于 amd64 是 QEMU 模拟的固有开销，进度长时间停在编译步骤属正常，耐心等待；
- 构建完成后 `docker images` 应看到两个镜像，注意 arm64 镜像的 SIZE 显示与实际无参考意义。

---

## 5. 提取二进制并生成校验和

与主文档 7.3 节相同逻辑（从镜像提取，保证与镜像同源），PowerShell 版命令：

```powershell
# amd64
docker create --name vr-bin vllm-router:v0.1.15-amd64
docker cp vr-bin:/usr/local/bin/vllm-router ./vllm-router-v0.1.15-linux-amd64
docker rm vr-bin

# arm64
docker create --name vr-bin vllm-router:v0.1.15-arm64
docker cp vr-bin:/usr/local/bin/vllm-router ./vllm-router-v0.1.15-linux-arm64
docker rm vr-bin
```

校验和（生成 `sha256sum -c` 兼容格式，目标 Linux 机器可直接校验）：

<!-- markdownlint-disable MD013 --><!-- 整行单命令，无法安全换行 -->
```powershell
"$((Get-FileHash -Algorithm SHA256 .\vllm-router-v0.1.15-linux-amd64).Hash.ToLower())  vllm-router-v0.1.15-linux-amd64" | Out-File -Encoding ascii vllm-router-v0.1.15-linux-amd64.sha256
"$((Get-FileHash -Algorithm SHA256 .\vllm-router-v0.1.15-linux-arm64).Hash.ToLower())  vllm-router-v0.1.15-linux-arm64" | Out-File -Encoding ascii vllm-router-v0.1.15-linux-arm64.sha256
```
<!-- markdownlint-enable MD013 -->

---

## 6. 验证

```powershell
# 1. 版本验证（--platform 让 QEMU 运行对应架构的容器）
docker run --rm --platform linux/amd64 vllm-router:v0.1.15-amd64 --version
docker run --rm --platform linux/arm64 vllm-router:v0.1.15-arm64 --version

# 2. 静态链接验证（预期输出包含 "statically linked"）
docker run --rm -v ${PWD}:/w alpine:3.22 sh -c "apk add -q file && file /w/vllm-router-v0.1.15-linux-arm64"
docker run --rm -v ${PWD}:/w alpine:3.22 sh -c "apk add -q file && file /w/vllm-router-v0.1.15-linux-amd64"

# 3. 冒烟测试（arm64 镜像同样能在本机跑起来，这正是 QEMU 的作用）
#    启动校验要求：空 worker 列表必须 --service-discovery，而它又必须配 --selector（任意占位值）。
#    本机没有 K8s 没关系：服务发现启动失败只记 warn 日志，服务照常监听（server.rs:1000）。
#    切忌用假地址 --worker-urls http://127.0.0.1:1：启动会先等 worker 健康（最长 600s），
#    期间端口不监听，curl 得到 000。
docker run -d --name vr-smoke -p 30000:30000 vllm-router:v0.1.15-amd64 `
  --host 0.0.0.0 --port 30000 --service-discovery --selector app=vllm
Start-Sleep 5    # 等服务完成端口监听；docker run -d 刚返回就 curl 会 connection refused（-s 下表现为空输出）
docker ps --filter name=vr-smoke    # 有输出=容器在运行；无输出=已退出，用 docker logs vr-smoke 查原因
curl.exe -s -w "%{http_code}`n" http://127.0.0.1:30000/liveness   # 预期 200 OK
docker rm -f vr-smoke
```

> **端点语义**（冒烟测试用 `/liveness`）：
>
> - `/liveness`：进程活着就返回 200，与后端无关；
> - `/health`：**所有** worker 都健康才 200，否则 503 `Unhealthy servers: [...]`；
> - `/readiness`：有健康后端才 200。
> 本测试 0 个 worker：`/liveness` 200，`/health` 503，`/readiness` 非 200——都是预期行为。
> **注意：`${PWD}` 是 PowerShell 语法。** 如果你用的是 cmd 窗口（提示符是 `E:\xxx>`），
> 需把所有 `${PWD}` 换成 `%cd%`，例如：
>
> ```cmd
> docker run --rm -v %cd%:/w alpine:3.22 sh -c "apk add -q file && file /w/vllm-router-v0.1.15-linux-amd64"
> ```
>
> 否则会报错 `"${PWD}" includes invalid characters for a local volume name`。

---

## 7. 导出

每个架构 3 个文件（镜像 tar + 二进制 + sha256）：

```powershell
docker save -o vllm-router_v0.1.15_amd64.tar vllm-router:v0.1.15-amd64
docker save -o vllm-router_v0.1.15_arm64.tar vllm-router:v0.1.15-arm64
dir vllm-router*
```

拷贝（scp / U 盘 / IM 文件传输均可）到目标机器后，加载与部署完全按主文档：

```bash
# 目标机器（x86 Ubuntu 或 鲲鹏 openEuler）
docker load -i vllm-router_v0.1.15_arm64.tar
sha256sum -c vllm-router-v0.1.15-linux-arm64.sha256   # 校验二进制

# test
docker run -d --name vr-smoke -p 30000:30000 vllm-router:v0.1.15-amd64 \
  --host 0.0.0.0 --port 30000 --service-discovery --selector app=vllm
curl -s -w "%{http_code}\n" http://127.0.0.1:30000/liveness
docker rm -f vr-smoke

# 之后按主文档 9.1 docker run 或 9.2 docker compose 部署
```

> Windows 导出的 tar 与 Linux 上导出的格式相同，`docker load` 无平台差异。

---

## 8. FAQ

**1. Windows/QEMU 构建的产物，和在真实 x86/鲲鹏机器上构建的有区别吗？**
没有。QEMU 只影响"编译过程在哪执行"，不改变编译产物——arm64 二进制的指令集、链接方式与鲲鹏机原生编译结果一致（同一份源码、同一份 Dockerfile、同一个 rust:1-alpine 工具链）。如不放心，可用第 6
节命令验证 `statically linked`，并在鲲鹏目标机上执行 `--version` 确认。

**2. arm64 构建中途失败/被杀死？**
按顺序排查：① 内存不足（Docker Desktop 崩溃或容器 OOM-killed：`docker buildx build` 报 "Killed"，WSL2 默认可用一半内存，32GB 机器一般不会遇到）；②
磁盘满（Settings→Resources→Advanced 查看虚拟磁盘位置剩余空间）；③ 偶发的 QEMU 问题（重新执行构建命令，BuildKit 会从缓存的层继续，不必从头来）。

**3. 构建网络慢/拉不动 rust:1-alpine？**
Docker Desktop → Settings → Docker Engine 里加 registry-mirrors（国内加速地址），或在 Settings → Proxies 配置本机代理。改动后 Apply &
Restart（这只重启 Docker 引擎，不影响 Windows 其他程序）。

**4. `docker buildx build` 报 "exec format error"？**
binfmt 未注册 arm64。执行第 2.3 节的 `tonistiigi/binfmt --install arm64` 后重试。

**5. 构建上下文相关诡异报错（copy 失败/路径 not found）？**
多半是中文路径问题。确认已按第 3 节把源码复制到纯英文路径（如 `E:\code\router`）再构建。

**6. 公司网络必须走代理，Docker Desktop 怎么配？**
只用 GUI：Settings → Resources → Proxies，填 Web/HTTP proxy 后 Apply。**不要**去改 daemon.json 或 systemd（Windows 上没有这些东西），这也是
Windows 方式更省心的地方——不会有服务器上 `127.0.0.1:3129` 那种残留代理坑。

**7. buildx 构建缓存在哪，换目录后还有效吗？**
缓存在 Docker Desktop 的 WSL2 虚拟磁盘内，跨 PowerShell 会话持续有效；同一份 `Dockerfile.static` 重复构建会命中缓存。删除缓存放空虚拟磁盘：Settings → Resources →
Advanced → Disk image size 区域的 Clean / Purge。
