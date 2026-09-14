# CI Pipeline - 构建工具库

## 项目概述

Python版本的构建工具库，支持本地构建和 Jenkins 流水线构建。

### 核心功能

- **动态构建代码**: Jenkinsfile 动态拉取最新构建代码
- **组件化构建**: 支持独立构建多个组件
- **镜像构建**: 支持构建Docker镜像和Helm包
- **配置管理**: 通过config.yaml管理构建参数
- **多架构支持**: 支持x86_64和aarch64架构

## 架构说明

### 新架构（推荐）

**特点**：

- ✅ Jenkinsfile 随业务代码走
- ✅ 构建代码动态拉取最新版本
- ✅ 配置文件分散到各业务项目
- ✅ 无需扫描 Jenkins 仓库

**使用方式**：

1. 业务项目根目录放置 `Jenkinsfile`
2. 业务项目维护自己的配置文件
3. Jenkinsfile 动态拉取 `ci-pipelines` 构建代码

**目录结构**：

```text
ci-pipelines/                    # 构建代码库
├── compile/                     # 构建脚本
├── vars/                        # 共享函数（预留，当前为空）
├── configs/                     # 配置模板
├── pipelines/                   # Jenkinsfile 模板
│   ├── Jenkinsfile-template    # 业务项目模板
│   └── Jenkinsfile-standard    # 标准流水线
└── MIGRATION.md                 # 迁移指南

your-business-project/           # 业务项目
├── Jenkinsfile                  # 项目流水线
├── jenkins-config/              # 项目配置
│   ├── config-master.yaml
│   ├── config-test.yaml
│   └── config-dev.yaml
└── src/                         # 业务代码
```

### 快速开始

#### 1. 为你的项目添加 Jenkinsfile

```bash
# 复制模板到你的项目
cp ci-pipelines/pipelines/Jenkinsfile-template your-project/Jenkinsfile

# 创建配置目录
mkdir -p your-project/jenkins-config
cp ci-pipelines/configs/config-template.yaml your-project/jenkins-config/config-master.yaml
```

#### 2. 修改配置文件

编辑 `jenkins-config/config-master.yaml`，根据项目需求修改组件配置。

#### 3. 提交到仓库

```bash
git add Jenkinsfile jenkins-config/
git commit -m "Add Jenkins pipeline"
git push
```

#### 4. 创建 Jenkins 任务

在 Jenkins 中创建 Multibranch Pipeline 任务，指向你的业务仓库。

详细步骤请参考 ci-pipelines 仓库文档

## 项目结构

```text
ci-pipelines/
├── README.md                    # 项目说明文档
├── config.yaml                 # 构建配置文件
├── build.py                    # 组件编译入口
├── compile/                    # 组件编译
│   ├── build.py                # 组件构建脚本
│   ├── utils/                  # 工具函数
│   ├── ray-adapter.sh          # Ray适配器构建脚本
│   ├── yuanrong-datasystem.sh  # 数据系统构建脚本
│   ├── yuanrong-functionsystem.sh # 函数系统构建脚本
│   ├── yuanrong-frontend.sh    # 前端构建脚本
│   ├── yuanrong.sh             # 主构建脚本
│   └── spring-adapter.sh        # Spring适配器构建脚本
└── k8s/                        # K8s部署
    ├── build.py                # K8s构建入口
    ├── build_docker.sh         # Docker镜像构建
    └── build_helm.sh           # Helm包构建
```

## 系统要求

- Linux系统 (x86_64/aarch64)
- Python 3.8+
- PyYAML (用于配置文件解析)
- Git
- Docker (用于镜像构建)
- Helm (用于打包)
- 构建工具链 (gcc, make, maven等)

安装PyYAML:

```bash
pip install pyyaml
```

## 使用指南

### 1. 配置构建参数

编辑 `config.yaml` 文件：

```yaml
# 工作目录
workspace: ""

# 构建版本
build_version: "9.9.9"

# Docker配置
docker:
  image_repo: "swr.cn-southwest-2.myhuaweicloud.com/yuanrong-dev"
  image_tag: "daily"
  base_image: "swr.cn-southwest-2.myhuaweicloud.com/yuanrong-dev/base-image-common:202603201044"
  base_runtime_image: "swr.cn-southwest-2.myhuaweicloud.com/yuanrong-dev/base-image-runtime:202603201044"

# Helm配置
helm:
  image_repo: "swr.cn-southwest-2.myhuaweicloud.com/yuanrong-dev"
  image_tag: "daily"
  chart_version: "1.0"
  obs_charts_path: "charts_dev"

# 组件配置
components:
  ray-adapter:
    repo: "https://gitcode.com/openeuler/ray-adapter.git"
    branch: "master"
    script: "ray-adapter.sh"
```

### 2. 编译组件

```bash
# 构建所有组件（按依赖顺序）
python3 build.py --all

# 构建单个组件
python3 build.py -n yuanrong-datasystem

# 指定版本
python3 build.py -n yuanrong-datasystem -v 1.0.0

# 指定Git分支
python3 build.py -n yuanrong-datasystem -b develop

# 设置工作目录
python3 build.py --all --workspace /tmp/build
```

### 3. 构建Docker镜像和Helm包

#### 流程说明

```text
组件编译 → 生成tar.gz压缩包 → 构建Docker镜像/Helm包
```

#### 步骤一：编译组件生成压缩包

```bash
python3 compile/build.py --all
```

编译完成后，压缩包位于 `{workspace}/yuanrong/output/openyuanrong-*.tar.gz`

#### 步骤二：构建Docker镜像

```bash
# 使用Python脚本（推荐）
python3 k8s/build.py docker -p /path/to/openyuanrong.tar.gz -a

# 或直接使用Shell脚本
cd k8s
./build_docker.sh -p /path/to/openyuanrong.tar.gz -a
```

#### 步骤三：构建Helm包

```bash
# 使用Python脚本（推荐）
python3 k8s/build.py helm -p /path/to/openyuanrong.tar.gz

# 或直接使用Shell脚本
cd k8s
./build_helm.sh -p /path/to/openyuanrong.tar.gz -v 1.0.0
```

## 组件说明

### 编译顺序（按依赖关系）

1. **ray-adapter** - Ray适配器
2. **yuanrong-datasystem** - 数据系统
3. **yuanrong-functionsystem** - 函数系统（依赖datasystem）
4. **yuanrong-frontend** - 前端
5. **yuanrong** - 主程序（依赖functionsystem、datasystem、frontend）
6. **spring-adapter** - Spring适配器（最后编译）

### 命令行参数

#### 组件编译 (compile/build.py)

```bash
  --all                构建所有组件
  -n, --name           组件名称
  -v, --version        构建版本
  -b, --branch         Git分支
  --workspace          工作目录
  --config             配置文件路径
```

#### K8s构建 (k8s/build.py)

```bash
  docker                构建Docker镜像
  helm                  构建Helm包
  -p, --package         压缩包路径或URL（必需）
  -a, --all             构建所有（仅docker）
  -n, --name            组件名（仅docker）
  -v, --version         Helm chart版本
```

## 注意事项

- 组件间有严格的依赖关系，建议使用 `--all` 参数按顺序构建
- 确保 `workspace` 目录有足够的磁盘空间
- ARM架构构建datasystem和functionsystem需要CANN支持
- 构建前请确保已安装所有必要的构建工具链
- Docker镜像和Helm包输出目录: `k8s/output/`
