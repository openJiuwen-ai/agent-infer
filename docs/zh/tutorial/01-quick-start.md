# 快速开始

本教程将安装 AgentInfer，并使用异步调度桥、API 中间件和内嵌 Progress-TTL 控制器启动 vLLM。请在
Linux 或 WSL 2 中运行，并准备能够运行所选模型的 CUDA GPU。

## 1. 准备 vLLM 环境

创建或激活运行 vLLM 的 Python 环境。AgentInfer 支持 Python 3.10 或更高版本以及 vLLM 0.23.0：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "vllm==0.23.0"
```

如果 CUDA 环境需要特定平台的安装方式，请按
[vLLM 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/)安装 `0.23.0` 版本。

## 2. 安装 AgentInfer

开发环境中，在已激活的 vLLM 环境内克隆仓库并以可编辑模式安装：

```bash
git clone https://gitcode.com/openJiuwen/agent-infer.git
cd agent-infer
python -m pip install -e .
```

发布环境中，下载 [AgentInfer 0.1.0 wheel][release-wheel]，激活目标 vLLM 环境并执行：

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

## 3. 启动 vLLM

选择一个未使用的本地 Unix socket 传递生命周期信号，然后使用当前 AgentInfer 服务栈启动 vLLM：

```bash
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle.sock

vllm serve meta-llama/Llama-3.1-8B-Instruct \
 --async-scheduling \
 --scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge \
 --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
 --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware \
 --additional-config \
 '{"agentcache":{"controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller"}}'
```

在源码仓中，[`examples/serve-progress-ttl.sh`](../../../examples/serve-progress-ttl.sh) 会运行相同服务配置。
需要更换默认模型时，设置 `MODEL=<model-name>`。

模型名称仅作示例，可替换为 vLLM 支持的其他模型。使用 `AgentCacheAsyncSchedulerBridge` 时必须保留
`--async-scheduling`；异步调度被禁用时，调度桥会拒绝该配置。

## 4. 验证服务

在另一个终端中等待模型加载完成，然后查询 vLLM 模型端点：

```bash
curl http://127.0.0.1:8000/v1/models
```

响应应列出 `meta-llama/Llama-3.1-8B-Instruct`。标准请求示例和服务参数见
[vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)。

接下来可按[接入 vLLM](../how-to/integrate-vllm.md)复用现有服务，或按
[运行基准测试](../how-to/run-benchmark.md)比较调度方案。

[release-wheel]: https://gitcode.com/openJiuwen/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
