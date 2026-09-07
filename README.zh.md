# AgentInfer

AgentInfer 为智能体工作流提供高效的缓存管理和请求调度能力，既可集成到 vLLM 等 LLM 推理引擎中，也可作为
推理引擎前置的请求路由器运行。

[English](README.md)

## 核心特性

- 提供异步和同步调度桥，用于将 AgentInfer 显式集成到 vLLM。
- 通过专用 API 中间件传递智能体身份和生命周期观测。
- 使用内嵌 Progress-TTL 控制器保留、暂停和恢复智能体程序。
- 支持以内嵌调度器或推理引擎前置请求路由器两种方式部署。
- 提供 AgentBench，用于在 vLLM 部署上运行并比较可复现的智能体工作负载。

## 相关文档

[文档导航](docs/README.md) · [快速开始](docs/zh/tutorial/01-quick-start.md) ·
[vLLM 接入](docs/zh/how-to/integrate-vllm.md) · [基准测试指南](docs/zh/how-to/run-benchmark.md) ·
[变更日志](CHANGELOG.md)

## 环境要求

- 操作系统：Linux，或安装了 WSL 2 的 Windows。
- Python：3.10 或更高版本，与 `pyproject.toml` 中的 `requires-python` 保持一致。
- 推理运行时：vLLM 0.23.0。
- 硬件：与 vLLM 兼容且能够运行所选模型的 CUDA GPU。

AgentInfer 必须与 vLLM 安装在同一 Python 环境中。

## 安装指南

### 从源码安装

开发环境中，先激活目标 vLLM 环境，再以可编辑模式安装 AgentInfer：

```bash
git clone https://gitcode.com/openJiuwen/agent-infer.git
cd agent-infer
python -m pip install -e .
```

### 安装发布包

下载 [AgentInfer 0.1.0 wheel][release-wheel]，然后在目标 vLLM 环境中安装：

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

## Quick Start

选择一个本地 Unix socket 传递 API 生命周期信号，然后使用 AgentInfer 异步调度桥、身份和生命周期中间件以及
内嵌 Progress-TTL 控制器启动 vLLM：

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

仓库中的等效示例位于 [`examples/serve-progress-ttl.sh`](examples/serve-progress-ttl.sh)。

安装后的 `vllm` 命令会将普通命令委托给上游 vLLM。显式的 `vllm bench serve --agentinfer` 命令进入
AgentBench；其他命令在未显式设置 `--scheduler-cls` 时使用 AgentInfer 的默认调度器。标准服务参数见
[vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)。

## License

本项目基于 [Apache License 2.0](LICENSE) 开源。

[release-wheel]: https://gitcode.com/openJiuwen/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
