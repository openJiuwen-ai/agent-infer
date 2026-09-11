# 接入 vLLM

本指南将当前 AgentInfer 服务栈接入已有的 vLLM 0.23.0 部署。AgentInfer 必须与 vLLM 安装在同一 Python
环境中。

## 将 AgentInfer 安装到 vLLM 环境

源码开发时，激活 vLLM 服务使用的环境，并以可编辑模式安装 AgentInfer：

```bash
python -m pip install -e .
```

发布部署时，下载 [AgentInfer 0.1.0 wheel][release-wheel]，然后执行：

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

确认两个包由同一解释器加载：

```bash
python -c "import agentinfer, vllm; print(vllm.__version__)"
```

命令应输出 `0.23.0`。

## 配置生命周期传输

为生命周期观测选择一个唯一的本地 Unix socket。缺少该变量时，生命周期中间件会拒绝启动：

```bash
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle.sock
```

不得复用其他服务正在使用的 socket。只有确认原进程已经停止后，才能删除遗留 socket。

## 启动 AgentInfer 服务路径

使用异步调度桥、两个 API 中间件和 Progress-TTL 控制器工厂启动 vLLM：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --async-scheduling \
  --scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware \
  --additional-config \
  '{"agentcache":{"controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller"}}'
```

根据部署需要追加张量并行、端口、模型专用工具解析和 Prefix Cache 等标准 vLLM 参数。上游参数见
[vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)。

## 了解 CLI 委托行为

AgentInfer 会安装一个委托型 `vllm` 控制台脚本：

- 显式的 `vllm bench serve --agentinfer` 命令进入 AgentBench。
- 其他命令委托给上游 vLLM，未设置 `--scheduler-cls` 时使用 AgentInfer 默认调度器。
- 显式传入的 `--scheduler-cls` 会被保留，如上面的服务启动命令所示。

旧版 `AgentAwareScheduler` 和 `agentinfer.LLM` 兼容接口仍然可用，但不会启用这里展示的显式
Progress-TTL 服务路径。

## 验证接入

模型加载完成后，在另一个终端查询服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

组件职责和选择关系见[架构概述](../explanation/architecture.md)，类签名见
[Python API 参考](../reference/python-api.md)。如需进行可复现的基线和候选组比较，请阅读
[运行基准测试](run-benchmark.md)。

[release-wheel]: https://gitcode.com/openJiuwen/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
