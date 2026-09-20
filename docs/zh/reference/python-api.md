# Python API 参考

本文记录 AgentInfer 与 vLLM 0.23.0 的公开集成接口。vLLM 自身 API 的完整参数以对应版本的 vLLM 文档为准。

当前服务路径使用显式的 Agent 感知调度器、API 中间件和内嵌 Progress-TTL 控制器。

## Agent 感知调度器

导入模块：
`agentinfer.agentcache.core.scheduler`

### `AgentCacheAsyncSchedulerBridge`

```python
class AgentCacheAsyncSchedulerBridge(vllm.v1.core.sched.async_scheduler.AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None: ...
```

在 vLLM 异步调度器和 AgentInfer 准入控制器之间传递请求、输出、取消和 Prefix Cache 观测。配置要求：

- vLLM 必须启用 `async_scheduling=true`，否则构造函数抛出 `ValueError`。
- 桥接器直接根据 `additional_config.agentcache` 构建内嵌 Progress-TTL 控制器。
- 生命周期事件可通过 `lifecycle_socket_path` 或 `AGENTCACHE_VLLM_LIFECYCLE_SOCKET` 接收。

### `AgentCacheSyncSchedulerBridge`

```python
class AgentCacheSyncSchedulerBridge(vllm.v1.core.sched.scheduler.Scheduler):
    def __init__(self, *args, **kwargs) -> None: ...
```

同步回退版本。vLLM 必须设置 `async_scheduling=false`，否则构造函数抛出 `ValueError`。控制器配置与异步版本
相同。

## API 中间件

导入模块：
`agentinfer.agentcache.core.api_adapter`

### `AgentCacheIdentityMiddleware`

```python
class AgentCacheIdentityMiddleware:
    def __init__(self, app: AsgiApp) -> None: ...
```

将 OpenAI Chat Completions 和 Anthropic Messages 请求体中受支持的智能体身份元数据传递到 vLLM
EngineCore。其他端点和不含受支持身份元数据的请求保持不变；无效元数据返回 HTTP 400。

### `AgentCacheLifecycleMiddleware`

```python
class AgentCacheLifecycleMiddleware:
    def __init__(self, app: AsgiApp) -> None: ...
```

在 vLLM 完成工具解析后观测响应生命周期，不重写 ASGI 响应消息。构造函数读取
`AGENTCACHE_VLLM_LIFECYCLE_SOCKET`，缺少该变量时抛出 `RuntimeError`。生命周期观测通过配置的本地 Unix
socket 发送给调度器。

## 控制器构建

导入路径：
`agentinfer.agentcache.core.controller.build_progress_ttl_controller`

```python
def build_progress_ttl_controller(
    backend_pool_info: BackendPoolInfo,
    settings: JsonMapping,
) -> ProgramScheduler[Request, ProgressTTLGlobalFactors, ProgressTTLProgramFactors]: ...
```

构建 Agent 感知调度器直接使用的内嵌 Progress-TTL 调度器。`settings` 是 `additional_config.agentcache` 映射，策略参数从
嵌套的 `progress_ttl` 对象读取。配置固定、已移除或未知的 Progress-TTL 字段时抛出 `ValueError`。

具体部署步骤见[接入 vLLM](../how-to/integrate-vllm.md)，组件协作关系见
[架构概述](../explanation/architecture.md)。

## Replay Python 入口

`agentinfer.agentbench.replay.config.load_replay_config(path: Path) -> ReplayBenchConfig`
读取 YAML 并相对于文件目录解析路径。文件读取失败抛出 `OSError`，无效 YAML 抛出 `yaml.YAMLError`，
配置不符合模型时抛出 `pydantic.ValidationError`。

`agentinfer.agentbench.replay.runner.run_replay`

```python
def run_replay(
    config: ReplayBenchConfig,
    *,
    cli_metadata: dict[str, object] | None = None,
) -> Path: ...
```

同步运行回放并返回产物目录。`config` 是已解析配置，`cli_metadata` 是可选的调用证据；
预留 trace 类型抛出 `NotImplementedError`，执行错误在记录失败产物后向调用方传播。
该同步入口内部使用 `asyncio.run`，不能在已有事件循环的线程内调用。

## Replay 采样参数

Anthropic Replay 元数据 `_agentinfer_replay_sampling` 中的 `seed`、`min_tokens` 和 `ignore_eos`
会传递到 vLLM 内部 Chat 请求；不含该元数据时保留 vLLM 默认采样行为。
