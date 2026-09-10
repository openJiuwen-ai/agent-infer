# Python API 参考

本文记录 AgentInfer 与 vLLM 0.23.0 的公开集成接口。vLLM 自身 API 的完整参数以对应版本的 vLLM 文档为准。

当前服务路径使用显式调度桥、API 中间件和内嵌 Progress-TTL 控制器。

## 调度桥

导入模块：
`agentinfer.agentcache.core.scheduler`

### `AgentCacheAsyncSchedulerBridge`

```python
class AgentCacheAsyncSchedulerBridge(vllm.v1.core.sched.async_scheduler.AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None: ...
```

在 vLLM 异步调度器和 AgentInfer 准入控制器之间传递请求、输出、取消和 Prefix Cache 观测。配置要求：

- vLLM 必须启用 `async_scheduling=true`，否则构造函数抛出 `ValueError`。
- `additional_config.agentcache.controller_factory` 是可选的 `module.attribute` 导入路径。
- 配置控制器后，可通过 `lifecycle_socket_path` 或 `AGENTCACHE_VLLM_LIFECYCLE_SOCKET` 接收生命周期事件。

### `AgentCacheSyncSchedulerBridge`

```python
class AgentCacheSyncSchedulerBridge(vllm.v1.core.sched.scheduler.Scheduler):
    def __init__(self, *args, **kwargs) -> None: ...
```

同步调度回退桥。vLLM 必须设置 `async_scheduling=false`，否则构造函数抛出 `ValueError`。控制器配置与异步桥
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

## 控制器工厂

导入路径：
`agentinfer.agentcache.core.factory.build_progress_ttl_controller`

```python
def build_progress_ttl_controller(
    backend_pool_info: BackendPoolInfo,
    settings: JsonMapping,
) -> ProgramScheduler[Request, ProgressTTLGlobalFactors, ProgressTTLProgramFactors]: ...
```

构建调度桥使用的内嵌 Progress-TTL 调度器。`settings` 是 `additional_config.agentcache` 映射，策略参数从
嵌套的 `progress_ttl` 对象读取。配置固定、已移除或未知的 Progress-TTL 字段时抛出 `ValueError`。

## 兼容接口

以下 API 保留用于兼容，但不会启用显式 Progress-TTL 服务路径：

| 接口 | 导入路径 | 行为 |
| --- | --- | --- |
| `AGENT_AWARE_SCHEDULER` | `agentinfer.AGENT_AWARE_SCHEDULER` | `AgentAwareScheduler` 的点分路径。 |
| `LLM` | `agentinfer.LLM` | 不修改上游签名的 `vllm.LLM` 薄封装。 |
| `AgentAwareScheduler` | `agentinfer.agentcache.core.scheduler.AgentAwareScheduler` | 将原生等待队列替换为 `AgentAwareQueue`。 |
| `AgentAwareQueue` | `agentinfer.agentcache.core.request_queue.AgentAwareQueue` | 兼容 FCFS 的队列扩展点。 |

导入 `agentinfer` 时会修补一次 vLLM `EngineArgs`。仅当 `scheduler_cls` 未设置时选择
`AgentAwareScheduler`，并保留任何显式调度器。缺少 vLLM 时，`agentinfer` 仍可导入，但不会导出 `LLM`。

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
