# 架构概述

AgentInfer 在 vLLM 原生调度和 KV Cache 所有权之外增加智能体工作流感知能力。它不替代模型执行、
token 级调度或物理 KV 管理，而是在请求进入原生等待队列前提供可扩展的准入决策。

## 集成层次

```text
客户端请求
                |
                v
AgentCacheIdentityMiddleware
                |
                v
vLLM API ---- AgentCacheLifecycleMiddleware ----> Unix lifecycle socket
                |                                                |
                v                                                v
AgentCacheAsyncSchedulerBridge <------------> Progress-TTL 控制器
                |
                v
vLLM 原生 waiting / running / KV 状态
```

当前服务路径显式组合调度桥、身份中间件、生命周期中间件和 Progress-TTL 控制器。控制器决定已识别的智能体
请求何时进入 vLLM 原生等待队列。

## 显式服务路径

每个配置组件负责一个独立边界：

- `AgentCacheIdentityMiddleware` 从 OpenAI Chat 或 Anthropic Messages 请求体解析受支持的身份，并传递到
        EngineCore。
- `AgentCacheLifecycleMiddleware` 在工具解析后观测响应完成事件，并通过本地 Unix socket 发出生命周期事实。
- `AgentCacheAsyncSchedulerBridge` 将 vLLM 异步调度连接到 AgentInfer 准入决策，并要求
        `async_scheduling=true`。
- `build_progress_ttl_controller` 根据 `additional_config.agentcache` 构建内嵌调度策略。

生命周期中间件要求设置 `AGENTCACHE_VLLM_LIFECYCLE_SOCKET`。除非
`additional_config.agentcache.lifecycle_socket_path` 显式覆盖，调度桥会从环境读取同一路径。

## Progress-TTL 调度桥

`AgentCacheAsyncSchedulerBridge` 和 `AgentCacheSyncSchedulerBridge` 通过组合辅助对象连接 vLLM 与
`EmbeddedSchedulerController`。控制器负责：

- 在原生准入前保留带智能体身份的请求。
- 根据后端容量和工作流进度执行策略周期。
- 向原生等待队列释放已准入请求，并保留原始排队时间。
- 观测流式输出、请求完成和可复用 Prefix Cache token。
- 取消尚未进入 vLLM 的请求，并将生命周期事件应用到对应 Program。

vLLM 继续拥有原生等待和运行集合、请求完成语义、模型执行以及 KV block 分配。AgentInfer 只读取受支持的
容量与 Prefix Cache 观测，不创建第二套物理 KV 状态。

## 生命周期边界

API 中间件将响应完成后解析出的工作流生命周期写入 Unix socket。每个调度桥在准入或策略周期前排空本地
receiver，并将事件交给控制器。socket 可以通过 `additional_config.agentcache.lifecycle_socket_path` 或
`AGENTCACHE_VLLM_LIFECYCLE_SOCKET` 配置。

该边界避免 HTTP 层直接修改调度器对象，也使多进程 vLLM 部署按数据并行 rank 消费对应事件。

## CLI 委托与兼容路径

AgentInfer 会安装一个 `vllm` 控制台脚本。显式的 `vllm bench serve --agentinfer` 命令进入 AgentBench；其他
命令原样委托给上游 vLLM。导入 AgentInfer 包时会修补一次 `EngineArgs`，因此未设置的 `scheduler_cls` 默认
选择旧版 `AgentAwareScheduler`，显式调度器则保持不变。

`AgentAwareScheduler`、`AgentAwareQueue` 和 `agentinfer.LLM` 仍作为兼容 FCFS 的扩展接口保留，但不会启用
Progress-TTL 控制器。当前部署应显式配置相应调度桥、中间件和控制器工厂。

## 基准子系统

BenchKit 是独立的验证工具，不是生产服务依赖。它管理配置、数据集、工作区、Agent 进程、Request Proxy、
证据采集和结果比较；生产调度模块不得导入 `agentinfer.agentbench`。

部署方法见[接入 vLLM](../how-to/integrate-vllm.md)，接口签名见[Python API 参考](../reference/python-api.md)，
开发边界和不变量见[内部设计索引](design-documents.md)。
