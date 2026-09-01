# Architecture

AgentInfer adds agent-workflow awareness around vLLM's native scheduling and KV Cache ownership. It does not replace
model execution, token-level scheduling, or physical KV management. Instead, it provides extensible admission
decisions before requests enter the native waiting queue.

## Integration Layers

```text
client request
        |
        v
AgentCacheIdentityMiddleware
        |
        v
vLLM API ---- AgentCacheLifecycleMiddleware ----> Unix lifecycle socket
        |                                                |
        v                                                v
AgentCacheAsyncSchedulerBridge <------------> Progress-TTL controller
        |
        v
native vLLM waiting / running / KV state
```

The current serving path explicitly composes the scheduler bridge, identity middleware, lifecycle middleware, and
Progress-TTL controller. The controller decides when identified agent requests enter vLLM's native waiting queue.

## Explicit Serving Path

Each configured component owns a distinct boundary:

- `AgentCacheIdentityMiddleware` resolves supported identity from OpenAI Chat or Anthropic Messages bodies and
  transports it to EngineCore.
- `AgentCacheLifecycleMiddleware` observes response completion after tool parsing and emits lifecycle facts through a
  local Unix socket.
- `AgentCacheAsyncSchedulerBridge` connects vLLM async scheduling to AgentInfer admission decisions. It requires
  `async_scheduling=true`.
- `build_progress_ttl_controller` constructs the embedded scheduling policy from `additional_config.agentcache`.

The lifecycle middleware requires `AGENTCACHE_VLLM_LIFECYCLE_SOCKET`. The bridge reads the same socket path from the
environment unless `additional_config.agentcache.lifecycle_socket_path` overrides it.

## Progress-TTL Scheduler Bridges

`AgentCacheAsyncSchedulerBridge` and `AgentCacheSyncSchedulerBridge` connect vLLM to an
`EmbeddedSchedulerController` through a composed helper. The controller:

- Retains requests with agent identity before native admission.
- Runs strategy cycles using backend capacity and workflow progress.
- Releases admitted requests to native waiting while preserving their original queue time.
- Observes streaming output, request completion, and reusable Prefix Cache tokens.
- Cancels requests not yet admitted to vLLM and applies lifecycle events to the corresponding Program.

vLLM continues to own native waiting and running sets, completion semantics, model execution, and KV block allocation.
AgentInfer consumes supported capacity and Prefix Cache observations without creating a second physical KV state.

## Lifecycle Boundary

API middleware writes workflow lifecycle facts parsed after response completion to a Unix socket. Each scheduler
bridge drains its local receiver before admission or strategy work and passes events to the controller. Configure the
socket through `additional_config.agentcache.lifecycle_socket_path` or `AGENTCACHE_VLLM_LIFECYCLE_SOCKET`.

This boundary prevents the HTTP layer from mutating scheduler objects directly and lets multi-process vLLM
deployments consume events for the matching data-parallel rank.

## CLI Delegation and Compatibility

AgentInfer installs a `vllm` console script. Explicit `vllm bench serve --agentinfer` commands enter AgentBench; other
commands delegate unchanged to upstream vLLM. Importing the AgentInfer package patches `EngineArgs` once, so an unset
`scheduler_cls` defaults to the legacy `AgentAwareScheduler`, while an explicit scheduler is preserved.

`AgentAwareScheduler`, `AgentAwareQueue`, and `agentinfer.LLM` remain FCFS-compatible extension surfaces. They do not
enable the Progress-TTL controller. Current deployments should explicitly configure the appropriate scheduler bridge,
middleware, and controller factory.

## Benchmark Subsystem

BenchKit is an independent validation tool, not a production service dependency. It manages configuration, datasets,
workspaces, agent processes, the Request Proxy, evidence collection, and comparison. Production scheduler modules
must not import `agentinfer.agentbench`.

See [Integrate with vLLM](../how-to/integrate-vllm.md) for deployment,
[Python API](../reference/python-api.md) for signatures, and
[Internal design documents](design-documents.md) for development boundaries and invariants.
