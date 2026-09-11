# Python API

This document describes AgentInfer's public integration surface for vLLM 0.23.0. Refer to the matching vLLM
documentation for the full upstream API.

The current serving path uses an explicit scheduler bridge, API middleware, and the embedded Progress-TTL controller.

## Scheduler Bridges

Import module:
`agentinfer.agentcache.core.scheduler`

### `AgentCacheAsyncSchedulerBridge`

```python
class AgentCacheAsyncSchedulerBridge(vllm.v1.core.sched.async_scheduler.AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None: ...
```

Transfers requests, outputs, cancellations, and Prefix Cache observations between the vLLM async scheduler and an
AgentInfer admission controller. Configuration requirements:

- vLLM must set `async_scheduling=true`; otherwise the constructor raises `ValueError`.
- `additional_config.agentcache.controller_factory` is an optional `module.attribute` import path.
- With a controller configured, lifecycle events can use `lifecycle_socket_path` or
  `AGENTCACHE_VLLM_LIFECYCLE_SOCKET`.

### `AgentCacheSyncSchedulerBridge`

```python
class AgentCacheSyncSchedulerBridge(vllm.v1.core.sched.scheduler.Scheduler):
    def __init__(self, *args, **kwargs) -> None: ...
```

The synchronous fallback bridge. vLLM must set `async_scheduling=false`; otherwise the constructor raises
`ValueError`. Controller configuration is the same as for the async bridge.

## API Middleware

Import module:
`agentinfer.agentcache.core.api_adapter`

### `AgentCacheIdentityMiddleware`

```python
class AgentCacheIdentityMiddleware:
    def __init__(self, app: AsgiApp) -> None: ...
```

Transports supported agent identity metadata from OpenAI Chat Completions and Anthropic Messages request bodies to
vLLM EngineCore. Other endpoints and requests without supported identity metadata pass through unchanged. Invalid
metadata returns an HTTP 400 response.

### `AgentCacheLifecycleMiddleware`

```python
class AgentCacheLifecycleMiddleware:
    def __init__(self, app: AsgiApp) -> None: ...
```

Observes response lifecycle after vLLM tool parsing without rewriting ASGI response messages. Its constructor reads
`AGENTCACHE_VLLM_LIFECYCLE_SOCKET` and raises `RuntimeError` when the variable is absent. Lifecycle observations are
sent to the scheduler through the configured local Unix socket.

## Controller Factory

Import path:
`agentinfer.agentcache.core.factory.build_progress_ttl_controller`

```python
def build_progress_ttl_controller(
    backend_pool_info: BackendPoolInfo,
    settings: JsonMapping,
) -> ProgramScheduler[Request, ProgressTTLGlobalFactors, ProgressTTLProgramFactors]: ...
```

Builds the embedded Progress-TTL scheduler used by the bridges. `settings` is the
`additional_config.agentcache` mapping; policy values are read from its nested `progress_ttl` object. Fixed, removed,
or unknown Progress-TTL fields raise `ValueError`.

## Compatibility Interfaces

The following APIs remain available for compatibility but do not enable the explicit Progress-TTL serving path:

| Interface | Import path | Behavior |
| --- | --- | --- |
| `AGENT_AWARE_SCHEDULER` | `agentinfer.AGENT_AWARE_SCHEDULER` | Dotted path for `AgentAwareScheduler`. |
| `LLM` | `agentinfer.LLM` | Thin `vllm.LLM` subclass with unchanged upstream signatures. |
| `AgentAwareScheduler` | `agentinfer.agentcache.core.scheduler.AgentAwareScheduler` | Replaces native waiting with `AgentAwareQueue`. |
| `AgentAwareQueue` | `agentinfer.agentcache.core.request_queue.AgentAwareQueue` | FCFS-compatible queue extension point. |

Importing `agentinfer` patches vLLM `EngineArgs` once. It selects `AgentAwareScheduler` only when `scheduler_cls` is
unset and preserves any explicit scheduler. Without vLLM, `agentinfer` remains importable but does not export `LLM`.

See [Integrate with vLLM](../how-to/integrate-vllm.md) for deployment steps and
[Architecture](../explanation/architecture.md) for component interactions.

## Replay Python entry points

`agentinfer.agentbench.replay.config.load_replay_config(path: Path) -> ReplayBenchConfig`
loads YAML and resolves paths relative to its directory. File errors raise `OSError`, malformed YAML raises
`yaml.YAMLError`, and invalid configuration raises `pydantic.ValidationError`.

`agentinfer.agentbench.replay.runner.run_replay`

```python
def run_replay(
    config: ReplayBenchConfig,
    *,
    cli_metadata: dict[str, object] | None = None,
) -> Path: ...
```

runs Replay synchronously and returns the artifact directory. `config` is resolved configuration; `cli_metadata`
is optional invocation evidence. Reserved trace types raise `NotImplementedError`; execution errors propagate after
failure artifacts are recorded. The synchronous entry uses `asyncio.run` and cannot run in a thread with an active
event loop.

## Synchronized server behavior

`AgentCacheIdentityMiddleware` transfers `seed`, `min_tokens`, and `ignore_eos` from Anthropic Replay metadata
`_agentinfer_replay_sampling` to the internal vLLM Chat request. Without that metadata, vLLM defaults are preserved.

The lifecycle sender targets the explicit `X-data-parallel-rank` when present. Without it, delivery fans out to
discovered internal-DP rank sockets to match vLLM load balancing. Invalid ranks do not emit a lifecycle signal.

Progress-TTL adds `ProgressTTLResumeOrder` (`mru` / `fcfs`), cold-prefill and decode cost models, and dynamic forced-resume
timeouts. See [Progress-TTL configuration](progress-ttl-config.md) for fields and migration notes.
