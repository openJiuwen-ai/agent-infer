#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=${AGENTCACHE_VLLM_LIFECYCLE_SOCKET:-/tmp/agentinfer-vllm-lifecycle.sock}

exec vllm serve "$MODEL" \
  --async-scheduling \
  --scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware \
  --additional-config \
  '{"agentcache":{"controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller"}}'