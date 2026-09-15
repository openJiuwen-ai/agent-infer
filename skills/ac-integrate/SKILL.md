---
name: ac-integrate
description: Use when plugging AgentInfer AgentCache into a vLLM serving deployment; discovers the installed integration surface, checks runtime capabilities, configures the adapter, and validates cache evidence.
allowed-tools:
  - Bash
  - Read
  - Write
  - Grep
  - Glob
  - Edit
---

# ac-integrate

Wire AgentInfer AgentCache into a vLLM serving deployment using the current
scheduler bridge and lifecycle middleware, version-check the engine, and
validate cache behavior from captured evidence.

## WHEN TO INVOKE

- The user asks to "integrate AgentInfer with vLLM",
  "plug AgentCache into my deployment", or "wire the scheduler bridge"
  for a serving engine.
- A new engine integration is being added to the project's scripts and configs.

Do NOT invoke for: adding a cache backend (use `ac-bootstrap`), benchmarking
(use `ac-benchmark`), or editing integration docs only.

## STEPS

1. **Detect the target engine and version before changing configuration**:

   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

   If vLLM is not importable, stop and report the missing environment instead of
   continuing with guessed APIs. Read the current checkout's requirements and
   runbook for its supported version (currently `0.23.0`). If the installed
   version differs, stop and resolve or explicitly approve the mismatch before
   relying on scheduler, middleware, or Prefix Cache APIs.

2. **Select deployment parameters from the target environment**. Read the
   current runbook and deployment config for model, tensor parallelism, tool
   parser, port, and environment variables. The following is the current Q3
   Linux/WSL baseline example, not a universal deployment recipe:

   ```bash
   export VLLM_ENABLE_CUDA_COMPATIBILITY=1
   export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
   export MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8

   vllm serve "$MODEL" \
     --tensor-parallel-size 2 \
     --enable-prompt-tokens-details \
     --enable-prefix-caching \
     --enable-auto-tool-choice \
     --tool-call-parser qwen3_coder
   ```

   Adapt shell syntax and socket paths for the deployment platform. The current
   benchmark launch flow assumes Linux or WSL because it uses Unix domain
   sockets and process/session tooling.

3. **Discover and preflight the installed integration surface**. Read the current
   checkout and runbook, then verify every selected symbol imports before launch.
   The repository package may expose only the embedded scheduler, while a
   separately supplied AgentInfer runtime build provides lifecycle middleware
   and progress-TTL integration.

   Current surfaces to inspect:

   - BenchKit CLI: `vllm bench serve --agentinfer`
   - Cache/vLLM adapters: `agentinfer/agentcache/`
   - Engine-neutral scheduling contracts: `agentinfer/scheduling/`
   - Focused integration tests: `tests/agentcache/` and `tests/agentbench/`

   For the external runtime-backed candidate documented in the current runbook,
   preflight its required symbols:

   ```bash
   python - <<'PY'
   from agentinfer.agentcache.core.api_adapter import (
       AgentCacheIdentityMiddleware,
       AgentCacheLifecycleMiddleware,
   )
   from agentinfer.agentcache.core.factory import build_progress_ttl_controller
   from agentinfer.agentcache.core.scheduler import AgentCacheAsyncSchedulerBridge

   print(AgentCacheAsyncSchedulerBridge)
   print(AgentCacheIdentityMiddleware, AgentCacheLifecycleMiddleware)
   print(build_progress_ttl_controller)
   PY
   ```

   If imports fail, stop: install/select the required runtime build or choose an
   integration path that is actually present in the checkout. Do not present the
   external runtime recipe as provided by the repository package alone.

   After preflight succeeds, follow `docs/en/how-to/run-benchmark.md` for the exact
   scheduler, middleware, lifecycle socket, and progress-TTL launch flags. Record
   the resolved classes and config instead of copying an old command. Preserve
   the boundary between engine-specific adapters and `agentinfer/scheduling/`
   contracts.

4. **Validate from benchmark evidence**, not request timing alone. Run identical
   cold baseline and candidate workloads through the Request Proxy, then compare:

   ```bash
   vllm bench serve --agentinfer compare \
     --baseline results/vllm/run1 \
     --candidate results/agentinfer/run1
   ```

   Inspect `manifest.json`, `summary.json`, `requests.jsonl`, and
   `evidence/vllm_metrics_*.prom`. Confirm the expected scheduler/middleware
   loaded, lifecycle events were captured, cold state is known, and cache metrics
   differ as expected. A faster repeated request by itself is not proof of a
   cache hit.

5. **Document deployment specifics**: engine/version, scheduler and middleware
   classes, endpoint shape, lifecycle socket, controller configuration, service
   logs, and artifact paths so the integration is reproducible.

## DON'T

- Don't assume a vLLM version's scheduler or cache API without checking it.
- Don't continue after a version or runtime-capability preflight fails.
- Don't present external runtime classes as bundled with the repository package.
- Don't omit middleware required by the selected, successfully preflighted path.
- Don't mutate the user's deployment config files without explicit confirmation.
- Don't infer cache hits from timing alone; require metrics and run artifacts.
- Don't declare integration done until the evidence validates the expected path.

## AFTER

Once the selected adapter path is wired and cache evidence is validated, invoke
`ac-review` to validate the diff against repo conventions before opening the PR.
