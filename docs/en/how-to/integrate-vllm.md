# Integrate with vLLM

This guide adds the current AgentInfer serving stack to an existing vLLM 0.23.0 deployment. AgentInfer and vLLM must
be installed in the same Python environment.

## Install AgentInfer into the vLLM Environment

For a source checkout, activate the environment used by the vLLM service and install AgentInfer in editable mode:

```bash
python -m pip install -e .
```

For a release deployment, download the [AgentInfer 0.1.0 wheel][release-wheel], then run:

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

Confirm that both packages resolve from the same interpreter:

```bash
python -c "import agentinfer, vllm; print(vllm.__version__)"
```

The command should print `0.23.0`.

## Configure Lifecycle Transport

Choose a unique local Unix socket for lifecycle observations. The lifecycle middleware refuses to start when this
variable is absent:

```bash
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-vllm-lifecycle.sock
```

Do not reuse a socket owned by another server. Remove a stale socket only after confirming that its previous process
has stopped.

## Start the AgentInfer Serving Path

Start vLLM with the async bridge, both API middleware components, and the Progress-TTL controller factory:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --async-scheduling \
  --scheduler-cls agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware \
  --additional-config \
  '{"agentcache":{"controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller"}}'
```

Append standard vLLM options such as tensor parallelism, port selection, model-specific tool parsing, and Prefix Cache
configuration as required by the deployment. See the
[vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) for upstream options.

### Configure Progress-TTL policy values

Policy settings belong to `additional_config.agentcache.progress_ttl`. The following example shows the principal
deployment-calibrated values and the work-ahead starvation bound introduced by the current policy:

```bash
--additional-config '{
  "agentcache": {
    "controller_factory": "agentinfer.agentcache.core.factory.build_progress_ttl_controller",
    "progress_ttl": {
      "ttl_min_seconds": 0.05,
      "ttl_max_seconds": 32,
      "resume_capacity_ratio": 1.0,
      "resume_order": "mru",
      "ttl_prefill_model_intercept_seconds": 0.042935,
      "ttl_prefill_model_linear_seconds_per_1k_tokens": 0.080027,
      "ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared": 0.00220962,
      "decode_step_fixed_seconds": 0.012112,
      "decode_step_seconds_per_request": 0.0006939,
      "decode_step_seconds_per_context_token": 2.2516e-7,
      "force_resume_timeout_scale": 3.0,
      "force_resume_timeout_min_seconds": 30,
      "force_resume_timeout_max_seconds": 300
    }
  }
}'
```

The Prefill and Decode coefficients are deployment-specific and should be obtained by calibration. Force-resume now
freezes a deadline when a request first waits: it divides the remaining rounds ahead of the request by recent
aggregate request throughput, multiplies the result by `force_resume_timeout_scale`, and clamps it to the configured
minimum and maximum. Until the bounded throughput window is complete, it conservatively uses the maximum timeout.

The policy rejects removed legacy keys instead of silently ignoring them. Replace
`ttl_prefill_seconds_per_1k_uncached_tokens` with the three quadratic Prefill coefficients and replace
`force_resume_timeout_seconds` with the scale, minimum, and maximum shown above. Remove
`target_min_segment_rounds`, `privileged_lookahead_rounds`, `pause_capacity_lookahead_rounds`,
`resume_fairness_weight`, `resume_resource_penalty_weight`, `capacity_safety_margin_tokens`,
`ttl_impact_multiplier`, `uncached_ratio_default`, and configurable `decode_buffer_tokens`.

## Understand CLI Delegation

AgentInfer installs a delegating `vllm` console script:

- Explicit `vllm bench serve --agentinfer` commands enter AgentBench.
- Other commands are delegated to upstream vLLM with AgentInfer's default scheduler unless `--scheduler-cls` is set.
- An explicit `--scheduler-cls` is preserved, as in the serving command above.

The legacy `AgentAwareScheduler` and `agentinfer.LLM` compatibility surfaces remain available, but they do not enable
the explicit Progress-TTL serving path shown here.

## Verify the Integration

After model loading completes, query the service from another terminal:

```bash
curl http://127.0.0.1:8000/v1/models
```

See [Architecture](../explanation/architecture.md) for component responsibilities and
[Python API](../reference/python-api.md) for class signatures. For a reproducible baseline/candidate comparison, see
[Run a benchmark](run-benchmark.md).

[release-wheel]: https://github.com/openjiuwen-ai/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
