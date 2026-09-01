# Quick Start

This tutorial installs AgentInfer and starts vLLM with the async scheduler bridge, API middleware, and embedded
Progress-TTL controller. Run it on Linux or WSL 2 with a CUDA GPU that supports the selected model.

## 1. Prepare the vLLM Environment

Create or activate the Python environment that runs vLLM. AgentInfer supports Python 3.10 or later and vLLM 0.23.0:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "vllm==0.23.0"
```

If your CUDA environment requires a platform-specific vLLM installation, follow the
[vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/) and install version `0.23.0`.

## 2. Install AgentInfer

For development, clone the repository and install it in editable mode in the active vLLM environment:

```bash
git clone https://gitcode.com/openJiuwen/agent-infer.git
cd agent-infer
python -m pip install -e .
```

For a release deployment, download the [AgentInfer 0.1.0 wheel][release-wheel], activate the target vLLM environment,
and run:

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

## 3. Start vLLM

Choose an unused local Unix socket for lifecycle signals, then start vLLM with the current AgentInfer serving stack:

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

From a source checkout, [`examples/serve-progress-ttl.sh`](../../../examples/serve-progress-ttl.sh) runs the same
serving configuration. Override its default model with `MODEL=<model-name>` when needed.

The model is illustrative and can be replaced with another model supported by vLLM. Keep `--async-scheduling` when
using `AgentCacheAsyncSchedulerBridge`; the bridge rejects configurations with async scheduling disabled.

## 4. Verify the Service

In another terminal, wait for model loading to finish and query the vLLM models endpoint:

```bash
curl http://127.0.0.1:8000/v1/models
```

The response should list `meta-llama/Llama-3.1-8B-Instruct`. See the
[vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) for standard request examples and serving
options.

Next, [integrate AgentInfer with vLLM](../how-to/integrate-vllm.md) in an existing service or
[run a benchmark](../how-to/run-benchmark.md) to compare schedulers.

[release-wheel]: https://gitcode.com/openJiuwen/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
