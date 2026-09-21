# Quick Start

This tutorial installs AgentInfer and starts vLLM with the async agent-aware scheduler, API middleware, and embedded
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
git clone https://github.com/openjiuwen-ai/agent-infer.git
cd agent-infer
python -m pip install -e .
```

For a release deployment, download the [AgentInfer 0.1.0 wheel][release-wheel], activate the target vLLM environment,
and run:

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

## 3. Start vLLM

Start vLLM with the current AgentInfer serving stack using a single flag:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --agentinfer
```

The flag injects async scheduling, the `AgentCacheAsyncSchedulerBridge`, the identity and lifecycle middleware, and
the Progress-TTL controller factory. The lifecycle socket defaults to `/tmp/agentinfer-vllm-lifecycle.sock` when
`AGENTCACHE_VLLM_LIFECYCLE_SOCKET` is unset; export a distinct socket per server instance on one host. For a
synchronous deployment, add `--no-async-scheduling` and the shim selects `AgentCacheSyncSchedulerBridge` instead.

From a source checkout, [`examples/serve-progress-ttl.sh`](../../../examples/serve-progress-ttl.sh) runs the same
serving configuration. Override its default model with `MODEL=<model-name>` when needed.

The model is illustrative and can be replaced with another model supported by vLLM. The agent-aware scheduler follows
the scheduling mode: `AgentCacheAsyncSchedulerBridge` for async scheduling, `AgentCacheSyncSchedulerBridge` for sync
scheduling; the async variant rejects configurations with async scheduling disabled.

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

[release-wheel]: https://github.com/openjiuwen-ai/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
