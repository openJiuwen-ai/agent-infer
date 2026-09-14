# AgentInfer

AgentInfer manages caches and schedules agent workflow requests in or in front of LLM serving engines such as vLLM.

[中文](README.zh.md)

## Core Features

- Provides async and sync scheduler bridges for explicit AgentInfer integration with vLLM.
- Transports agent identity and lifecycle observations through dedicated API middleware.
- Retains, pauses, and resumes agent programs with the embedded Progress-TTL controller.
- Supports deployment as an in-engine scheduler or as a request router in front of serving engines.
- Includes AgentBench for running and comparing reproducible agent workloads against vLLM deployments.

## Related Documentation

[Documentation](docs/README.md) · [Quick start](docs/en/tutorial/01-quick-start.md) ·
[vLLM integration](docs/en/how-to/integrate-vllm.md) · [Benchmark guide](docs/en/how-to/run-benchmark.md) ·
[Changelog](CHANGELOG.md)

## Requirements

- Operating system: Linux, or Windows with WSL 2.
- Python: 3.10 or later, matching `requires-python` in `pyproject.toml`.
- Inference runtime: vLLM 0.23.0.
- Hardware: a CUDA GPU supported by vLLM and large enough for the selected model.

Install AgentInfer in the same Python environment as vLLM.

## Installation

### Install from source

For development, activate the target vLLM environment and install AgentInfer in editable mode:

```bash
git clone https://github.com/openjiuwen-ai/agent-infer.git
cd agent-infer
python -m pip install -e .
```

### Install a release package

Download the [AgentInfer 0.1.0 wheel][release-wheel], then install it in the target vLLM environment:

```bash
python -m pip install agentinfer-0.1.0-py3-none-any.whl
```

## Quick Start

Choose a local Unix socket for API lifecycle signals, then start vLLM with the AgentInfer async scheduler bridge,
identity and lifecycle middleware, and embedded Progress-TTL controller:

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

The equivalent repository example is available at [`examples/serve-progress-ttl.sh`](examples/serve-progress-ttl.sh).

The installed `vllm` command delegates ordinary commands to upstream vLLM. Explicit
`vllm bench serve --agentinfer` commands enter AgentBench; other commands use AgentInfer's default scheduler unless
`--scheduler-cls` is set explicitly. See the
[vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) for standard serving options.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

This product serves solely as a workflow orchestration tool and does not embed any AI model capabilities. When
users integrate AI models for specific business scenarios, they shall bear full responsibility for compliance
obligations under the EU AI Act and other relevant regulatory frameworks.

[release-wheel]: https://github.com/openjiuwen-ai/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
