# AgentInfer

AgentInfer manages caches and schedules agent workflow requests in or in front of LLM serving engines such as vLLM.

[中文](README.zh.md)

## Core Features

- **Semantic Router**: a programmable Mixture-of-Models router for heterogeneous LLM inference.
- **Router**: a high-performance, lightweight router for large-scale vLLM deployments.
- **Agent Cache**: vLLM plugins that manage request scheduling and Ascend NPU-native KV cache pooling and transfer.
- **AgentBench**: benchmarks agentic workloads through real agent runs or trace-dataset replay.

## Architecture

![AgentInfer architecture](docs/assets/agentinfer-architecture.png)

## Related Documentation

[Documentation](docs/README.md) · [Quick start](docs/en/tutorial/01-quick-start.md) ·
[vLLM integration](docs/en/how-to/integrate-vllm.md) · [Benchmark guide](docs/en/how-to/run-benchmark.md) ·
[AgentRouter middleware patch](agentinfer/agentrouter/README.md) · [Changelog](CHANGELOG.md)

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

Start the AgentInfer serving path with a single flag:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --agentinfer
```

The lifecycle socket defaults to `/tmp/agentinfer-vllm-lifecycle.sock` when
`AGENTCACHE_VLLM_LIFECYCLE_SOCKET` is unset; export a distinct socket per server instance on one host.
The equivalent repository example is available at [`examples/serve-progress-ttl.sh`](examples/serve-progress-ttl.sh).

The installed `vllm` command delegates ordinary commands to upstream vLLM. Explicit
`vllm bench serve --agentinfer` commands enter AgentBench; `vllm serve MODEL --agentinfer` activates the
AgentInfer serving path; other commands are delegated unchanged.
See the [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) for standard serving options.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

This product serves solely as a workflow orchestration tool and does not embed any AI model capabilities. When
users integrate AI models for specific business scenarios, they shall bear full responsibility for compliance
obligations under the EU AI Act and other relevant regulatory frameworks.

[release-wheel]: https://github.com/openjiuwen-ai/agent-infer/releases/download/0.1.0/agentinfer-0.1.0-py3-none-any.whl
