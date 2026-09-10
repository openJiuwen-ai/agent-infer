# BenchKit tests

Unit tests under this directory require no running vLLM, Router, Claude Code, tmux, or accelerator.
`run-baseline-smoke.sh` validates one externally started vLLM baseline. `run-scheduler-e2e-compare.sh` cold-compares
the upstream async scheduler and AgentInfer scheduler bridge. `run-router-e2e-compare.sh` exercises the
candidate through a transparent Router configured as its backend endpoint; it uses no AgentBench Router control
protocol. Both comparison scripts require explicit `REPO`, `VENV`, and `MODEL` values; use
`TENSOR_PARALLEL_SIZE` and `VLLM_EXTRA_ARGS` for deployment-specific vLLM options. The Scheduler comparison also
requires an AgentInfer runtime build that provides `AgentCacheAsyncSchedulerBridge`,
`AgentCacheLifecycleMiddleware`, and the progress-TTL controller in `VENV`.

The Scheduler comparison defaults its Progress-TTL cost model to the checked NPU/GLM reference values. GPU or other
NPU deployments can override `PRIVILEGED_MAX_CONTEXT_TOKENS`, the three `TTL_PREFILL_MODEL_*` values,
`TTL_DECODE_THROUGHPUT_ALPHA`, and the three `DECODE_STEP_*` values without changing the script. `VLLM_ENV_SCRIPT` and
`VLLM_EXTRA_ARGS` configure the engine environment and deployment arguments shared by the baseline and candidate.

Before a manual run, record the branch/commit, host, virtualenv, ports, configs, dataset, artifact directories,
explicit tmux names, commands, stop conditions, tested agent profiles and concurrency shapes, and all omissions.
See the [Quickstart](../../docs/en/how-to/run-benchmark.md).
