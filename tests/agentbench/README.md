# BenchKit tests

Unit tests under this directory require no running vLLM, Router, Claude Code, tmux, or accelerator.
`run-baseline-smoke.sh` validates one externally started vLLM baseline. `run-scheduler-e2e-compare.sh` cold-compares
the upstream async scheduler and AgentInfer scheduler bridge. `run-router-e2e-compare.sh` is reserved for the
Router-backed candidate. Both comparison scripts require explicit `REPO`, `VENV`, and `MODEL` values; use
`TENSOR_PARALLEL_SIZE` and `VLLM_EXTRA_ARGS` for deployment-specific vLLM options. The Scheduler comparison also
requires an AgentInfer runtime build that provides `AgentCacheAsyncSchedulerBridge`,
`AgentCacheLifecycleMiddleware`, and the progress-TTL controller in `VENV`.

Before a manual run, record the branch/commit, host, virtualenv, ports, configs, dataset, artifact directories,
explicit tmux names, commands, stop conditions, tested agent profiles and concurrency shapes, and all omissions.
See the [Quickstart](../../docs/benchmarks/quickstart.md).
