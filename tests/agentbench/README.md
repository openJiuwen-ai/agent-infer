# AgentBench tests

AgentBench validation is grouped by responsibility and dependency cost:

1. **Function/unit** tests cover parsing, validation, aggregation, profiles, and
   state transitions without live sockets or external processes.
2. **Module/contract** tests cover owned module boundaries with local fakes,
   ASGI transport, temporary files, and local git repositories. This includes
   runtime dispatch, runner lifecycle, Request Proxy, workspaces, artifacts,
   and dependency-direction contracts.
3. **Hermetic benchmark smoke** in `test_benchmark_smoke.py` runs the production
   CLI/config/dataset/workspace/dispatch/proxy-server/trace/finalization path
   with helpers from `helpers/benchmark_smoke.py`. A test runtime and loopback
   model backend replace external software. The smoke covers single-task
   artifacts plus concurrent task isolation and aggregation; it does not cover
   the production proxy's multiprocessing owner or model quality.
4. **Environment E2E, correctness, and performance** use the shell launchers in
   this directory and the external SWE-bench evaluator. They require real agent
   runtimes, serving backends, Linux/tmux, or accelerators and do not run in
   hermetic PR CI.

The Python groups require no running vLLM, Router, Claude Code, JiuwenSwarm,
tmux, GPU, accelerator, or Internet:

```bash
python -m pytest -q -m "not benchmark_smoke" tests/agentbench
python -m pytest -q tests/agentbench/test_benchmark_smoke.py
```

Four JavaScript bridge checks in `agents/dsh/test_dsh_instance.py` additionally
require Node.js on `PATH` (Node.js 20+ recommended). If `node` is missing, only
those checks are skipped with an explicit reason; the Python-only DSH checks
still run. GitHub CPU CI installs Node.js 20 and executes the bridge assertions.
Other CI environments should install Node.js before pytest for the same coverage.

`run-baseline-smoke.sh` validates one externally started vLLM baseline.
`run-scheduler-e2e-compare.sh` cold-compares the upstream async scheduler and
AgentInfer scheduler bridge. `run-router-e2e-compare.sh` exercises the candidate
through a transparent Router configured as its backend endpoint; it uses no
AgentBench Router control protocol. Both comparison scripts require explicit
`REPO`, `VENV`, and `MODEL` values; use `TENSOR_PARALLEL_SIZE` and
`VLLM_EXTRA_ARGS` for deployment-specific vLLM options. The Scheduler comparison
also requires an AgentInfer runtime build that provides
`AgentCacheAsyncSchedulerBridge`, `AgentCacheLifecycleMiddleware`, and the
progress-TTL controller in `VENV`.

The Scheduler comparison defaults its Progress-TTL cost model to the checked
NPU/GLM reference values. GPU or other NPU deployments can override
`PRIVILEGED_MAX_CONTEXT_TOKENS`, the three `TTL_PREFILL_MODEL_*` values,
`TTL_DECODE_THROUGHPUT_ALPHA`, and the three `DECODE_STEP_*` values without
changing the script. The destination controller factory must support these
policy fields. `VLLM_ENV_SCRIPT` and `VLLM_EXTRA_ARGS` configure the engine
environment and deployment arguments shared by both arms.

Before a manual run, record the branch/commit, host, virtualenv, ports, configs,
dataset, artifact directories, explicit tmux names, commands, stop conditions,
tested agent profiles and concurrency shapes, and all omissions. See the
[Quickstart](../../docs/en/how-to/run-benchmark.md).
