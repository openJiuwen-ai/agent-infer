# Run a Benchmark

This guide compares a cold upstream vLLM async scheduler with the AgentInfer Progress-TTL scheduler bridge using the
same Claude Code workload. BenchKit records both arms through the same Request Proxy to avoid observation-path bias.

## Prepare the Environment

The workflow requires:

- Linux, Python 3.10 or later, and vLLM 0.23.0.
- CUDA GPUs capable of running the target model; the script defaults to two tensor-parallel workers.
- Claude Code, tmux, and Git, with Claude Code authentication completed.
- A model that supports Anthropic `POST /v1/messages`.

Create an environment and install AgentInfer from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "vllm==0.23.0"
python -m pip install -e .
```

## Prepare SWE-bench Data

```bash
vllm bench serve --agentinfer prepare swebench \
  --output-dir agentinfer/agentbench/data/swebench
```

The command creates `instances.jsonl`, `task-lists/default.txt`, and `manifest.json`.

## Run One Cold Comparison

The repository script starts the baseline, runs the workload, stops the service completely, starts the candidate,
and prints the comparison report:

```bash
REPO=$PWD \
VENV=$PWD/.venv \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
TASK_NUM=1 \
CONCURRENCY=1 \
bash tests/agentbench/run-scheduler-e2e-compare.sh
```

Set `TENSOR_PARALLEL_SIZE=1` when the model fits on one GPU. Set `VLLM_EXTRA_ARGS` to append deployment-specific vLLM
arguments.

The script performs this sequence:

1. Starts the baseline with `vllm.v1.core.sched.async_scheduler.AsyncScheduler` and `--async-scheduling`.
2. Runs BenchKit with requests sent directly to the baseline vLLM server.
3. Stops the baseline process to clear process-local metrics and Prefix Cache state.
4. Starts the candidate with `AgentCacheAsyncSchedulerBridge`, the Progress-TTL controller, and lifecycle middleware.
5. Runs the same tasks, model, concurrency, and configuration, then compares the two finalized result directories.

If the model or virtual environment is elsewhere, set `MODEL` and `VENV` to absolute paths. The script rejects an
existing tmux session or lifecycle socket so an earlier run cannot contaminate the experiment.

## Scale Up the Experiment

After the one-task smoke run succeeds, increase task count and concurrency:

```bash
REPO=$PWD \
VENV=$PWD/.venv \
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
TASK_NUM=64 \
CONCURRENCY=32 \
bash tests/agentbench/run-scheduler-e2e-compare.sh
```

Use repeated cold runs for release decisions. Give every run a new result directory and keep the commit, task set and
order, model, agent profile, timeouts, tensor parallelism, vLLM arguments, and hardware identical.

## Compare Again or Summarize

Compare one or more finalized runs:

```bash
vllm bench serve --agentinfer compare \
  --baseline results/vllm/run1 results/vllm/run2 \
  --candidate results/agentinfer/run1 results/agentinfer/run2
```

Export run summaries to CSV:

```bash
vllm bench serve --agentinfer summarize \
  results/agentinfer/run1 results/agentinfer/run2 \
  --output-csv combined-summary.csv
```

See [Benchmark CLI](../reference/benchmark-cli.md) for command options,
[Benchmark configuration](../reference/benchmark-config.md) for YAML fields, and
[Run artifacts](../reference/run-artifacts.md) for output files. `completed` means that the agent process completed,
not that its patch is correct; read [Benchmark methodology](../explanation/benchmark-methodology.md) before making a
release claim.

## Select an agent and transparent Router

The default example is `agentinfer/agentbench/configs/swebench_vllm.yaml`. Claude Code supports `single` and
`plan-subagent` and requires the Claude CLI and tmux. JiuwenSwarm supports `code.normal` and requires its CLI; DSH
supports `single` and `plan-subagent` and requires the DeepSeek Harness CLI. Change the runtime, profile,
executable, and endpoint together:

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --agent-type jiuwenswarm --agent-profile code.normal \
  --agent-executable jiuwenswarm --endpoint /v1/chat/completions \
  --task-num 1 --result-dir results/jiuwenswarm-smoke
```

For DSH use `--agent-type dsh --agent-profile single --agent-executable dsh --endpoint /v1/chat/completions`.
For Router runs use `swebench_agentinfer.yaml`, with `backend.base_url` pointing to the Router and
`backend.metrics_url` directly to vLLM. The Router forwards transparently without the old registration/cleanup API.

For captured traces, see [Trace Replay](run-trace-replay.md).

## Local smoke validation

Run `python -m pytest tests/agentbench/test_benchmark_smoke.py -q` without a model service to validate the
production CLI, configuration, workspace, dispatch, request proxy, and finalization. The local test runtime and
loopback backend do not establish model quality or performance. The scheduler comparison script uses fields
supported by the destination controller factory.
