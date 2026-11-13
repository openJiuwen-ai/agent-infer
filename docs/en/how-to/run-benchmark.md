# Run a Benchmark

This guide compares a cold upstream vLLM async scheduler with the AgentInfer Progress-TTL agent-aware scheduler using
the same Claude Code workload. BenchKit records both arms through the same Request Proxy to avoid observation-path bias.
It also covers SWE-bench Verified correctness evaluation, including automatic scoring after a run.

## Prepare the Environment

The workflow requires:

- Linux, Python 3.10 or later, and vLLM 0.23.0.
- CUDA GPUs capable of running the target model; the script defaults to two tensor-parallel workers.
- Claude Code, tmux, and Git, with Claude Code authentication completed.
- A model that supports Anthropic `POST /v1/messages`.

Run Claude's `plan-subagent` profile as a non-root user (`id -u` must not be `0`). This profile uses
`--permission-mode bypassPermissions`; Claude rejects root/sudo execution with
`--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons`.
Use an account that can read the code and dataset and write the workspace and results. With `--with-correctness`,
the same account also needs Docker access; verify it with `docker info`. Otherwise, run standalone `evaluate`
after generation in an environment with Docker access.

At startup, the framework selects `Yes, I trust this folder` in Claude's workspace trust dialog and confirms only
once that option is highlighted, including when `No, exit` is the default. Unrecognized selections are not confirmed.
While `esc to interrupt` is visible, Claude is treated as busy even if its input prompt is also shown.
Plan approval recognizes wrapped text and case changes, including `Yes, and switch to BYPASS PERMISSIONS`.
If the same approval prompt remains visible for 5 seconds, confirmation is retried once to handle an ignored initial key.
File edits and other highlighted Yes confirmations use the same bounded retry. It resets when the prompt clears;
confirmation is not retried when Yes is no longer selected.

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

## Experimental Terminal-Bench 4.0.0 single-task run

The initial path supports Claude Code `single` through Harbor and Docker. Confirm `docker compose version` works.
Install Harbor in a separate Python 3.12 environment, for example with
`uv tool install --python 3.12 harbor==0.25.0`, then prepare the pinned dataset:

```bash
vllm bench serve --agentinfer prepare terminalbench \
  --output-dir agentinfer/agentbench/data/terminalbench
```

Set the actual model URL and name in `agentinfer/agentbench/configs/terminalbench_vllm.yaml`, then run one task:

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/terminalbench_vllm.yaml \
  --agent-type claude --agent-profile single \
  --agent-executable claude --endpoint /v1/messages \
  --task-num 1 --result-dir results/claude-smoke
```

Use a result directory that does not exist yet. Preparation exports all task packages; `default.txt` is currently
a CPU candidate set selected from task resource fields, not a fully oracle-validated subset. The run retains raw
Harbor logs, per-task `verifier.json`, `correctness.json`, request facts, and the summary. Summary
`correctness.available` is true only when every selected task has a valid reward; agent completion does not imply
task success. The container must reach the model gateway on the Docker bridge. Validate networking and image
availability with one task first; JiuwenSwarm, other profiles, GPU, and multi-service tasks still need adapters.

DSH `single` and `plan-subagent` use the same dataset and run command. The runner copies pinned DSH
`0.2.1-rc.2` and a supported host Node runtime into each task container. Set `DSH_NODE_EXECUTABLE`
to a Node 22.19+ binary if the default `node` is older. For a two-task run, set `--task-num 2`,
`--agent-type dsh --agent-executable dsh --endpoint /v1/chat/completions`, and either
`--agent-profile single` or `--agent-profile plan-subagent`. Use a separate `--result-dir` for each profile.
The host `dsh` executable must also resolve to version `0.2.1-rc.2`.

See [RFC #76](https://github.com/openJiuwen-ai/agent-infer/issues/76) for the design scope and additional agent adapters.

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

## Evaluate SWE-bench Verified correctness

Evaluate existing AgentBench patches using official SWE-bench **4.1.0**. Only
tasks with outcome=completed are selected; the finalized run may contain failed
tasks. No model requests are made. Version 5 requires a different dataset schema.
The evaluation runner targets Linux and uses POSIX process groups for bounded
shutdown. On Windows, run it inside a Linux environment such as WSL2.

The configuration file is optional. By default, evaluation uses the Python
environment running AgentInfer, which must have swebench==4.1.0 installed:

```bash
vllm bench serve --agentinfer evaluate --result-path results/run-a
```

Defaults are 2 workers, a 1800-second instance timeout and a 7200-second overall
timeout. Use --config only for a separate interpreter or custom settings.

### Setup and evaluation

To use a separate evaluator with Docker access, create evaluation.yaml:

```bash
uv venv --python 3.12 /path/to/swebench-venv
uv pip install --python /path/to/swebench-venv/bin/python 'swebench==4.1.0'
```

```yaml
harness_python: /path/to/swebench-venv/bin/python
expected_harness_version: "4.1.0"
max_workers: 2
timeout_seconds: 1800
overall_timeout_seconds: 7200
cleanup_timeout_seconds: 30
```

```bash
vllm bench serve --agentinfer evaluate \
  --result-path results/run-a \
  --config evaluation.yaml \
  --output-dir results/run-a/evaluations/eval-001
```

The source needs finalized manifest/summary files, a consistent task index,
per-task results/prompts and the local Verified dataset referenced by the manifest.
Task IDs, repository and base commit are checked. With no completed tasks, no
score is produced. Relative interpreter paths resolve against the configuration
file; unknown fields and unsupported harness versions are rejected.
Preflight checks the installed SWE-bench version before importing evaluator modules
and collecting environment details; a version mismatch prevents scoring from starting.

Omit --output-dir to create a unique directory under the run's evaluations/.
Existing directories are not overwritten. Original patches, manifest and
performance summary are preserved. First use downloads official Docker images.
Finalization removes containers left by this evaluation in one batch. Cleanup
failures or timeouts are recorded in the evaluation manifest and mark the evaluation failed.

### Evaluate automatically after a run

Add `--with-correctness` to `run` to generate patches and evaluate them in one command:

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --result-dir results/run-a \
  --with-correctness
```

For a separate evaluator, also pass `--evaluation-config evaluation.yaml`.
`--config` selects the run configuration; `--evaluation-config` selects the evaluation
configuration described above. Omitting the latter uses the current Python environment
and default evaluation settings. Passing it without `--with-correctness` exits with code 2.
Evaluation configuration is loaded and validated before generation. Checks of Docker,
the installed SWE-bench version and other scoring requirements remain in the evaluation phase.

After performance aggregation, artifact writes and resource cleanup finish, completed tasks
in the actual run directory are evaluated. Results go under that directory's `evaluations/`;
the result path and correctness summary are printed. Scoring time is excluded from run timing,
and evaluation uses its own timeout and worker settings. The original performance summary
and patches are preserved.

Without the flag, run behavior is unchanged. Run exceptions or interruption skip automatic
evaluation and propagate the original failure. Valid finalized artifacts can still be scored
later using standalone `evaluate`. If some tasks fail but orchestration finishes normally,
only completed tasks are scored. With no completed tasks, the command exits 1 without a score.
Normal unresolved or empty-patch results exit 0; this does not mean every patch is correct.
Evaluation errors and interruption use the exit codes below.

### Results

Each selected task gets one record in task_results.jsonl. The correctness.json
counts contain exactly total, resolved, unresolved, empty_patch and error.
The only accuracy metric is resolved_rate = resolved / total; errors remain in
the denominator. This describes the completed-task subset, not all Verified tasks.

An absent patch is only empty when the result and DSH export status confirm
no_changes. Otherwise it is an error. Failed source tasks are excluded even when
they have patches. F2P/P2P details stay in official per-task reports under raw/.
Commands, logs, versions, image identities and hashes are retained.

Exit codes: 0 for finalized evaluation without errors (unresolved is valid),
1 for input/execution/evaluation errors, 2 for invalid configuration, and 130
for interruption. Errors still produce summaries once the task set is known.

### Comparison

```bash
vllm bench serve --agentinfer compare-correctness \
  --baseline results/baseline/evaluations/eval-001/correctness.json \
  --candidate results/candidate/evaluations/eval-001/correctness.json
```

Comparison checks hashes, task sets, dataset contents, recorded generation
configuration/prompts, evaluator environment and image identity. Incompatible or
incomplete evidence yields insufficient_evidence; it never silently intersects
task sets. An unrecorded model checkpoint revision cannot be independently verified.

The JSON report lists regressions and improvements even when total accuracy is
unchanged. Any task regression fails the comparison; improvements do not offset
regressions. Exit codes: 0 (passed), 1 (regression detected), 2 (insufficient
evidence). This is a fixed-task check, not a statistical capability claim.

See the [official guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
and [benchmark methodology](../explanation/benchmark-methodology.md).

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

DSH live runs require **0.2.1-rc.2** and Node `^22.19.0 || >=24`.
Put a supported Node on `PATH` first; activating a Python virtual environment does not switch Node.
DSH and runtime preflight look up `node` through `PATH` even when `--agent-executable` is an absolute path.
Install the required DSH version in the external environment and check versions:

```bash
node --version
npm install --global @deepseek-ai/dsh@0.2.1-rc.2
DSH_EXECUTABLE="$(command -v dsh)"
"$DSH_EXECUTABLE" --version
```

Prepare the SWE-bench data and vLLM service described above, then set dataset paths, backend address, and model name
in `swebench_vllm.yaml`. Merge these fields into that YAML, keeping `agent:` as an uncommented top-level section:

```yaml
agent:
  type: dsh
  profile: plan-subagent
  context_window: 262144
  max_tokens: 8192
backend:
  endpoint: /v1/chat/completions
```

Adjust the example context and output capacities to the actual backend. Check `max_model_len` from `/v1/models`
for the context limit. Declaring a smaller `context_window` restricts the client's available output budget even when
the model service supports a larger context. See [Benchmark Configuration](../reference/benchmark-config.md).

This example runs two tasks at task concurrency one; subagents may explore in parallel within a task:

```bash
RESULT_DIR="results/dsh-plan-subagent-$(date +%Y%m%d-%H%M%S)"
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --base-url http://127.0.0.1:8000 \
  --agent-type dsh --agent-profile plan-subagent --endpoint /v1/chat/completions \
  --agent-executable "$DSH_EXECUTABLE" \
  --task-num 2 --max-concurrency 1 --result-dir "$RESULT_DIR"
```

BenchKit creates the result directory; it must not already exist. For a global installation, replace
`--agent-executable` with the actual DSH path. Use `--agent-profile single` for a single agent.
The `plan-subagent` profile enforces initial plan mode, read-only child tools, and plan-review approval.
The lead must wait for every child to complete successfully before submitting a plan for approval.
Only explicit plan-review questions are approved automatically.

The lead may execute at most 200 steps (`agent_step_limit`); Explore subagents may execute at most 100 steps.
A fourth identical read within the last eight tool calls
(same file, offset, and limit) stops the task. These limits produce `subagent_step_limit` and `subagent_read_loop`;
other unsuccessful child endings produce `subagent_failed`.
If the lead tries to finish in plan mode after all children have completed, it receives one recovery reminder
to submit its plan. Finishing again without approval produces `plan_not_approved`. The reminder does not exit
plan mode or approve the plan. Model length truncation is reported as `max_tokens`.
For both roles, four equivalent argument-validation errors or inactive-plan exit errors from the same tool
within the last eight tool results produce `validation_error_loop`. Successful calls age old errors out of the window.
Calling `exit_plan_mode` after plan mode closes returns guidance to continue implementation or reply with the final
result; it is not a task-completion tool.
These controls run on DSH lifecycle events; the overall task timeout still applies.

DSH uses pi-ai Chat Completions through the Request Proxy and receives the prompt on stdin.
Runtime versions and the npm lockfile hash are recorded in `evidence/dsh-runtime.json`.
DSH runs currently reject a nonempty `backend.api_key_env`.

The default DSH cleanup budget is 10 seconds. Configure it through YAML's `experiment.cleanup_timeout_seconds`.
For example, increase it for large logs or slow storage:

```yaml
experiment:
  cleanup_timeout_seconds: 60.0
```

Log collection, patch flush waiting, and patch export share this budget. Flush waiting is limited to the smaller of
`patch_flush_seconds` and half the budget. An exhausted budget is reported as `cleanup_timeout`; existing execution
timeout or cancellation reasons are preserved. Cleanup details are recorded in
`tasks/<instance_id>/dsh-run-status.json`; successful completion still requires validated session evidence.

For Router runs use `swebench_agentinfer.yaml`, with `backend.base_url` pointing to the Router and
`backend.metrics_url` directly to vLLM. The Router forwards transparently without the old registration/cleanup API.

## Historical DSH execution record (2026-10-29)

The run `test-dsh-20261029-102951` exercised 16 SWE-bench Verified Astropy tasks
at task concurrency 8, with a 3600-second per-task timeout and no overall timeout.
It used DSH 0.2.1-rc.2, Node 24.22.1, Python 3.12.13 and
Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 in the plan-subagent profile.

| Observation | Result |
| --- | --- |
| Agent tasks completed / failed | 14 / 2 |
| Tasks with exported patches | 14 |
| Run wall time | 3614.25 seconds |
| Model requests successful / failed | 1971 / 1 |
| Request Proxy TTFT mean / P95 | 1.201 / 2.887 seconds |

Task 13033 timed out after a child repeatedly read the same source; its final
request was cancelled during cleanup. Task 13579 failed because the lead ended
without completing plan approval. These are observations from that dated run,
not outcomes guaranteed by the current runtime safeguards.

No official correctness evaluation was available for this run. The 87.5% agent
completion rate is not a SWE-bench resolved rate, and its long-tail behavior does
not establish normal-workload scaling. The [Chinese run record](../../zh/how-to/run-benchmark.md)
retains the full parameters, task evidence and reproduction example.

For captured traces, see [Trace Replay](run-trace-replay.md).

## Local smoke validation

Run `python -m pytest tests/agentbench/test_benchmark_smoke.py -q` without a model service to validate the
production CLI, configuration, workspace, dispatch, request proxy, and finalization. The local test runtime and
loopback backend do not establish model quality or performance. The scheduler comparison script uses fields
supported by the destination controller factory.

## ZCode

The [RFC #71](https://github.com/openJiuwen-ai/agent-infer/issues/71) adapter supports `single` and
`plan-subagent` on Linux. Build source commit `a9fe401eede6e6e5e01be9048679c206fe6f02d9`
(CLI `0.17.1`) with Node `24.14.0` and pnpm `10.33.2`, following ZCode's build instructions.
Point `agent.executable` at an executable launcher that runs the built `zcode.cjs` with a fixed Node path
and forwards all arguments.
The launcher must preserve `NODE_OPTIONS`. A Node preload module reads `prompt.txt` inside the process,
avoiding the OS limit on a single command argument; preflight verifies this capability.
Native SEA executables are not supported by this launch path.

Only CLI `0.17.1` is currently supported. Preflight requires an exact `--version` match; other versions,
including `0.17.10`, fail with `Expected ZCode CLI 0.17.1, got ...` before any task starts.
Rebuild using the verified source and toolchain above, or validate compatibility before updating the supported
version. Do not bypass the check. Preflight verifies the CLI version and required capabilities, **not the Git
commit that produced the bundle**. `launcher_sha256` hashes the launcher, not `zcode.cjs`; retain the actual
bundle hash and build provenance when reproducing experiments. All CLI probes use a temporary working directory,
private HOME and ZCode state paths, excluding inherited personal configuration and credentials.

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --agent-type zcode --agent-profile single \
  --agent-executable /absolute/path/to/zcode-launcher \
  --endpoint /v1/chat/completions --task-num 1 --result-dir results/zcode-single
```

Set `context_window` and `max_tokens` to the served model's limits. Only unauthenticated local backends
are supported; leave `backend.api_key_env` empty. Each task has independent ZCode state and a local adapter
endpoint that attaches canonical identity before forwarding to the Request Proxy. This endpoint also owns
synchronous Hook decisions, using the official CLI without patching ZCode.

`single` disables delegation. `plan-subagent` defaults to one read-only Explore delegation before plan approval
and lead implementation. Explore exposes and permits only Read, Glob and Grep, returning its report
as final text; task Hooks handle the lead's tool policy and approvals. Background work is disabled.
Permission modes are not OS sandboxes; use an
external sandbox or container for production benchmarks.

Set positive limits under `agent` in YAML, or override them with `--max-subagents` and
`--max-concurrent-subagents`:

```yaml
agent:
  type: zcode
  profile: plan-subagent
  max_subagents: 4
  max_concurrent_subagents: 2
```

This permits up to four accepted exploration launches per task, with at most two running together.
Both limits default to 1; the lead need not use the entire budget. Launch children in parallel in one response,
or in successive batches. Excess calls are rejected before execution, without queuing. Failed launches still
consume the total budget. Once a report is available, the lead may explore further or submit a plan;
approval requires all accepted children to finish without failure. No delegation is allowed after approval.
Non-default limits apply only to ZCode `plan-subagent`; `single` always disables delegation.
`--max-concurrency` controls task concurrency separately. `zcode-lineage.json` records limits, accepted launch
count, peak concurrency and delegation statuses, retaining unfinished calls on timeout or cancellation.

Task artifacts include raw `transcript.jsonl`, `zcode.stderr.log`, `zcode-hooks.jsonl`, `zcode-lineage.json`,
`zcode-config.json` and `zcode-run-status.json`. Patches use the existing `model.patch` format and exclude
only the current session's generated plan file. Execution completion is separate from independent scoring.

Native ZCode `turn.failed` details are preserved as `native_error` in `zcode-run-status.json`;
context capacity/compaction failures use `model_context_exceeded`. The advertised tools follow the
planning phase and constrain delegation to the task-owned read-only profile. Explore has eight tool
rounds, followed by a report without tools. When the child concurrency limit exceeds one, initial exploration
uses `tool_choice: required` and `parallel_tool_calls: true` to permit multiple delegations in one response.
Single-child delegation and approval retain named tool choice. Required choice means at least one call;
it does not force the model to fill all slots. Verify actual overlap and peak concurrency in the artifacts;
conversation messages are preserved. Prepare the task's test dependencies in its execution environment
before starting the agent.
Preflight requires the exact CLI version `0.17.1`. Nonzero exits without a more specific error are classified
as `harness_error` rather than an unapproved plan.
The model SSE stream emits a comment heartbeat every 15 idle seconds to prevent premature HTTP body timeouts.
Heartbeats produce no model content; task deadlines and ZCode's model progress checks remain active.
`zcode-run-status.json` records `prompt_loaded`, `execution_phase`, `model_stream_error_count` and
`last_model_error`, preserving earlier SDK stream errors and their log location even when the task times out.
Malformed auxiliary log records, missing fields and read errors are reported in `model_log_warnings` without
overriding task outcomes or cancellation. Error counts cover successfully parsed records only; a zero count with
parsing warnings does not establish that no model error occurred.
