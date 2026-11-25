# Run a Benchmark

This guide covers installation and benchmark execution for Claude Code, JiuwenSwarm, DSH, and ZCode.
BenchKit records model requests through the Request Proxy and saves task results, logs, and code patches.
It also covers SWE-bench Verified correctness evaluation, including automatic scoring after a run.

## Prepare the Environment

The workflow requires:

- Linux, Python 3.10 or later, and vLLM 0.23.0.
- CUDA GPUs capable of running the target model.
- Git and the CLI and runtime dependencies of your selected agent.
- A model service with tool calling: `/v1/messages` for Claude Code, `/v1/chat/completions` for the other three agents.

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

## Install an Agent

Install only the agent you need. These commands target Linux Bash. Agent installation does not install the test
dependencies of benchmark tasks. AgentInfer and the agent may use separate environments; pass the full executable
path with `--agent-executable`.

| Agent | Adapter requirements | Additional dependencies |
| --- | --- | --- |
| Claude Code | No pinned CLI version; record `claude --version` and validate one task first | tmux, bubblewrap, socat |
| JiuwenSwarm | `code.normal`; CLI capability checks, no pinned package version | Separate Python environment, jiuwenbox-server, bubblewrap |
| DSH | Exactly `0.2.1-rc.2`; other versions fail preflight | Node `^22.19.0` or `>=24`, a working local sandbox |
| ZCode | Exactly CLI `0.17.1`, built from the source commit below | Node `24.14.0`, pnpm `10.33.2`, an external sandbox or container for benchmarks |

### System Dependencies and Node

On Ubuntu/Debian, install the following packages; use equivalent packages on other distributions.
Claude uses `tmux` and `socat`. The `bubblewrap` package provides `bwrap`; build tools support native dependencies.

```bash
sudo apt-get update
sudo apt-get install -y git curl tmux bubblewrap socat build-essential pkg-config python3-dev python3-venv
```

DSH and ZCode can both use Node `24.14.0`. The following uses the
[official nvm installer](https://github.com/nvm-sh/nvm#install--update-script); skip the first line if nvm is installed:

```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.41.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"
nvm install 24.14.0
nvm use 24.14.0
node --version
npm --version
```

Use this Node in the terminal that runs the benchmark. Activating a Python environment does not switch Node.
DSH preflight resolves `node` through `PATH`, even when the DSH executable uses an absolute path.

### Claude Code

Use the [official native installer](https://code.claude.com/docs/en/setup#install-a-specific-version),
which does not require Node:

```bash
curl -fsSL https://claude.ai/install.sh | bash -s stable
export PATH="$HOME/.local/bin:$PATH"
CLAUDE_EXECUTABLE="$(command -v claude)"
"$CLAUDE_EXECUTABLE" --version
tmux -V
bwrap --version
socat -V
```

Use `stable` for an initial installation. For reproducible runs, replace it with a validated version number and
save the version output. This adapter does not pin Claude; installation alone does not establish compatibility.
For unauthenticated local vLLM, the adapter supplies a placeholder credential, so Anthropic login is unnecessary.
Provide credentials when the backend requires them. Run `plan-subagent` as a non-root user because it uses
`bypassPermissions`.

### JiuwenSwarm

The official Python package is [`workswarm`](https://pypi.org/project/workswarm/0.2.8/).
This installation example pins `0.2.8`, whose published package requires Python `>=3.11,<3.14`.
It assumes Python 3.12 is already installed:

```bash
JIUWEN_VENV="$HOME/.venvs/agentbench-jiuwenswarm"
python3.12 -m venv "$JIUWEN_VENV"
"$JIUWEN_VENV/bin/python" -m pip install "workswarm==0.2.8"
"$JIUWEN_VENV/bin/python" -m pip show workswarm
JIUWEN_EXECUTABLE="$JIUWEN_VENV/bin/jiuwenswarm"
"$JIUWEN_EXECUTABLE" --help
"$JIUWEN_EXECUTABLE" chat --help
test -x "$JIUWEN_VENV/bin/jiuwenswarm-init"
test -x "$JIUWEN_VENV/bin/jiuwenswarm-start"
"$JIUWEN_VENV/bin/jiuwenbox-server" --help
```

See the [official JiuwenSwarm repository](https://github.com/openJiuwen-ai/jiuwenswarm) for package and entry points.
`chat --help` must include `--jsonl`, `--mode`, `--session`, `--cwd`, `--project-dir`, `--gateway-url`, and `--timeout`.
If an entry point or option is missing, select a version that provides it. This guide has not validated `0.2.8`
with a real benchmark task.

Pass `"$JIUWEN_EXECUTABLE"` to `--agent-executable`. AgentBench initializes each task, writes isolated configuration,
and starts its services. Do not run `jiuwenswarm-init` or `jiuwenswarm-start` manually beforehand.

### DSH

Prepare Node as above, then install the supported
[`@deepseek-ai/dsh@0.2.1-rc.2`](https://www.npmjs.com/package/@deepseek-ai/dsh/v/0.2.1-rc.2):

```bash
node --version
npm install --global @deepseek-ai/dsh@0.2.1-rc.2
DSH_EXECUTABLE="$(command -v dsh)"
"$DSH_EXECUTABLE" --version
"$DSH_EXECUTABLE" --profile headless --help
```

The version must be `0.2.1-rc.2`; headless help must include JSON output, stdin, and session options.
AgentBench also checks sandbox operation before execution. Python `zstandard`, needed to read session artifacts,
is an AgentInfer dependency and belongs in the Python environment running the benchmark.

### Build ZCode and Create a Launcher

With Node `24.14.0` selected, follow the [build instructions for the supported ZCode source][zcode-build]:

```bash
node --version
npm install --global pnpm@10.33.2
pnpm --version
ZCODE_SOURCE="$HOME/.local/src/agentbench-zcode"
mkdir -p "$HOME/.local/src"
git clone https://github.com/zai-org/ZCode.git "$ZCODE_SOURCE"
git -C "$ZCODE_SOURCE" checkout --detach a9fe401eede6e6e5e01be9048679c206fe6f02d9
(
  cd "$ZCODE_SOURCE"
  pnpm install --frozen-lockfile
  pnpm --filter '@zcode/cli...' build
)
```

Installation downloads dependencies and runtime resources; do not skip install scripts.
Keep the source checkout and adjacent resources after building, rather than copying only `zcode.cjs`.
Create a launcher that fixes the absolute paths to Node and the bundle:

```bash
ZCODE_NODE="$(command -v node)"
ZCODE_CLI="$ZCODE_SOURCE/apps/zcode-cli/packages/cli/dist/zcode.cjs"
ZCODE_EXECUTABLE="$HOME/.local/bin/agentbench-zcode"
mkdir -p "$HOME/.local/bin"
printf '#!/bin/bash\nexec %q %q "$@"\n' "$ZCODE_NODE" "$ZCODE_CLI" > "$ZCODE_EXECUTABLE"
chmod +x "$ZCODE_EXECUTABLE"
"$ZCODE_EXECUTABLE" --version
git -C "$ZCODE_SOURCE" rev-parse HEAD
sha256sum "$ZCODE_CLI"
```

The version must print `0.17.1`; the repository version `3.17.2` is not the CLI version.
Pass `"$ZCODE_EXECUTABLE"` to `--agent-executable`. This launcher preserves `NODE_OPTIONS` and forwards all arguments
so the adapter can load long prompts. It only launches ZCode; use an external sandbox or container for benchmarks.
See [ZCode](#zcode) below for version checks and execution limits.

[zcode-build]: https://github.com/zai-org/ZCode/blob/a9fe401eede6e6e5e01be9048679c206fe6f02d9/apps/zcode-cli/docs/usage/01-quickstart.md

## Prepare SWE-bench Data

```bash
vllm bench serve --agentinfer prepare swebench \
  --output-dir agentinfer/agentbench/data/swebench
```

The command creates `instances.jsonl`, `task-lists/default.txt`, and `manifest.json`.

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

## Select an Agent and Transparent Router

The default example is `agentinfer/agentbench/configs/swebench_vllm.yaml`. Claude Code supports `single` and
`plan-subagent` and requires the Claude CLI and tmux. JiuwenSwarm supports `code.normal`; DSH and ZCode support
`single` and `plan-subagent`. [Install the CLI](#install-an-agent), then change the runtime, profile,
executable, and endpoint together:

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --agent-type jiuwenswarm --agent-profile code.normal \
  --agent-executable "$JIUWEN_EXECUTABLE" --endpoint /v1/chat/completions \
  --task-num 1 --result-dir results/jiuwenswarm-run
```

See [DSH](#dsh) above for installation and version requirements. The examples use the saved `DSH_EXECUTABLE`.

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

Adjust task count and task concurrency with `--task-num` and `--max-concurrency`; use a new result directory each time.
DSH currently supports only unauthenticated model services; leave `backend.api_key_env` empty.
Runtime versions are recorded in `evidence/dsh-runtime.json`. See [Run artifacts](../reference/run-artifacts.md)
for step limits and loop termination reasons.

The default DSH cleanup budget is 10 seconds. Configure it through YAML's `experiment.cleanup_timeout_seconds`.
For example, increase it for large logs or slow storage:

```yaml
experiment:
  cleanup_timeout_seconds: 60.0
```

Log collection, patch flush waiting, and patch export share this budget. An exhausted budget is reported as
`cleanup_timeout`; see `tasks/<instance_id>/dsh-run-status.json` for details.

For Router runs use `swebench_agentinfer.yaml`, with `backend.base_url` pointing to the Router and
`backend.metrics_url` directly to vLLM. The Router forwards transparently without the old registration/cleanup API.

## ZCode

Complete the [source build and launcher setup](#build-zcode-and-create-a-launcher) above.
The [RFC #71](https://github.com/openJiuwen-ai/agent-infer/issues/71) adapter supports `single` and
`plan-subagent` on Linux. Native SEA executables are not supported by this launch path.

Preflight requires exactly CLI `0.17.1`. Other versions fail with `Expected ZCode CLI 0.17.1, got ...`
before tasks start. It checks the version and required capabilities, **not the Git commit that produced the bundle**.
`launcher_sha256` hashes the launcher; also retain the `zcode.cjs` hash and source version for reproducibility.
Preflight uses a temporary working directory, private HOME, and isolated ZCode state paths.

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/swebench_vllm.yaml \
  --agent-type zcode --agent-profile single \
  --agent-executable "$ZCODE_EXECUTABLE" \
  --endpoint /v1/chat/completions --task-num 1 --result-dir results/zcode-single
```

Set `context_window` and `max_tokens` to the served model's limits. Only unauthenticated local backends
are supported; leave `backend.api_key_env` empty. Each task has independent configuration, state, and logs.

`single` disables delegation. `plan-subagent` defaults to one read-only Explore delegation before plan approval
and lead implementation. Explore can use only Read, Glob, and Grep, with at most eight tool rounds before
returning a report. Background work is disabled in both profiles.
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
Both limits default to 1; the lead need not use the entire budget. The concurrency limit must not exceed the total limit.
Launch children in parallel in one response,
or in successive batches. Excess calls are rejected before execution, without queuing. Failed launches still
consume the total budget. Once a report is available, the lead may explore further or submit a plan;
approval requires all accepted children to finish without failure. No delegation is allowed after approval.
Non-default limits apply only to ZCode `plan-subagent`; `single` always disables delegation.
`--max-concurrency` controls task concurrency separately. `zcode-lineage.json` records limits, accepted launch
count, peak concurrency and delegation statuses, retaining unfinished calls on timeout or cancellation.

Task artifacts include raw `transcript.jsonl`, `zcode.stderr.log`, `zcode-hooks.jsonl`, `zcode-lineage.json`,
`zcode-config.json` and `zcode-run-status.json`. Patches use the existing `model.patch` format and exclude
only the current session's generated plan file. Execution completion is separate from independent scoring.

For failures, inspect `native_error`, `last_model_error`, and `model_log_warnings` in `zcode-run-status.json`,
which identify native errors, model request errors, and log parsing issues. Then consult the raw logs.
A zero error count with parsing warnings does not establish that no model error occurred.

## Compare or Summarize Results

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
