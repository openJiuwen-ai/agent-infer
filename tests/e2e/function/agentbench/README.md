# AgentBench local smoke tests

One entry tests Claude Code, DSH, JiuwenSwarm, and ZCode against a local coding
task. Functional cases no longer download SWE-bench metadata or GitHub task
repositories. Performance benchmarks continue to use their existing datasets.

## Task and preparation

Each case runs one task with concurrency one. The `smoke_task/` directory
contains `addition.py`, with a broken `add(a, b)` function and three assertions.
The agent fixes the function, runs `python addition.py`, and finishes with a
short report. Only the Python standard library is needed.

Before each bench run, the suite copies the task files into a fresh local Git
cache and creates a JSONL index, selection file, and generated benchmark YAML.
These inputs remain under the case result_root/smoke-inputs directory. Normal
BenchKit workspace preparation uses the preseeded commit without fetching a
remote repository. Keep the task source intentionally broken.

The generated YAML declares dataset.name=smoke. All benchmark entries use the dedicated
tests/e2e/function/agentbench/smoke_vllm.yaml template, with runtime and endpoint
overrides in JSON. Dataset paths are filled in at runtime.

## Agents and requirements

| Agent type | Agent | Profiles | Endpoint |
| --- | --- | --- | --- |
| claude | Claude Code | single, plan-subagent | /v1/messages |
| dsh | DSH | single, plan-subagent | /v1/chat/completions |
| jiuwenswarm | JiuwenSwarm | code.normal | /v1/chat/completions |
| zcode | ZCode | single, plan-subagent | /v1/chat/completions |

The single case file, cases/qwen3_coder_agents_1x1.json, declares the shared
model deployment, hardware marks, and result root once. Its benchmark_params
array contains seven agent/profile entries. Pytest collects each entry as an
independent item, with IDs ending in agent/profile, such as claude/single.

Run on Linux with GPU access, vLLM, Git, and the selected agent installed.
Claude needs tmux; DSH and ZCode need Node.js; JiuwenSwarm needs its companion
commands, jiuwenbox-server, bubblewrap, and host support for its sandbox.
Agent dependency/version checks reuse each runtime's preflight before starting
vLLM. Git is checked separately because the local fixture creates a repository.
ZCode must be CLI 0.17.1 with the pinned build documented in the
[benchmark guide](../../../../docs/en/how-to/run-benchmark.md).

The default case uses Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 on two L20s.
Adjust model paths, executables, serving options, and hardware marks for your machine.
The 1x1 in a case name means one task and one concurrent task, not GPU count.
The local suite requires task-num=1 and max-concurrency=1.

Pytest, fixture preparation, and vLLM run as the invoking user (root in CI).
When pytest is root, pass --benchmark-run-as-user test-user for plan-subagent
bench subprocesses. Create the account first. Single and code.normal bench
subprocesses remain root. Agent executables must be accessible to that account.
The user is selected only by the command, not by JSON. A root-launched
plan-subagent case fails before model startup if this option is missing or
names root; single and code.normal use the invoking user without this option.
Pytest already running as an ordinary user can run plan-subagent directly.

## Run

Run all seven cases in one pytest process to reuse the model service:

```bash
pytest -s -v -m "L20 and cards_2 and ready" tests/e2e/function/agentbench/test_smoke_task.py \
  --benchmark-run-as-user test-user
```

Run one agent/profile from the shared case file:

```bash
pytest -s -v tests/e2e/function/agentbench/test_smoke_task.py \
  --test-config-file tests/e2e/function/agentbench/cases/qwen3_coder_agents_1x1.json \
  -k "zcode and plan" \
  --benchmark-run-as-user test-user
```

Use -k claude or -k dsh to run both profiles of one agent. Omitting -k selects
all seven entries. Results and generated inputs use the shared root
test-results/function/agentbench, with a distinct run ID for every entry.

Each case has its own task workspace, agent instance, Request Proxy, and run
artifacts. The session fixture holds one VllmServer from
tests/e2e/helpers/server.py and starts it lazily on the first benchmark entry.
Entries in one session must share the deployment and bind settings; a mismatch
is reported instead of switching services. Teardown closes the process when
pytest exits. Separate pytest invocations do not share a service; do not
use pytest-xdist for this suite if only one model instance is desired.

The ready CI pipeline runs all profiles in one functional job and one pytest
invocation. It uses the existing l20_2 preset and the shared CI runtime image.
Performance and replay suites retain their previous service lifecycle.

## Validation and evidence

The run must finalize, complete its one task, have no failed tasks, and record
at least one model request. Request failure counts are not pass criteria.
Common files must exist: agents.csv,
distribution_samples.csv, manifest.json, requests.jsonl, sessions.csv,
summary.json, task_index.json, proxy.stdout.log, and proxy.stderr.log. The
evidence, tasks, and workspaces directories must exist. Extra files/directories
and empty files are allowed. Agent-specific artifact contents are not graded.

A task's completed status is the runtime outcome, not an independent grading of
the patch. The smoke prompt asks the agent to run the supplied checks. A
plan-subagent profile exercises that runtime policy but does not independently
assert child delegation here.

The model server log is retained under the first case's vllm-logs directory;
each bench run keeps its own evidence and task artifacts. Fixture/cache tests
can run on CPU. Full real-agent GPU execution has not been run in the Windows
development environment.
