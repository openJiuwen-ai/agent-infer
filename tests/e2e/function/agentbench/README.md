# AgentBench local smoke tests

One entry tests Claude Code, DSH, JiuwenSwarm, and ZCode against a local coding
task. Functional cases no longer download SWE-bench metadata or GitHub task
repositories. Performance benchmarks continue to use their existing datasets.

## Task and preparation

Each case runs one task with concurrency one. The fixture contains a broken
add(a, b) function and three unittest checks in check_arithmetic.py. The agent
must read the files, fix arithmetic.py, run python check_arithmetic.py, and
finish with a short report. Only the Python standard library is needed.

Before each bench run, the suite copies the fixture into a fresh local Git
cache and creates a JSONL index, selection file, and generated benchmark YAML.
These inputs remain under the case result_root/smoke-inputs directory. Normal
BenchKit workspace preparation uses the preseeded commit without fetching a
remote repository. Do not edit the intentionally broken source fixture.

The generated YAML declares dataset.name=smoke. All cases use the dedicated
agentinfer/agentbench/configs/smoke_vllm.yaml template, with runtime and endpoint
overrides in JSON. Dataset paths are filled in at runtime.

## Agents and requirements

| Case directory | Agent | Profiles | Endpoint |
| --- | --- | --- | --- |
| cases/claude/ | Claude Code | single, plan-subagent | /v1/messages |
| cases/dsh/ | DSH | single, plan-subagent | /v1/chat/completions |
| cases/jiuwenswarm/ | JiuwenSwarm | code.normal | /v1/chat/completions |
| cases/zcode/ | ZCode | single, plan-subagent | /v1/chat/completions |

Run on Linux with GPU access, vLLM, Git, and the selected agent installed.
Claude needs tmux; DSH and ZCode need Node.js; JiuwenSwarm needs its companion
commands, jiuwenbox-server, bubblewrap, and host support for its sandbox.
ZCode must be CLI 0.17.1 with the pinned build documented in the
[benchmark guide](../../../../docs/en/how-to/run-benchmark.md).

Cases use Qwen/Qwen3-Coder-30B-A3B-Instruct in BF16 on two L20s. Adjust model
paths, executables, serving options, and hardware marks for your machine.
The 1x1 in a case name means one task and one concurrent task, not GPU count.
The local suite requires task-num=1 and max-concurrency=1.

Pytest, fixture preparation, and vLLM run as the invoking user (root in CI).
When pytest is root, pass --benchmark-run-as-user test-user for plan-subagent
bench subprocesses. Create the account first. Single and code.normal bench
subprocesses remain root. Agent executables must be accessible to that account.

## Run

Run all seven cases in one pytest process to reuse the model service:

```bash
pytest -s -v -m "L20 and cards_2 and ready" tests/e2e/function/agentbench/test_smoke.py \
  --benchmark-run-as-user test-user
```

Run just one profile using a complete case path:

```bash
pytest -s -v tests/e2e/function/agentbench/test_smoke.py \
  --test-config-file tests/e2e/function/agentbench/cases/zcode/qwen3_coder_zcode_1x1_plan_subagent.json \
  --benchmark-run-as-user test-user
```

Each case has its own task workspace, agent instance, Request Proxy, and run
artifacts. A session fixture starts vLLM lazily and reuses it for matching
server configurations. Changing the model, serving arguments, server environment,
bind address, or deployment scenario restarts the service. Teardown closes it
when pytest exits. Separate pytest invocations do not share a service; do not
use pytest-xdist for this suite if only one model instance is desired.

The ready CI pipeline runs all profiles in one functional job and one pytest
invocation. It uses the existing l20_2 preset and the shared CI runtime image.
Performance and replay suites retain their previous service lifecycle.

## Validation and evidence

The run must finalize, complete its one task, and have no failed tasks. Request
failure counts are not pass criteria. Common files must exist: agents.csv,
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
