# AgentBench functional E2E

One test entry runs Claude Code, DSH, JiuwenSwarm, or ZCode according to the selected
case JSON. Shared helpers prepare SWE-bench Verified, manage vLLM, run BenchKit,
and collect artifacts.

## Cases

| Directory | Agent | Profiles | Endpoint |
| --- | --- | --- | --- |
| `cases/claude/` | Claude Code | single, plan-subagent | /v1/messages |
| `cases/dsh/` | DSH | single, plan-subagent | /v1/chat/completions |
| `cases/jiuwenswarm/` | JiuwenSwarm | code.normal | /v1/chat/completions |
| `cases/zcode/` | ZCode | single, plan-subagent | /v1/chat/completions |

Each case selects `benchmark_params.config`, `agent-type`,
`agent-executable`, `agent-profile`, and `endpoint`. The YAML supplies the
remaining BenchKit settings. Case overrides are validated together before
launching the run.

## Requirements

Run on Linux with GPU access, vLLM, Git, and dataset/model access. Edit the case
model path, serving options, executable, and hardware marks for your host.
All examples specify one L20, tensor parallelism one, a 32K context limit,
and at most four active sequences. The model uses its Hugging Face repository ID.
The `2x1` in case names means two tasks and one concurrent task, not two GPUs.
Cases omit a fixed serving port so the shared helper selects an available
loopback port when jobs share a host.

| Agent | Additional requirements |
| --- | --- |
| Claude Code | Authenticated Claude CLI, tmux; vLLM with /v1/messages support |
| DSH | DSH CLI and Node.js compatible with the repository runtime |
| JiuwenSwarm | JiuwenSwarm CLI, jiuwenswarm-init, jiuwenswarm-start, jiuwenbox-server, bwrap, and sandbox support on the host |
| ZCode | ZCode CLI 0.17.1 and Node.js; launcher must preserve NODE_OPTIONS for prompt loading |

BenchKit also performs runtime-specific CLI and sandbox preflight checks.
Claude `plan-subagent` uses `bypassPermissions` and requires a non-root
account. When pytest is root, configure `benchmark-run-as-user` in the case
or pass `--benchmark-run-as-user USER`; the shared helper applies sudo
wrapping for `plan-subagent`. This option does not wrap `code.normal`.
ZCode plan-subagent also uses the shared ordinary-user benchmark option.
ZCode requires external sandboxing for production benchmarks; see the
[ZCode runtime guide](../../../../docs/en/how-to/run-benchmark.md).

## Run

```bash
pytest -s tests/e2e/function/agentbench/test_swebench.py \
  --test-config-file tests/e2e/function/agentbench/cases/claude/qwen3_coder_claude_2x1_single.json
pytest -s tests/e2e/function/agentbench/test_swebench.py \
  --test-config-file tests/e2e/function/agentbench/cases/dsh/qwen3_coder_dsh_2x1_plan_subagent.json
pytest -s tests/e2e/function/agentbench/test_swebench.py \
  --test-config-file tests/e2e/function/agentbench/cases/jiuwenswarm/qwen3_coder_jiuwenswarm_2x1_code_normal.json
pytest -s tests/e2e/function/agentbench/test_swebench.py \
  --test-config-file tests/e2e/function/agentbench/cases/zcode/qwen3_coder_zcode_2x1_plan_subagent.json \
  --benchmark-run-as-user test-user
```

Use `--task-num` and `--max-concurrency` to override the load, and
`-m 'cuda and cards_1'` to select by case hardware marks. Without a case
file the test skips. Missing prerequisites fail an explicitly selected case.

## Validation and evidence

The Buildkite ready pipeline has separate DSH, Claude Code, JiuwenSwarm, and
ZCode jobs on the `l20_1` preset. Each uses the unified entry and the
`cuda and cards_1` marker selection. Claude Code and ZCode each run both
profiles; JiuwenSwarm runs code.normal. All four jobs explicitly invoke pytest
as root in the pipeline YAML. Dataset preparation and vLLM remain root;
only plan-subagent bench subprocesses switch to `test-user` through the
shared helper. Single and code.normal bench subprocesses remain root.
ZCode's installed bundle uses a shebang pointing to the pinned Node binary;
it does not require a separate launcher script and preserves NODE_OPTIONS.
The image installs Claude Code 2.1.323, JiuwenSwarm 0.2.3 in a separate Python
environment, and ZCode 0.17.1 from the pinned source/toolchain above.
JiuwenSwarm's host/container must permit bubblewrap user namespaces.

All agents must finalize the run, complete every selected task, and record
requests with zero failed requests. All agents also require non-empty run
artifacts, the expected number of completed task index rows, non-empty task
transcripts, and task/workspace paths confined to the run directory.
Claude Code requires its isolated `claude_state/` configuration directory;
JiuwenSwarm requires `jiuwenswarm-instance/home/`; DSH requires `dsh-home/`;
ZCode requires `zcode-home/`.
These post-run path checks do not prove OS-level sandbox isolation.
All four agents reject nonzero failure summaries (including startup failures,
startup timeouts, and harness errors), failure evidence on completed task rows,
and isolated state directories found outside `tasks/`.
DSH additionally retains its HMR startup regression checks from
[PR #75](https://github.com/openJiuwen-ai/agent-infer/pull/75).
ZCode additionally requires its run-status, lineage, configuration, launch
command, hooks, and stderr artifacts. Status must confirm successful execution
and prompt loading; lineage must contain sessions with no failure or pending
work. Empty stderr is valid, and single mode need not create any subagents.

These checks verify execution rather than SWE-bench patch correctness.
A plan-subagent profile does not prove that the model actually delegated work.
Results and vLLM logs remain under the case's `result_root`. Inspect the run
summary, request records, task transcripts, and service logs when reporting
results. Full GPU validation has not been run in the Windows development
environment.
