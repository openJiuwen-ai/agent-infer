# SWE-bench Verified accuracy E2E

Covers [PR #64](https://github.com/openJiuwen-ai/agent-infer/pull/64): automatic scoring, independent scoring of
existing runs, and per-task regression comparison.

Run on Linux with AgentInfer, vLLM, the selected agent CLI, Verified/test data, Docker access,
and `swebench==4.1.0` in the evaluation Python environment. Use a non-root benchmark account for `plan-subagent`.
See the [benchmark guide](../../../docs/en/how-to/run-benchmark.md).

The Qwen3-Coder-30B-A3B-Instruct baseline and AgentInfer JSON examples use BF16 on two L20 GPUs (TP=2),
a 125,536-token server context, and two tasks per agent with one concurrent agent. Agent executables are
resolved from `PATH`; CI must install each selected CLI and expose it on the benchmark user's `PATH`.
Adjust model paths, benchmark user, and hardware parameters for your host. Add `"evaluation-config":
"/absolute/path/evaluation.yaml"` to each selected `benchmark_params` entry for a separate evaluation environment.

## Automatic scoring

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_run_with_correctness \
  --test-config-file tests/e2e/acc/cases/qwen3_coder_2x1_baseline_correctness.json \
  --benchmark-run-as-user test-user
```

Use `qwen3_coder_2x1_agentinfer_correctness.json` for AgentInfer. Supports `--task-num`, `--max-concurrency`, and
`--benchmark-run-as-user` overrides. No case hardcodes a benchmark username.
When pytest runs as root, Claude `plan-subagent` requires `--benchmark-run-as-user USER`
with an existing non-root account that can access Docker. When running as that account directly, omit the option.

## Grouped agent cases

`cases/` contains two files, `qwen3_coder_2x1_baseline_correctness.json` and
`qwen3_coder_2x1_agentinfer_correctness.json`. Each defines one `server_params` object and a `benchmark_params`
array for Claude Code, DSH, ZCode, and JiuwenSwarm. Pytest expands the array into four independent tests and
reuses the same server across them. The server stops before switching files; running the full matrix in one
pytest process starts serve twice. A failed agent test does not prevent later tests from running, unless pytest's
`-x` or `--maxfail` option stops the session. Agent selection with `-k` is preserved.

The server context uses Claude's existing 125,536-token setting for both groups. The other agents previously
used a 65,536-token server context; their benchmark configurations and generation budgets are unchanged.
Configuration names follow PR #92: Claude uses `swebench_vllm.yaml`; DSH, ZCode, and JiuwenSwarm use
`swebench_vllm_dsh.yaml`, `swebench_vllm_zcode.yaml`, and `swebench_vllm_jiuwenswarm.yaml`, respectively.
The latter three configs set a 65,536-token agent context and an 8,192-token output budget.
Accuracy scoring is enabled by `with-correctness` in each case, sharing the agent configs with functional tests.

| Agent | Selector | Profile | Executable on PATH |
| ----- | ------------- | ------- | ------------------ |
| Claude Code | `-k claude` | `plan-subagent` | `claude` |
| DSH | `-k dsh` | `plan-subagent` | `dsh` |
| ZCode | `-k zcode` | `plan-subagent` | `zcode-launcher` |
| JiuwenSwarm | `-k jiuwenswarm` | `code.normal` | `jiuwenswarm` |

The three additional agents use `/v1/chat/completions`; Claude uses `/v1/messages`.
Install DSH `0.2.1-rc.2` with a supported Node runtime. ZCode requires the documented Node-based CLI `0.17.1`
launcher, named `zcode-launcher` on `PATH`, preserving `NODE_OPTIONS`; the native SEA binary is unsupported.
JiuwenSwarm requires `jiuwenbox-server` and `bwrap`. See the benchmark guide for runtime setup.
Run the additional agents as the intended benchmark account; when pytest runs as root, DSH and ZCode
`plan-subagent` cases can use `--benchmark-run-as-user USER`.

For example, run DSH with automatic scoring:

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_run_with_correctness \
  --test-config-file tests/e2e/acc/cases/qwen3_coder_2x1_agentinfer_correctness.json \
  -k dsh --benchmark-run-as-user test-user
```

Compare baseline and AgentInfer results from the same agent and profile.

## Nightly CI

The nightly pipeline uses the `agentinfer-accuracy` Compose service and invokes pytest directly.
Its two jobs run all agents for baseline and AgentInfer respectively, each with one serve startup.
Inside that container, run one group with:

```bash
pytest -v tests/e2e/acc/test_correctness.py::test_run_with_correctness -k baseline \
  --correctness-ci --benchmark-run-as-user test-user \
  --correctness-evaluation-config /opt/agentbench/evaluation.yaml
```

Replace `baseline` with `agentinfer` for the other group. Use `-k dsh` to run just DSH in both groups.
`--correctness-ci` requires root in the accuracy container. Its fixtures prepare the benchmark user,
Docker access, runtime checks, shared model cache, and dependency records. Manual runs omit this flag.
The evaluation config overrides the loaded case in memory; the JSON files are preserved.
The pipeline uploads pytest logs and JUnit reports per serve group, dependency records per agent,
and benchmark/scoring artifacts per agent and scenario.
Shared pytest arguments come from the accuracy hardware preset's `PYTEST_ADDOPTS`; each job only names its
test target and group selector. CI reports default to `nightly-logs/<scenario>/`; selecting just one agent
uses `nightly-logs/<agent>/`. Explicit report paths take precedence when supplied.

Jiuwen runs `jiuwenbox-server` as a local process in the benchmark environment. The CI preflight checks
that the benchmark user can create the required bubblewrap PID/user namespaces before its benchmark starts.
The accuracy suite uses the existing agent adapters without extra tool filtering or sandbox path mappings.
Unexpected confirmation events still fail the task.

## Result directories

Each benchmark saves its runs under `test-results/acc/<agent>/<scenario>/`, where `<scenario>` is
`baseline` or `agentinfer`. Each run gets a unique subdirectory; automatic scoring writes
`evaluations/<evaluation_id>/correctness.json` inside that run directory. Shared server logs are kept
under `test-results/acc/<scenario>/vllm-logs/`. The JSON's `result_root` sets the base directory
(`test-results/acc`); pytest adds the agent and scenario for benchmark artifacts.

For example, Claude baseline results are saved under `test-results/acc/claude/baseline/`.
When running BenchKit directly, use `--result-dir` to select the run directory explicitly.

## Score an existing run

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_evaluate_existing_run \
  --correctness-run-dir /absolute/path/run \
  --correctness-evaluation-config /absolute/path/evaluation.yaml
```

Evaluation config is optional and defaults to the current Python environment. Each evaluation creates a new
`evaluations/<evaluation_id>/` directory. Original data and complete run artifacts must remain accessible; an inference
server is not required.

## Compare correctness

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_compare_correctness \
  --correctness-baseline /path/baseline/evaluations/eval-id/correctness.json \
  --correctness-candidate /path/candidate/evaluations/eval-id/correctness.json
```

Task sets, generation parameters, evaluation configurations, and image identities must be compatible. Previously
resolved tasks regressing, missing evidence, or incompatible evidence fail the test.

## Acceptance and evidence

Scoring checks completed-task coverage, counts, absence of errors and environment failures, finalized manifests,
evidence hashes, official reports and test output, and byte equality between original and frozen patches. Counts and
resolved rate are printed. Unresolved tasks and empty patches are allowed; the sample has no fixed accuracy threshold.
Per-task comparison detects regressions.

Automatic scoring E2E additionally requires every configured task to complete and enter scoring. A failed task
such as `confirmation_hang` fails the case even when the remaining tasks have valid scores. Independent evaluation
of existing runs retains its completed-task scope.

Keep exact commands, console logs, run artifacts, and evaluation reports with accuracy claims. Missing input options
cause skips; skips do not establish successful accuracy validation.
