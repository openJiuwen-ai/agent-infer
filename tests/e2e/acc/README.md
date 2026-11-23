# SWE-bench Verified accuracy E2E

Covers [PR #64](https://github.com/openJiuwen-ai/agent-infer/pull/64): automatic scoring, independent scoring of
existing runs, and per-task regression comparison.

Run on Linux with AgentInfer, vLLM, the selected agent CLI, Verified/test data, Docker access,
and `swebench==4.1.0` in the evaluation Python environment. Use a non-root benchmark account for `plan-subagent`.
See the [benchmark guide](../../../docs/en/how-to/run-benchmark.md).

The baseline and AgentInfer JSON examples use `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` on two L20 GPUs (TP=2),
a 65,536-token server context, and two tasks per agent with one concurrent agent. Agent executables are
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

Claude and DSH reuse `swebench_vllm.yaml` and `swebench_vllm_dsh.yaml`, respectively.
ZCode and JiuwenSwarm use `swebench_vllm_zcode.yaml` and `swebench_vllm_jiuwenswarm.yaml`, respectively.
The cases retain each repository config's agent budgets and enable accuracy scoring with `with-correctness`.
The serve context remains 65,536 tokens; DSH currently inherits the 262,144-token agent context default.
Ports are omitted so pytest reserves a free loopback port, then shares it among the agents in that file.

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

## Ready CI

The `ready` pipeline runs only JiuwenSwarm accuracy, covering baseline and AgentInfer with two tasks each.
Its `agentinfer-accuracy-jiuwen` Compose service builds the `accuracy-jiuwen` target, which adds the scorer
and Jiuwen SDK to the common CI dependencies. Claude and ZCode CLIs are installed only in the full nightly image.
Pytest and Jiuwen run as root; the `code.normal` profile does not require `--benchmark-run-as-user`.
Inside that container, the job runs:

```bash
pytest -v tests/e2e/acc/test_correctness.py::test_run_with_correctness -k jiuwenswarm \
  --correctness-ci --correctness-evaluation-config /opt/agentbench/evaluation.yaml
```

The pipeline records pytest output in `ready-logs/jiuwenswarm/pytest.log`, dependency records in
`nightly-logs/jiuwenswarm/`, and all benchmark/scoring artifacts in `test-results/acc/jiuwenswarm/`.
Server logs are also uploaded. Known sandbox failures fail the job; this configuration does not fix the
tracked Jiuwen sandbox issue.

## Nightly CI

### Nightly triggers

The bootstrap in `.buildkite/pipeline.yml` follows vLLM-Omni's
[nightly entry points](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/bootstrap-upload-steps.yml):

| Trigger | Configuration | Code tested |
| ------- | ------------- | ----------- |
| PR label | Add `nightly-test` to an open PR | The PR revision checked out by Buildkite |
| Scheduled build | Buildkite Schedule on `main`, with `NIGHTLY=1` | Latest `main` (`HEAD`) |
| Manual build | New Build with `NIGHTLY=1` | The selected branch and commit |

Ordinary PR builds and main-branch pushes do not run nightly without these conditions.
The `ready` and `nightly-test` labels select their pipelines independently. When both labels are present,
both pipelines run in the same build; execution order follows their dependencies and agent availability.
Their GPU image tags are separate: ordinary ready jobs use `agentinfer-ci:<build-id>-gpu`, Jiuwen accuracy uses
`agentinfer-ci:<build-id>-ready-accuracy-gpu`, and nightly uses `agentinfer-ci:<build-id>-nightly-gpu`.
All use the same Dockerfile: ordinary ready/CPU jobs build `ci`, ready accuracy builds `accuracy-jiuwen`,
and nightly builds `accuracy`. The targets share `ci-deps`, preserving the existing dependency installs
and package mirror. Both accuracy targets also share the scoring and Jiuwen dependency layers.
Accuracy builds reuse those cached layers without forcing another base-image pull with `--pull`.
The nightly file also applies the bootstrap's trigger condition to its outer group, covering image build,
both accuracy jobs, and cleanup. Uploading that file directly still requires the nightly label or `NIGHTLY=1`.

Keep the Buildkite pipeline's entry point at `.buildkite/pipeline.yml`. In its GitHub settings:

- Enable **Build when pull request labels are changed**.
- Disable **Skip when pull request has existing build for commit and branch**, so adding a label can trigger
  another build of the same commit.

After merging the bootstrap changes into `main`, create and enable a schedule in
**Pipeline Settings > Schedules** with these values:

| Setting | Value |
| ------- | ----- |
| Description | Nightly accuracy |
| Cron interval | `0 2 * * * Asia/Shanghai` (daily at 02:00 Beijing time) |
| Branch | `main` |
| Commit | `HEAD` |
| Environment variables | `NIGHTLY=1` |

Buildkite manages schedules outside the step YAML; committing `test-nightly.yml` alone does not create a timer.
See [Buildkite scheduled builds](https://buildkite.com/docs/pipelines/configure/workflows/scheduled-builds).
Use the schedule's **Run now** action to check the trigger before its next scheduled run. The schedule targets
`main`; manual builds can select another branch with `NIGHTLY=1`.

### Accuracy jobs

The nightly pipeline uses the `agentinfer-accuracy` Compose service and invokes pytest directly.
Its two jobs run Claude, DSH, ZCode, and JiuwenSwarm for baseline and AgentInfer respectively,
each with one serve startup. Full agent coverage means eight cases, with two tasks in each case;
it does not select the entire SWE-bench Verified dataset.
Inside that container, run one group with:

```bash
pytest -v tests/e2e/acc/test_correctness.py::test_run_with_correctness -k baseline \
  --correctness-ci --benchmark-run-as-user test-user \
  --correctness-evaluation-config /opt/agentbench/evaluation.yaml
```

Replace `baseline` with `agentinfer` for the other group. Use `-k dsh` to run just DSH in both groups.
`--correctness-ci` requires root in the accuracy container. Its fixtures prepare Docker access, runtime checks,
the shared model cache, and dependency records. The requested benchmark user is prepared for `plan-subagent`;
Jiuwen's `code.normal` profile continues to run as root. Manual runs omit this flag.
The evaluation config overrides the loaded case in memory; the JSON files are preserved.
The pipeline uploads pytest logs per serve group, dependency records per agent,
and benchmark/scoring artifacts per agent and scenario.
Both jobs use the `l20_2` hardware preset. Each job declares its accuracy `PYTEST_ADDOPTS` and Compose plugin
configuration in `test-nightly.yml`; job-level plugins override the hardware preset's defaults.
The pytest command names its test target and group selector. The jobs set their log files to
`nightly-logs/<scenario>/pytest.log`. Selecting just one agent defaults to `nightly-logs/<agent>/pytest.log`.
Explicit log paths take precedence. JUnit output is generated only when requested with `--junitxml`.

Jiuwen runs `jiuwenbox-server` as a local process in the benchmark environment. The CI preflight checks
that its execution account can create the required bubblewrap PID/user namespaces before its benchmark starts.
It also rejects completed runs whose `jiuwenbox-server.stderr.log` contains `bwrap: Can't mount proc`.
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
