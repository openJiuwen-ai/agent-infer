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
  --test-config-file tests/e2e/acc/cases/qwen3_coder_fp8_2x1_baseline_correctness.json \
  --benchmark-run-as-user test-user
```

Use `qwen3_coder_fp8_2x1_agentinfer_correctness.json` for AgentInfer. Supports `--task-num`, `--max-concurrency`, and
`--benchmark-run-as-user` overrides. No case hardcodes a benchmark username.
When pytest runs as root, Claude `plan-subagent` requires `--benchmark-run-as-user USER`
with an existing non-root account that can access Docker. When running as that account directly, omit the option.

## Grouped agent cases

`cases/` contains two files, `qwen3_coder_fp8_2x1_baseline_correctness.json` and
`qwen3_coder_fp8_2x1_agentinfer_correctness.json`. Each defines one `server_params` object and a `benchmark_params`
array for Claude Code, DSH, ZCode, and JiuwenSwarm. Pytest expands the array into four independent tests and
reuses the same server across them. The server stops before switching files; running the full matrix in one
pytest process starts serve twice. A failed agent test does not prevent later tests from running, unless pytest's
`-x` or `--maxfail` option stops the session. Agent selection with `-k` is preserved.

The common `mark.ci` is `nightly`. JiuwenSwarm's AgentInfer entry overrides only its CI lane with
`"mark": {"ci": "ready"}` inside `benchmark_params`; its baseline entry inherits `nightly`.
Expanded cases inherit the common hardware/card
markers. Per-benchmark mark overrides may contain only `ci`; they are consumed by E2E collection and are not
passed to BenchKit. Ready selects only Jiuwen's AgentInfer case; nightly selects the other seven cases.

Claude and DSH reuse `swebench_vllm.yaml` and `swebench_vllm_dsh.yaml`, respectively.
ZCode and JiuwenSwarm use `swebench_vllm_zcode.yaml` and `swebench_vllm_jiuwenswarm.yaml`, respectively.
The cases enable accuracy scoring with `with-correctness` and reuse the repository YAMLs.
DSH entries override `context-window` to match the serve limit through the CLI; the shared DSH YAML is preserved.
`context-window` and `max-tokens` in `benchmark_params` map to the corresponding CLI budget overrides.
For Claude, the accuracy fixture sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the case's serve limit and
`CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192`, including when bench runs through sudo.
When pytest runs as root, the shared benchmark helper reclaims the configured repo cache before each case,
hands it to the requested benchmark user, and returns it to root in `finally`, including when a case fails.
Ownership changes stay within the cache and do not follow symlinks. Non-root pytest runs keep their ownership.
Ports are omitted so pytest reserves a free loopback port, then shares it among the agents in that file.
The referenced YAML keeps `backend.base_url: http://127.0.0.1:8000` as an example. E2E passes the selected
port to `vllm serve --port` and overrides that YAML value with the same `--base-url` in the benchmark command.
The proxy's `request_proxy.listen_url: http://127.0.0.1:0` already requests an OS-assigned port.

| Agent | Selector | CI lane | Profile | Executable on PATH |
| ----- | ------------- | ------- | ------- | ------------------ |
| Claude Code | `-k claude` | `nightly` | `plan-subagent` | `claude` |
| DSH | `-k dsh` | `nightly` | `plan-subagent` | `dsh` |
| ZCode | `-k zcode` | `nightly` | `plan-subagent` | `zcode-launcher` |
| JiuwenSwarm | `-k jiuwenswarm` | `ready` (AgentInfer), `nightly` (baseline) | `code.normal` | `jiuwenswarm` |

The three additional agents use `/v1/chat/completions`; Claude uses `/v1/messages`.
Install DSH `0.2.1-rc.2` with a supported Node runtime. ZCode requires the documented Node-based CLI `0.17.1`
launcher, named `zcode-launcher` on `PATH`, preserving `NODE_OPTIONS`; the native SEA binary is unsupported.
JiuwenSwarm requires `jiuwenbox-server` and `bwrap`. See the benchmark guide for runtime setup.
Run the additional agents as the intended benchmark account; when pytest runs as root, DSH and ZCode
`plan-subagent` cases can use `--benchmark-run-as-user USER`.

For example, run DSH with automatic scoring:

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_run_with_correctness \
  --test-config-file tests/e2e/acc/cases/qwen3_coder_fp8_2x1_agentinfer_correctness.json \
  -k dsh --benchmark-run-as-user test-user
```

Compare baseline and AgentInfer results from the same agent and profile.

## Ready CI

The `ready` pipeline runs only JiuwenSwarm's AgentInfer accuracy case with two tasks.
Its `agentinfer-accuracy-jiuwen` Compose service builds the `accuracy-jiuwen` target, which adds the scorer
and Jiuwen SDK to the common CI dependencies. Claude and ZCode CLIs are installed only in the full nightly image.
Pytest and Jiuwen run as root; the `code.normal` profile does not require `--benchmark-run-as-user`.
Inside that container, the job runs:

```bash
pytest -v tests/e2e/acc/test_correctness.py::test_run_with_correctness -k jiuwenswarm \
  -m "L20 and cards_2 and ready" \
  --correctness-ci --correctness-evaluation-config /opt/agentbench/evaluation.yaml
```

The pipeline puts shared invocation logs under `tests/run/acc/jiuwenswarm-<timestamp>/`, complete benchmark
artifacts under `tests/run/acc/jiuwenswarm-agentinfer-<timestamp>/`, and exported score summaries under
`tests/result/acc/jiuwenswarm-agentinfer-<timestamp>/`. It uploads both the run and result directories.
Known sandbox failures fail the job; this configuration does not fix the tracked Jiuwen sandbox issue.

Jiuwen runs `jiuwenbox-server` as a local process in the benchmark environment. The CI preflight checks
that its execution account can create the required bubblewrap PID/user namespaces before its benchmark starts.
It also rejects completed runs whose `jiuwenbox-server.stderr.log` contains `bwrap: Can't mount proc`.
The accuracy suite uses the existing agent adapters without extra tool filtering or sandbox path mappings.
Unexpected confirmation events still fail the task.

## Nightly CI

### Nightly triggers

The bootstrap and nightly group use the same `NIGHTLY=1` or `nightly-test` condition as vLLM-Omni's
[nightly pipeline](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/cuda/test-nightly.yml):

| Trigger | Configuration | Code tested |
| ------- | ------------- | ----------- |
| PR label | Add `nightly-test` to an open PR | The PR revision checked out by Buildkite |
| Scheduled build | Buildkite Schedule on `main`, with `NIGHTLY=1` | Latest `main` (`HEAD`) |
| Manual build | New Build with `NIGHTLY=1` | The selected branch and commit |

`nightly-test` is a GitHub PR trigger label; `nightly` is a pytest marker. The trigger label starts the
pipeline, while pytest markers select the cases inside its jobs. Both `ready` and `nightly` markers are registered.

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
Its baseline job runs Claude, DSH, ZCode, and JiuwenSwarm; its AgentInfer job runs Claude, DSH, and ZCode.
Each job starts serve once. Nightly covers seven cases, with two tasks in each case;
JiuwenSwarm's AgentInfer case runs only in ready CI. Nightly does not select the entire SWE-bench Verified dataset.
Inside that container, run one group with:

```bash
pytest -v tests/e2e/acc/test_correctness.py::test_run_with_correctness -k baseline \
  -m "L20 and cards_2 and nightly" \
  --correctness-ci --benchmark-run-as-user test-user \
  --correctness-evaluation-config /opt/agentbench/evaluation.yaml
```

Replace `baseline` with `agentinfer` for the other group. Use `-k dsh` to run just DSH in both groups.
`--correctness-ci` requires root in the accuracy container. Its fixtures prepare Docker access, runtime checks,
the shared model cache, and dependency records. The requested benchmark user is prepared for `plan-subagent`.
Manual runs omit `--correctness-ci`.
The evaluation config overrides the loaded case in memory; the JSON files are preserved.
The pipeline uploads shared invocation logs, complete benchmark/scoring artifacts per agent and scenario,
and the corresponding exported score summaries.
Both jobs use the `l20_2` hardware preset. Each job declares its accuracy `PYTEST_ADDOPTS` and Compose plugin
configuration in `test-nightly.yml`; job-level plugins override the hardware preset's defaults.
The pytest command names its test target and group selector. Each job automatically saves `pytest.log` under
`tests/run/acc/<scenario>-<timestamp>/`. Selecting just one agent uses that agent's name in place of the scenario.
Explicit log paths take precedence. JUnit output is generated only when requested with `--junitxml`.

## Result directories

Automatic scoring uses paired directories for each agent and scenario. The feature is `<agent>-<scenario>`,
for example `claude-baseline`. Both directories share the same timestamp and unique suffix:

```text
tests/run/acc/claude-baseline-20261124-103000-a1b2c3/
  manifest.json
  task_index.json
  tasks/
  evaluations/<evaluation_id>/
    correctness.json
    task_results.jsonl
    evaluation_manifest.json
    harness.stdout.log
    harness.stderr.log
    inputs/
    raw/

tests/result/acc/claude-baseline-20261124-103000-a1b2c3/
  correctness.json
```

`run` keeps the complete benchmark and original evaluation evidence, including task logs, patches, official reports,
and raw harness logs. `result/correctness.json` is a byte-for-byte copy of the original summary for convenient
inspection.
The matching directory name under `tests/run/acc/` identifies the complete original benchmark and evaluation.
Export happens before the pytest assertions, including when bench exits nonzero after producing an error summary.
Runs that fail before producing a score have no exported `correctness.json`.

Shared pytest/server logs and dependency records sit under `tests/run/acc/<selection>-<timestamp>/`.
The selection is an agent (`-k jiuwenswarm`), scenario (`-k baseline` or `-k agentinfer`), or `all` for a full run
or other keyword expressions. Selecting one JSON without a recognized keyword uses its scenario.
This directory contains `pytest.log`, `<agent>/*-requirements.txt` with `--correctness-ci`, and
`<scenario>/vllm-logs/` for shared workers. It is printed as `[correctness] artifacts=...`.
Manual and CI runs use the same layout. The JSON's `result_root` sets the raw run base directory;
cases in one invocation must share it. Score exports use `tests/result/acc/`.
`--collect-only` does not create artifact directories. Existing functional/performance suites retain their
current output paths.
Generated `tests/run/` and `tests/result/` directories are ignored by Git and excluded from pytest collection.
Explicit `--log-file` and `--junitxml` paths take precedence over the automatic layout.
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
Use the original `correctness.json` under `tests/run/acc/.../evaluations/<evaluation_id>/` for this command:
its sibling manifests, task results, frozen inputs, and raw evidence are required for validation.
The exported `tests/result/acc/.../correctness.json` provides a summary. Find the complete evaluation under the
same-named directory in `tests/run/acc/`.

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
