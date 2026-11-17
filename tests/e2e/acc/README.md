# SWE-bench Verified accuracy E2E

Covers [PR #64](https://github.com/openJiuwen-ai/agent-infer/pull/64): automatic scoring, independent scoring of
existing runs, and per-task regression comparison.

Run on Linux with AgentInfer, vLLM, the selected agent CLI, Verified/test data, Docker access,
and `swebench==4.1.0` in the evaluation Python environment. Use a non-root benchmark account. See the [benchmark
guide](../../../docs/en/how-to/run-benchmark.md).

The Qwen3-Coder-30B-A3B-Instruct baseline and AgentInfer JSON examples use BF16 on two L20 GPUs (TP=2),
a 65,536-token context, and two tasks with one concurrent agent. The agent executable is
`claude`, resolved from `PATH`; CI must install Claude CLI and expose it on the benchmark user's `PATH`.
Adjust model paths, benchmark user, and hardware parameters for your host. Add `"evaluation-config":
"/absolute/path/evaluation.yaml"` to `benchmark_params` for a separate evaluation environment.

## Automatic scoring

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_run_with_correctness \
  --test-config-file tests/e2e/acc/cases/claude/qwen3_coder_2x1_baseline_correctness.json
```

Use `qwen3_coder_2x1_agentinfer_correctness.json` for AgentInfer. Supports `--task-num`, `--max-concurrency`, and
`--benchmark-run-as-user` overrides.

## Agent cases

Each agent has a baseline and an AgentInfer case; replace `<scenario>` with `baseline` or `agentinfer`.
Configuration names follow PR #92: Claude uses `swebench_vllm.yaml`; DSH, ZCode, and JiuwenSwarm use
`swebench_vllm_dsh.yaml`, `swebench_vllm_zcode.yaml`, and `swebench_vllm_jiuwenswarm.yaml`, respectively.
The latter three configs set a 65,536-token agent context and an 8,192-token output budget.
Accuracy scoring is enabled by `with-correctness` in each case, sharing the agent configs with functional tests.

| Agent | Case path under `cases/` | Profile | Executable on PATH |
| ----- | ------------- | ------- | ------------------ |
| Claude Code | `claude/qwen3_coder_2x1_<scenario>_correctness.json` | `plan-subagent` | `claude` |
| DSH | `dsh/qwen3_coder_dsh_2x1_<scenario>_correctness.json` | `plan-subagent` | `dsh` |
| ZCode | `zcode/qwen3_coder_zcode_2x1_<scenario>_correctness.json` | `plan-subagent` | `zcode-launcher` |
| JiuwenSwarm | `jiuwenswarm/qwen3_coder_jiuwenswarm_2x1_<scenario>_correctness.json` | `code.normal` | `jiuwenswarm` |

The three additional agents use `/v1/chat/completions`; Claude uses `/v1/messages`.
Install DSH `0.2.1-rc.2` with a supported Node runtime. ZCode requires the documented Node-based CLI `0.17.1`
launcher, named `zcode-launcher` on `PATH`, preserving `NODE_OPTIONS`; the native SEA binary is unsupported.
JiuwenSwarm requires `jiuwenbox-server` and `bwrap`. See the benchmark guide for runtime setup.
Run the additional agents as the intended benchmark account; when pytest runs as root, DSH and ZCode
`plan-subagent` cases can use `--benchmark-run-as-user USER`.

For example, run DSH with automatic scoring:

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_run_with_correctness \
  --test-config-file tests/e2e/acc/cases/dsh/qwen3_coder_dsh_2x1_agentinfer_correctness.json
```

Compare baseline and AgentInfer results from the same agent and profile.

## Result directories

Each case saves its runs under `test-results/acc/<agent>/<scenario>/`, where `<scenario>` is
`baseline` or `agentinfer`. Each run gets a unique subdirectory; automatic scoring writes
`evaluations/<evaluation_id>/correctness.json` inside that run directory. Server logs are kept
under the same result root in `vllm-logs/`.

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

Keep exact commands, console logs, run artifacts, and evaluation reports with accuracy claims. Missing input options
cause skips; skips do not establish successful accuracy validation.
