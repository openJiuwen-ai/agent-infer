# SWE-bench Verified accuracy E2E

Covers [PR #64](https://github.com/openJiuwen-ai/agent-infer/pull/64): automatic scoring, independent scoring of
existing runs, and per-task regression comparison.

Run on Linux with AgentInfer, vLLM, Claude CLI, Verified/test data, Docker access, and `swebench==4.1.0` in the
evaluation Python environment. Use a non-root benchmark account. See the [benchmark
guide](../../../docs/en/how-to/run-benchmark.md).

The GLM-5.2 baseline and AgentInfer JSON examples select two tasks with one concurrent agent. Adjust model paths, agent
executable, benchmark user, and hardware parameters for your host. Add `"evaluation-config":
"/absolute/path/evaluation.yaml"` to `benchmark_params` for a separate evaluation environment.

## Automatic scoring

```bash
python -m pytest -s -v tests/e2e/acc/test_correctness.py::test_run_with_correctness \
  --test-config-file tests/e2e/acc/cases/glm52_2x1_baseline_correctness.json
```

Use `glm52_2x1_agentinfer_correctness.json` for AgentInfer. Supports `--task-num`, `--max-concurrency`, and
`--benchmark-run-as-user` overrides.

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
