# E2E functional tests (`tests/e2e/function`)

Correctness-oriented GPU/NPU E2E checks live here. They are **not** performance A/B suites (`tests/e2e/perf/`).

Default CI skips all of `tests/e2e` (`pytest --ignore=tests/e2e`). Run these on a machine with vLLM,
weights, and (for replay) trace files.

## Position in the repo

```text
tests/e2e/
├── helpers/          # shared: case JSON, vLLM lifecycle, validate_completed_run
├── perf/             # BenchKit plan-subagent; baseline vs AgentInfer
└── function/         # correctness E2E (replay, AgentBench agents, …)
    ├── agentbench/
    │   └── dsh/      # DSH + SWE-bench Verified
    └── tracereplay/
        ├── cases/    # inferact / tracelab / agentx case JSON
        ├── helpers/
        └── test_replay.py
```

## DSH + SWE-bench (`agentbench/dsh/`)

Functional BenchKit run (not replay): Qwen3-Coder-30B-A3B-Instruct-FP8 on 2× L20 + DSH on SWE-bench Verified.

| Case JSON | profile | task-num | max-concurrency | timeout | Source |
| --------- | ------- | -------- | --------------- | ------- | ------ |
| `glm52_dsh_8x4.json` | `single` | 8 | 4 | 14400 | [issue #14](https://github.com/openJiuwen-ai/agent-infer/issues/14) HMR regression |
| `glm52_dsh_8x4_plan_subagent.json` | `plan-subagent` | 8 | 4 | 3600 | [PR #62](https://github.com/openJiuwen-ai/agent-infer/pull/62) DSH `plan-subagent` path |

```bash
pytest -s -v tests/e2e/function/agentbench/dsh/test_dsh_swebench.py::test_dsh_swebench \
  --test-config-file tests/e2e/function/agentbench/dsh/cases/glm52_dsh_8x4_plan_subagent.json
```

The `single` case uses the same pytest entry with `cases/glm52_dsh_8x4.json`.
`plan-subagent` on root should pass `--benchmark-run-as-user` (or edit `benchmark-run-as-user` in the case JSON).

Case JSON sets `"functional_validation": "dsh_swebench"`; post-run checks live in
[`../helpers/functional.py`](../helpers/functional.py) (summary/manifest, no HMR startup failure, DSH trace,
`dsh-home` confinement).

| Concern | `perf/` | `function/agentbench/dsh` | `function/tracereplay` |
| ------- | ------- | ------------------------- | ---------------------- |
| Bench command | `vllm bench serve --agentinfer run` | same | `vllm bench serve --agentinfer replay` |
| Default YAML | `swebench_vllm.yaml` (in case) | `swebench_vllm_dsh.yaml` | `replay_*.yaml` |
| Pass criteria | Completes; perf compare elsewhere | Completes + DSH artifact checks | Cold replay × N + compare run 1 vs 2 |

## Trace replay (`tracereplay/`)

Inferact / TraceLab / AgentX cold-start repeatability. Layout and commands:
[`tracereplay/README.md`](tracereplay/README.md).
