# Run complete benchmark validation

Start with the smoke test that needs no model service. It exercises the production CLI, configuration, dataset,
workspace, runtime dispatch, request proxy, trace, and artifact finalization:

```bash
python -m pytest tests/agentbench/test_benchmark_smoke.py -q
```

It uses a local test runtime and loopback backend, covering single-task artifacts and concurrent isolation.
It does not validate model quality or proxy multiprocessing ownership.

## Run a live scenario

`tests/e2e/` provides a JSON-driven pytest entry point, server lifecycle management, and run validation.
Copy and edit a suitable case, checking `serve_env`, `server_params`, `benchmark_params`, and `result_root`.
Bundled GLM cases contain deployment-specific NPU paths and parallelism settings; adapt them to the current host.

```bash
python -m pytest -s -v tests/e2e/run_benchmark.py \
  --test-config-file /path/to/case.json \
  --task-num 1 --max-concurrency 1
```

Baseline and candidate cases are independent files and run separately. The selected model, vLLM, agent CLI, dataset,
and isolation dependencies must be available. For `plan-subagent` invoked as root, use `--benchmark-run-as-user USER`
to select the actual agent user. Choose an unused port and a new result directory; this E2E harness manages service
processes and must not share a port with unrelated workloads.

Afterward, compare the finalized directories with `vllm bench serve --agentinfer compare`.
Service logs and run artifacts live under the configured result root. Execution completion is not SWE-bench patch
correctness. See [E2E tests](../../../tests/e2e/README.md) for case fields and
[benchmark methodology](../explanation/benchmark-methodology.md) for comparison requirements.
