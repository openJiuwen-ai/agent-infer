# Evaluate SWE-bench Verified correctness

Evaluate existing AgentBench patches using official SWE-bench **4.1.0**. Only
tasks with outcome=completed are selected; the finalized run may contain failed
tasks. No model requests are made. Version 5 requires a different dataset schema.
The evaluation runner targets Linux and uses POSIX process groups for bounded
shutdown. On Windows, run it inside a Linux environment such as WSL2.

The configuration file is optional. By default, evaluation uses the Python
environment running AgentInfer, which must have swebench==4.1.0 installed:

```bash
vllm bench serve --agentinfer evaluate --result-path results/run-a
```

Defaults are 2 workers, a 1800-second instance timeout and a 7200-second overall
timeout. Use --config only for a separate interpreter or custom settings.

## Setup and evaluation

To use a separate evaluator with Docker access, create evaluation.yaml:

```bash
uv venv --python 3.12 /path/to/swebench-venv
uv pip install --python /path/to/swebench-venv/bin/python 'swebench==4.1.0'
```

```yaml
harness_python: /path/to/swebench-venv/bin/python
expected_harness_version: "4.1.0"
max_workers: 2
timeout_seconds: 1800
overall_timeout_seconds: 7200
cleanup_timeout_seconds: 30
```

```bash
vllm bench serve --agentinfer evaluate \
  --result-path results/run-a \
  --config evaluation.yaml \
  --output-dir results/run-a/evaluations/eval-001
```

The source needs finalized manifest/summary files, a consistent task index,
per-task results/prompts and the local Verified dataset referenced by the manifest.
Task IDs, repository and base commit are checked. With no completed tasks, no
score is produced. Relative interpreter paths resolve against the configuration
file; unknown fields and unsupported harness versions are rejected.
Preflight checks the installed SWE-bench version before importing evaluator modules
and collecting environment details; a version mismatch prevents scoring from starting.

Omit --output-dir to create a unique directory under the run's evaluations/.
Existing directories are not overwritten. Original patches, manifest and
performance summary are preserved. First use downloads official Docker images.
Finalization removes containers left by this evaluation in one batch. Cleanup
failures or timeouts are recorded in the evaluation manifest and mark the evaluation failed.

## Results

Each selected task gets one record in task_results.jsonl. The correctness.json
counts contain exactly total, resolved, unresolved, empty_patch and error.
The only accuracy metric is resolved_rate = resolved / total; errors remain in
the denominator. This describes the completed-task subset, not all Verified tasks.

An absent patch is only empty when the result and DSH export status confirm
no_changes. Otherwise it is an error. Failed source tasks are excluded even when
they have patches. F2P/P2P details stay in official per-task reports under raw/.
Commands, logs, versions, image identities and hashes are retained.

Exit codes: 0 for finalized evaluation without errors (unresolved is valid),
1 for input/execution/evaluation errors, 2 for invalid configuration, and 130
for interruption. Errors still produce summaries once the task set is known.

## Comparison

```bash
vllm bench serve --agentinfer compare-correctness \
  --baseline results/baseline/evaluations/eval-001/correctness.json \
  --candidate results/candidate/evaluations/eval-001/correctness.json
```

Comparison checks hashes, task sets, dataset contents, recorded generation
configuration/prompts, evaluator environment and image identity. Incompatible or
incomplete evidence yields insufficient_evidence; it never silently intersects
task sets. An unrecorded model checkpoint revision cannot be independently verified.

The JSON report lists regressions and improvements even when total accuracy is
unchanged. Any task regression fails the comparison; improvements do not offset
regressions. Exit codes: 0 (passed), 1 (regression detected), 2 (insufficient
evidence). This is a fixed-task check, not a statistical capability claim.

See the [official guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
and [benchmark methodology](../explanation/benchmark-methodology.md).
