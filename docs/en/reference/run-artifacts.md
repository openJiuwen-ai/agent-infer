# Run Artifacts

BenchKit writes every run to a new result directory. Raw request facts, service captures, task outcomes, patches, and
environment evidence are authoritative; `summary.json` is their normalized aggregation.

## Directory Layout

```text
<result_dir>/
  manifest.json
  summary.json
  requests.jsonl
  evidence/
    environment.json
    source_control.json
    vllm_metrics_start.prom
    vllm_metrics_end.prom
  tasks/<instance_id>/
    result.json
    model.patch
    transcript.jsonl
    claude-settings.json
    claude-launch-command.txt
    terminal-*.log
  workspaces/<instance_id>/
```

Evidence that cannot be collected is marked unavailable or not applicable in `manifest.json` or `summary.json`. Its
optional file may be absent; missing evidence is never silently converted to numeric zero or success.

## Top-Level Files

| File | Content |
| --- | --- |
| `manifest.json` | Resolved run configuration, CLI metadata, lifecycle status, and evidence availability. |
| `summary.json` | Normalized task, request, vLLM, correctness, and source-health metrics. |
| `requests.jsonl` | One immutable request-fact row per proxied request. |

`compare` reads only finalized, compatible summaries. It rejects incomplete, incorrectly paired, or incompatible
baseline and candidate summaries.

## `evidence/`

| File | Content |
| --- | --- |
| `environment.json` | Host, Python, and runtime environment evidence. |
| `source_control.json` | Branch, commit, and worktree state. |
| `vllm_metrics_start.prom` | Raw vLLM Prometheus metrics captured before the workload. |
| `vllm_metrics_end.prom` | Raw vLLM Prometheus metrics captured after the workload. |

A cold-start claim requires service-start evidence or matching start metrics. When cold state cannot be established,
the comparison reports a warning and the run should not support a release-gate claim.

## `tasks/<instance_id>/`

| File | Content |
| --- | --- |
| `result.json` | Agent outcome, termination reason, topology, and patch status. |
| `model.patch` | Agent-produced patch; it can be absent when no patch was generated. |
| `transcript.jsonl` | Normalized agent conversation events. |
| `claude-settings.json` | Non-secret Claude Code settings used for the task. |
| `claude-launch-command.txt` | Redacted launch command. |
| `terminal-*.log` | Periodic terminal captures and final output. |

Credentials enter through the launch environment and must not appear in settings, command, or log artifacts. If a
secret appears, do not share the result directory and rotate the credential immediately.

## Workspaces and Correctness

`workspaces/<instance_id>/` stores the task checkout and agent working directory. `completed` means only that the agent
execution contract completed. To determine whether `model.patch` resolves a SWE-bench task, apply it at the task's
`base_commit` and execute the `FAIL_TO_PASS` and `PASS_TO_PASS` tests.

See [Benchmark methodology](../explanation/benchmark-methodology.md) for evaluation boundaries and fair-comparison
requirements.
