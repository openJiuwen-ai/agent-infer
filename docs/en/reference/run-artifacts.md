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

When the `vllm` block in `summary.json` is missing a metrics snapshot, `reason` explains why the capture failed
(for example `metrics_capture_failed: <collector error>`, typically a misconfigured `metrics_url` or an unreachable
metrics endpoint). When there is no collector error, `reason` is `missing_metrics_snapshot`.

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

## Multiple runtimes and Replay artifacts

The run directory also includes `task_index.json`, mapping tasks to agent sessions and summarizing failures. Claude
settings and terminal files apply to Claude only; JiuwenSwarm and DSH use their own isolated directories and
execution evidence.

Replay additionally writes `replay-source-analysis.json`, `replay-plan.json`, and `replay-execution.json`, or
`replay-error.json` on failure. These describe planning, execution, and audit facts, not task correctness. See
[Trace Replay](../how-to/run-trace-replay.md).

DSH 0.2 also records `evidence/dsh-runtime.json` (versions, capabilities, installation lock hash) and task-local
`dsh-effective-config.yaml`, `dsh-lineage.json`, `dsh-run-status.json`, and `dsh-processes.jsonl`.
`dsh-sessions/raw/index.json` links unchanged raw logs to decoded sessions; `transcript.jsonl` is selected by the
stdout root identity. `dsh-launch-command.json` excludes the stdin prompt. Timeout and cancellation retain available
logs and patches; a cancelled outer result may still have `has_patch=false`, so inspect `dsh-run-status.json`.
Lineage validation never rewrites `requests.jsonl`.

For streaming request facts, an SDK closing HTTP after OpenAI `[DONE]` or Anthropic `message_stop` is recorded as
success. Cancellation before the terminal protocol event remains an error.

### DSH execution-control evidence

`dsh-lineage.json` records at most one plan recovery reminder in `plan_recoveries`.
When a guard stops execution, `failure` contains its `reason`, `session_id`, and `message`.
`dsh-run-status.json` retains the same diagnostic; `result.json.termination_reason` records the final classification.
Execution timeouts and external cancellation keep their own final classification. Existing cleanup still collects
raw session logs and any generated patch.

| Classification | Meaning |
| --- | --- |
| `agent_step_limit` | The lead exceeded its 200-step execution budget. |
| `subagent_step_limit` | A child exceeded its 100-step exploration budget. |
| `subagent_read_loop` | The fourth identical read in the last eight tool calls. |
| `subagent_failed` | A child ended with a reason other than `completed`. |
| `plan_not_approved` | The plan protocol was not fulfilled, including after one recovery reminder. |
| `validation_error_loop` | Four equivalent argument-validation or inactive-plan exit errors within eight tool results. |
| `max_tokens` | The model or DSH turn reported length truncation. |

Session collection reads all sessions in the task-local `DSH_HOME` and validates session IDs and parent links;
directory names and log modification times do not select the lead session.
