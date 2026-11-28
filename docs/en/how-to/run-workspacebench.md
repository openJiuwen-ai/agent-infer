# Workspace-Bench Full with AgentBench

Run these commands at the repository root on Linux. Preparation downloads both Full task trees and
both role filesystem archives (about 38 GB compressed for the archives alone). Reserve space for extraction, the
role snapshots and concurrent task copies.

```bash
DATA=agentinfer/agentbench/data/workspacebench
vllm bench serve --agentinfer prepare workspacebench --output-dir "$DATA"
```

Preparation accepts `--upstream-checkout` pointing to an existing checkout of the pinned official revision.
Retry an interrupted prepare with the same destination.
Each language contains 388 tasks. The default selection pairs IDs 2 and 1 across English and Chinese. Select all
776 language-specific instances using `--selection-path "$DATA/task-lists/all.txt"`.

## Run in a container

Use the prepared upstream checkout as the first Docker build context.
Set `WSB_UPSTREAM` to your external checkout path if used.

```bash
WSB_UPSTREAM="$DATA/upstream"
docker build -t workspace-bench:local -f "$WSB_UPSTREAM/evaluation/docker/Dockerfile" "$WSB_UPSTREAM"
docker build -t agentinfer/workspacebench:3fbd0f1 -f docker/workspacebench.Dockerfile .
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/workspacebench_jiuwenswarm.yaml \
  --base-url http://127.0.0.1:8000 --model YOUR_MODEL --result-dir results/wsb-jiuwen
```

Jiuwen uses WorkSwarm 0.2.8 and `agent.code.normal` in a non-root, disposable container. The runtime uses the host
network for the local model proxy. JiuwenBox requires nested unprivileged user and mount namespaces. The Jiuwen
container drops all capabilities and relaxes Docker's outer seccomp, AppArmor and system-path restrictions for its
inner bubblewrap sandbox. The displayed source workspace is a writable copy of the complete role snapshot.

The `single` profile for `claude` and `dsh` uses the upstream SDK adapters. Claude needs `--endpoint /v1/messages`.
DSH, Jiuwen and ZCode need `/v1/chat/completions`. To run ZCode, extend the image with the supported Node-based
0.17.1 launcher, set `AGENTBENCH_WORKSPACEBENCH_IMAGE` to that image, and supply
`--agent-type zcode --agent-profile single --agent-executable zcode`. Executable paths for Jiuwen and ZCode refer
to the inside of the image. [ZCode setup](run-benchmark.md#zcode)

## Run JiuwenSwarm on the host

Prepare the inputs once using the same `prepare` command. Each run copies the complete role workspace into a fresh
directory per task, collects deliverables and trace, and cleans up the working copy. The official runner uses
copy-on-write when available and falls back to copying file contents.

Activate the Python environment with AgentInfer, WorkSwarm 0.2.8 and OpenJiuwen 0.1.20. Install bubblewrap and the
office tools required by your selected tasks. JiuwenBox continues to sandbox the agent's file and command tools.

```bash
vllm bench serve --agentinfer run \
  --config agentinfer/agentbench/configs/workspacebench_jiuwenswarm.yaml \
  --workspacebench-execution host \
  --base-url http://127.0.0.1:8000 --model YOUR_MODEL --result-dir results/wsb-jiuwen-host
```

For a checkout supplied externally during prepare, pass `--workspacebench-checkout /path/to/Workspace-Bench`.
Host execution uses the pinned runner from that checkout or the prepared data directory's `upstream/` checkout.
The optional grading stage uses the same judge image as Docker runs. Set `AGENTBENCH_WORKSPACEBENCH_IMAGE` to override it.

## Official rubric grading

Set `JUDGE_BASE_URL`, `JUDGE_MODEL`, and `JUDGE_API_KEY` for your trusted Anthropic-compatible judge. Run
`vllm bench serve --agentinfer evaluate --result-path results/wsb-jiuwen` or add `--with-correctness` to the run
command. The judge sees the task inputs, execution evidence and candidate deliverables after performance
measurement finishes. A judge configuration can also be supplied using `evaluate --config` or
`run --evaluation-config`. Its fields are `baseUrl`, `model`, `apiKey`, optional filename-safe `model_name`, and
optional per-task `timeout_seconds`. Values can reference environment variables using `${VAR}`.

Each run records its selection and execution environment in `workspacebench.json`. Each task's `official/<id>/` contains
`output/`, `agent.json` and native logs under `raw/`. Every evaluation writes a new
`evaluations/<id>/correctness.json`, plus the official rubric and dependency graph files per scored task. Retain
the dataset inputs for regrading. Container runs default to their recorded image.
The report gives the proportion of passed rubrics among scored rubrics and
the proportion of selected tasks with a complete judgment, separately for each language. Incomplete
evaluations exit nonzero and retain available scores.
[Official evaluation code](https://github.com/OpenDataBox/Workspace-Bench/tree/3fbd0f1a136720fece86786545983e26642c3db2/evaluation)
