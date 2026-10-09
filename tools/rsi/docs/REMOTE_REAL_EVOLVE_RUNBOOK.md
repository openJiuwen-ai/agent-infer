# Local Frontier → remote real-vLLM runbook

This is the operational contract for a user-configured GPU host. Search, source
authoring, adoption, rollback, and follow-up evolution stay on the local Codex
host. The remote host is only an isolated real-vLLM executor.

## Fixed remote layout

Choose a workspace writable by your SSH account and set `VE_REMOTE_WORKSPACE`.
The following is an example layout, not a preconfigured deployment:

```text
/workspace/vllm-evolve/
├── miniforge3/                 pinned Miniforge installation
├── envs/vllm-evolve/           isolated Python/vLLM environment
├── repo/
│   ├── releases/<git-sha>/     immutable committed source trees
│   └── current -> releases/... atomic active-source link
├── staging/                    content-addressed inputs
├── runs/<run-id>/              immutable experiment evidence
├── locks/                      one lock per selected GPU
├── cache/{huggingface,xdg}/    project-owned caches
└── environment/                package locks and health evidence
```

Nothing overwrites an existing release or run directory.

## Run a frozen validation suite

Copy `config/remote/gpu.env.example` to a local `.env` file and set the SSH
alias and paths for your host. Do not commit credentials or host-specific settings.
Prepare and freeze the candidate, source-level ablation, formal workloads and
calibration configuration before invoking the suite:

```bash
scripts/remote/run_real_e2e.sh \
  --env .env \
  --policy /path/to/frozen_winner.py \
  --control /path/to/source_level_ablation.py \
  --workloads /path/to/formal_burstgpt_workloads \
  --frozen-bench-config /path/to/calibration_locked_bench_config.json \
  --required-action-counter deferred_request_actions
```

This entry point runs the frozen real-vLLM suite. It does not run Frontier,
author candidates, or replace missing calibration with a low-load fallback.
The required action counter must match the mechanism being validated.

After the first healthy deployment, `--skip-bootstrap` avoids repeating the
idempotent installer.

## Environment and activation

The bootstrap pins and verifies the Miniforge installer and the vLLM 0.21.0
CUDA 12.9 wheel by SHA-256, creates Python 3.12 in an isolated prefix, installs
the exact committed project tree, and records:

```text
environment/healthcheck.json
environment/conda-explicit.txt
environment/pip-freeze.txt
environment/nvidia-smi-q.txt
environment/activate.sh
```

Activation on the box:

```bash
source /workspace/vllm-evolve/environment/activate.sh
```

The environment, source release, model cache, input workload, and every run
are therefore independently inspectable.

## Two-GPU hard boundary

Real evolution and the formal suite default to `--gpus auto`. The controller
queries the remote inventory once, excludes devices with a compute process or
more than 1024 MiB in use, selects the lowest-index same-model/same-capacity
group required by the topology, and freezes its IDs and UUIDs for the entire
round. Baseline, candidate, and mechanism control therefore never drift across
physical GPUs. Use `--gpus config` to honor a frozen config or pass explicit
IDs when needed.

The resolved `VE_GPUS` must contain one or two unique device indices/UUIDs. The
local dispatcher and remote worker both reject zero, duplicate, or more than
two devices, and tensor parallel size cannot exceed the visible count. Before
launch, the worker takes an advisory per-GPU lock and repeats the busy check. It
records:

```text
gpu_before.json
gpu_samples.jsonl
gpu_summary.json
gpu_after.json
```

Unknown jobs are never preempted. If no suitable idle group exists, the attempt
fails closed and records why each GPU was excluded.

## Measurement and adoption

Every real round uses the same model, exact token IDs, arrival offsets, engine
configuration, seed set, and request count for the default scheduler,
candidate, and mechanism control. It covers:

- `burstgpt_test`;
- `stress_test_moderate`;
- `stress_test_severe`.

The default real model is `Qwen/Qwen2.5-0.5B-Instruct`: its declared 32K
context covers the unchanged 4480-token severe requests. The earlier
`facebook/opt-125m` default is intentionally not used here because its 2048
position limit would require truncating the workload and invalidate the
simulator-to-real comparison.

The report includes request and output-token throughput, completion/error
rates, TTFT/TPOT/E2E p50/p95/p99, CV, wall time, GPU utilization, and memory
peaks. Adoption requires all of:

- the plugin loaded, reordered the real vLLM waiting queue, and never fell back;
- median gain versus the strong baseline ≥3%;
- positive gain on at least two scenarios and no scenario below −2%;
- no completion loss and primary-metric CV ≤0.10;
- measured fixed-set quality non-regression;
- the same cross-scenario gate against the no-decode-signal control, proving
  the decode-aware mechanism rather than incidental coefficient tuning.

Frontier output remains `simulator_nonqualifying`; only the real-vLLM suite can
produce adoption evidence.

## Failure, cleanup, and feedback

Each worker has a bounded timeout, owns a unique port and process group, and
kills only its own vLLM process carrying that port. Success and failure both
return `status.json`, stdout/stderr, server log, hashes, config, workload, and
GPU evidence to the local artifact directory.

If the candidate is rejected, Codex reads `result.json`, identifies which real
bottleneck or scenario invalidated the simulated hypothesis, and feeds that
evidence into the next local authoring/evolution round. The next candidate gets
a new source hash and isolated remote run. No remote process authors or edits
candidate source.
