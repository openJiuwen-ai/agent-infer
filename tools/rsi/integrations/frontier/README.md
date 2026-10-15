# Frontier `ve_policy` bridge

`ve_policy.patch` adds a `ve_policy` replica scheduler to a pinned local
[NetX-lab/Frontier](https://github.com/NetX-lab/Frontier) checkout. It preserves the parent
`vllm_v1` capacity, token-budget, and KV-admission model, but delegates waiting-queue
order/admission to an external vllm-evolve `schedule_batch` policy. Candidate source therefore
executes inside a real `python -m frontier.main` process and can change simulator results.

This is a search evaluator, not production vLLM. All resulting artifacts remain
`source=frontier_sim`, `outcome_class=simulator_nonqualifying`, and at most `sim_winner`.

## Pinned setup

The reference experiment used:

- Frontier commit `a4b22df8211864bf229258ecdfbe680f048f2d77`
- patch SHA256 `a8d0262c033ad79bfab299cf7ee96ffacd9b0efb534b15ad7eaa98a6fd56ca3e`
- Frontier's independent Python 3.12 virtual environment

From the vllm-evolve root:

```bash
export VE_FRONTIER_REPO=/absolute/path/to/Frontier
export VE_FRONTIER_PYTHON="$VE_FRONTIER_REPO/.venv/bin/python"
# Apply once, or prove that exactly this patch is already present.
python integrations/frontier/ensure_patch.py "$VE_FRONTIER_REPO"
```

On PowerShell:

```powershell
.\integrations\frontier\apply_patch.ps1 -FrontierRepo C:\path\to\Frontier
```

The patch modifies/adds:

- `frontier/types/replica_scheduler_type.py`
- `frontier/scheduler/replica_scheduler/ve_policy_replica_scheduler.py`
- `frontier/scheduler/replica_scheduler/replica_scheduler_registry.py`
- `frontier/config/config.py`
- the Frontier base cluster scheduler's prefix-cache allowlist

## Runtime contract

| Environment variable | Meaning |
|---|---|
| `VE_POLICY_PATH` | absolute `.py` defining `schedule_batch`; missing is a hard failure |
| `VE_POLICY_ROOT` | vllm-evolve root added to `sys.path` |
| `VE_MARKER_DIR` | directory receiving `ve_policy_marker.json` |

Select the scheduler with `--replica_scheduler_config_type ve_policy`. Its knobs inherit
`vllm_v1`, prefixed with `--ve_policy_scheduler_config_*`, including batch cap, token cap, and
prefix caching. The vllm-evolve runner records exact `cwd`, `argv`, seed, policy path, and policy
root in `frontier_command.json` before every invocation.

The policy view exposes immutable request information:

- request id, arrival time, remaining prompt and requested output tokens;
- prefill/decode state;
- neutral session id;
- `has_prefix_hint`, which is true only when Frontier received a non-empty trace prefix hash.

Official BurstGPT conversion always uses one neutral session and empty prefix hashes because the
dataset does not contain those facts. Prefix hints are enabled only for separately labeled
synthetic workloads.

## Ordering and bounded defer

`ScheduleDecision` has two admission channels:

| Channel | Trigger | Bridge behavior |
|---|---|---|
| forgotten | id is in neither `prefill_batch` nor `defer_ids` | append in parent FCFS order behind chosen ids |
| explicit defer | id appears in `defer_ids` | keep visible at the queue tail |

A forgotten request can therefore never disappear. Consecutive explicit defers are counted; at
the default limit of eight scheduling events, the request is forced to the head and its counter is
reset. This anti-starvation algorithm is single-sourced in `ve_defer.py`. If it cannot be imported,
the bridge ignores the defer request and appends FCFS, preferring loss of optimization over
stranding.

## Honesty marker

At exit, the process writes:

```json
{
  "scheduler": "ve_policy",
  "policy_sha256": "...",
  "invocations": 99,
  "fallbacks": 0,
  "defers": 0,
  "forced": 0
}
```

A candidate trial is invalid if the marker is missing, its SHA differs from the source, it has zero
invocations, or every invocation fell back. The local evolution runner also requires no completion
loss and reports defer/forced counts to expose starvation risk.

## Reproduce the local evolution

```bash
ve frontier-evolve \
  --burstgpt /absolute/path/to/BurstGPT_1.csv \
  --out runs/frontier_local_evolution \
  --seeds 0,1,2 \
  --fragment-size 32 \
  --slo-ttft-ms 200 \
  --max-num-seqs 4 \
  --generations 2 \
  --population 5 \
  --max-total-evals 10
```

See `reports/frontier_local_evolution.md` for the reference command, hashes, lineage, baselines,
held-out metrics, failure history, and limitations.
