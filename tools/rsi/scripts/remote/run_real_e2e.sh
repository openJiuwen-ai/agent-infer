#!/usr/bin/env bash
# Local owner for a frozen, saturated, real-vLLM held-out A/B/control suite.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/remote/run_real_e2e.sh \
    --policy /path/to/frozen_winner.py \
    --control /path/to/source_level_ablation.py \
    --workloads /path/to/formal_burstgpt_workloads \
    --frozen-bench-config /path/to/calibration_locked_bench_config.json \
    --required-action-counter deferred_request_actions \
    [--gpus auto|config|0,1] \
    [--env config/remote/gpu.env.example] \
    [--out runs/real_vllm/suites/<name>] \
    [--skip-bootstrap]

This entry point never runs Frontier or materializes a low-load fallback.
Candidate authoring/evolution, workload calibration, winner freeze, and
source-level ablation must already be complete.
EOF
}

env_file="config/remote/gpu.env.example"
policy=""
control=""
workloads=""
frozen_bench_config=""
out=""
skip_bootstrap=0
gpus="auto"
required_action_counters=()

while (( $# )); do
  case "$1" in
    --env)
      env_file="$2"
      shift 2
      ;;
    --policy)
      policy="$2"
      shift 2
      ;;
    --control)
      control="$2"
      shift 2
      ;;
    --workloads)
      workloads="$2"
      shift 2
      ;;
    --frozen-bench-config)
      frozen_bench_config="$2"
      shift 2
      ;;
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --required-action-counter)
      required_action_counters+=("$2")
      shift 2
      ;;
    --out)
      out="$2"
      shift 2
      ;;
    --skip-bootstrap)
      skip_bootstrap=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${env_file}" ]]; then
  echo "ERROR: environment file does not exist: ${env_file}" >&2
  exit 2
fi
if [[ ! -f "${policy}" || ! -f "${control}" ]]; then
  echo "ERROR: --policy and --control must both name readable files" >&2
  exit 2
fi
if [[ ! -d "${workloads}" ]]; then
  echo "ERROR: --workloads must name the three formal BurstGPT workloads" >&2
  exit 2
fi
if [[ ! -f "${frozen_bench_config}" ]]; then
  echo "ERROR: --frozen-bench-config must name the calibration lock JSON" >&2
  exit 2
fi
if (( ${#required_action_counters[@]} == 0 )); then
  echo "ERROR: declare at least one --required-action-counter" >&2
  exit 2
fi

set -a
# shellcheck source=/dev/null
source "${env_file}"
set +a

: "${VE_REMOTE:=gpu-host}"
: "${VE_REMOTE_WORKSPACE:=/workspace/vllm-evolve}"
: "${VE_REMOTE_REPO:=${VE_REMOTE_WORKSPACE}/repo/current}"
: "${VE_MINIFORGE_PREFIX:=${VE_REMOTE_WORKSPACE}/miniforge3}"
: "${VE_CONDA_ENV:=${VE_REMOTE_WORKSPACE}/envs/vllm-evolve}"
: "${VE_CONDA_SH:=${VE_MINIFORGE_PREFIX}/etc/profile.d/conda.sh}"
: "${VE_LOCAL_PYTHON:=./.venv/bin/python}"
export VE_REMOTE VE_REMOTE_WORKSPACE VE_REMOTE_REPO VE_MINIFORGE_PREFIX
export VE_CONDA_ENV VE_CONDA_SH

if [[ ! -x "${VE_LOCAL_PYTHON}" ]]; then
  echo "ERROR: local Python is not executable: ${VE_LOCAL_PYTHON}" >&2
  exit 2
fi

frozen_value() {
  "${VE_LOCAL_PYTHON}" -c \
    'import functools, sys; from vllm_evolve.engine.real_evolve_suite import load_frozen_bench_config; config=load_frozen_bench_config(sys.argv[1])[0]; print(functools.reduce(getattr, sys.argv[2].split("."), config))' \
    "${frozen_bench_config}" "$1"
}

# The immutable formal config, not mutable shell defaults, owns the target
# host/environment/GPU/model identity.
VE_REMOTE="$(frozen_value runner.remote)"
VE_REMOTE_WORKSPACE="$(frozen_value runner.remote_workspace)"
VE_REMOTE_REPO="$(frozen_value runner.remote_repo)"
VE_CONDA_ENV="$(frozen_value runner.conda_env)"
VE_CONDA_SH="$(frozen_value runner.conda_sh)"
VE_GPUS="$(frozen_value environment.gpus)"
VE_MODEL="$(frozen_value engine.model)"
export VE_REMOTE VE_REMOTE_WORKSPACE VE_REMOTE_REPO VE_CONDA_ENV VE_CONDA_SH
export VE_GPUS VE_MODEL

"${VE_LOCAL_PYTHON}" - "${frozen_bench_config}" <<'PY'
import sys
from vllm_evolve.engine.real_evolve_suite import load_frozen_bench_config

config, provenance = load_frozen_bench_config(sys.argv[1])
print(
    "Frozen formal config:",
    provenance["bench_config_sha256"],
    "GPUs=" + config.environment.gpus,
    "TP=" + str(config.engine.tensor_parallel_size),
)
PY

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="${out:-runs/real_vllm/suites/${stamp}}"

action_counter_args=()
for counter in "${required_action_counters[@]}"; do
  action_counter_args+=(--required-action-counter "${counter}")
done

scripts/remote/sync_committed_tree.sh

if (( ! skip_bootstrap )); then
  ssh -o BatchMode=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=40 \
    "${VE_REMOTE}" env \
    VE_REMOTE_WORKSPACE="${VE_REMOTE_WORKSPACE}" \
    VE_REMOTE_REPO="${VE_REMOTE_REPO}" \
    VE_MINIFORGE_PREFIX="${VE_MINIFORGE_PREFIX}" \
    VE_CONDA_ENV="${VE_CONDA_ENV}" \
    VE_GPUS="${VE_GPUS}" \
    bash "${VE_REMOTE_REPO}/scripts/remote/bootstrap_vllm_env.sh"
fi

"${VE_LOCAL_PYTHON}" -m vllm_evolve.engine.real_evolve_suite \
  --policy "${policy}" \
  --control "${control}" \
  --workloads "${workloads}" \
  --out "${out}" \
  --frozen-bench-config "${frozen_bench_config}" \
  "${action_counter_args[@]}" \
  --gpus "${gpus}"

echo "Machine-readable result: ${out}/result.json"
echo "Human report: ${out}/report.md"
