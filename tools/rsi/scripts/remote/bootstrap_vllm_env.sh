#!/usr/bin/env bash
# Reproducible environment bootstrap. Run as the workspace owner on your GPU host.
set -euo pipefail

workspace="${VE_REMOTE_WORKSPACE:-/workspace/vllm-evolve}"
repo="${VE_REMOTE_REPO:-${workspace}/repo/current}"
env_prefix="${VE_CONDA_ENV:-${workspace}/envs/vllm-evolve}"
miniforge_prefix="${VE_MINIFORGE_PREFIX:-${workspace}/miniforge3}"
evidence_dir="${workspace}/environment"

miniforge_version="26.3.2-2"
miniforge_name="Miniforge3-${miniforge_version}-Linux-x86_64.sh"
miniforge_url="https://github.com/conda-forge/miniforge/releases/download/${miniforge_version}/${miniforge_name}"
miniforge_sha256="42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94"

vllm_version="0.21.0"
vllm_wheel="vllm-${vllm_version}+cu129-cp38-abi3-manylinux_2_34_x86_64.whl"
vllm_url="https://github.com/vllm-project/vllm/releases/download/v${vllm_version}/vllm-${vllm_version}%2Bcu129-cp38-abi3-manylinux_2_34_x86_64.whl"
vllm_sha256="920777691e340df7a8328adfb1e57b9996dbb537edfb654dd32f70844f5f423d"

if [[ ! -d "${workspace}" || ! -w "${workspace}" ]]; then
  echo "ERROR: workspace must already exist and be writable: ${workspace}" >&2
  exit 2
fi
if [[ ! -f "${repo}/pyproject.toml" ]]; then
  echo "ERROR: synchronized vllm-evolve checkout is missing: ${repo}" >&2
  exit 2
fi

mkdir -p \
  "${workspace}/downloads" \
  "${workspace}/envs" \
  "${workspace}/cache/huggingface" \
  "${workspace}/cache/xdg" \
  "${workspace}/runs" \
  "${workspace}/staging" \
  "${workspace}/locks" \
  "${evidence_dir}"

installer="${workspace}/downloads/${miniforge_name}"
if [[ ! -f "${installer}" ]]; then
  curl --fail --location --retry 5 --output "${installer}" "${miniforge_url}"
fi
printf '%s  %s\n' "${miniforge_sha256}" "${installer}" | sha256sum --check -
if [[ ! -x "${miniforge_prefix}/bin/conda" ]]; then
  bash "${installer}" -b -p "${miniforge_prefix}"
fi

# shellcheck source=/dev/null
source "${miniforge_prefix}/etc/profile.d/conda.sh"
if [[ ! -x "${env_prefix}/bin/python" ]]; then
  conda create --yes --prefix "${env_prefix}" python=3.12 pip
fi

wheel="${workspace}/downloads/${vllm_wheel}"
if [[ ! -f "${wheel}" ]]; then
  curl --fail --location --retry 5 --output "${wheel}" "${vllm_url}"
fi
printf '%s  %s\n' "${vllm_sha256}" "${wheel}" | sha256sum --check -

conda run --prefix "${env_prefix}" python -m pip install --upgrade pip
conda run --prefix "${env_prefix}" python -m pip install \
  "${wheel}" \
  --extra-index-url https://download.pytorch.org/whl/cu129
conda run --prefix "${env_prefix}" python -m pip install --editable "${repo}[dev]"
conda run --prefix "${env_prefix}" python -m pip check

export CUDA_VISIBLE_DEVICES="${VE_GPUS:-3}"
IFS=',' read -r -a selected_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#selected_gpus[@]} < 1 || ${#selected_gpus[@]} > 2 )); then
  echo "ERROR: VE_GPUS must select one or two devices, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${selected_gpus[@]}" | sort -u | wc -l | tr -d ' ')" \
      != "${#selected_gpus[@]}" ]]; then
  echo "ERROR: VE_GPUS contains duplicate devices: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_CACHE="${workspace}/cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${workspace}/cache/huggingface/hub"
export TRANSFORMERS_CACHE="${workspace}/cache/huggingface/transformers"
export XDG_CACHE_HOME="${workspace}/cache/xdg"
export PYTHONUNBUFFERED=1

conda list --prefix "${env_prefix}" --explicit > "${evidence_dir}/conda-explicit.txt"
conda run --prefix "${env_prefix}" python -m pip freeze --all \
  > "${evidence_dir}/pip-freeze.txt"
nvidia-smi -q > "${evidence_dir}/nvidia-smi-q.txt"

conda run --no-capture-output --prefix "${env_prefix}" python - \
  > "${evidence_dir}/healthcheck.json" <<'PY'
import json
import os
import platform

import torch
import vllm
from vllm.v1.core.sched.scheduler import Scheduler

payload = {
    "ok": bool(torch.cuda.is_available()),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "visible_gpu_count": torch.cuda.device_count(),
    "visible_gpu_names": [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ],
    "scheduler_api": f"{Scheduler.__module__}.{Scheduler.__name__}",
}
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["ok"] or not 1 <= payload["visible_gpu_count"] <= 2:
    raise SystemExit(1)
PY

cat > "${evidence_dir}/activate.sh" <<EOF
#!/usr/bin/env bash
source "${miniforge_prefix}/etc/profile.d/conda.sh"
conda activate "${env_prefix}"
export CUDA_VISIBLE_DEVICES="\${VE_GPUS:-3}"
export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_CACHE="${workspace}/cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${workspace}/cache/huggingface/hub"
export TRANSFORMERS_CACHE="${workspace}/cache/huggingface/transformers"
export XDG_CACHE_HOME="${workspace}/cache/xdg"
EOF
chmod 0755 "${evidence_dir}/activate.sh"

echo "Environment ready: ${env_prefix}"
echo "Health evidence: ${evidence_dir}/healthcheck.json"
