#!/usr/bin/env bash
# One-task baseline smoke: Claude Code -> Request Proxy -> vLLM /v1/messages.
set -euo pipefail

REPO=${REPO:-$(pwd)}
CONFIG=${CONFIG:-$REPO/agentinfer/agentbench/configs/swebench_vllm.yaml}
VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000}
TASK_NUM=${TASK_NUM:-1}
CONCURRENCY=${CONCURRENCY:-1}
RESULT_DIR=${RESULT_DIR:-$REPO/agentinfer/agentbench/results/baseline-smoke-$(date +%Y%m%d_%H%M%S)}

command -v vllm >/dev/null || { printf 'BLOCKED: vllm executable is unavailable\n' >&2; exit 2; }
command -v claude >/dev/null || { printf 'BLOCKED: Claude Code executable is unavailable\n' >&2; exit 2; }
curl -fsS "$VLLM_BASE_URL/v1/models" >/dev/null || {
  printf 'BLOCKED: vLLM is not ready at %s\n' "$VLLM_BASE_URL" >&2
  exit 2
}

exec vllm bench serve --agentinfer run \
  --config "$CONFIG" \
  --base-url "$VLLM_BASE_URL" \
  --task-num "$TASK_NUM" \
  --max-concurrency "$CONCURRENCY" \
  --result-dir "$RESULT_DIR"
