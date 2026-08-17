#!/usr/bin/env bash
# Cold upstream async scheduler -> cold AgentInfer scheduler bridge -> compare.
set -euo pipefail

REPO=${REPO:?set REPO to the AgentInfer checkout}
VENV=${VENV:?set VENV to the benchmark virtualenv}
MODEL=${MODEL:?set MODEL to the model path}
AGENT_BIN=${AGENT_BIN:-${CLAUDE_BIN:-claude}}
TASK_NUM=${TASK_NUM:-1}
CONCURRENCY=${CONCURRENCY:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
VLLM_ENV_SCRIPT=${VLLM_ENV_SCRIPT:-}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_TMUX=${VLLM_TMUX:-agentinfer-scheduler-e2e-vllm}
LOG_DIR=${LOG_DIR:-$REPO/benchkit-logs}
CONFIG=${CONFIG:-$REPO/agentinfer/agentbench/configs/swebench_vllm.yaml}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_DIR=${BASELINE_DIR:-$REPO/agentinfer/agentbench/results/scheduler-baseline-$TIMESTAMP}
CANDIDATE_DIR=${CANDIDATE_DIR:-$REPO/agentinfer/agentbench/results/scheduler-candidate-$TIMESTAMP}
BASELINE_LOG=$LOG_DIR/scheduler-baseline-$TIMESTAMP.log
CANDIDATE_LOG=$LOG_DIR/scheduler-candidate-$TIMESTAMP.log
UPSTREAM_SCHEDULER=vllm.v1.core.sched.async_scheduler.AsyncScheduler
AGENTCACHE_SCHEDULER=agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge
AGENTCACHE_IDENTITY_MIDDLEWARE=agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware
AGENTCACHE_LIFECYCLE_MIDDLEWARE=agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware
LIFECYCLE_SOCKET=${LIFECYCLE_SOCKET:-/tmp/agentinfer-vllm-lifecycle-$TIMESTAMP.sock}
BACKEND_ID=${BACKEND_ID:-vllm-local}
SCHEDULE_INTERVAL_SECONDS=${SCHEDULE_INTERVAL_SECONDS:-5}
PRIVILEGED_MAX_CONTEXT_TOKENS=${PRIVILEGED_MAX_CONTEXT_TOKENS:-85000}
TTL_PREFILL_MODEL_INTERCEPT_SECONDS=${TTL_PREFILL_MODEL_INTERCEPT_SECONDS:-0.0}
TTL_PREFILL_MODEL_LINEAR_SECONDS_PER_1K_TOKENS=${TTL_PREFILL_MODEL_LINEAR_SECONDS_PER_1K_TOKENS:-0.0}
TTL_PREFILL_MODEL_QUADRATIC_SECONDS_PER_1K_TOKENS_SQUARED=${TTL_PREFILL_MODEL_QUADRATIC_SECONDS_PER_1K_TOKENS_SQUARED:-0.001}
TTL_DECODE_THROUGHPUT_ALPHA=${TTL_DECODE_THROUGHPUT_ALPHA:-1.0}
DECODE_STEP_FIXED_SECONDS=${DECODE_STEP_FIXED_SECONDS:-0.03}
DECODE_STEP_SECONDS_PER_REQUEST=${DECODE_STEP_SECONDS_PER_REQUEST:-0.0}
DECODE_STEP_SECONDS_PER_CONTEXT_TOKEN=${DECODE_STEP_SECONDS_PER_CONTEXT_TOKEN:-5.2631579e-8}
SESSION_STARTED=false
VLLM_ENV_PREFIX=
if [[ -n $VLLM_ENV_SCRIPT ]]; then
  VLLM_ENV_PREFIX="source '$VLLM_ENV_SCRIPT' && "
fi

mkdir -p "$LOG_DIR"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python - <<'PY'
from agentinfer.agentcache.core.api_adapter import AgentCacheIdentityMiddleware, AgentCacheLifecycleMiddleware
from agentinfer.agentcache.core.scheduler import AgentCacheAsyncSchedulerBridge
PY

if tmux has-session -t "$VLLM_TMUX" 2>/dev/null; then
  printf 'BLOCKED: tmux session already exists: %s\n' "$VLLM_TMUX" >&2
  exit 2
fi
if [[ -e $LIFECYCLE_SOCKET || -e ${LIFECYCLE_SOCKET}.dp0 ]]; then
  printf 'BLOCKED: lifecycle socket path already exists: %s[.dp0]\n' "$LIFECYCLE_SOCKET" >&2
  exit 2
fi

stop_vllm() {
  if [[ $SESSION_STARTED == true ]]; then
    tmux kill-session -t "$VLLM_TMUX" 2>/dev/null || true
    SESSION_STARTED=false
  fi
}
cleanup() {
  stop_vllm
  rm -f "$LIFECYCLE_SOCKET" "${LIFECYCLE_SOCKET}.dp0"
}
trap cleanup EXIT

wait_http() {
  local url=$1 attempts=$2
  for _ in $(seq 1 "$attempts"); do
    curl -fsS --max-time 2 "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  printf 'Service did not become ready: %s\n' "$url" >&2
  return 1
}

start_baseline() {
  tmux new-session -d -s "$VLLM_TMUX" \
    "${VLLM_ENV_PREFIX}source '$VENV/bin/activate' && vllm serve '$MODEL' --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' --async-scheduling --scheduler-cls '$UPSTREAM_SCHEDULER' --enable-prefix-caching --enable-prompt-tokens-details --port '$VLLM_PORT' $VLLM_EXTRA_ARGS 2>&1 | tee '$BASELINE_LOG'"
  SESSION_STARTED=true
  wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" 180
}

start_candidate() {
  local additional_config
  additional_config=$(printf '{"agentcache":{"backend_id":"%s","lifecycle_socket_path":"%s","controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller","schedule_interval_seconds":%s,"progress_ttl":{"target_max_segment_rounds":14,"resume_capacity_ratio":1.0,"pause_capacity_ratio":1.0,"privileged_max_context_tokens":%s,"privileged_ttl_seconds":5,"paused_program_ttl_seconds":1800,"ttl_prefill_model_intercept_seconds":%s,"ttl_prefill_model_linear_seconds_per_1k_tokens":%s,"ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared":%s,"ttl_decode_throughput_alpha":%s,"decode_step_fixed_seconds":%s,"decode_step_seconds_per_request":%s,"decode_step_seconds_per_context_token":%s}}}' "$BACKEND_ID" "$LIFECYCLE_SOCKET" "$SCHEDULE_INTERVAL_SECONDS" "$PRIVILEGED_MAX_CONTEXT_TOKENS" "$TTL_PREFILL_MODEL_INTERCEPT_SECONDS" "$TTL_PREFILL_MODEL_LINEAR_SECONDS_PER_1K_TOKENS" "$TTL_PREFILL_MODEL_QUADRATIC_SECONDS_PER_1K_TOKENS_SQUARED" "$TTL_DECODE_THROUGHPUT_ALPHA" "$DECODE_STEP_FIXED_SECONDS" "$DECODE_STEP_SECONDS_PER_REQUEST" "$DECODE_STEP_SECONDS_PER_CONTEXT_TOKEN")
  tmux new-session -d -s "$VLLM_TMUX" \
    "${VLLM_ENV_PREFIX}source '$VENV/bin/activate' && export AGENTCACHE_VLLM_LIFECYCLE_SOCKET='$LIFECYCLE_SOCKET' && vllm serve '$MODEL' --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' --async-scheduling --scheduler-cls '$AGENTCACHE_SCHEDULER' --middleware '$AGENTCACHE_IDENTITY_MIDDLEWARE' --middleware '$AGENTCACHE_LIFECYCLE_MIDDLEWARE' --additional-config '$additional_config' --enable-prefix-caching --enable-prompt-tokens-details --port '$VLLM_PORT' $VLLM_EXTRA_ARGS 2>&1 | tee '$CANDIDATE_LOG'"
  SESSION_STARTED=true
  wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" 180
  [[ -S $LIFECYCLE_SOCKET || -S ${LIFECYCLE_SOCKET}.dp0 ]] || {
    printf 'BLOCKED: lifecycle socket was not created: %s[.dp0]\n' "$LIFECYCLE_SOCKET" >&2
    exit 2
  }
}

run_arm() {
  local result_dir=$1
  vllm bench serve --agentinfer run \
    --config "$CONFIG" \
    --base-url "http://127.0.0.1:$VLLM_PORT" \
    --agent-executable "$AGENT_BIN" \
    --task-num "$TASK_NUM" \
    --max-concurrency "$CONCURRENCY" \
    --result-dir "$result_dir"
}

cd "$REPO"
start_baseline
run_arm "$BASELINE_DIR"
stop_vllm

start_candidate
run_arm "$CANDIDATE_DIR"
stop_vllm

vllm bench serve --agentinfer compare \
  --baseline "$BASELINE_DIR" \
  --candidate "$CANDIDATE_DIR" \
  | tee "$LOG_DIR/scheduler-compare-$TIMESTAMP.txt"
