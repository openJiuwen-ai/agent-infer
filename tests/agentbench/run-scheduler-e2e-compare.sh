#!/usr/bin/env bash
# Cold upstream async scheduler -> cold AgentInfer scheduler bridge -> compare.
set -euo pipefail

REPO=${REPO:?set REPO to the AgentInfer checkout}
VENV=${VENV:?set VENV to the benchmark virtualenv}
MODEL=${MODEL:?set MODEL to the model path}
CLAUDE_BIN=${CLAUDE_BIN:-claude}
TASK_NUM=${TASK_NUM:-1}
CONCURRENCY=${CONCURRENCY:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
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
AGENTCACHE_MIDDLEWARE=agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware
LIFECYCLE_SOCKET=${LIFECYCLE_SOCKET:-/tmp/agentinfer-vllm-lifecycle-$TIMESTAMP.sock}
BACKEND_ID=${BACKEND_ID:-vllm-local}
SCHEDULE_INTERVAL_SECONDS=${SCHEDULE_INTERVAL_SECONDS:-5}
SESSION_STARTED=false

mkdir -p "$LOG_DIR"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python - <<'PY'
from agentinfer.agentcache.core.api_adapter import AgentCacheLifecycleMiddleware
from agentinfer.agentcache.core.scheduler import AgentCacheAsyncSchedulerBridge
PY

if tmux has-session -t "$VLLM_TMUX" 2>/dev/null; then
  printf 'BLOCKED: tmux session already exists: %s\n' "$VLLM_TMUX" >&2
  exit 2
fi
if [[ -e $LIFECYCLE_SOCKET ]]; then
  printf 'BLOCKED: lifecycle socket path already exists: %s\n' "$LIFECYCLE_SOCKET" >&2
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
  rm -f "$LIFECYCLE_SOCKET"
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
    "source '$VENV/bin/activate' && vllm serve '$MODEL' --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' --async-scheduling --scheduler-cls '$UPSTREAM_SCHEDULER' --enable-prefix-caching --enable-prompt-tokens-details --port '$VLLM_PORT' $VLLM_EXTRA_ARGS 2>&1 | tee '$BASELINE_LOG'"
  SESSION_STARTED=true
  wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" 180
}

start_candidate() {
  local additional_config
  additional_config=$(printf '{"agentcache":{"backend_id":"%s","lifecycle_socket_path":"%s","controller_factory":"agentinfer.agentcache.core.factory.build_progress_ttl_controller","schedule_interval_seconds":%s,"progress_ttl":{"target_min_segment_rounds":9,"target_max_segment_rounds":14,"resume_capacity_ratio":0.9,"pause_capacity_ratio":0.95,"pause_capacity_lookahead_rounds":2,"privileged_lookahead_rounds":14,"privileged_max_context_tokens":262144,"ttl_impact_multiplier":2,"paused_program_ttl_seconds":1800}}}' "$BACKEND_ID" "$LIFECYCLE_SOCKET" "$SCHEDULE_INTERVAL_SECONDS")
  tmux new-session -d -s "$VLLM_TMUX" \
    "source '$VENV/bin/activate' && export AGENTCACHE_VLLM_LIFECYCLE_SOCKET='$LIFECYCLE_SOCKET' && vllm serve '$MODEL' --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' --async-scheduling --scheduler-cls '$AGENTCACHE_SCHEDULER' --middleware '$AGENTCACHE_MIDDLEWARE' --additional-config '$additional_config' --enable-prefix-caching --enable-prompt-tokens-details --port '$VLLM_PORT' $VLLM_EXTRA_ARGS 2>&1 | tee '$CANDIDATE_LOG'"
  SESSION_STARTED=true
  wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" 180
  [[ -S $LIFECYCLE_SOCKET ]] || {
    printf 'BLOCKED: lifecycle socket was not created: %s\n' "$LIFECYCLE_SOCKET" >&2
    exit 2
  }
}

run_arm() {
  local result_dir=$1
  vllm bench serve --agentinfer run \
    --config "$CONFIG" \
    --base-url "http://127.0.0.1:$VLLM_PORT" \
    --agent-executable "$CLAUDE_BIN" \
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
