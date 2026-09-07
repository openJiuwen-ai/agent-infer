#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

# Cold baseline -> cold candidate -> compare. Services use explicit tmux sessions.
set -euo pipefail

REPO=${REPO:?set REPO to the AgentInfer checkout}
VENV=${VENV:?set VENV to the benchmark virtualenv}
MODEL=${MODEL:?set MODEL to the model path}
CLAUDE_BIN=${CLAUDE_BIN:-claude}
TASK_NUM=${TASK_NUM:-1}
CONCURRENCY=${CONCURRENCY:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
VLLM_PORT=${VLLM_PORT:-8000}
ROUTER_PORT=${ROUTER_PORT:-8400}
VLLM_TMUX=${VLLM_TMUX:-agentinfer-router-e2e-vllm}
ROUTER_TMUX=${ROUTER_TMUX:-agentinfer-router-e2e-router}
LOG_DIR=${LOG_DIR:-$REPO/benchkit-logs}
BASELINE_CONFIG=${BASELINE_CONFIG:-$REPO/agentinfer/agentbench/configs/swebench_vllm.yaml}
CANDIDATE_CONFIG=${CANDIDATE_CONFIG:-$REPO/agentinfer/agentbench/configs/swebench_agentinfer.yaml}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_DIR=${BASELINE_DIR:-$REPO/agentinfer/agentbench/results/e2e-baseline-$TIMESTAMP}
CANDIDATE_DIR=${CANDIDATE_DIR:-$REPO/agentinfer/agentbench/results/e2e-candidate-$TIMESTAMP}

mkdir -p "$LOG_DIR"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

stop_services() {
  tmux kill-session -t "$ROUTER_TMUX" 2>/dev/null || true
  tmux kill-session -t "$VLLM_TMUX" 2>/dev/null || true
}
trap stop_services EXIT

wait_http() {
  local url=$1 attempts=$2
  for _ in $(seq 1 "$attempts"); do
    curl -fsS --max-time 2 "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  printf 'Service did not become ready: %s\n' "$url" >&2
  return 1
}

start_vllm() {
  tmux new-session -d -s "$VLLM_TMUX" \
    "source '$VENV/bin/activate' && vllm serve '$MODEL' --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' --enable-prefix-caching --enable-prompt-tokens-details --port '$VLLM_PORT' $VLLM_EXTRA_ARGS 2>&1 | tee '$LOG_DIR/vllm-$TIMESTAMP.log'"
  wait_http "http://127.0.0.1:$VLLM_PORT/v1/models" 180
}

start_router() {
  tmux new-session -d -s "$ROUTER_TMUX" \
    "source '$VENV/bin/activate' && vllm router --backends 'http://127.0.0.1:$VLLM_PORT' --port '$ROUTER_PORT' 2>&1 | tee '$LOG_DIR/router-$TIMESTAMP.log'"
  wait_http "http://127.0.0.1:$ROUTER_PORT/health" 60
}

cd "$REPO"
start_vllm
vllm bench serve --agentinfer run --config "$BASELINE_CONFIG" --base-url "http://127.0.0.1:$VLLM_PORT" --agent-executable "$CLAUDE_BIN" --task-num "$TASK_NUM" --max-concurrency "$CONCURRENCY" --result-dir "$BASELINE_DIR"
tmux kill-session -t "$VLLM_TMUX"

start_vllm
start_router
vllm bench serve --agentinfer run --config "$CANDIDATE_CONFIG" --base-url "http://127.0.0.1:$VLLM_PORT" --router-url "http://127.0.0.1:$ROUTER_PORT" --enabled --agent-executable "$CLAUDE_BIN" --task-num "$TASK_NUM" --max-concurrency "$CONCURRENCY" --result-dir "$CANDIDATE_DIR"

vllm bench serve --agentinfer compare --baseline "$BASELINE_DIR" --candidate "$CANDIDATE_DIR" | tee "$LOG_DIR/compare-$TIMESTAMP.txt"
