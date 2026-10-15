"""Run one model-eval round: start vLLM, generate, judge.

Merged Python port of run_office_head.sh (head launcher) and node_office.sh
(compute-node lifecycle). Replaces the two-script srun pattern with a single
config-driven entry point that uses common.run_on_node() for launcher
abstraction (local / srun / ssh / docker).

The full vLLM lifecycle runs on the target node inside one bash -c string:
  start vLLM -> ready gate (curl /v1/models) -> multi_benchmark_eval.py
  -> stop vLLM -> mcq_judge.py

Generation uses the eval venv python (has the openai client); judging uses the
judge venv python (stdlib only). All model/path values are injected into the
node command via environment variables, mirroring the original node_office.sh
env contract (srun propagates env to the remote task).

Outputs land in workspace/<task_id>/results/<served_name>/<suffix>/:
  records.jsonl, judged.jsonl, summary.json, analysis.json,
  vllm.log, gen.log, judge.log
"""

import argparse
import atexit
import os
import signal
import sys
from pathlib import Path

from common import (
    check_port,
    cleanup_vllm,
    get_workspace,
    load_config,
    run_on_node,
)

DEFAULT_MAX_LEN = 16384
DEFAULT_GPU_UTIL = 0.90
DEFAULT_CONCURRENCY = 8
READY_POLL_INTERVAL = 5  # seconds between /v1/models probes
DEFAULT_SUFFIX = "accept_office"


def build_node_command(ready_iters, ready_timeout, stop_wait, concurrency):
    """Build the bash -c string that runs the full vLLM lifecycle on the node.

    The script only references environment variables ($OUT, $MODEL_DIR, ...);
    the caller (run_eval_round) is responsible for setting them. Bash braces
    are doubled to escape the f-string, and a raw string preserves the
    backslash line-continuations and the grep escape sequences verbatim.
    """
    return rf"""set -uo pipefail
mkdir -p "$OUT"
export PATH="$VENV_BIN:$PATH"

cleanup() {{
  pkill -f "vllm serve $MODEL_DIR" 2>/dev/null || true
  sleep 5
  pkill -9 -f "vllm serve $MODEL_DIR" 2>/dev/null || true
}}
trap cleanup EXIT

echo "[node] $(date '+%F %T') host=$(hostname) starting vLLM: $MODEL_TAG (TP=$TP)"
KV_ARGS=""
if [ -n "${{KV_DTYPE:-}}" ]; then KV_ARGS="--kv-cache-dtype $KV_DTYPE"; fi

"$VENV_BIN/vllm" serve "$MODEL_DIR" \
  --served-model-name "$MODEL_TAG" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  $KV_ARGS \
  > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!

READY=0
for i in $(seq 1 {ready_iters}); do
  sleep {READY_POLL_INTERVAL}
  if grep -q "Address already in use" "$OUT/vllm.log" 2>/dev/null; then
    echo "[FATAL] port $PORT already in use"; tail -20 "$OUT/vllm.log"; exit 1
  fi
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "\"id\":\"$MODEL_TAG\""; then
    READY=1; break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[FATAL] vLLM process died"; tail -60 "$OUT/vllm.log"; exit 1
  fi
done
if [ "$READY" != "1" ]; then
  echo "[FATAL] vLLM not ready after {ready_timeout}s"; tail -40 "$OUT/vllm.log"; exit 1
fi
echo "[node] vLLM ready (~$((i * {READY_POLL_INTERVAL}))s)"

echo "[node] $(date '+%F %T') generating on tasks: $TASKS_FILE"
CTK_ARG=""
if [ -n "${{CHAT_TEMPLATE_KWARGS:-}}" ]; then
  CTK_ARG="--chat-template-kwargs $CHAT_TEMPLATE_KWARGS"
fi
"$BENCH_PY" "$SR_EVAL/multi_benchmark_eval.py" \
  --split "$TASKS_FILE" \
  --out "$OUT/records.jsonl" \
  --endpoint "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL_TAG" \
  --params "$PARAMS" \
  --concurrency "${{CONC:-{concurrency}}}" \
  $CTK_ARG 2>&1 | tee "$OUT/gen.log"
GEN_RC=${{PIPESTATUS[0]}}

echo "[node] stopping vLLM"
kill $VLLM_PID 2>/dev/null || true
sleep {stop_wait}
pkill -f "vllm serve $MODEL_DIR" 2>/dev/null || true
sleep 5

if [ "$GEN_RC" != "0" ]; then
  echo "[FATAL] generation failed rc=$GEN_RC"; exit 1
fi

echo "[node] $(date '+%F %T') judging (MCQ exact match + analysis.json)"
"$JUDGE_PY" "$SR_EVAL/mcq_judge.py" \
  --records "$OUT/records.jsonl" \
  --tasks "$TASKS_FILE" \
  --out "$OUT/judged.jsonl" \
  --summary "$OUT/summary.json" \
  --analysis "$OUT/analysis.json" 2>&1 | tee "$OUT/judge.log"
JUDGE_RC=${{PIPESTATUS[0]}}
if [ "$JUDGE_RC" != "0" ]; then
  echo "[FATAL] judge failed rc=$JUDGE_RC"; exit 1
fi

echo "[node] DONE $(date '+%F %T')"
echo "[node] Outputs: records.jsonl, judged.jsonl, summary.json, analysis.json"
"""


def run_eval_round(config, workspace, model_key, tasks_file, out_suffix):
    """Run one eval round for a single model preset.

    Returns (result, out_dir) where result is the subprocess.CompletedResult
    from run_on_node (or None on a pre-flight abort), and out_dir is the
    results Path for the round.
    """
    model = config["models"][model_key]
    out_dir = workspace / "results" / model["served_name"] / out_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_cfg = config.get("vllm", {})
    port = vllm_cfg["port"]
    ready_timeout = int(vllm_cfg.get("ready_timeout", 900))
    stop_wait = int(vllm_cfg.get("stop_wait", 15))
    ready_iters = max(1, ready_timeout // READY_POLL_INTERVAL)
    concurrency = int(model.get("concurrency", DEFAULT_CONCURRENCY))

    # Pre-flight port check (meaningful only for the local launcher; for
    # remote launchers the node-side "Address already in use" gate handles it).
    launcher_type = config.get("launcher", {}).get("type", "local")
    if launcher_type == "local" and not check_port(port):
        print(f"[FATAL] port {port} already in use locally; aborting round")
        return None, out_dir

    cmd = build_node_command(ready_iters, ready_timeout, stop_wait, concurrency)

    judge_venv = config["python"]["judge_venv"]
    venv_bin = os.path.dirname(judge_venv)
    env = {
        "VENV_BIN": venv_bin,
        "BENCH_PY": config["python"]["eval_venv"],
        "JUDGE_PY": judge_venv,
        "SR_EVAL": config["paths"]["sr_eval_dir"],
        "MODEL_DIR": model["model_dir"],
        "MODEL_TAG": model["served_name"],
        "TP": str(model["tensor_parallel"]),
        "PORT": str(port),
        "OUT": str(out_dir),
        "PARAMS": model["gen_params"],
        "KV_DTYPE": str(model.get("kv_cache_dtype", "")),
        "TASKS_FILE": str(tasks_file),
        "CONC": str(concurrency),
        "MAX_LEN": str(model.get("max_model_len", DEFAULT_MAX_LEN)),
        "GPU_UTIL": str(model.get("gpu_memory_utilization", DEFAULT_GPU_UTIL)),
        "CHAT_TEMPLATE_KWARGS": str(model.get("chat_template_kwargs", "")),
    }

    # Forward the model's GPU count onto the launcher config so run_on_node's
    # srun branch can emit --gres=gpu:N if it honors launcher.gpus. This is a
    # no-op for the current run_on_node and is harmless for other launchers.
    config.setdefault("launcher", {})["gpus"] = model.get("n_gpu", model["tensor_parallel"])

    # Best-effort cleanup: the node command's own `trap cleanup EXIT` handles
    # the remote vLLM; this covers the local-launcher case and any unexpected
    # exit of the Python driver while a local vLLM is still running.
    model_dir = model["model_dir"]
    atexit.register(cleanup_vllm, model_dir)

    def _on_signal(signum, _frame):
        cleanup_vllm(model_dir)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[run-eval] model={model_key} tag={model['served_name']} tp={model['tensor_parallel']} out={out_dir}")
    result = run_on_node(config, cmd, env=env)
    if result.returncode != 0:
        print(f"[run-eval] FAILED rc={result.returncode} (see {out_dir}/vllm.log, gen.log, judge.log)")
    else:
        print(f"[run-eval] OK -> {out_dir}")
    return result, out_dir


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="config/run_config.yaml",
        help="Path to run_config.yaml (default: %(default)s)",
    )
    ap.add_argument("--task-id", required=True, help="Workspace task ID")
    ap.add_argument(
        "--model",
        required=True,
        help="Model preset key in run_config.yaml (e.g. small|large)",
    )
    ap.add_argument(
        "--tasks-file",
        required=True,
        help="Task JSONL file (path valid on the target node)",
    )
    ap.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help="Output subdir under results/<tag>/ (default: %(default)s)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.model not in config.get("models", {}):
        print(
            f"[FATAL] --model '{args.model}' not in run_config models "
            f"(available: {list(config.get('models', {}).keys())})"
        )
        sys.exit(2)
    workspace = get_workspace(config, args.task_id)
    result, _out_dir = run_eval_round(config, workspace, args.model, Path(args.tasks_file), args.suffix)
    if result is None or result.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
