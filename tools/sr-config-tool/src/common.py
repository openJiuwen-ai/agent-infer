"""Common utilities for sr-config-tool.

Provides:
- Sampling constants and subject lists (shared by build_tasks / split_evalset)
- build_prompt() — MCQ prompt formatter (B4 protocol)
- load_config() — read run_config.yaml
- get_workspace() — create/return task workspace paths
- run_on_node() — launcher abstraction (local/srun/ssh/docker)
- check_port() / cleanup_vllm() — safety helpers
- log_step() — step execution wrapper with logging
"""

import hashlib
import json
import os
import random
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

# ============================================================================
# Sampling constants (experiment design, generally not changed by users)
# ============================================================================

CMMLU_SUBJECTS = [
    "business_ethics",
    "chinese_civil_service_exam",
    "college_law",
    "construction_project_management",
    "economics",
    "international_law",
    "jurisprudence",
    "legal_and_moral_basis",
    "management",
    "marketing",
    "professional_accounting",
    "professional_law",
    "public_relations",
    "sociology",
]

CEVAL_SUBJECTS = [
    "accountant",
    "business_administration",
    "civil_servant",
    "college_economics",
    "law",
    "legal_professional",
    "tax_accountant",
]

PREFIX = {"mmlu": "mmlu", "mmlu_pro": "mmlu-pro", "cmmlu": "cmmlu", "ceval": "ceval"}
WIDTH = {"mmlu": 4, "mmlu_pro": 5, "cmmlu": 4, "ceval": 4}


def task_id_of(dataset, category, idx):
    """Generate a deterministic task ID from dataset/category/index."""
    return f"{PREFIX[dataset]}-{category}-{idx:0{WIDTH[dataset]}d}"


def build_prompt(question, options):
    """MCQ prompt formatter (B4 protocol, verbatim from sr-baseline)."""
    lines = [
        "The following is a multiple choice question. Read it carefully and pick the best option.",
        "",
        f"Question: {question}",
        "",
        "Options:",
    ]
    for i, opt in enumerate(options):
        lines.append(f"({chr(ord('A') + i)}) {opt}")
    lines += [
        "",
        "First, immediately output your chosen answer on its own line in the "
        "exact form 'Answer: X' where X is the letter of the option. Then, if "
        "you wish, add at most 2 short sentences of reasoning. Do NOT restate "
        "the options. Always provide the answer even if you are unsure.",
    ]
    return "\n".join(lines)


# ============================================================================
# Config loading
# ============================================================================


def load_config(config_path="config/run_config.yaml"):
    """Load run_config.yaml and expand ~ in path values."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"config not found: {config_path}\n"
            "Copy config/run_config.example.yaml to config/run_config.yaml first, "
            "or pass --config <path>."
        ) from None
    # Expand user home in known path fields
    for section in ("paths",):
        if section in cfg:
            for k, v in cfg[section].items():
                if isinstance(v, str):
                    cfg[section][k] = os.path.expanduser(v)
    if "python" in cfg:
        for k, v in cfg["python"].items():
            if isinstance(v, str):
                cfg["python"][k] = os.path.expanduser(v)
    if "sr_stack" in cfg:
        for k, v in cfg["sr_stack"].items():
            if isinstance(v, str):
                cfg["sr_stack"][k] = os.path.expanduser(v)
    if "models" in cfg:
        for _model_key, model_cfg in cfg["models"].items():
            if "model_dir" in model_cfg:
                model_cfg["model_dir"] = os.path.expanduser(model_cfg["model_dir"])
            if "gen_params" in model_cfg:
                model_cfg["gen_params"] = os.path.expanduser(model_cfg["gen_params"])
    return cfg


# ============================================================================
# Workspace management
# ============================================================================


def generate_task_id(name="office-calib"):
    """Generate a task ID like 'office-calib-20260923-a3f9b2'."""
    date_str = datetime.now().strftime("%Y%m%d")
    suffix = hashlib.sha256(f"{time.time()}{random.random()}".encode()).hexdigest()[:6]
    return f"{name}-{date_str}-{suffix}"


def get_workspace(config, task_id):
    """Return workspace Path for a task, creating subdirs if needed."""
    ws_root = Path(config["paths"]["workspace_dir"])
    ws = ws_root / task_id
    for subdir in ("tasks", "results", "logs", "config_output"):
        (ws / subdir).mkdir(parents=True, exist_ok=True)
    return ws


def load_state(workspace):
    """Load state.json for a task, or return empty state."""
    state_path = workspace / "state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": None, "current": None}


def save_state(workspace, state):
    """Save state.json for a task."""
    state_path = workspace / "state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def update_task_index(config, task_id, status, current_step=None):
    """Update the external task index table."""
    ws_root = Path(config["paths"]["workspace_dir"])
    index_path = ws_root / "task_index.json"
    index = []
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    # Find or create entry
    entry = None
    for e in index:
        if e["task_id"] == task_id:
            entry = e
            break
    if entry is None:
        entry = {
            "task_id": task_id,
            "created": datetime.now().isoformat(),
            "status": status,
            "current_step": current_step,
            "workspace": str(ws_root / task_id),
        }
        index.append(entry)
    else:
        entry["status"] = status
        if current_step:
            entry["current_step"] = current_step
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ============================================================================
# Launcher abstraction
# ============================================================================


def run_on_node(config, cmd, env=None):
    """Execute a command on the target node based on launcher type.

    Args:
        config: loaded run_config dict
        cmd: shell command string to execute
        env: optional dict of environment variables

    Returns:
        subprocess.CompletedResult
    """
    launcher = config.get("launcher", {})
    ltype = launcher.get("type", "local")

    # Merge env
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    if ltype == "local":
        return subprocess.run(cmd, shell=True, env=full_env, text=True, check=False)
    elif ltype == "srun":
        parts = [
            "srun",
            "-p",
            launcher.get("partition", "h20"),
        ]
        # Only pin to a specific node if configured; otherwise let SLURM
        # auto-select a node with available GPU resources.
        node = launcher.get("node", "")
        if node:
            parts += ["-w", node]
        parts += [
            "--cpus-per-task",
            str(launcher.get("cpus_per_task", 16)),
            "--mem",
            launcher.get("mem", "64G"),
            "--time",
            launcher.get("time_limit", "06:00:00"),
        ]
        gpus = launcher.get("gpus")
        if gpus:
            parts.append(f"--gres=gpu:{gpus}")
        return subprocess.run([*parts, "bash", "-c", cmd], env=full_env, text=True, check=False)
    elif ltype == "ssh":
        node = launcher.get("node", "localhost")
        ssh_key = launcher.get("ssh_key", "")
        ssh_parts = ["ssh"]
        if ssh_key:
            ssh_parts += ["-i", ssh_key]
        ssh_parts += [node, cmd]
        return subprocess.run(ssh_parts, env=full_env, text=True, check=False)
    elif ltype == "docker":
        container = launcher.get("container", "")
        return subprocess.run(
            ["docker", "exec", container, "bash", "-c", cmd],
            env=full_env,
            text=True,
            check=False,
        )
    else:
        raise ValueError(f"Unknown launcher type: {ltype}")


# ============================================================================
# Safety helpers
# ============================================================================


def check_port(port, host="127.0.0.1"):
    """Check if a TCP port is already in use. Returns True if available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex((host, port))
        return result != 0  # port available if connect fails
    finally:
        sock.close()


def cleanup_vllm(model_dir):
    """Kill any lingering vLLM process serving the given model dir."""
    subprocess.run(
        ["pkill", "-f", f"vllm serve {model_dir}"],
        capture_output=True,
        check=False,
    )
    time.sleep(3)
    subprocess.run(
        ["pkill", "-9", "-f", f"vllm serve {model_dir}"],
        capture_output=True,
        check=False,
    )


# ============================================================================
# Step execution wrapper
# ============================================================================


def log_step(workspace, step_name, cmd_fn, dry_run=False):
    """Execute a step function, logging stdout/stderr to workspace/logs/.

    Args:
        workspace: Path to task workspace
        step_name: e.g. 'build-tasks', 'run-eval-small'
        cmd_fn: callable that returns a subprocess.CompletedResult or runs logic
        dry_run: if True, only print the step name without executing

    Returns:
        The return value of cmd_fn, or None if dry_run
    """
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step_name}.log"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] === {step_name} ===")
    if dry_run:
        # In dry-run mode, still call cmd_fn so it can print the command
        result = cmd_fn()
        return result

    result = cmd_fn()

    # Write log
    with open(log_path, "w", encoding="utf-8") as f:
        if hasattr(result, "stdout"):
            f.write(result.stdout or "")
            if result.stderr:
                f.write("\n--- stderr ---\n")
                f.write(result.stderr)
        else:
            f.write(str(result or ""))

    if hasattr(result, "returncode") and result.returncode != 0:
        print(f"  [FAILED] returncode={result.returncode}")
        print(f"  Log: {log_path}")
        return result

    print(f"  [OK] Log: {log_path}")
    return result
