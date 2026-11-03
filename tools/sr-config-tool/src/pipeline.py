"""Pipeline orchestrator for sr-config-tool.

Provides subcommands that chain the full evaluation-to-config workflow.
Each step is logged and recorded in state.json for resume capability.

Usage:
    python pipeline.py --task-id my-exp build-tasks
    python pipeline.py --task-id my-exp run-eval --model small
    python pipeline.py --task-id my-exp --resume build-tasks
    python pipeline.py --list-tasks
    python pipeline.py --dry-run build-tasks
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    generate_task_id,
    get_workspace,
    load_config,
    load_state,
    log_step,
    save_state,
    update_task_index,
)


def run_step(config, workspace, step_name, step_fn, dry_run=False, force=False):
    """Run a pipeline step with state tracking and logging.

    Args:
        dry_run: print commands without executing; do NOT touch state.json
        force: rerun even if step is already in state["completed"]
    """
    state = load_state(workspace)

    # Check if already completed (resume mode)
    if step_name in state["completed"] and not dry_run and not force:
        print(f"[skip] {step_name} already completed (use --force to rerun)")
        return True

    # In dry-run mode, run the step fn so it can print its command, but do NOT
    # mutate state.json — otherwise a dry-run leaves fake "completed" entries
    # that block real runs later.
    if dry_run:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] === {step_name} (dry-run) ===")
        step_fn(config, workspace, dry_run=True)
        return True

    # Mark as current
    state["current"] = step_name
    save_state(workspace, state)
    task_id = workspace.name
    update_task_index(config, task_id, "running", step_name)

    def wrapped_fn():
        return step_fn(config, workspace, dry_run=False)

    result = log_step(workspace, step_name, wrapped_fn, dry_run=False)

    # Update state
    state = load_state(workspace)
    if result is not None and hasattr(result, "returncode") and result.returncode != 0:
        state["failed"] = step_name
        state["current"] = None
        save_state(workspace, state)
        update_task_index(config, task_id, "failed", step_name)
        print(f"\n[FAILED] Step '{step_name}' failed.")
        print(f"  Log: {workspace}/logs/{step_name}.log")
        print("  To resume: re-run this command with --resume (completed steps are skipped)")
        return False

    if step_name not in state["completed"]:
        state["completed"].append(step_name)
    state["failed"] = None
    state["current"] = None
    save_state(workspace, state)
    update_task_index(config, task_id, "running", None)
    return True


# ============================================================================
# Step implementations
# ============================================================================


def _run_streaming(cmd_parts):
    """Run a subprocess with real-time stdout streaming.

    Prints output to terminal in real-time AND caches it so log_step can
    write it to the log file. stderr is merged into stdout.
    """
    proc = subprocess.Popen(
        cmd_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    return subprocess.CompletedProcess(
        args=cmd_parts,
        returncode=proc.returncode,
        stdout="".join(lines),
        stderr="",
    )


def _run_script(config, script_name, extra_args=None, dry_run=False):
    """Run a src/ script via subprocess with --config and --task-id."""
    py = config["python"]["eval_venv"]
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{script_name}.py")
    cmd_parts = [py, script_path, "--config", config.get("_config_path", "config/run_config.yaml")]
    if extra_args:
        cmd_parts.extend(extra_args)
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd_parts)}")
        return None
    return _run_streaming(cmd_parts)


def step_build_tasks(config, workspace, dry_run=False):
    return _run_script(config, "build_tasks", extra_args=["--task-id", workspace.name], dry_run=dry_run)


def step_run_eval(config, workspace, dry_run=False, model="small", tasks_file=None, suffix="accept_office"):
    if tasks_file is None:
        tasks_file = str(workspace / "tasks" / "accept_office.jsonl")
    args = ["--task-id", workspace.name, "--model", model, "--suffix", suffix, "--tasks-file", tasks_file]
    return _run_script(config, "run_eval", extra_args=args, dry_run=dry_run)


def _run_local_script(config, cmd_parts, dry_run=False):
    """Run a command locally (not via srun/ssh) for CPU-only data processing steps.

    These steps (differentiation_check, cost_aware_calib, result_to_config) only
    read/write jsonl files and should run on the head node, not queue for GPU
    compute nodes via srun.
    """
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd_parts)}")
        return None
    return _run_streaming(cmd_parts)


def step_check_differentiation(config, workspace, dry_run=False):
    """Run differentiation_check.py from SR (CPU-only, local)."""
    sr_eval = config["paths"]["sr_eval_dir"]
    small_tag = config["models"]["small"]["served_name"]
    large_tag = config["models"]["large"]["served_name"]
    small_judged = str(workspace / "results" / small_tag / "accept_office" / "judged.jsonl")
    large_judged = str(workspace / "results" / large_tag / "accept_office" / "judged.jsonl")
    out_path = str(workspace / "results" / "differentiation_verdict.json")
    cmd_parts = [
        config["python"]["eval_venv"],
        f"{sr_eval}/differentiation_check.py",
        "--small",
        small_judged,
        "--large",
        large_judged,
        "--gap-redline",
        "0.08",
        "--saturation-threshold",
        "0.95",
        "--output",
        out_path,
    ]
    return _run_local_script(config, cmd_parts, dry_run=dry_run)


def step_split_evalset(config, workspace, dry_run=False):
    return _run_script(config, "split_evalset", extra_args=["--task-id", workspace.name], dry_run=dry_run)


def step_combine_records(config, workspace, dry_run=False):
    return _run_script(config, "combine_records", extra_args=["--task-id", workspace.name], dry_run=dry_run)


def step_think_gate(config, workspace, dry_run=False):
    """Run think_gate_eval.py in both oracle and cv modes.

    think_gate_eval.py expects a records-dir layout:
      final/direct-s/records.jsonl   (small, nothink = B1)
      final/direct-l/records.jsonl   (large, think   = B2)
      nthink/direct-l/records.jsonl  (large, nothink = B2')
      final/sr-auto/records.jsonl    (SR auto, optional, oracle only)
    The pipeline produces per-model subdirs instead, so we create
    symlinks mapping the pipeline layout to the expected layout.
    """
    results_dir = workspace / "results"
    small_tag = config["models"]["small"]["served_name"]
    large_tag = config["models"]["large"]["served_name"]

    # Expected layout -> actual pipeline path
    mapping = {
        "final/direct-s": f"{small_tag}/calib_train",
        "final/direct-l": f"{large_tag}/calib_train",
        "nthink/direct-l": f"{large_tag}/calib_nthink",
    }
    for rel_dst, rel_src in mapping.items():
        dst = results_dir / rel_dst
        src = results_dir / rel_src
        if not src.exists():
            print(f"[think-gate] WARNING: missing {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        print(f"[think-gate] symlink {dst} -> {src}")

    for mode in ("oracle", "cv"):
        print(f"[think-gate] running --mode {mode}")
        rc = _run_script(
            config, "think_gate_eval", extra_args=["--task-id", workspace.name, "--mode", mode], dry_run=dry_run
        )
        if rc is None:
            continue
        if rc.returncode != 0:
            return rc
    return None


def step_cost_calib(config, workspace, dry_run=False):
    """Run cost_aware_calib.py from SR (CPU-only, local)."""
    sr_eval = config["paths"]["sr_eval_dir"]
    small_tag = config["models"]["small"]["served_name"]
    large_tag = config["models"]["large"]["served_name"]
    small_judged = str(workspace / "results" / small_tag / "calib_train" / "judged.jsonl")
    large_judged = str(workspace / "results" / large_tag / "calib_train" / "judged.jsonl")
    out_path = str(workspace / "results" / "calib_cost_office.json")
    token_costs_dir = str(workspace / "results" / "token_costs")
    cmd_parts = [
        config["python"]["eval_venv"],
        f"{sr_eval}/cost_aware_calib.py",
        "--model",
        f"{small_tag},{small_judged}",
        "--model",
        f"{large_tag},{large_judged}",
        "--out",
        out_path,
        "--token-costs-dir",
        token_costs_dir,
    ]
    return _run_local_script(config, cmd_parts, dry_run=dry_run)


def step_gen_config(config, workspace, dry_run=False):
    """Run result_to_config.py from SR (CPU-only, local).

    result_to_config.py expects <model_name>_direct/analysis.json or
    <model_name>_cot/analysis.json under --results-dir. The pipeline produces
    <served_name>/<run_type>/analysis.json, so we create symlinks mapping
    to the expected layout.
    """
    sr_eval = config["paths"]["sr_eval_dir"]
    small_tag = config["models"]["small"]["served_name"]
    large_tag = config["models"]["large"]["served_name"]
    analysis_dir = workspace / "results" / "analysis_by_model"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Create model_name_direct symlinks pointing to accept_office analysis
    for tag in (small_tag, large_tag):
        src = workspace / "results" / tag / "accept_office" / "analysis.json"
        if not src.exists():
            print(f"[gen-config] WARNING: missing {src}")
            continue
        dst = analysis_dir / f"{tag}_direct" / "analysis.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        print(f"[gen-config] symlink {dst} -> {src}")

    nothink_dir = str(workspace / "results" / large_tag / "calib_nthink")
    token_costs_dir = str(workspace / "results" / "token_costs")
    out_yaml = str(workspace / "config_output" / "config-office.yaml")
    cmd_parts = [
        config["python"]["eval_venv"],
        f"{sr_eval}/result_to_config.py",
        "--results-dir",
        str(analysis_dir),
        "--nothink-results-dir",
        nothink_dir,
        "--cost-lambda",
        "0.1",
        "--token-costs-dir",
        token_costs_dir,
        "--output-file",
        out_yaml,
    ]
    return _run_local_script(config, cmd_parts, dry_run=dry_run)


def step_join(config, workspace, dry_run=False):
    """Run join_three_way.py with explicit paths.

    If dryrun_office.json (signals) is missing, generate a fallback that
    uses each record's dataset as the domain.
    """
    import json as _json

    sr_config = str(workspace / "config_output" / "config-office.yaml")
    signals = workspace / "results" / "dryrun_office.json"
    records_dir = str(workspace / "results")
    out = str(workspace / "results" / "join_three_way.json")

    # Generate fallback signals if signal_predict was not run
    if not signals.exists():
        print("[join] dryrun_office.json not found, generating fallback signals")
        small_tag = config["models"]["small"]["served_name"]
        recs_path = workspace / "results" / small_tag / "calib_train" / "records.jsonl"
        if recs_path.exists():
            sigs = {}
            with open(recs_path, encoding="utf-8") as f:
                for line in f:
                    r = _json.loads(line)
                    sigs[r["task_id"]] = {
                        "domain": r.get("dataset", "_NONE"),
                        "confidence": 1.0,
                    }
            with open(signals, "w", encoding="utf-8") as f:
                _json.dump(sigs, f, indent=2)
            print(f"[join] wrote {len(sigs)} fallback signals to {signals}")
        else:
            print(f"[join] WARNING: {recs_path} not found, signals may fail")

    return _run_script(
        config,
        "join_three_way",
        extra_args=["--sr-config", sr_config, "--signals", str(signals), "--records-dir", records_dir, "--out", out],
        dry_run=dry_run,
    )


def step_convert(config, workspace, dry_run=False):
    """Convert eval_office.jsonl into the bench testset format (stage 6b precondition)."""
    return _run_script(config, "convert_eval_to_bench", extra_args=["--task-id", workspace.name], dry_run=dry_run)


def step_e2e(config, workspace, dry_run=False):
    """Run E2E test, ensuring the bench testset exists first.

    Prefers converting eval_office.jsonl via convert_eval_to_bench.py (which
    reconstructs question/options and re-checks answers); falls back to a raw
    copy only if conversion fails, printing a warning that the integrity
    asserts were skipped.
    """
    testset = workspace / "results" / "office_eval_testset.jsonl"
    if not testset.exists():
        eval_set = workspace / "tasks" / "eval_office.jsonl"
        if not eval_set.exists():
            print(f"[e2e] WARNING: {eval_set} not found, e2e may fail")
        elif dry_run:
            step_convert(config, workspace, dry_run=True)
        else:
            print("[e2e] testset missing, running convert first")
            rc = step_convert(config, workspace, dry_run=False)
            if not testset.exists():
                import shutil

                rc_info = f" (rc={rc.returncode})" if rc is not None and hasattr(rc, "returncode") else ""
                print(f"[e2e] WARNING: convert failed{rc_info}; falling back to raw copy")
                print("[e2e] WARNING: raw copy skips the answer/option-count integrity asserts")
                testset.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(eval_set, testset)
                print(f"[e2e] copied {eval_set} -> {testset}")
    return _run_script(config, "run_e2e", extra_args=["--task-id", workspace.name], dry_run=dry_run)


# ============================================================================
# Lambda wrappers for step_run_eval variants
# ============================================================================


def _make_run_eval_step(model, tasks_filename, suffix):
    """Create a step function for run-eval with fixed parameters."""

    def _step(config, workspace, dry_run=False):
        tasks_file = str(workspace / "tasks" / tasks_filename) if tasks_filename else None
        return step_run_eval(config, workspace, dry_run=dry_run, model=model, tasks_file=tasks_file, suffix=suffix)

    return _step


# ============================================================================
# CLI
# ============================================================================


def list_tasks(config_path):
    """List all historical tasks."""
    config = load_config(config_path)
    ws_root = config["paths"]["workspace_dir"]
    index_path = Path(ws_root) / "task_index.json"
    if not index_path.exists():
        print("No tasks found.")
        return
    with open(index_path, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"{'Task ID':<45} {'Status':<12} {'Current Step':<25} {'Created'}")
    print("-" * 110)
    for t in tasks:
        print(
            f"{t['task_id']:<45} {t.get('status', '?'):<12} "
            f"{t.get('current_step', '-') or '-':<25} "
            f"{t.get('created', '?')}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="sr-config-tool pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/run_config.yaml", help="Config file path")
    parser.add_argument("--task-id", default=None, help="Task ID for workspace isolation")
    parser.add_argument("--resume", action="store_true", help="Resume from last failed step")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--list-tasks", action="store_true", help="List all historical tasks")
    parser.add_argument("--force", action="store_true", help="Force rerun even if step completed")

    subparsers = parser.add_subparsers(dest="command")

    # build-tasks
    subparsers.add_parser("build-tasks", help="Stage 1a: Sample acceptance tasks")

    # run-eval
    p_run = subparsers.add_parser("run-eval", help="Stage 1b/2a: Run model eval + judge")
    p_run.add_argument("--model", required=True, choices=["small", "large"])
    p_run.add_argument("--tasks-file", default=None)
    p_run.add_argument("--suffix", default="accept_office")

    # check-differentiation
    subparsers.add_parser("check-differentiation", help="Stage 1c: Differentiation check")

    # split-evalset
    subparsers.add_parser("split-evalset", help="Stage 1d: Split train/eval")

    # run-calib (same as run-eval but with calib suffix)
    p_calib = subparsers.add_parser("run-calib", help="Stage 2a: Run calibration eval")
    p_calib.add_argument("--model", required=True, choices=["small", "large"])

    # combine-records
    subparsers.add_parser("combine-records", help="Stage 2b: Combine records")

    # run-nothink
    subparsers.add_parser("run-nothink", help="Stage 2c: Run nothink eval")

    # think-gate
    subparsers.add_parser("think-gate", help="Stage 4b: Think gate eval")

    # cost-calib
    subparsers.add_parser("cost-calib", help="Stage 4a: Cost-aware calibration")

    # gen-config
    subparsers.add_parser("gen-config", help="Stage 5c: Generate SR config")

    # join
    subparsers.add_parser("join", help="Stage 6a: Offline join evaluation")

    # convert
    subparsers.add_parser("convert", help="Stage 6b precondition: Convert eval set to bench testset format")

    # e2e
    subparsers.add_parser("e2e", help="Stage 6b: E2E final validation")

    args = parser.parse_args()

    if args.list_tasks:
        list_tasks(args.config)
        return

    if not args.command:
        parser.print_help()
        return

    # Load config
    config = load_config(args.config)
    config["_config_path"] = os.path.abspath(args.config)

    # Generate or use task ID
    task_id = args.task_id or generate_task_id()
    workspace = get_workspace(config, task_id)
    print(f"Task ID: {task_id}")
    print(f"Workspace: {workspace}")

    # Resume: skip completed steps
    if args.resume:
        state = load_state(workspace)
        if state["completed"]:
            print(f"Resuming. Completed steps: {state['completed']}")
        if state.get("failed"):
            print(f"Last failed at: {state['failed']}")

    # Map commands to steps
    step_map = {
        "build-tasks": ("build-tasks", step_build_tasks),
        "check-differentiation": ("check-differentiation", step_check_differentiation),
        "split-evalset": ("split-evalset", step_split_evalset),
        "combine-records": ("combine-records", step_combine_records),
        "think-gate": ("think-gate", step_think_gate),
        "cost-calib": ("cost-calib", step_cost_calib),
        "gen-config": ("gen-config", step_gen_config),
        "join": ("join", step_join),
        "convert": ("convert", step_convert),
        "e2e": ("e2e", step_e2e),
    }

    if args.command == "run-eval":
        step_name = f"run-eval-{args.model}"
        step_fn = _make_run_eval_step(args.model, None, args.suffix)
    elif args.command == "run-calib":
        step_name = f"run-calib-{args.model}"
        step_fn = _make_run_eval_step(args.model, "train_fresh_office.jsonl", "calib_fresh")
    elif args.command == "run-nothink":
        step_name = "run-nothink"
        step_fn = _make_run_eval_step("large", "train_office.jsonl", "calib_nthink")
    elif args.command in step_map:
        step_name, step_fn = step_map[args.command]
    else:
        print(f"Unknown command: {args.command}")
        return

    success = run_step(
        config,
        workspace,
        step_name,
        step_fn,
        dry_run=args.dry_run,
        force=args.force,
    )

    if success:
        update_task_index(config, task_id, "completed")
        print(f"\n[OK] Step '{step_name}' completed successfully.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
