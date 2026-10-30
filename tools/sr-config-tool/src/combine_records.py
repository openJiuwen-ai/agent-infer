"""Combine accept_office + calib_fresh records per model, then judge the full
train set with mcq_judge.py.

Python port of calib_combine.sh. Instead of hardcoding model tags, this reads
them dynamically from run_config.yaml (config['models']). For each model it
concatenates the two record sets into a calib_train/records.jsonl, then runs
mcq_judge.py via the judge venv to produce judged.jsonl + summary.json +
analysis.json (SR format).

Runs locally on the head node (no launcher needed): record merging is pure
Python, and judging is a stdlib-only mcq_judge.py call.

Outputs per model (workspace/<task_id>/results/<served_name>/calib_train/):
  records.jsonl, judged.jsonl, summary.json, analysis.json, judge.log
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import get_workspace, load_config

ACCEPT_SUFFIX = "accept_office"
FRESH_SUFFIX = "calib_fresh"
TRAIN_SUFFIX = "calib_train"
DEFAULT_TRAIN_TASKS_NAME = "train_office.jsonl"


def merge_records(sources, out_path):
    """Concatenate record JSONL files into out_path. Returns lines written.

    Missing sources are warned and skipped so a partial round (one of the two
    record sets absent) still produces a usable, smaller train set.
    """
    count = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for src in sources:
            if not src.exists():
                print(f"[combine] WARN missing {src}, skipping")
                continue
            with open(src, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        w.write(line)
                        count += 1
    return count


def run_judge(config, records_path, tasks_file, out_dir):
    """Run mcq_judge.py via the judge venv on the merged records."""
    judge_py = config["python"]["judge_venv"]
    judge_script = str(Path(config["paths"]["sr_eval_dir"]) / "mcq_judge.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        judge_py,
        judge_script,
        "--records",
        str(records_path),
        "--tasks",
        str(tasks_file),
        "--out",
        str(out_dir / "judged.jsonl"),
        "--summary",
        str(out_dir / "summary.json"),
        "--analysis",
        str(out_dir / "analysis.json"),
    ]
    print(f"[combine] judging via {judge_py} {judge_script}")
    log_path = out_dir / "judge.log"
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result.returncode


def combine_model(config, workspace, model_key, train_tasks_file):
    """Merge accept + fresh records for one model, then judge the train set."""
    model = config["models"][model_key]
    tag = model["served_name"]
    results = workspace / "results"
    accept_rec = results / tag / ACCEPT_SUFFIX / "records.jsonl"
    fresh_rec = results / tag / FRESH_SUFFIX / "records.jsonl"
    out_dir = results / tag / TRAIN_SUFFIX
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rec = out_dir / "records.jsonl"

    count = merge_records([accept_rec, fresh_rec], out_rec)
    print(f"[combine] {tag}: {count} records merged -> {out_rec}")

    rc = run_judge(config, out_rec, train_tasks_file, out_dir)
    if rc != 0:
        print(f"[combine] FAILED {tag} judge rc={rc} (see {out_dir / 'judge.log'})")
    else:
        print(f"[combine] OK {tag} -> {out_dir}")
    return rc


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="config/run_config.yaml",
        help="Path to run_config.yaml (default: %(default)s)",
    )
    ap.add_argument("--task-id", required=True, help="Workspace task ID")
    ap.add_argument(
        "--tasks-file",
        default=None,
        help=f"Ground-truth train tasks JSONL. Default: <workspace>/tasks/{DEFAULT_TRAIN_TASKS_NAME}",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    workspace = get_workspace(config, args.task_id)

    if args.tasks_file:
        train_tasks_file = Path(args.tasks_file)
    else:
        train_tasks_file = workspace / "tasks" / DEFAULT_TRAIN_TASKS_NAME
    if not train_tasks_file.exists():
        print(f"[FATAL] train tasks file not found: {train_tasks_file}")
        sys.exit(1)

    overall = 0
    for model_key in config.get("models", {}):
        rc = combine_model(config, workspace, model_key, train_tasks_file)
        if rc != 0:
            overall = 1
    print(f"[combine] DONE (rc={overall})")
    sys.exit(overall)


if __name__ == "__main__":
    main()
