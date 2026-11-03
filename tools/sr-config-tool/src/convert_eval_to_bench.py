"""Convert eval_office.jsonl into the bench (sr_baseline_eval.py) testset
format: {id, source, category, question, options, answer, answer_type}.

task_id encodes the origin (prefix-category-row_index), so question/options
are reconstructed deterministically from the source datasets — identical to
what build_prompt() consumed when the prompt_user field was created.

Dataset root and workspace paths come from run_config.yaml (via
common.load_config / get_workspace) instead of being hardcoded.

Run on head node with bench-venv python.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from common import PREFIX, get_workspace, load_config

# prefix as it appears in a task_id -> dataset key used to load frames
PREFIX2DS = {prefix: ds for ds, prefix in PREFIX.items()}


def load_frames(data_dir):
    frames = {}
    subjects = sorted(
        p.name
        for p in (data_dir / "mmlu").iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name not in ("all", "auxiliary_train")
        and (p / "test-00000-of-00001.parquet").exists()
    )
    for s in subjects:
        df = pd.read_parquet(data_dir / "mmlu" / s / "test-00000-of-00001.parquet")
        frames[("mmlu", s)] = [
            (str(r["question"]), [str(c) for c in list(r["choices"])], chr(ord("A") + int(r["answer"])))
            for _, r in df.iterrows()
        ]
    df = pd.read_parquet(data_dir / "mmlu_pro" / "data" / "test-00000-of-00001.parquet")
    for cat, g in df.groupby("category"):
        frames[("mmlu_pro", str(cat))] = [
            (str(r["question"]), [str(o) for o in list(r["options"])], str(r["answer"]))
            for _, r in g.sort_index().iterrows()
        ]
    for f in (data_dir / "cmmlu" / "test").glob("*.csv"):
        df = pd.read_csv(f)
        frames[("cmmlu", f.stem)] = [
            (str(r["Question"]), [str(r[c]) for c in ("A", "B", "C", "D")], str(r["Answer"])) for _, r in df.iterrows()
        ]
    for d in (data_dir / "ceval").iterdir():
        if d.is_dir() and (d / "val-00000-of-00001.parquet").exists():
            df = pd.read_parquet(d / "val-00000-of-00001.parquet")
            frames[("ceval", d.name)] = [
                (str(r["question"]), [str(r[c]) for c in ("A", "B", "C", "D")], str(r["answer"]))
                for _, r in df.iterrows()
            ]
    return frames


def parse_args():
    ap = argparse.ArgumentParser(description="Convert eval_office.jsonl to bench testset format")
    ap.add_argument("--config", default="config/run_config.yaml", help="Path to run_config.yaml")
    ap.add_argument(
        "--task-id",
        default="office-calib",
        help="Task ID whose workspace holds tasks/eval_office.jsonl and receives office_eval_testset.jsonl",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    workspace = get_workspace(config, args.task_id)
    data_dir = Path(config["paths"]["data_dir"])

    frames = load_frames(data_dir)
    out = []
    missing = 0
    in_path = workspace / "tasks" / "eval_office.jsonl"
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            # task_id = <prefix>-<category>-<idx>
            tid = t["task_id"]
            # prefixes: mmlu-pro (contains '-'), mmlu, cmmlu, ceval
            if tid.startswith("mmlu-pro-"):
                rest = tid[len("mmlu-pro-") :]
                ds = "mmlu_pro"
            else:
                ds = PREFIX2DS[tid.split("-", 1)[0]]
                rest = tid[len(tid.split("-", 1)[0]) + 1 :]
            cat, idx = rest.rsplit("-", 1)
            idx = int(idx)
            rows = frames.get((ds, cat))
            if rows is None or idx >= len(rows):
                missing += 1
                continue
            q, opts, ans = rows[idx]
            assert ans == t["answer"], f"answer mismatch {tid}"
            assert len(opts) == t["n_options"], f"option count mismatch {tid}"
            out.append(
                {
                    "id": tid,
                    "source": t["dataset"],
                    "category": cat,
                    "question": q,
                    "options": opts,
                    "answer": ans,
                    "answer_type": "letter",
                }
            )
    out_path = workspace / "results" / "office_eval_testset.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[convert] {len(out)} items (missing={missing}) -> {out_path}")
    print(Counter(r["source"] for r in out))


if __name__ == "__main__":
    main()
