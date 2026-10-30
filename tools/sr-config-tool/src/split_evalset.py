"""Build the office train/test split (acceptance locked to train).

Refactored from sr-baseline/office-calib/build_split_office.py: paths and
sampling constants come from run_config.yaml. Reads
workspace/{task_id}/tasks/accept_office.jsonl (from build_tasks.py), expands
a fresh pool per dataset (fresh_multiplier * per-subject count, excluding
acceptance ids), splits fresh by train_ratio -> fresh_train + fresh_eval.
Final: train = acceptance + fresh_train, eval = fresh_eval.

Outputs to workspace/{task_id}/tasks/:
  train_office.jsonl, train_fresh_office.jsonl, eval_office.jsonl + manifest.

Run:
    python split_evalset.py --config config/run_config.yaml --task-id <id>
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from common import (
    CEVAL_SUBJECTS,
    CMMLU_SUBJECTS,
    build_prompt,
    get_workspace,
    load_config,
    task_id_of,
)


def load_frames(data_dir):
    """Return {dataset: {category: [task_stub, ...]}} (stub: no prompt yet,
    indexed by list position == original row index)."""
    frames = {"mmlu": {}, "mmlu_pro": {}, "cmmlu": {}, "ceval": {}}

    subjects = sorted(
        p.name
        for p in (data_dir / "mmlu").iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name not in ("all", "auxiliary_train")
        and (p / "test-00000-of-00001.parquet").exists()
    )
    for subj in subjects:
        df = pd.read_parquet(data_dir / "mmlu" / subj / "test-00000-of-00001.parquet")
        frames["mmlu"][subj] = [
            {
                "dataset": "mmlu",
                "category": subj,
                "language": "en",
                "answer": chr(ord("A") + int(row["answer"])),
                "n_options": len(list(row["choices"])),
                "question": str(row["question"]),
                "options": [str(c) for c in list(row["choices"])],
            }
            for _, row in df.iterrows()
        ]

    df = pd.read_parquet(data_dir / "mmlu_pro" / "data" / "test-00000-of-00001.parquet")
    for cat, g in df.groupby("category"):
        frames["mmlu_pro"][str(cat)] = [
            {
                "dataset": "mmlu_pro",
                "category": str(cat),
                "language": "en",
                "answer": str(row["answer"]),
                "n_options": len(list(row["options"])),
                "question": str(row["question"]),
                "options": [str(o) for o in list(row["options"])],
            }
            for _, row in g.sort_index().iterrows()
        ]

    for subj in CMMLU_SUBJECTS:
        df = pd.read_csv(data_dir / "cmmlu" / "test" / f"{subj}.csv")
        frames["cmmlu"][subj] = [
            {
                "dataset": "cmmlu",
                "category": subj,
                "language": "zh",
                "answer": str(row["Answer"]),
                "n_options": 4,
                "question": str(row["Question"]),
                "options": [str(row[c]) for c in ("A", "B", "C", "D")],
            }
            for _, row in df.iterrows()
        ]

    for subj in CEVAL_SUBJECTS:
        df = pd.read_parquet(data_dir / "ceval" / subj / "val-00000-of-00001.parquet")
        frames["ceval"][subj] = [
            {
                "dataset": "ceval",
                "category": subj,
                "language": "zh",
                "answer": str(row["answer"]),
                "n_options": 4,
                "question": str(row["question"]),
                "options": [str(row[c]) for c in ("A", "B", "C", "D")],
            }
            for _, row in df.iterrows()
        ]

    return frames


def main():
    parser = argparse.ArgumentParser(description="Build office train/test split from run_config.yaml")
    parser.add_argument("--config", default="config/run_config.yaml", help="Path to run_config.yaml")
    parser.add_argument("--task-id", default="office-calib", help="Task ID for workspace isolation")
    args = parser.parse_args()

    config = load_config(args.config)
    sampling = config["sampling"]
    data_dir = Path(config["paths"]["data_dir"])
    seed = sampling["seed"]
    train_ratio = sampling["train_ratio"]

    fresh_multiplier = sampling["fresh_multiplier"]
    fresh_n = {
        "mmlu": fresh_multiplier * sampling["mmlu_per_subject"],
        "mmlu_pro": fresh_multiplier * sampling["mmlu_pro_per_category"],
        "cmmlu": fresh_multiplier * sampling["cmmlu_per_subject"],
        "ceval": fresh_multiplier * sampling["ceval_per_subject"],
    }

    workspace = get_workspace(config, args.task_id)
    out_dir = workspace / "tasks"

    acc_path = out_dir / "accept_office.jsonl"
    with open(acc_path, encoding="utf-8") as f:
        acc_tasks = [json.loads(line) for line in f]
    acc_ids = {t["task_id"] for t in acc_tasks}

    frames = load_frames(data_dir)
    rng = random.Random(seed)

    fresh_train, fresh_test = [], []
    clamp_notes = []
    for dataset in ("mmlu", "mmlu_pro", "cmmlu", "ceval"):
        for cat, stubs in sorted(frames[dataset].items()):
            pool = [i for i in range(len(stubs)) if task_id_of(dataset, cat, i) not in acc_ids]
            want = fresh_n[dataset]
            n_fresh = min(want, len(pool))
            if n_fresh < want:
                clamp_notes.append(f"{dataset}/{cat}: pool {len(pool)} < {want}, sampled {n_fresh}")
            fresh = rng.sample(pool, n_fresh)
            n_tr = round(n_fresh * train_ratio)
            train_idx = set(rng.sample(fresh, n_tr))
            for i in fresh:
                stub = dict(stubs[i])
                tid = task_id_of(dataset, cat, i)
                rec = {
                    "task_id": tid,
                    "dataset": stub["dataset"],
                    "category": stub["category"],
                    "language": stub["language"],
                    "difficulty": None,
                    "answer": stub["answer"],
                    "n_options": stub["n_options"],
                    "prompt_user": build_prompt(stub["question"], stub["options"]),
                }
                (fresh_train if i in train_idx else fresh_test).append(rec)

    def dump(path, records):
        ordered = sorted(records, key=lambda r: r["task_id"])
        with open(path, "w", encoding="utf-8") as f:
            for r in ordered:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return ordered

    train = dump(out_dir / "train_office.jsonl", acc_tasks + fresh_train)
    dump(out_dir / "train_fresh_office.jsonl", fresh_train)
    evalr = dump(out_dir / "eval_office.jsonl", fresh_test)

    # integrity checks
    train_ids = {r["task_id"] for r in train}
    eval_ids = {r["task_id"] for r in evalr}
    assert not (train_ids & eval_ids), "train/eval overlap!"
    assert not (eval_ids & acc_ids), "acceptance leaked into eval!"

    def counts(records):
        c = {}
        for r in records:
            c[r["dataset"]] = c.get(r["dataset"], 0) + 1
        return c

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "fresh_multiplier": fresh_multiplier,
        "design": (
            "acceptance locked into train; fresh pool = "
            "fresh_multiplier * per-subject count per dataset "
            "(C-Eval clamped by pool size), split by train_ratio -> "
            "train = acc + fresh_train, eval = fresh_eval"
        ),
        "n_train": len(train),
        "n_eval": len(evalr),
        "n_acceptance_reused": len(acc_tasks),
        "n_fresh_train": len(fresh_train),
        "n_fresh_eval": len(fresh_test),
        "by_dataset_train": counts(train),
        "by_dataset_eval": counts(evalr),
        "sha256_train": sha(out_dir / "train_office.jsonl"),
        "sha256_train_fresh": sha(out_dir / "train_fresh_office.jsonl"),
        "sha256_eval": sha(out_dir / "eval_office.jsonl"),
        "clamped_strata": clamp_notes,
        "discipline": (
            "eval_office.jsonl is FROZEN: dry-run/tuning/calibration "
            "use train only; eval runs once at the final experiment"
        ),
    }
    with open(out_dir / "split_office_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
