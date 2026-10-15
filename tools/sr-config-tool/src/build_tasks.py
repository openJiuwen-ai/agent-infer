"""Build office acceptance tasks (MMLU / MMLU-Pro / CMMLU / C-Eval).

Refactored from sr-baseline/office-calib/build_tasks_office.py: paths and
sampling constants come from run_config.yaml instead of hardcoded module
globals. Outputs to workspace/{task_id}/tasks/accept_office.jsonl with a
manifest (sha256 + by_dataset + by_language).

Run:
    python build_tasks.py --config config/run_config.yaml --task-id <id>
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
)


def emit(records, task, task_id):
    rec = dict(task)
    rec["task_id"] = task_id
    rec["prompt_user"] = build_prompt(task["question"], task["options"])
    del rec["question"]
    del rec["options"]
    records.append(rec)


def sample_mmlu(records, data_dir, seed, per_subject):
    subjects = sorted(
        p.name
        for p in (data_dir / "mmlu").iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and p.name not in ("all", "auxiliary_train")
        and (p / "test-00000-of-00001.parquet").exists()
    )
    rng = random.Random(seed)
    for subj in subjects:
        df = pd.read_parquet(data_dir / "mmlu" / subj / "test-00000-of-00001.parquet")
        n = min(per_subject, len(df))
        idx = rng.sample(range(len(df)), n)
        for i in idx:
            row = df.iloc[i]
            emit(
                records,
                {
                    "dataset": "mmlu",
                    "category": subj,
                    "language": "en",
                    "difficulty": None,
                    "answer": chr(ord("A") + int(row["answer"])),
                    "n_options": len(list(row["choices"])),
                    "question": str(row["question"]),
                    "options": [str(c) for c in list(row["choices"])],
                },
                f"mmlu-{subj}-{i:04d}",
            )


def sample_mmlu_pro(records, data_dir, seed, per_category):
    df = pd.read_parquet(data_dir / "mmlu_pro" / "data" / "test-00000-of-00001.parquet")
    rng = random.Random(seed)
    for cat, g in sorted(df.groupby("category")):
        n = min(per_category, len(g))
        idx = rng.sample(sorted(g.index.tolist()), n)
        for i in idx:
            row = df.loc[i]
            emit(
                records,
                {
                    "dataset": "mmlu_pro",
                    "category": str(cat),
                    "language": "en",
                    "difficulty": None,
                    "answer": str(row["answer"]),
                    "n_options": len(list(row["options"])),
                    "question": str(row["question"]),
                    "options": [str(o) for o in list(row["options"])],
                },
                f"mmlu-pro-{cat}-{i:05d}",
            )


def sample_cmmlu(records, data_dir, seed, per_subject):
    rng = random.Random(seed)
    for subj in CMMLU_SUBJECTS:
        df = pd.read_csv(data_dir / "cmmlu" / "test" / f"{subj}.csv")
        n = min(per_subject, len(df))
        idx = rng.sample(range(len(df)), n)
        for i in idx:
            row = df.iloc[i]
            opts = [str(row[c]) for c in ("A", "B", "C", "D")]
            emit(
                records,
                {
                    "dataset": "cmmlu",
                    "category": subj,
                    "language": "zh",
                    "difficulty": None,
                    "answer": str(row["Answer"]),
                    "n_options": 4,
                    "question": str(row["Question"]),
                    "options": opts,
                },
                f"cmmlu-{subj}-{i:04d}",
            )


def sample_ceval(records, data_dir, seed, per_subject):
    rng = random.Random(seed)
    for subj in CEVAL_SUBJECTS:
        df = pd.read_parquet(data_dir / "ceval" / subj / "val-00000-of-00001.parquet")
        n = min(per_subject, len(df))
        idx = rng.sample(range(len(df)), n)
        for i in idx:
            row = df.iloc[i]
            opts = [str(row[c]) for c in ("A", "B", "C", "D")]
            emit(
                records,
                {
                    "dataset": "ceval",
                    "category": subj,
                    "language": "zh",
                    "difficulty": None,
                    "answer": str(row["answer"]),
                    "n_options": 4,
                    "question": str(row["question"]),
                    "options": opts,
                },
                f"ceval-{subj}-{i:04d}",
            )


def main():
    parser = argparse.ArgumentParser(description="Build office acceptance tasks from run_config.yaml")
    parser.add_argument("--config", default="config/run_config.yaml", help="Path to run_config.yaml")
    parser.add_argument("--task-id", default="office-calib", help="Task ID for workspace isolation")
    args = parser.parse_args()

    config = load_config(args.config)
    sampling = config["sampling"]
    data_dir = Path(config["paths"]["data_dir"])
    seed = sampling["seed"]

    workspace = get_workspace(config, args.task_id)
    out_dir = workspace / "tasks"

    records = []
    sample_mmlu(records, data_dir, seed, sampling["mmlu_per_subject"])
    sample_mmlu_pro(records, data_dir, seed, sampling["mmlu_pro_per_category"])
    sample_cmmlu(records, data_dir, seed, sampling["cmmlu_per_subject"])
    sample_ceval(records, data_dir, seed, sampling["ceval_per_subject"])
    records.sort(key=lambda r: r["task_id"])

    out = out_dir / "accept_office.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "seed": seed,
        "n_total": len(records),
        "by_dataset": {
            ds: sum(1 for r in records if r["dataset"] == ds) for ds in ("mmlu", "mmlu_pro", "cmmlu", "ceval")
        },
        "by_language": {lang: sum(1 for r in records if r["language"] == lang) for lang in ("en", "zh")},
        "sha256": hashlib.sha256(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode("utf-8")
        ).hexdigest(),
        "source_files": {
            "mmlu": str(data_dir / "mmlu" / "<subject>" / "test-00000-of-00001.parquet"),
            "mmlu_pro": str(data_dir / "mmlu_pro" / "data" / "test-00000-of-00001.parquet"),
            "cmmlu": str(data_dir / "cmmlu" / "test" / "<subject>.csv"),
            "ceval": str(data_dir / "ceval" / "<subject>" / "val-00000-of-00001.parquet"),
        },
    }
    with open(out_dir / "accept_office_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
