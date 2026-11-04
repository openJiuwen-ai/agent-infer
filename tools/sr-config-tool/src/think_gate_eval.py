"""Think Gate Evaluation Script
===========================

Offline analysis of think-gate decisions for three-way routing
(4B / V4-think / V4-nothink). Two complementary modes:

  --mode oracle  : Upper-bound analysis. Reports the best achievable acc
                  if every question picked its best form (oracle), the
                  greedy quality upper bound (pick cheapest correct form),
                  and a bucket-level three-way simulation using B4's
                  selected_decision as the bucket tag (self-eval, optimistic).

  --mode cv      : Generalization check via half-split cross-validation.
                  Splits the test set in half, calibrates a think-gate on
                  half A (bucket-level gain threshold), evaluates on half B,
                  then swaps. Reports acc/token/latency across threshold
                  sweep {0, -2, -3}.

Both modes read the same records.jsonl layout under --records-dir:
  final/direct-s    B1 4B (small, nothink)
  final/direct-l    B2 V4-think (large, think)
  nthink/direct-l   B2' V4-nothink (large, nothink)
  final/sr-auto     B4 SR auto-routing (for bucket tags, oracle mode only)

Usage:
    # oracle upper bound
    python think_gate_eval.py --mode oracle \\
        --records-dir results

    # half-split CV with threshold sweep
    python think_gate_eval.py --mode cv \\
        --records-dir results \\
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import get_workspace, load_config


def load_records(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_all(records_dir):
    """Load B1/B2/B2n/B4 records keyed by question task_id.

    Prefers judged.jsonl (has is_correct) over records.jsonl (raw inference).
    """
    rd = Path(records_dir)

    def _load(subdir):
        judged = rd / subdir / "judged.jsonl"
        if judged.exists():
            return {r["task_id"]: r for r in load_records(judged)}
        return {r["task_id"]: r for r in load_records(rd / subdir / "records.jsonl")}

    b1 = _load("final/direct-s")
    b2 = _load("final/direct-l")
    b2n = _load("nthink/direct-l")
    try:
        b4 = _load("final/sr-auto")
    except FileNotFoundError:
        b4 = {}
    return b1, b2, b2n, b4


def summarize(name, src, ids):
    """Print acc/token/latency for a single form across all ids."""
    n = len(ids)
    a = sum(1 for i in ids if src[i]["is_correct"])
    t = sum(src[i]["completion_tokens"] or 0 for i in ids)
    w = sum(src[i]["latency_s"] for i in ids)
    print(f"  {name:8s}: acc={a / n * 100:.2f}%  tok={t:,}  serial={w:.0f}s")


def mode_oracle(records_dir):
    """Three-way oracle upper bound + greedy quality + bucket simulation."""
    b1, b2, b2n, b4 = load_all(records_dir)
    ids = sorted(b2.keys())
    n = len(ids)
    print(f"[oracle] n={n} questions, 3 forms + B4 bucket tags")

    # 1) Three-way oracle + exclusive correct
    any_ok = 0
    only = {"4B": 0, "V4t": 0, "V4n": 0, "none": 0}
    for i in ids:
        o = (b1[i]["is_correct"], b2[i]["is_correct"], b2n[i]["is_correct"])
        if any(o):
            any_ok += 1
        winners = [k for k, v in zip(("4B", "V4t", "V4n"), o, strict=False) if v]
        if len(winners) == 1:
            only[winners[0]] += 1
        elif not winners:
            only["none"] += 1

    print("\n== 1) Three-way oracle ==")
    print(f"  any form correct: {any_ok}/{n} = {any_ok / n * 100:.2f}%")
    print(f"  exclusive correct: 4B={only['4B']}  V4t={only['V4t']}  V4n={only['V4n']}  all_wrong={only['none']}")

    # 2) Greedy quality upper bound (pick cheapest correct: 4B > V4n > V4t)
    acc = tok = wall = 0
    for i in ids:
        if b1[i]["is_correct"]:
            r = b1[i]
        elif b2n[i]["is_correct"]:
            r = b2n[i]
        elif b2[i]["is_correct"]:
            r = b2[i]
        else:
            r = b2n[i]  # all wrong -> pick cheapest
        acc += 1 if r["is_correct"] else 0
        tok += r["completion_tokens"] or 0
        wall += r["latency_s"]
    print("\n== 2) Greedy quality upper bound (4B > V4n > V4t priority) ==")
    print(f"  acc={acc / n * 100:.2f}%  comp_tok={tok:,}  serial_wall={wall:.0f}s")

    # 3) Bucket-level three-way simulation (B4 tags, self-eval, optimistic)
    if not b4:
        print("\n== 3) Bucket-level sim skipped (no B4 sr-auto records) ==")
    else:
        buckets = defaultdict(list)
        for i in ids:
            buckets[b4[i].get("selected_decision") or "_NONE"].append(i)
        print(f"\n== 3) Bucket-level three-way sim ({len(buckets)} buckets, self-eval, optimistic) ==")
        for mode in ("greedy", "quality_first"):
            acc = tok = wall = 0
            for mem in buckets.values():
                best = None
                for name, src in (("4B", b1), ("V4n", b2n), ("V4t", b2)):
                    c = sum(1 for i in mem if src[i]["is_correct"])
                    t = sum(src[i]["completion_tokens"] or 0 for i in mem)
                    w = sum(src[i]["latency_s"] for i in mem)
                    if mode == "greedy":
                        key = (c, -t)
                        if best is None or key > best[0]:
                            best = (key, name, src, c, t, w)
                    elif best is None or c > best[3] or (c == best[3] and t < best[4]):
                        best = (None, name, src, c, t, w)
                chosen_src = best[2]
                t = best[4]
                w = best[5]
                for i in mem:
                    r = chosen_src[i]
                    acc += 1 if r["is_correct"] else 0
                tok += t
                wall += w
            print(f"  {mode:14s}: acc={acc / n * 100:.2f}%  comp_tok={tok:,}  serial_wall={wall:.0f}s")

    # 4) Baselines
    print("\n== 4) Baselines ==")
    for name, src in (("B1 4B", b1), ("B2 V4t", b2), ("B2' V4n", b2n)):
        summarize(name, src, ids)


def mode_cv(records_dir, seed, thresholds):
    """Half-split CV with threshold sweep."""
    _b1, b2, b2n, _ = load_all(records_dir)
    ids = sorted(b2.keys())
    n = len(ids)
    print(f"[cv] n={n} questions, half-split CV with seed={seed}")

    random.seed(seed)
    random.shuffle(ids)
    half_a, half_b = set(ids[: n // 2]), set(ids[n // 2 :])

    def bucket(r):
        return (r.get("source", r.get("dataset", "")), r["category"])

    def sim(calib_ids, eval_ids, thresh):
        """Calibrate gate on calib_ids (gain<=thresh -> think), eval on eval_ids."""
        gain = defaultdict(lambda: [0, 0])  # [think_correct, nothink_correct]
        for i in calib_ids:
            k = bucket(b2[i])
            if b2[i]["is_correct"]:
                gain[k][0] += 1
            if b2n[i]["is_correct"]:
                gain[k][1] += 1
        gate = {k for k in gain if (gain[k][1] - gain[k][0]) <= thresh}
        acc = tok = wall = n_think = 0
        for i in eval_ids:
            use_t = bucket(b2[i]) in gate
            r = b2[i] if use_t else b2n[i]
            n_think += 1 if use_t else 0
            acc += 1 if r["is_correct"] else 0
            tok += r["completion_tokens"] or 0
            wall += r["latency_s"]
        m = len(eval_ids)
        return acc / m, tok, wall, n_think / m

    print("\n== Half-split CV (merged) + threshold sweep ==")
    print(f"{'thresh':>7s} {'acc':>7s} {'think%':>7s} {'comp_tok':>10s} {'serial_wall':>9s}")
    for th in thresholds:
        accs = toks = walls = shares = 0.0
        for calib, ev in ((half_a, half_b), (half_b, half_a)):
            a, t, w, s = sim(calib, ev, th)
            accs += a * len(ev)
            toks += t
            walls += w
            shares += s * len(ev)
        print(f"{th:>7d} {accs / n * 100:6.2f}% {shares / n * 100:6.1f}% {toks:>10,} {walls:>8.0f}s")

    print("\n== Baselines ==")
    for name, src in (("B2 think", b2), ("B2' nothink", b2n)):
        summarize(name, src, ids)


def main(config=None, workspace=None, dry_run=False, mode="oracle", records_dir=None, seed=42, thresholds=None):
    """Run think-gate evaluation.

    Callable from the pipeline (config/workspace/dry_run) or standalone CLI.
    ``records_dir`` defaults to the workspace results directory.
    """
    if config is None:
        config = load_config()
    if records_dir is None:
        if workspace is None:
            raise ValueError("either workspace or records_dir must be provided")
        records_dir = workspace / "results"
    records_dir = Path(records_dir)
    if dry_run:
        print(f"[dry-run] think_gate_eval mode={mode} records_dir={records_dir}")
        return None
    if mode == "oracle":
        mode_oracle(records_dir)
    else:
        mode_cv(records_dir, seed, thresholds or [0, -2, -3])


def cli():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config", default="config/run_config.yaml", help="run_config.yaml path (default: config/run_config.yaml)"
    )
    ap.add_argument(
        "--mode", required=True, choices=("oracle", "cv"), help="oracle: upper bound analysis; cv: half-split CV"
    )
    ap.add_argument(
        "--records-dir",
        default=None,
        help="Directory with final/direct-s, final/direct-l, "
        "nthink/direct-l, final/sr-auto subdirs "
        "(default: <workspace>/results)",
    )
    ap.add_argument(
        "--task-id",
        default=None,
        help="Task ID to resolve --records-dir via the workspace (used when --records-dir is omitted)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for CV split (default: 42)")
    ap.add_argument(
        "--thresholds", type=int, nargs="+", default=[0, -2, -3], help="Gain thresholds for CV sweep (default: 0 -2 -3)"
    )
    args = ap.parse_args()

    config = load_config(args.config)
    if args.records_dir:
        records_dir = Path(args.records_dir)
    elif args.task_id:
        records_dir = get_workspace(config, args.task_id) / "results"
    else:
        records_dir = Path(config["paths"]["workspace_dir"]) / "results"
    main(config=config, records_dir=records_dir, mode=args.mode, seed=args.seed, thresholds=args.thresholds)


if __name__ == "__main__":
    cli()
