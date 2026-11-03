"""三路路由离线 join 评估(无需 vLLM).

模拟"用某个配置 yaml 跑测试集会怎样"——读 yaml 的
routing.signals.domains[].model_scores 得到域→形态映射,结合 dry-run
域信号 + 逐题三形态实测记录,算配置的 acc/token/latency.

输入:
  --config         : sr-config-tool run_config.yaml(提供
                     sr_stack.routing_overhead_s)
  --sr-config      : SR v0.3 yaml(routing.signals.domains[].model_scores
                     每个域按 score 排序,最高者即 winner;use_reasoning
                     决定 think 形态)
  --signals        : dry-run 域信号 json({task_id: {domain, confidence}},
                     signal_predict.py 产出)
  --records-dir    : 逐题实测目录(含 final/direct-s, final/direct-l,
                     nthink/direct-l, final/sr-auto 等子目录的 records.jsonl)
输出:配置 acc/token/串行 wall(+路由开销/题),参照 B1/B2/B2'/B4/B4',
域标签交叉验证,形态分布.
"""

import argparse
import json
import os

import yaml
from common import load_config


def load_records(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_judged_or_records(records_dir, subdir):
    """Load judged.jsonl (has is_correct) or fall back to records.jsonl."""
    for fname in ("judged.jsonl", "records.jsonl"):
        p = os.path.join(records_dir, subdir, fname)
        if os.path.exists(p):
            return {r["task_id"]: r for r in load_records(p)}
    raise FileNotFoundError(f"neither judged.jsonl nor records.jsonl in {records_dir}/{subdir}")


def parse_form_from_score(model_ref):
    """根据 model_ref 的 model 名 + use_reasoning 推断形态 key.

    约定(与 gen_config_office_v3.py form_model_ref 一致):
      qwen3.5-4b  -> 4B(小模型,always nothink)
      dsv4-flash + use_reasoning=true  -> V4t
      dsv4-flash + use_reasoning=false -> V4n
    """
    model = model_ref.get("model", "")
    use_r = model_ref.get("use_reasoning", False)
    if "qwen3.5-4b" in model or "4b" in model.lower():
        return "4B"
    if "dsv4" in model or "v4" in model.lower():
        return "V4t" if use_r else "V4n"
    return "V4n"  # 保守默认


def load_domain_winners_from_yaml(config_path):
    """解析 yaml 的 routing.signals.domains[].model_scores,返回
    {domain_name: winner_form_key}.

    model_scores 在 SR StaticSelector 里按模型名建 map(后写覆盖),所以
    必须按模型名去重——保留该模型最高分形态(与 gen_config_office_v3.py
    seen/ms 去重逻辑一致).
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    domains = (cfg.get("routing", {}) or {}).get("signals", {}).get("domains", [])
    winners = {}
    for d in domains:
        name = d["name"]
        scores = d.get("model_scores", [])
        if not scores:
            continue
        # 按分数降序排序
        ranked = sorted(scores, key=lambda m: -float(m.get("score", 0)))
        seen_models = set()
        for m in ranked:
            model = m.get("model", "")
            if model in seen_models:
                continue
            seen_models.add(model)
            # 第一个出现的(最高分)就是 winner 形态
            winners[name] = parse_form_from_score(m)
            break
    return winners


def main(
    config=None,
    workspace=None,
    dry_run=False,
    sr_config=None,
    signals_path=None,
    records_dir=None,
    routing_overhead=None,
    out=None,
):
    """Run offline three-way join evaluation.

    Callable from the pipeline (config/workspace/dry_run) or standalone CLI.
    ``routing_overhead`` defaults to config['sr_stack']['routing_overhead_s'].
    """
    if config is None:
        config = load_config()
    if routing_overhead is None:
        routing_overhead = config["sr_stack"]["routing_overhead_s"]
    if workspace is not None:
        ws = workspace
        if sr_config is None:
            sr_config = ws / "config_output" / "config-office.yaml"
        if records_dir is None:
            records_dir = ws / "results"
        if signals_path is None:
            signals_path = ws / "results" / "dryrun_office.json"
        if out is None:
            out = ws / "results" / "join_three_way.json"

    if dry_run:
        print(
            f"[dry-run] join_three_way sr_config={sr_config} "
            f"signals={signals_path} records_dir={records_dir} "
            f"routing_overhead={routing_overhead}"
        )
        return None

    with open(signals_path, encoding="utf-8") as f_sigs:
        sigs = json.load(f_sigs)
    winners = load_domain_winners_from_yaml(sr_config)
    print(f"[join] sr-config: {sr_config}")
    print(f"[join] {len(winners)} domain winners parsed from yaml")
    for d, w in sorted(winners.items()):
        print(f"  {d:22s} -> {w}")

    b1 = load_judged_or_records(records_dir, "final/direct-s")
    b2 = load_judged_or_records(records_dir, "final/direct-l")
    b2n = load_judged_or_records(records_dir, "nthink/direct-l")
    try:
        b4 = load_judged_or_records(records_dir, "final/sr-auto")
    except FileNotFoundError:
        b4 = {}
    recs = {"4B": b1, "V4t": b2, "V4n": b2n}

    ids = sorted(sigs)
    n = len(ids)
    nerr = sum(1 for i in ids if sigs[i].get("domain") in ("_ERR", None))
    print(f"[join] n={n} 信号错误={nerr}")

    # 域标签交叉验证:dry-run 域 vs B4 selected_decision
    agree = 0
    for i in ids:
        d = sigs[i].get("domain") or "_NONE"
        sd = (b4.get(i) or {}).get("selected_decision") or "default_to_large"
        if d == sd or (d == "_NONE" and sd == "default_to_large"):
            agree += 1
    print(f"[join] 域标签一致性(dry-run vs B4 selected_decision): {agree}/{n}")

    # 用 yaml 解析出的 winner 评估
    acc = tok = wall = noans = 0
    share = {"4B": 0, "V4t": 0, "V4n": 0}
    missing_signal = 0
    by_dom = {}
    for i in ids:
        d = sigs[i].get("domain") or "_NONE"
        form = winners.get(d, "V4n")  # 兜底用 V4n(与原脚本默认一致)
        if d not in winners:
            missing_signal += 1
        share[form] += 1
        r = recs[form][i]
        acc += 1 if r["is_correct"] else 0
        noans += 1 if r["is_correct"] is None else 0
        tok += r["completion_tokens"] or 0
        wall += r["latency_s"]
        b = by_dom.setdefault(d, [0, 0])
        b[0] += 1
        b[1] += 1 if r["is_correct"] else 0
    wall += routing_overhead * n

    print("\n== 配置模拟结果(基于 yaml winner 映射)==")
    print(
        f"{'acc':>7s} {'tok':>10s} {'串行wall':>9s} {'noans':>5s} "
        f"{'%4B':>5s} {'%V4t':>5s} {'%V4n':>5s} {'缺域信号':>10s}"
    )
    print(
        f"{acc / n * 100:6.2f}% {tok:10,d} {wall:8.0f}s {noans:5d} "
        f"{share['4B'] / n * 100:4.1f}% {share['V4t'] / n * 100:4.1f}% "
        f"{share['V4n'] / n * 100:4.1f}% {missing_signal:10d}"
    )

    # 参照
    print(f"\n{'参照':10s}")
    for name, p in (
        ("B1 4B", "final/direct-s"),
        ("B2 V4t", "final/direct-l"),
        ("B2' V4n", "nthink/direct-l"),
        ("B4 SR", "final/sr-auto"),
        ("B4' SR-n", "nthink/sr-auto"),
    ):
        judged_p = os.path.join(records_dir, p, "judged.jsonl")
        rec_p = os.path.join(records_dir, p, "records.jsonl")
        full_p = judged_p if os.path.exists(judged_p) else rec_p
        if not os.path.exists(full_p):
            continue
        rs = load_records(full_p)
        a = sum(1 for r in rs if r.get("is_correct")) / len(rs) * 100
        t = sum(r["completion_tokens"] or 0 for r in rs)
        print(f"  {name:9s} acc={a:6.2f}%  tok={t:10,d}")

    # 逐域明细
    print("\n逐域明细:")
    for d in sorted(by_dom, key=lambda dk: -by_dom[dk][0]):
        m = by_dom[d]
        w = winners.get(d, "?")
        print(f"  {d:22s} n={m[0]:3d} acc={m[1] / m[0] * 100:6.2f}%  -> {w}")

    if out:
        out_data = {
            "config": str(sr_config),
            "n": n,
            "routing_s_per_req": routing_overhead,
            "policy": {
                "acc": round(acc / n * 100, 2),
                "tok": tok,
                "wall": round(wall, 1),
                "noans": noans,
                "share": share,
                "missing_signal": missing_signal,
            },
            "domain_winners": winners,
            "domain_agree_with_b4": agree,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n[join] saved -> {out}")


def cli():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config", default="config/run_config.yaml", help="run_config.yaml path (default: config/run_config.yaml)"
    )
    ap.add_argument(
        "--sr-config", required=True, help="SR v0.3 yaml config path (routing.signals.domains[].model_scores)"
    )
    ap.add_argument("--signals", required=True, help="dry-run signals json path")
    ap.add_argument(
        "--records-dir",
        required=True,
        help="records dir with final/direct-s, final/direct-l, nthink/direct-l, final/sr-auto subdirs",
    )
    ap.add_argument(
        "--routing-overhead",
        type=float,
        default=None,
        help="per-request routing latency in s (default: config sr_stack.routing_overhead_s)",
    )
    ap.add_argument("--out", default="", help="output json path (default: skip)")
    args = ap.parse_args()

    config = load_config(args.config)
    main(
        config=config,
        sr_config=args.sr_config,
        signals_path=args.signals,
        records_dir=args.records_dir,
        routing_overhead=args.routing_overhead,
        out=args.out or None,
    )


if __name__ == "__main__":
    cli()
