#!/usr/bin/env python3
"""SR baseline 统一评测 harness(本地 vLLM 版,stdlib 实现,无需 pip 安装).

对同一份测试集,用 5 种模式之一发起流式请求,逐请求记录延迟/TTFT/
token/答案对错/SR 路由头/收发时间戳,汇总输出 summary.json(含任务时长,
延迟,TTFT 的 mean/p50/p95/p99 等关键指标,及按实际后端模型分组的延迟).

支持 --concurrency N 并发压测(默认 1=串行,与历史口径一致).并发>1 时
request_interval 不生效;每请求的 t_sent_s/t_done_s(epoch 秒)用于与
router 日志(routing_latency_ms / signal_phase_timing / decision_phase_timing)
及 vLLM /metrics 按时间窗对齐做端到端分解.

后端为**自建的两个 vLLM 实例**(小模型 :8000 / 大模型 :8001),
不依赖任何外部 API——延迟测量不受第三方共享负载干扰.

模式:
  direct-s    直连本地 vLLM 小模型实例(质量/成本下界)
  direct-l    直连本地 vLLM 大模型实例(质量/成本上界)
  random      逐请求掷硬币(固定种子)在小/大模型直连端点间二选一
  sr-auto     经 SR 路由,model=auto(★ 被测主角)
  sr-fixed-s  经 SR 但 model 固定为小模型(测 SR 纯转发开销)

用法示例(在 vLLM/SR 同一台机器上运行,避免网络噪声):
  # 小/大模型名必须与两个 vLLM 实例的 --served-model-name 一致
  python3 bench/sr_baseline_eval.py --mode direct-s --runs 1 \
      --small-model qwen3.8-27b --large-model glm-5.3

  # 打难度标签(见 bench/sr_baseline_calibrate.py)后正式测量
  python3 bench/sr_baseline_eval.py --mode sr-auto --runs 3 \
      --testset results/baseline/testset_labeled.jsonl

输出目录默认 results/baseline/<mode>/,含:
  per_request.csv   逐请求明细
  records.jsonl     逐请求明细(JSON Lines,供 calibrate/汇总脚本读取)
  summary.json      汇总指标
"""

import argparse
import csv
import json
import math
import os
import random
import re
import ssl
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

HTTP_BAD_REQUEST = 400
HTTP_SERVER_ERROR = 500
RANDOM_THRESHOLD = 0.5

ANSWER_LETTER_RE = re.compile(r"(?:answer(?:\s+is)?\s*[::]?\s*|答案\s*[是::]*\s*)[((\[]?\s*([A-J])\b", re.IGNORECASE)

CSV_FIELDS = [
    "id",
    "source",
    "category",
    "difficulty",
    "expected_model",
    "run",
    "mode",
    "model_requested",
    "model_used",
    "selected_decision",
    "cache_hit",
    "is_correct",
    "answer_expected",
    "answer_extracted",
    "answer_type",
    "ttft_s",
    "latency_s",
    "prompt_tokens",
    "completion_tokens",
    "error",
    "t_sent_s",
    "t_done_s",
]


# ---------------------------------------------------------------- prompt / 评分


def build_prompt(item: dict[str, Any]) -> str:
    """客观题 prompt(MMLU / MMLU-Pro 通用,选项数 4~10 自适应).

    Supports two item formats:
    - bench: has 'question' and 'options' list -> build prompt from scratch
    - eval:  has 'prompt_user' (pre-built prompt) -> use directly
    """
    if "prompt_user" in item and "question" not in item:
        return item["prompt_user"]
    lines = [
        "The following is a multiple choice question. Read it carefully and pick the best option.",
        "",
        f"Question: {item['question']}",
        "",
        "Options:",
    ]
    for i, opt in enumerate(item["options"]):
        lines.append(f"({chr(ord('A') + i)}) {opt}")
    lines += [
        "",
        "First, immediately output your chosen answer on its own line in the "
        "exact form 'Answer: X' where X is the letter of the option. Then, if "
        "you wish, add at most 2 short sentences of reasoning. Do NOT restate "
        "the options. Always provide the answer even if you are unsure.",
    ]
    return "\n".join(lines)


def extract_letter(text: str) -> str | None:
    """从模型回复中提取选项字母(沿用 bench/router_reason_bench.py 的模式,并加兜底)."""
    if not text:
        return None
    matches = ANSWER_LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    # 兜底:最后一个只有 "(X)" / "X" 的行
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip("*_`").strip()
        if re.fullmatch(r"\(?[A-J]\)?", stripped):
            return stripped.strip("()").upper()
    return None


def score_answer(item: dict[str, Any], text: str) -> tuple[bool | None, str | None]:
    atype = item.get("answer_type", "letter")
    if atype == "contains":
        if not text:
            return None, None
        ok = str(item["answer"]).lower() in text.lower()
        return ok, item["answer"]
    letter = extract_letter(text)
    if letter is None:
        return None, None
    return letter == item["answer"], letter


# ---------------------------------------------------------------- HTTP / 流式请求


def stream_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body: dict[str, Any] | None = None,
    use_stream_options: bool = True,
) -> dict[str, Any]:
    """发一条流式 chat 请求,返回 TTFT/延迟/文本/usage/响应头.

    extra_body: 附加到请求体的字段(如 chat_template_kwargs 关闭思考模式).
    use_stream_options: 附加 stream_options={"include_usage": true};
    部分网关不识别该字段会报 4xx,此时去掉该字段重试一次(vLLM 原生支持).
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if use_stream_options:
        body["stream_options"] = {"include_usage": True}
    if extra_body:
        body.update(extra_body)

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    ctx = ssl.create_default_context()
    # 计时起点在 urlopen 之前:latency_s/ttft_s 覆盖 SR 路由 + vLLM 排队/
    # prefill(响应头到达即 urlopen 返回,之前起表会把这段全部漏在窗口外)
    t0 = time.perf_counter()
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except urllib.error.HTTPError as exc:
        if use_stream_options and HTTP_BAD_REQUEST <= exc.code < HTTP_SERVER_ERROR:
            # 端点可能不支持 stream_options,去掉重试一次
            return stream_chat_completion(
                base_url,
                model,
                prompt,
                api_key,
                max_tokens,
                temperature,
                timeout,
                extra_body=extra_body,
                use_stream_options=False,
            )
        raise

    ttft: float | None = None
    usage: dict[str, Any] | None = None
    chunks: list[str] = []
    headers: dict[str, str] = {}
    with resp_ctx as resp:
        for key in ("x-vsr-selected-model", "x-vsr-selected-decision", "x-vsr-cache-hit"):
            val = resp.headers.get(key)
            if val is not None:
                headers[key] = val
        for raw in resp:
            if ttft is None:
                ttft = time.perf_counter() - t0  # 首个 SSE 字节到达
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    chunks.append(content)
    total = time.perf_counter() - t0
    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "latency_s": round(total, 4),
        "text": "".join(chunks),
        "usage": usage,
        "headers": headers,
    }


# ---------------------------------------------------------------- 统计


def pct(sorted_xs: list[float], p: float) -> float | None:
    if not sorted_xs:
        return None
    k = (len(sorted_xs) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return round(sorted_xs[int(k)], 4)
    return round(sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f), 4)


def stats_block(values: list[float]) -> dict[str, float | None]:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": round(statistics.fmean(xs), 4),
        "p50": pct(xs, 50),
        "p95": pct(xs, 95),
        "p99": pct(xs, 99),
        "max": pct(xs, 100),
    }


def flat_metrics(values: list[float], prefix: str) -> dict[str, float | None]:
    """生成 {prefix}_mean_seconds / p50 / p95 / p99 扁平指标."""
    xs = sorted(v for v in values if v is not None)
    return {
        f"{prefix}_mean_seconds": round(statistics.fmean(xs), 4) if xs else None,
        f"{prefix}_p50_seconds": pct(xs, 50),
        f"{prefix}_p95_seconds": pct(xs, 95),
        f"{prefix}_p99_seconds": pct(xs, 99),
    }


# ---------------------------------------------------------------- 主流程


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SR baseline evaluation harness (local vLLM backends, 5 modes, streaming)")
    ap.add_argument("--mode", required=True, choices=["direct-s", "direct-l", "random", "sr-auto", "sr-fixed-s"])
    ap.add_argument(
        "--testset", default="results/baseline/testset.jsonl", help="测试集 JSONL(校准后建议用 testset_labeled.jsonl)"
    )
    ap.add_argument("--small-base-url", default="http://127.0.0.1:8000/v1", help="小模型 vLLM 实例的 base URL")
    ap.add_argument("--large-base-url", default="http://127.0.0.1:8001/v1", help="大模型 vLLM 实例的 base URL")
    ap.add_argument("--sr-base-url", default="http://127.0.0.1:8888/v1", help="SR 路由入口 base URL")
    ap.add_argument("--small-model", default="qwen3.8-27b", help="小模型的 served-model-name")
    ap.add_argument("--large-model", default="glm-5.3", help="大模型的 served-model-name")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key(本地 vLLM 未启用 --api-key 时填任意非空值即可)",
    )
    ap.add_argument("--runs", type=int, default=1, help="重复轮数(轮间打乱题目顺序)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--request-interval", type=float, default=0.2, help="请求间隔秒数(给 vLLM 留调度间隙;仅 concurrency=1 生效)"
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发请求数(默认 1=串行,与历史口径一致;>1 为闭环并发压测,request_interval 不生效)",
    )
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=3, help="正式测量前对每个涉及端点发的预热请求数(排除冷启动)")
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="注入 chat_template_kwargs 同时关闭 Qwen3(enable_thinking)与 GLM(thinking.type=disabled)的思考模式",
    )
    ap.add_argument(
        "--large-extra-body",
        default="",
        help="JSON 字符串:目标模型为 large(direct-l / random 正面 / "
        "sr-auto 全局 merge)时并入请求体.例:"
        '\'{"reasoning_effort":"none"}\'.用于关闭大模型思考等'
        "仅作用于 large 的参数;不认识的键会被各家模板忽略",
    )
    ap.add_argument(
        "--small-extra-body",
        default="",
        help="JSON 字符串:目标模型为 small(direct-s / random 反面)时并入请求体(small 流专用参数注入)",
    )
    ap.add_argument("--output-dir", default="results/baseline")
    ap.add_argument("--tag", default="", help="输出子目录名(默认用 mode)")
    return ap.parse_args()


def resolve_endpoint(mode: str, args: argparse.Namespace, coin: bool | None) -> tuple[str, str]:
    """返回 (base_url, model).coin 是 random 模式本请求的掷硬币结果."""
    if mode == "direct-s":
        return args.small_base_url, args.small_model
    if mode == "direct-l":
        return args.large_base_url, args.large_model
    if mode == "random":
        if coin:
            return args.small_base_url, args.small_model
        return args.large_base_url, args.large_model
    if mode == "sr-auto":
        return args.sr_base_url, "auto"
    if mode == "sr-fixed-s":
        return args.sr_base_url, args.small_model
    raise ValueError(f"unknown mode: {mode}")


def warmup_targets(mode: str, args: argparse.Namespace) -> list[tuple[str, str]]:
    """本模式涉及的(去重后的)端点列表."""
    if mode == "direct-s":
        return [(args.small_base_url, args.small_model)]
    if mode == "direct-l":
        return [(args.large_base_url, args.large_model)]
    if mode == "random":
        return [(args.small_base_url, args.small_model), (args.large_base_url, args.large_model)]
    if mode == "sr-auto":
        return [(args.sr_base_url, "auto")]
    return [(args.sr_base_url, args.small_model)]


def load_testset(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise SystemExit(f"测试集为空: {path}")
    return items


def main() -> None:
    args = parse_args()
    if args.small_model == args.large_model:
        raise SystemExit("小/大模型的 served-model-name 不能相同(无法区分路由结果)")

    extra_body: dict[str, Any] | None = None
    if args.disable_thinking:
        # 同时携带两家的开关:Qwen3 模板读 enable_thinking,
        # GLM 模板读 thinking.type;不认识的键会被各自模板忽略
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "thinking": {"type": "disabled"},
            }
        }

    large_extra_body: dict[str, Any] | None = None
    if args.large_extra_body:
        large_extra_body = json.loads(args.large_extra_body)
    small_extra_body: dict[str, Any] | None = None
    if args.small_extra_body:
        small_extra_body = json.loads(args.small_extra_body)

    def select_extra(model: str) -> dict[str, Any] | None:
        """按目标模型选请求体扩展.
        - small 流: small_extra_body(若设)否则 extra_body
        - large 流: large_extra_body(若设)否则 extra_body
        - auto 流: small/large 两份 merge 到一个全局体(SR 路由后两家都会收到,
          未识别键被各自模板忽略);若 large 侧显式关思考,则剔除合并结果中
          的 thinking 键(F3 烟测:thinking.type 会破坏 V4 的
          enable_thinking 关思考语义)."""
        if large_extra_body is None and small_extra_body is None:
            return extra_body
        if model == args.large_model and large_extra_body is not None:
            return large_extra_body
        if model == args.large_model:
            return extra_body
        if model == "auto":
            merged = dict(extra_body) if extra_body else {}
            for src in (small_extra_body, large_extra_body):
                if not src:
                    continue
                for k, v in src.items():
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
            ctk = merged.get("chat_template_kwargs")
            if (
                isinstance(ctk, dict)
                and large_extra_body
                and "enable_thinking" in large_extra_body.get("chat_template_kwargs", {})
            ):
                ctk.pop("thinking", None)
            return merged or None
        return small_extra_body if small_extra_body is not None else extra_body

    items = load_testset(args.testset)
    out_dir = os.path.join(args.output_dir, args.tag or args.mode)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 预热:排除 vLLM 冷启动 / 首次编译带来的首请求异常延迟
    if args.warmup > 0:
        for base_url, model in warmup_targets(args.mode, args):
            for i in range(args.warmup):
                try:
                    stream_chat_completion(
                        base_url, model, "Reply with OK.", args.api_key, 16, 0.0, args.timeout, extra_body=extra_body
                    )
                    print(f"[warmup] {base_url} model={model} #{i + 1} ok")
                except Exception as exc:
                    print(f"[warmup] {base_url} model={model} #{i + 1} FAILED: {exc}(继续,但请检查端点是否正常)")

    coin_rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    print_lock = threading.Lock()
    t_start = time.perf_counter()

    def execute_item(run: int, item: dict[str, Any]) -> dict[str, Any]:
        """执行单条请求并返回记录(串行/并发共用).

        t_sent_s/t_done_s 为 epoch 秒,用于与 router 日志,vLLM /metrics
        按时间窗对齐做端到端延迟分解(SR 段 vs 后端段 vs 网络残差).
        """
        coin = coin_rng.random() < RANDOM_THRESHOLD if args.mode == "random" else None
        base_url, model = resolve_endpoint(args.mode, args, coin)
        prompt = build_prompt(item)

        rec: dict[str, Any] = {
            "id": item["id"],
            "source": item["source"],
            "category": item["category"],
            "difficulty": item.get("difficulty"),
            "expected_model": item.get("expected_model"),
            "run": run,
            "mode": args.mode,
            "model_requested": model,
            "model_used": None,
            "selected_decision": None,
            "cache_hit": None,
            "is_correct": None,
            "answer_expected": item.get("answer"),
            "answer_extracted": None,
            "answer_type": item.get("answer_type", "letter"),
            "ttft_s": None,
            "latency_s": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": None,
            "t_sent_s": None,
            "t_done_s": None,
        }

        rec["t_sent_s"] = round(time.time(), 3)
        try:
            result = stream_chat_completion(
                base_url,
                model,
                prompt,
                args.api_key,
                args.max_tokens,
                args.temperature,
                args.timeout,
                extra_body=select_extra(model),
            )
            usage = result.get("usage") or {}
            headers = result.get("headers") or {}
            rec["model_used"] = headers.get("x-vsr-selected-model") or model
            rec["selected_decision"] = headers.get("x-vsr-selected-decision")
            rec["cache_hit"] = headers.get("x-vsr-cache-hit")
            rec["ttft_s"] = result["ttft_s"]
            rec["latency_s"] = result["latency_s"]
            rec["prompt_tokens"] = usage.get("prompt_tokens")
            rec["completion_tokens"] = usage.get("completion_tokens")
            ok, extracted = score_answer(item, result.get("text", ""))
            rec["is_correct"] = ok
            rec["answer_extracted"] = extracted
            if extracted is None:
                # no-answer 时保留回复尾部,便于排查截断 / 格式问题
                rec["response_tail"] = (result.get("text") or "")[-200:]
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["t_done_s"] = round(time.time(), 3)
        return rec

    def print_status(rec: dict[str, Any]) -> None:
        if rec["error"] is not None:
            line = f"  [{rec['run'] + 1}] {rec['id']:<28} ERROR {rec['error']}"
        else:
            status = "ok" if rec["is_correct"] else ("no-answer" if rec["answer_extracted"] is None else "wrong")
            line = (
                f"  [{rec['run'] + 1}] {rec['id']:<28} {status:<9} "
                f"lat={rec['latency_s']}s ttft={rec['ttft_s']}s "
                f"model={rec['model_used']}"
            )
        with print_lock:
            print(line, flush=True)

    if args.concurrency <= 1:
        for run in range(args.runs):
            order = list(range(len(items)))
            random.Random(args.seed + run).shuffle(order)
            print(f"[run {run + 1}/{args.runs}] 共 {len(order)} 题")
            for qi in order:
                rec = execute_item(run, items[qi])
                records.append(rec)
                print_status(rec)
                if args.request_interval > 0:
                    time.sleep(args.request_interval)
    else:
        print(f"[concurrency] {args.concurrency} 并发(request_interval 不生效)")
        tasks: list[tuple[int, dict[str, Any]]] = []
        for run in range(args.runs):
            order = list(range(len(items)))
            random.Random(args.seed + run).shuffle(order)
            print(f"[run {run + 1}/{args.runs}] 共 {len(order)} 题(并发提交)")
            tasks.extend((run, items[qi]) for qi in order)

        def worker(run: int, item: dict[str, Any]) -> None:
            rec = execute_item(run, item)
            with print_lock:
                records.append(rec)
            print_status(rec)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(worker, run, item) for run, item in tasks]
            for future in futures:
                future.result()  # 传播 execute_item 未捕获的意外异常

    wall = time.perf_counter() - t_start

    # ------------------------------------------------------------ 汇总
    ok_recs = [r for r in records if r["error"] is None]
    latencies = [r["latency_s"] for r in ok_recs if r["latency_s"] is not None]
    ttfts = [r["ttft_s"] for r in ok_recs if r["ttft_s"] is not None]

    def accuracy(recs: list[dict[str, Any]]) -> float | None:
        # 口径:no-answer(提取不到答案) 也算错.分母=全部请求,分子=答对的.
        # 因为 prompt 已明确要求给出选项字母,提不到即视为不符合要求.
        if not recs:
            return None
        correct = sum(1 for r in recs if r["is_correct"] is True)
        return round(correct / len(recs), 4)

    summary: dict[str, Any] = {
        "mode": args.mode,
        "tag": args.tag or args.mode,
        "testset": args.testset,
        "model_s": args.small_model,
        "model_l": args.large_model,
        "runs": args.runs,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "disable_thinking": args.disable_thinking,
        "total_requests": len(records),
        "successful_requests": len(ok_recs),
        "failed_requests": len(records) - len(ok_recs),
        "success_rate": round(len(ok_recs) / len(records), 4) if records else None,
        "run_wall_time_seconds": round(wall, 3),
        "request_throughput_per_second": round(len(ok_recs) / wall, 4) if wall > 0 else None,
    }
    # ---- 用户指定的关键指标(任务时长 = 每题端到端时长;单请求任务下与 latency 同值)
    lat_stats = stats_block(latencies)
    summary["mean_task_duration_seconds"] = lat_stats["mean"]
    summary["p50_task_duration_seconds"] = lat_stats["p50"]
    summary["p95_task_duration_seconds"] = lat_stats["p95"]
    summary["p99_task_duration_seconds"] = lat_stats["p99"]
    summary.update(flat_metrics(latencies, "latency"))
    summary.update(flat_metrics(ttfts, "ttft"))
    summary["concurrency"] = args.concurrency
    # ---- 按实际后端模型分组(客户端视角:不同模型各自承担的延迟水平;
    # 与 router 日志 routing_latency_ms / vLLM /metrics 对齐后可做分解对账)
    by_model: dict[str, dict[str, Any]] = {}
    for r in ok_recs:
        model_key = r.get("model_used") or "unknown"
        slot = by_model.setdefault(model_key, {"count": 0, "latency": [], "ttft": []})
        slot["count"] += 1
        if r.get("latency_s") is not None:
            slot["latency"].append(r["latency_s"])
        if r.get("ttft_s") is not None:
            slot["ttft"].append(r["ttft_s"])
    summary["latency_by_model"] = {
        m: {"count": s["count"], **stats_block(s["latency"])} for m, s in by_model.items()
    } or None
    summary["ttft_by_model"] = {m: {"count": s["count"], **stats_block(s["ttft"])} for m, s in by_model.items()} or None
    # ---- token / 质量
    summary["prompt_tokens_total"] = sum(r["prompt_tokens"] or 0 for r in ok_recs) or None
    summary["completion_tokens_total"] = sum(r["completion_tokens"] or 0 for r in ok_recs) or None
    summary["accuracy"] = accuracy(ok_recs)
    summary["no_answer_requests"] = sum(1 for r in ok_recs if r["is_correct"] is None)
    by_diff: dict[str, Any] = {}
    for diff in ("easy", "hard", "too_hard", "unlabeled"):
        pool = [r for r in ok_recs if (r["difficulty"] or "unlabeled") == diff]
        if pool:
            by_diff[diff] = {
                "count": len(pool),
                "accuracy": accuracy(pool),
                "latency_p50_seconds": stats_block([r["latency_s"] for r in pool])["p50"],
            }
    summary["accuracy_by_difficulty"] = by_diff or None
    # ---- 大模型调用占比(算力消耗代理)
    large_calls = sum(1 for r in ok_recs if r["model_used"] == args.large_model)
    summary["large_model_request_share"] = round(large_calls / len(ok_recs), 4) if ok_recs else None
    # ---- 路由指标(sr-auto + 已标注测试集)
    if args.mode == "sr-auto" and any(r["expected_model"] for r in ok_recs):
        routed = [r for r in ok_recs if r["expected_model"] in ("S", "L")]
        if routed:
            hit = 0
            confusion: dict[str, int] = {}
            for r in routed:
                actual = "L" if r["model_used"] == args.large_model else "S"
                if actual == r["expected_model"]:
                    hit += 1
                key = f"expected_{r['expected_model']}_actual_{actual}"
                confusion[key] = confusion.get(key, 0) + 1
            summary["routing_accuracy"] = round(hit / len(routed), 4)
            summary["routing_confusion_matrix"] = confusion
            summary["routing_evaluated_requests"] = len(routed)
    summary["errors"] = [r["error"] for r in records if r["error"]][:5]

    # ------------------------------------------------------------ 写文件
    with open(os.path.join(out_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "per_request.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[done] 明细与汇总已写入 {out_dir}/")


if __name__ == "__main__":
    main()
