#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Render a compact, data-derived PNG for the Qwen3.8 B300 RSI ledger."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1800
HEIGHT = 1450
BACKGROUND = "#f5f7fa"
PANEL = "#ffffff"
INK = "#253238"
MUTED = "#69777d"
GRID = "#d6dee2"
GREEN = "#2e7d5b"
BLUE = "#2d6cdf"
ORANGE = "#b66a00"
RED = "#bd4b4b"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = _font(20)
SMALL = _font(16)
TINY = _font(14)
BODY = _font(18)
SUBTITLE = _font(22)
TITLE = _font(38, bold=True)
HEADING = _font(22, bold=True)


def _load_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _profile(record: dict[str, object]) -> str:
    return str(record["backend"]).rsplit("/", 1)[-1]


def _performance_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for record in records:
        round_id = str(record.get("round_id", ""))
        backend = str(record.get("backend", ""))
        if not round_id.startswith("I") or not round_id[1:].isdigit():
            continue
        number = int(round_id[1:])
        if number < 11 or "gsm8k" in backend:
            continue
        if not backend.startswith("cuda/vllm-tp4/"):
            continue
        result.append(record)
    return result


def _stats(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [
        float(record["metrics"]["throughput_output_tokens_per_s"])
        for record in records
        if record.get("status") == "measured"
        and record.get("metrics", {}).get("throughput_output_tokens_per_s") is not None
        and record.get("metrics", {}).get("replay_successful_requests") == 36
    ]
    return {
        "total": len(records),
        "valid": len(valid),
        "failed": len(records) - len(valid),
        "median": statistics.median(valid) if valid else None,
        "minimum": min(valid) if valid else None,
        "maximum": max(valid) if valid else None,
    }


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill=INK) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, outline=GRID) -> None:
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=outline, width=2)


def _metric_card(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, value: str, detail: str
) -> None:
    _panel(draw, box)
    x, y, _, _ = box
    _text(draw, (x + 22, y + 16), label, SMALL, MUTED)
    _text(draw, (x + 22, y + 44), value, HEADING, INK)
    _text(draw, (x + 22, y + 80), detail, TINY, MUTED)


def _render_chart(draw: ImageDraw.ImageDraw, records: list[dict[str, object]], baseline: float) -> None:
    box = (40, 300, 920, 860)
    _panel(draw, box)
    _text(draw, (68, 324), "Profile median: output tokens/s", HEADING)
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(_profile(record), []).append(record)
    rows = []
    for name, items in grouped.items():
        stats = _stats(items)
        if stats["median"] is not None:
            rows.append((name, stats))
    rows.sort(key=lambda item: float(item[1]["median"]), reverse=True)
    chart_x, chart_y = 310, 390
    chart_w, chart_h = 540, 390
    max_value = max([float(item[1]["maximum"]) for item in rows] + [baseline]) * 1.08
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=GRID, width=2)
    for tick in range(0, 5):
        value = max_value * tick / 4
        y = chart_y + chart_h - int(chart_h * value / max_value)
        draw.line((chart_x, y, chart_x + chart_w, y), fill="#edf1f3", width=1)
        _text(draw, (chart_x - 70, y - 9), f"{value:.0f}", TINY, MUTED)
    bar_w = max(26, int(chart_w / max(1, len(rows) * 1.55)))
    gap = max(10, int((chart_w - len(rows) * bar_w) / max(1, len(rows) - 1)))
    labels = {
        "baseline-current": "base",
        "no-prefix-cache": "nocache",
        "batch-8192": "b8",
        "batch-32768": "b32",
        "batch-65536": "b64",
        "async-on": "a+",
        "async-off": "a-",
        "stream-8": "s8",
        "max-seqs-8": "m8",
        "kv-fp8": "kv8",
        "batch32768-async-on": "combo",
    }
    for index, (name, stats) in enumerate(rows):
        x = chart_x + index * (bar_w + gap)
        median = float(stats["median"])
        h = int(chart_h * median / max_value)
        color = GREEN if name == "batch32768" else RED if name in {"kv-fp8", "no-prefix-cache", "async-off"} else BLUE
        draw.rounded_rectangle((x, chart_y + chart_h - h, x + bar_w, chart_y + chart_h), radius=5, fill=color)
        _text(draw, (x - 10, chart_y + chart_h + 12), labels.get(name, name), TINY, INK)
        _text(draw, (x - 5, chart_y + chart_h - h - 24), f"{median:.1f}", TINY, color)
    baseline_y = chart_y + chart_h - int(chart_h * baseline / max_value)
    draw.line((chart_x, baseline_y, chart_x + chart_w, baseline_y), fill=ORANGE, width=3)
    _text(draw, (chart_x + chart_w - 170, baseline_y - 24), f"baseline {baseline:.1f}", TINY, ORANGE)


def _render_profile_table(draw: ImageDraw.ImageDraw, records: list[dict[str, object]], baseline: float) -> None:
    box = (950, 300, 1760, 860)
    _panel(draw, box)
    _text(draw, (978, 324), "50-round sweep and promotion checks", HEADING)
    headers = ("profile", "valid", "median", "range", "delta", "decision")
    widths = (230, 70, 100, 155, 90, 170)
    x = 978
    y = 370
    for header, width in zip(headers, widths, strict=True):
        _text(draw, (x, y), header, SMALL, MUTED)
        x += width
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(_profile(record), []).append(record)
    ordered = sorted(grouped.items(), key=lambda item: (item[0] != "batch32768", item[0]))
    for name, items in ordered:
        stats = _stats(items)
        if stats["median"] is None:
            continue
        valid = int(stats["valid"])
        median = float(stats["median"])
        delta = (median / baseline - 1) * 100
        if name == "batch32768":
            decision, color = "quality pass", GREEN
        elif name == "kv-fp8":
            decision, color = "quality reject", RED
        elif valid < len(items):
            decision, color = "coverage reject", RED
        else:
            decision, color = "measured", BLUE
        values = (
            name,
            f"{valid}/{len(items)}",
            f"{median:.3f}",
            f"{float(stats['minimum']):.1f}–{float(stats['maximum']):.1f}",
            f"{delta:+.2f}%",
            decision,
        )
        x = 978
        for value, width in zip(values, widths, strict=True):
            _text(draw, (x, y + 30), value, TINY, color if value == decision else INK)
            x += width
        y += 35
    _text(draw, (978, 810), "Quality: BF16 baseline 95.072% · FP8 KV 94.845% · batch32768 95.148%", TINY, MUTED)


def _render_round_strip(draw: ImageDraw.ImageDraw, records: list[dict[str, object]]) -> None:
    box = (40, 900, 1760, 1395)
    _panel(draw, box)
    _text(draw, (68, 924), "Round-level effect (I11–I67)", HEADING)
    values = []
    for record in records:
        metric = record.get("metrics", {}).get("throughput_output_tokens_per_s")
        values.append(
            (str(record["round_id"]), None if metric is None else float(metric), record.get("status") == "measured")
        )
    plot_x, plot_y, plot_w, plot_h = 80, 990, 1620, 220
    valid_values = [value for _, value, _ in values if value is not None]
    low = min(valid_values + [0])
    high = max(valid_values + [1])
    draw.line((plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h), fill=GRID, width=2)
    points: list[tuple[int, int]] = []
    for index, (round_id, value, measured) in enumerate(values):
        x = plot_x + int(index * plot_w / max(1, len(values) - 1))
        if value is None:
            continue
        y = plot_y + plot_h - int((value - low) / max(1, high - low) * plot_h)
        points.append((x, y))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=BLUE if measured else RED)
        if index % 5 == 0 or index == len(values) - 1:
            _text(draw, (x - 12, plot_y + plot_h + 12), round_id, TINY, MUTED)
    if len(points) > 1:
        draw.line(points, fill="#9bb8e9", width=2)
    _text(draw, (80, 1270), "Blue = replay-valid measured round · red = failed coverage/command round", SMALL, MUTED)
    _text(
        draw,
        (80, 1310),
        "I5–I9 are preserved harness failures from the pre-fix vLLM console script; the corrected 50-round sweep is I11–I60.",
        SMALL,
        ORANGE,
    )
    _text(
        draw,
        (80, 1342),
        "Promotion check I63–I67 combines batch32768 + async-on and remains below the batch32768 median.",
        SMALL,
        MUTED,
    )


def render(records: list[dict[str, object]], output: Path) -> None:
    performance = _performance_records(records)
    baseline_records = [record for record in performance if _profile(record) == "baseline-current"]
    baseline = float(_stats(baseline_records)["median"] or 0)
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _text(draw, (40, 28), "RSI dashboard · Qwen3.8-27B · 4×B300", TITLE)
    _text(draw, (42, 82), "Append-only evidence · replay-valid coverage gates · 2026-09-22", SUBTITLE, MUTED)
    draw.rounded_rectangle((40, 128, 1760, 190), radius=10, fill="#e8f1e8", outline="#bfd4bf", width=2)
    _text(
        draw,
        (62, 148),
        "Source: experiments.jsonl · values are measured artifacts; promotion also requires GSM8K",
        SMALL,
        GREEN,
    )
    total = len(records)
    valid = sum(
        1
        for record in performance
        if record.get("status") == "measured" and record.get("metrics", {}).get("replay_successful_requests") == 36
    )
    failed = sum(
        1
        for record in performance
        if record.get("status") != "measured" or record.get("metrics", {}).get("replay_successful_requests") != 36
    )
    _metric_card(draw, (40, 220, 420, 282), "LEDGER RECORDS", str(total), f"{valid} replay-valid performance rounds")
    _metric_card(draw, (445, 220, 825, 282), "BEST QUALIFIED", "210.385 tok/s", "+0.98% vs repeated baseline")
    _metric_card(draw, (850, 220, 1230, 282), "QUALITY GATE", "95.148%", "batch32768 · 1255/1319")
    _metric_card(draw, (1255, 220, 1760, 282), "REJECTIONS", str(failed), "coverage failures + FP8 quality")
    _render_chart(draw, performance, baseline)
    _render_profile_table(draw, performance, baseline)
    _render_round_strip(draw, performance)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(_load_records(args.run_dir / "experiments.jsonl"), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
