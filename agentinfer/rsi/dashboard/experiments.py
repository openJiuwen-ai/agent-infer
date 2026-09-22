# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Read-only HTML renderer for the append-only experiment ledger."""

import json
from html import escape

from agentinfer.rsi.experiments import METRICS


def render_experiments(records: list[dict[str, object]]) -> str:
    """Render only reported records; demo state is never substituted."""

    cards = []
    for record in records:
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        keys = (*METRICS, *(key for key in metrics if key not in METRICS))
        rows = "".join(
            f"<tr><th scope='row'>{escape(str(key))}</th><td>"
            f"{escape('Not measured (null)' if metrics.get(key) is None else str(metrics.get(key)))}</td></tr>"
            for key in keys
        )
        config = record["config"]
        assert isinstance(config, dict)
        precision = config.get("precision") or "Not recorded (null)"
        baseline = record.get("baseline_round_id") or "Not specified (null)"
        reason = record.get("failure_reason") or "None reported (null)"
        next_test = record.get("next_test") or "Not specified (null)"
        evidence = (
            "".join(
                f"<li><code>{escape(str(item['path']))}</code> — SHA-256: "
                f"<code>{escape(str(item.get('sha256') or 'Not recorded (null)'))}</code></li>"
                for item in record["evidence"]
            )
            or "<li>No evidence references</li>"
        )
        cards.append(
            f"<article><h2>{escape(str(record['round_id']))} "
            f"<span class='status {escape(str(record['status']))}'>{escape(str(record['status']))}</span></h2>"
            f"<p>{escape(str(record['timestamp']))} · {escape(str(record['model']))} · {escape(str(record['backend']))}</p>"
            f"<p>Component version: <code>{escape(str(record['component_version']))}</code></p>"
            f"<p>Model precision (config.precision): {escape(str(precision))}</p>"
            f"<p>Baseline reference: {escape(str(baseline))} (no automatic comparison)</p>"
            f"<p><strong>Hypothesis:</strong> {escape(str(record['hypothesis']))}</p>"
            f"<p><strong>Change:</strong> {escape(str(record['change']))}</p>"
            f"<table><caption>Reported measurements · GSM8K accuracy is a fraction in [0, 1]</caption>"
            f"<thead><tr><th scope='col'>Metric / unit</th><th scope='col'>Value</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            f"<p><strong>Failure / blocking reason:</strong> {escape(str(reason))}</p>"
            f"<p><strong>Next test:</strong> {escape(str(next_test))}</p>"
            f"<h3>Evidence references (contents and hashes are not verified here)</h3><ul>{evidence}</ul>"
            f"<details><summary>Full record, config and commands</summary>"
            f"<pre>{escape(json.dumps(record, indent=2, allow_nan=False))}</pre></details></article>"
        )
    content = "".join(cards) or "<p>No real experiment records. No measurements are available.</p>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>RSI Dashboard · Real experiments</title><style>"
        "body{margin:2rem auto;padding:0 1rem;max-width:1050px;background:#f3f4ee;color:#202e28;"
        "font-family:Arial,'DejaVu Sans',sans-serif;font-size:16px;line-height:1.6;font-weight:400}"
        "h1,h2,h3,p,li,summary,th,td,caption,code,pre{color:#202e28}a{color:#245439}"
        "article,.notice{background:#fffefa;"
        "border:1px solid #c8d0c3;border-radius:8px;padding:1.2rem;margin:1.2rem 0}"
        "article,pre,code{overflow-wrap:anywhere}pre{white-space:pre-wrap}"
        "table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.5rem;border:1px solid #c8d0c3}"
        "caption{text-align:left}.status{font-size:1rem;border:1px solid;padding:.2rem .5rem;border-radius:4px}"
        ".measured{color:#245439}.blocked,.not_run{color:#785019}.failed{color:#943c2d}"
        "summary{cursor:pointer}</style></head><body><main><h1>Real experiments · Read only</h1>"
        "<p><a href='/experiments.html'>Refresh records</a> · <a href='/api/rsi/experiments'>Read JSON API</a></p>"
        "<div class='notice'>Source: experiments.jsonl only. No demo or simulated values. "
        "Status is reported by the experiment author: measured does not mean accepted or independently verified. "
        "Missing values remain null; missing GSM8K accuracy leaves the quality gate incomplete. "
        "No production connection, acceptance decision or activation is provided.</div>"
        f"{content}</main></body></html>"
    )
