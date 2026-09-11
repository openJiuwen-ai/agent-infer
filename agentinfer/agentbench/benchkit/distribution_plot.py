# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Render benchmark distribution samples as a headless summary PNG."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_RUN_COLORS = ("#2B6CB0", "#D1495B", "#2A9D8F", "#7A5195", "#E07A1F", "#5B6770")
_RUN_HATCHES = ("", "////", "\\\\", "xxxx", "....", "++++")
_ROLE_HATCHES = {"lead": "", "subagent": "////"}
_LATENCY_BUCKET_EDGES = np.asarray((10.0, 30.0, 60.0, 120.0, 300.0))
_LATENCY_BUCKET_LABELS = ("0–10s", "10–30s", "30–60s", "60–120s", "120–300s", "≥300s")
_LATENCY_BUCKET_COLORS = ("#DCEAF7", "#A9CBE8", "#6FA8D6", "#367DB5", "#F0A35E", "#C94C4C")
_FIGURE_BACKGROUND = "#F4F7FA"
_AXES_BACKGROUND = "#FFFFFF"
_TEXT_COLOR = "#263746"
_GRID_COLOR = "#D7E0E8"

_REQUIRED_COLUMNS = {
    "run_id",
    "level",
    "metric",
    "value",
    "instance_id",
    "actor_role",
    "value_status",
}


def write_distribution_plots(samples_path: Path, output_dir: Path) -> None:
    """Render the distribution summary from normalized samples."""

    write_distribution_plots_from_rows(_load_samples(samples_path), output_dir)


def write_distribution_plots_from_rows(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Render the distribution summary from validated sample rows."""

    _validate_samples(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(rows, output_dir / "distribution_summary.png")
    _write_task_request_breakdown(rows, output_dir / "task_request_breakdown.png")


def _load_samples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"distribution samples missing columns: {sorted(missing)}")
        return list(reader)


def _validate_samples(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row["value_status"] == "valid":
            try:
                float(row["value"])
            except ValueError as exc:
                raise ValueError(f"invalid distribution value: {row['value']}") from exc


def _values(rows: list[dict[str, str]], level: str, metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["level"] == level and row["metric"] == metric and row["value_status"] == "valid":
            grouped[row["run_id"]].append(float(row["value"]))
    return dict(grouped)


def _run_colors(groups: dict[str, object]) -> dict[str, str]:
    return {run_id: _RUN_COLORS[index % len(_RUN_COLORS)] for index, run_id in enumerate(groups)}


def _style_axis(axis) -> None:
    axis.set_facecolor(_AXES_BACKGROUND)
    axis.tick_params(colors=_TEXT_COLOR, labelsize=8)
    axis.xaxis.label.set_color(_TEXT_COLOR)
    axis.yaxis.label.set_color(_TEXT_COLOR)
    axis.title.set_color(_TEXT_COLOR)
    for spine in axis.spines.values():
        spine.set_color(_GRID_COLOR)


def _plot_ecdf(axis, groups: dict[str, list[float]], title: str, xlabel: str) -> None:
    if not groups:
        _no_samples(axis, title)
        return
    colors = _run_colors(groups)
    for run_id, values in groups.items():
        ordered = np.sort(values)
        fractions = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.step(
            ordered,
            fractions,
            where="post",
            color=colors[run_id],
            linewidth=1.8,
            label=f"{run_id} (n={len(values)})",
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Cumulative fraction")
    axis.set_ylim(0, 1.02)
    axis.grid(color=_GRID_COLOR, alpha=0.65, linewidth=0.7)
    axis.legend(fontsize="small")


def _plot_probability_histogram(axis, groups: dict[str, list[float]], title: str, xlabel: str) -> None:
    if not groups:
        _no_samples(axis, title)
        return
    combined = np.asarray([value for values in groups.values() for value in values], dtype=float)
    limit = float(np.quantile(combined, 0.99))
    visible = combined[combined <= limit]
    edges = np.histogram_bin_edges(visible, bins="fd")
    if len(edges) < 2 or edges[0] == edges[-1]:
        center = float(visible[0])
        width = max(abs(center) * 0.05, 1e-9)
        edges = np.asarray([center - width, center + width])
    colors = _run_colors(groups)
    for run_id, values in groups.items():
        plotted = np.asarray([value for value in values if value <= limit], dtype=float)
        weights = np.full(len(plotted), 1 / len(values))
        axis.hist(
            plotted,
            bins=edges,
            weights=weights,
            histtype="step",
            linewidth=1.8,
            color=colors[run_id],
            label=f"{run_id} (n={len(values)})",
        )
    axis.set_title(f"{title} (x ≤ combined P99)")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Fraction of requests per bin")
    axis.grid(color=_GRID_COLOR, alpha=0.65, linewidth=0.7)
    axis.legend(fontsize="small")


def _boxplot(axis, groups: dict[str, list[float]], title: str, ylabel: str) -> None:
    if not groups:
        _no_samples(axis, title)
        return
    labels = list(groups)
    colors = _run_colors(groups)
    result = axis.boxplot([groups[label] for label in labels], showfliers=False, patch_artist=True)
    for label, box in zip(labels, result["boxes"], strict=True):
        box.set_facecolor(colors[label])
        box.set_edgecolor(colors[label])
        box.set_alpha(0.42)
    for label, median in zip(labels, result["medians"], strict=True):
        median.set_color(colors[label])
        median.set_linewidth(1.8)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=_GRID_COLOR, alpha=0.65, linewidth=0.7)
    axis.tick_params(axis="x", rotation=20)


def _agent_role_boxplot(axis, rows: list[dict[str, str]], metric: str, title: str, ylabel: str) -> None:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["level"] != "agent" or row["metric"] != metric or row["value_status"] != "valid":
            continue
        groups[row["run_id"]][row["actor_role"]].append(float(row["value"]))
    if not groups:
        _no_samples(axis, title)
        return

    run_ids = list(groups)
    observed_roles = {role for run in groups.values() for role in run}
    roles = [role for role in ("lead", "subagent") if role in observed_roles]
    roles.extend(sorted(observed_roles - set(roles)))
    colors = _run_colors(groups)
    width = min(0.7 / len(roles), 0.3)
    offsets = (np.arange(len(roles)) - (len(roles) - 1) / 2) * width
    for role, offset in zip(roles, offsets, strict=True):
        positions = []
        values = []
        for run_index, run_id in enumerate(run_ids, start=1):
            if role in groups[run_id]:
                positions.append(run_index + offset)
                values.append(groups[run_id][role])
        result = axis.boxplot(values, positions=positions, widths=width * 0.85, showfliers=False, patch_artist=True)
        for position, box in zip(positions, result["boxes"], strict=True):
            run_id = run_ids[round(position - offset) - 1]
            box.set_facecolor(colors[run_id])
            box.set_edgecolor(colors[run_id])
            box.set_alpha(0.5)
            box.set_hatch(_ROLE_HATCHES.get(role, ".."))
        for median in result["medians"]:
            median.set_color(_TEXT_COLOR)
            median.set_linewidth(1.6)

    role_handles = [
        Patch(facecolor="#B8C4CE", edgecolor=_TEXT_COLOR, hatch=_ROLE_HATCHES.get(role, ".."), label=role)
        for role in roles
    ]

    axis.set_xticks(range(1, len(run_ids) + 1), run_ids)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=_GRID_COLOR, alpha=0.65, linewidth=0.7)
    axis.tick_params(axis="x", rotation=20)
    axis.legend(handles=role_handles, title="Actor role", fontsize="small")


def _task_request_buckets(
    rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, list[str]], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    run_ids = []
    grouped: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for row in rows:
        if (
            row["level"] != "request"
            or row["metric"] != "latency_seconds"
            or row["value_status"] != "valid"
            or row["instance_id"] == "N/A"
        ):
            continue
        run_id = row["run_id"]
        task_id = row["instance_id"]
        if run_id not in grouped:
            run_ids.append(run_id)
        counts, latency = grouped[run_id].setdefault(
            task_id,
            (np.zeros(len(_LATENCY_BUCKET_LABELS)), np.zeros(len(_LATENCY_BUCKET_LABELS))),
        )
        value = float(row["value"])
        bucket = int(np.searchsorted(_LATENCY_BUCKET_EDGES, value, side="right"))
        counts[bucket] += 1
        latency[bucket] += value
    ranked_tasks = {
        run_id: sorted(grouped[run_id], key=lambda task_id: (grouped[run_id][task_id][1].sum(), task_id))
        for run_id in run_ids
    }
    return run_ids, ranked_tasks, grouped


def _plot_task_stacks(
    axis,
    run_ids: list[str],
    ranked_tasks: dict[str, list[str]],
    grouped: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    value_index: int,
    title: str,
    ylabel: str,
) -> None:
    rank_count = max(map(len, ranked_tasks.values()))
    positions = np.arange(rank_count, dtype=float)
    width = min(0.82 / len(run_ids), 0.22)
    offsets = (np.arange(len(run_ids)) - (len(run_ids) - 1) / 2) * width
    for run_index, (run_id, offset) in enumerate(zip(run_ids, offsets, strict=True)):
        values = np.asarray([grouped[run_id][task_id][value_index] for task_id in ranked_tasks[run_id]])
        run_positions = positions[: len(values)] + offset
        bottom = np.zeros(len(values))
        for bucket, color in enumerate(_LATENCY_BUCKET_COLORS):
            axis.bar(
                run_positions,
                values[:, bucket],
                width=width * 0.9,
                bottom=bottom,
                color=color,
                edgecolor=_TEXT_COLOR,
                linewidth=0.25,
                hatch=_RUN_HATCHES[run_index % len(_RUN_HATCHES)],
            )
            bottom += values[:, bucket]
    tick_step = max(1, int(np.ceil(rank_count / 32)))
    tick_indexes = np.arange(0, rank_count, tick_step)
    if tick_indexes[-1] != rank_count - 1:
        tick_indexes = np.append(tick_indexes, rank_count - 1)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(tick_indexes, tick_indexes + 1)
    axis.set_xlabel("Within-run task latency rank (ascending)")
    axis.grid(axis="y", color=_GRID_COLOR, alpha=0.65, linewidth=0.7)
    axis.set_axisbelow(True)


def _write_task_request_breakdown(rows: list[dict[str, str]], path: Path) -> None:
    run_ids, ranked_tasks, grouped = _task_request_buckets(rows)
    rank_count = max(map(len, ranked_tasks.values()), default=0)
    width = max(20.0, rank_count * 0.32)
    figure, axes = plt.subplots(2, 1, figsize=(width, 12), sharex=True, facecolor=_FIGURE_BACKGROUND)
    for axis in axes:
        _style_axis(axis)
    if not run_ids or not ranked_tasks:
        for axis, title in zip(
            axes,
            (
                "Requests by within-run task latency rank",
                "Summed request latency by within-run task latency rank",
            ),
            strict=True,
        ):
            _no_samples(axis, title)
    else:
        _plot_task_stacks(
            axes[0],
            run_ids,
            ranked_tasks,
            grouped,
            value_index=0,
            title="Requests by within-run task latency rank",
            ylabel="Request count",
        )
        _plot_task_stacks(
            axes[1],
            run_ids,
            ranked_tasks,
            grouped,
            value_index=1,
            title="Summed request latency by within-run task latency rank",
            ylabel="Summed latency (seconds)",
        )
        bucket_handles = [
            Patch(facecolor=color, edgecolor=_TEXT_COLOR, label=label)
            for label, color in zip(_LATENCY_BUCKET_LABELS, _LATENCY_BUCKET_COLORS, strict=True)
        ]
        run_handles = [
            Patch(
                facecolor="white",
                edgecolor=_TEXT_COLOR,
                hatch=_RUN_HATCHES[index % len(_RUN_HATCHES)],
                label=run_id,
            )
            for index, run_id in enumerate(run_ids)
        ]
        figure.legend(
            handles=bucket_handles,
            title="Request latency bucket",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.96),
            ncols=len(bucket_handles),
            fontsize="small",
        )
        figure.legend(
            handles=run_handles,
            title="Run",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.96),
            ncols=min(len(run_handles), 2),
            fontsize="small",
        )
    figure.suptitle(
        "Task request breakdown — tasks ranked independently within each run",
        fontsize=16,
        color=_TEXT_COLOR,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_summary(rows: list[dict[str, str]], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=_FIGURE_BACKGROUND)
    for axis in axes.flat:
        _style_axis(axis)
    _plot_ecdf(
        axes[0, 0],
        _values(rows, "task", "task_duration_seconds"),
        "Task duration",
        "seconds",
    )
    _plot_probability_histogram(
        axes[0, 1],
        _values(rows, "request", "latency_seconds"),
        "Request latency",
        "seconds",
    )
    _plot_probability_histogram(
        axes[0, 2],
        _values(rows, "request", "ttft_seconds"),
        "Request TTFT",
        "seconds",
    )
    _plot_probability_histogram(
        axes[1, 0],
        _values(rows, "request", "tpot_seconds"),
        "Request TPOT",
        "seconds/token",
    )
    _boxplot(
        axes[1, 1],
        _values(rows, "session", "request_ttft_seconds_p50"),
        "Per-session TTFT P50",
        "seconds",
    )
    _agent_role_boxplot(
        axes[1, 2],
        rows,
        "request_latency_seconds_p50",
        "Per-agent latency P50 by role",
        "seconds",
    )
    figure.suptitle("AgentInfer distribution summary", fontsize=16, color=_TEXT_COLOR, fontweight="bold")
    _save(figure, path)


def _no_samples(axis, title: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
