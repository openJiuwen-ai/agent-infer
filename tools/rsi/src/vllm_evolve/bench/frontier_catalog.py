"""P2 measurement catalog — query Frontier's FULL output, with one shared metric evaluator.

Lower-level than ``frontier_sim.parse_frontier_metrics``: ``load_catalog`` reads EVERY per-request
column + the whole ``system_metrics.json`` + the per-batch ledger + the ``ve_policy`` marker into
one ``FrontierCatalog``. ``evaluate(MetricExpr, catalog)`` is the SINGLE evaluator shared by
experiment metric requests (P1) and hypothesis adjudication (P3) — so a metric a prediction
references is either genuinely computable or **explicitly missing**, never an invented number.

Honesty (no fabrication): a column Frontier did not emit, or an aggregate with no numeric data,
returns ``value=None`` plus a non-empty ``missing`` list. The caller (adjudication) turns any
missing into ``inconclusive``; it never substitutes a value.

Layer: ``bench/`` — must NOT import ``core/`` or ``engine/`` (the frozen-core direction guard scans
every ``bench/**`` file). All latency columns/stats are in MILLISECONDS (confirmed on a real run).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

_AGGS = ("p50", "p90", "p99", "mean", "max", "min", "count", "rate")
_GROUP_REDUCES = ("max", "mean", "min")


@dataclass(frozen=True)
class MetricExpr:
    """A structured metric expression — the SHARED, typed contract between P1 (experiment metric
    requests) and P3 (adjudication), so the same expression yields the same value in both places.
    ``from_obj`` normalizes a plain mapping (CLI/JSON input) into this type. bench-layer, no
    core/engine import.

    Fields: ``column`` (a catalog column), ``agg`` in p50|p90|p99|mean|max|min|count|rate,
    ``group_by`` (optional column to partition by), ``group_reduce`` in max|mean|min (reduce across
    groups; default max — e.g. worst per-tenant p99), ``units`` (free-text metadata)."""

    column: str
    agg: str
    group_by: str | None = None
    group_reduce: str = "max"
    units: str | None = None

    @classmethod
    def from_obj(cls, obj) -> MetricExpr:
        if isinstance(obj, MetricExpr):
            return obj
        m = obj or {}
        return cls(column=m.get("column"), agg=m.get("agg"), group_by=m.get("group_by"),
                   group_reduce=m.get("group_reduce") or "max", units=m.get("units"))

    def is_valid(self) -> bool:
        return bool(self.column) and self.agg in _AGGS and self.group_reduce in _GROUP_REDUCES

    def to_dict(self) -> dict:
        return {"column": self.column, "agg": self.agg, "group_by": self.group_by,
                "group_reduce": self.group_reduce, "units": self.units}


def metrics_dir(out_dir: str, run_id: str) -> Path:
    """Locate the dir holding Frontier's metric files for THIS ``run_id``. Frontier nests by model +
    serving-mode + run_id (e.g. ``<out_dir>/<model_slug>/online_serving/<run_id>/``). A run_id-named
    ancestor wins; if none matches, fall back ONLY when there is exactly one candidate. If several
    candidates exist and none matches ``run_id``, return a non-existent path so the catalog reads
    empty (explicit missing) rather than silently loading the WRONG run — multiple arms/seeds can
    share an out_dir, so guessing would let adjudication cite a different experiment's numbers."""
    base = Path(out_dir)
    for fname in ("system_metrics.json", "request_metrics.csv"):
        hits = sorted(base.rglob(fname))
        if not hits:
            continue
        matches = [p for p in hits if run_id in p.parts]
        if matches:
            return matches[0].parent
        if len(hits) == 1:
            return hits[0].parent
        return base / f"__no_run_match_{run_id}__"   # ambiguous, no run_id match -> refuse to guess
    return base


@dataclass
class FrontierCatalog:
    """Everything Frontier wrote for one run, loaded once and queried many ways."""

    out_dir: str
    run_id: str
    dir: Path
    system: dict = field(default_factory=dict)        # full system_metrics.json
    rows: list = field(default_factory=list)          # per-request: list[dict[str,str]]
    columns: list = field(default_factory=list)       # request_metrics.csv header
    ledger: list = field(default_factory=list)        # per-batch stage ledger rows
    marker: dict | None = None                        # ve_policy honesty marker (or None)

    def has_column(self, name: str) -> bool:
        return name in self.columns


@dataclass
class CatalogValue:
    """The result of ``evaluate``: a real value, or None + the list of what was missing."""

    value: float | None
    missing: list = field(default_factory=list)
    n: int = 0

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.missing


def load_catalog(out_dir: str, run_id: str) -> FrontierCatalog:
    """Read all of Frontier's real output files into a queryable catalog. Missing files -> empty
    parts (never fabricated). The ``ve_policy`` marker lands in the out_dir root (VE_MARKER_DIR),
    not the nested metrics dir — check both."""
    mdir = metrics_dir(out_dir, run_id)

    system: dict = {}
    sm_path = mdir / "system_metrics.json"
    if sm_path.is_file():
        system = json.loads(sm_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    columns: list[str] = []
    rm_path = mdir / "request_metrics.csv"
    if rm_path.is_file():
        with rm_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]

    ledger: list[dict] = []
    lg_path = mdir / "frontier_stage_batch_ledger.jsonl"
    if lg_path.is_file():
        for line in lg_path.read_text(encoding="utf-8").splitlines():
            try:
                ledger.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    marker = None
    for cand in (Path(out_dir) / "ve_policy_marker.json", mdir / "ve_policy_marker.json"):
        if cand.is_file():
            marker = json.loads(cand.read_text(encoding="utf-8"))
            break

    return FrontierCatalog(out_dir=out_dir, run_id=run_id, dir=mdir, system=system,
                           rows=rows, columns=columns, ledger=ledger, marker=marker)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _percentile(values: list, q: float) -> float:
    """Linear-interpolated percentile (q in [0,100]); ``values`` must be non-empty. Pure-python,
    deterministic — matches numpy.percentile's default ('linear') without importing it here."""
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(xs):
        return float(xs[-1])
    return float(xs[lo] + (xs[lo + 1] - xs[lo]) * frac)


def _aggregate(agg: str, values: list, catalog: FrontierCatalog) -> float | None:
    """Apply one aggregate to numeric ``values``. None when there is no data to aggregate
    (count is the exception — zero rows is a real count of 0)."""
    if agg == "count":
        return float(len(values))
    if not values:
        return None
    if agg == "mean":
        return sum(values) / len(values)
    if agg == "max":
        return float(max(values))
    if agg == "min":
        return float(min(values))
    if agg in ("p50", "p90", "p99"):
        return _percentile(values, float(agg[1:]))
    if agg == "rate":
        dur = _num((catalog.system.get("throughput_metrics") or {}).get("total_duration_seconds"))
        if dur is None or dur <= 0:
            return None
        return len(values) / dur
    return None


def evaluate(expr, catalog: FrontierCatalog) -> CatalogValue:
    """Evaluate a ``MetricExpr`` (or a plain mapping, normalized by ``from_obj``) against
    a catalog. Shared verbatim by P1 (experiment metric requests) and P3 (adjudication), so the same
    expression yields the same value in both places. A referenced column the catalog lacks ->
    ``value=None`` + ``missing=[column]`` (never invented)."""
    spec = MetricExpr.from_obj(expr)
    if not spec.is_valid():
        return CatalogValue(
            None, missing=[f"bad_metric_expr:{spec.column}:{spec.agg}:{spec.group_reduce}"])
    column, agg, group_by = spec.column, spec.agg, spec.group_by

    missing: list[str] = []
    if not catalog.has_column(column):
        missing.append(column)
    if group_by and not catalog.has_column(group_by):
        missing.append(group_by)
    if missing:
        return CatalogValue(None, missing=missing)

    rows = catalog.rows

    def _col_values(subset: list) -> list:
        return [v for v in (_num(r.get(column)) for r in subset) if v is not None]

    if group_by:
        groups: dict = {}
        for r in rows:
            groups.setdefault(r.get(group_by), []).append(r)
        per_group = []
        for sub in groups.values():
            g = _aggregate(agg, _col_values(sub), catalog)
            if g is not None:
                per_group.append(g)
        if not per_group:
            return CatalogValue(None, missing=[f"{column}:no-data"])
        value = {"max": max, "min": min,
                 "mean": lambda xs: sum(xs) / len(xs)}[spec.group_reduce](per_group)
        return CatalogValue(float(value), n=len(per_group))

    vals = _col_values(rows)
    value = _aggregate(agg, vals, catalog)
    if value is None:
        return CatalogValue(None, missing=[f"{column}:no-data"])
    return CatalogValue(float(value), n=len(vals))
