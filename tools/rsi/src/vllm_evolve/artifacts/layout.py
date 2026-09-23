"""Run-centric artifact layout — the working-area bundle for a single run.

A run lives at ``runs/<target>/<YYYYMMDD-HHMMSS>-<mode>-<run_id>/`` and holds a
self-describing ``manifest.json`` plus slots for ``policy.py``,
``eval_result.json``, ``profile.json``, ``design.md`` (snapshot), ``run.jsonl``,
and ``bench/`` + ``gpu/`` subdirs.

Artifact layout rules:
* **Stable run_id** = sha256 of the *canonical* manifest **identity subset**
  (``_RUNID_EXCLUDED`` removes the mutable / self-referential fields), so a run's
  id never changes as ``status`` / ``ended_at`` evolve. ``policy_sha`` is a
  separate field — the run_id is NOT the policy source sha (that is what the
  legacy ``keep`` archive path uses; the two id schemes must not be confused).
* **Atomic writes** — manifest is written to a temp file then ``os.replace``-d,
  so a db sha check never reads a half-written file.
* **Path-safe components** — ``target`` / ``mode`` are slugged; a run dir can
  never escape ``runs_root``.
* **Pure** — no DB, no agent/LLM imports. ``runs/`` is the WORKING area, not the
  adoption trust boundary (that stays ``archive_policies/`` + db sha; PLAN §3.0).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RUNS_ROOT = "runs"

# Mutable / self-referential manifest fields — excluded from the run_id preimage
# so the id is stable across the run's lifecycle.
_RUNID_EXCLUDED = frozenset({"run_id", "manifest_sha", "status", "started_at", "ended_at"})

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _slug(text: str) -> str:
    """Path-safe single component (collapse unsafe runs, strip leading/trailing . and -)."""
    s = _UNSAFE.sub("-", str(text)).strip("-.")
    return s or "x"


def _utc_stamp(when: datetime | None = None) -> str:
    return (when or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")


def compute_run_id(fields: dict) -> str:
    """Stable run_id = sha256(canonical(identity subset of manifest))[:12]."""
    preimage = {k: v for k, v in fields.items() if k not in _RUNID_EXCLUDED}
    return hashlib.sha256(_canonical_json(preimage).encode("utf-8")).hexdigest()[:12]


def manifest_sha(manifest: dict) -> str:
    """Integrity sha of a manifest (over everything except ``manifest_sha`` itself)."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + os.replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


@dataclass(frozen=True)
class RunLayout:
    """Handle to a created run bundle."""

    run_dir: Path
    run_id: str
    manifest_path: Path

    @property
    def bench_dir(self) -> Path:
        return self.run_dir / "bench"

    @property
    def gpu_dir(self) -> Path:
        return self.run_dir / "gpu"

    def read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


def create_run(
    *,
    mode: str,
    target: str,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    policy_sha: str = "",
    vllm_version: str = "",
    hardware_id: str = "",
    git_sha: str = "",
    design_sha: str = "",
    spec: dict | None = None,
    when: datetime | None = None,
) -> RunLayout:
    """Create ``runs/<target>/<ts>-<mode>-<run_id>/`` with manifest + bench/gpu dirs.

    ``run_id`` is derived from the immutable identity (mode/target/policy_sha/
    vllm_version/hardware_id/git_sha/design_sha/spec); two runs differing only by
    e.g. hardware get distinct ids (supports the mode-c version/hardware matrix).
    """
    identity = {
        "mode": mode,
        "target": target,
        "policy_sha": policy_sha,
        "vllm_version": vllm_version,
        "hardware_id": hardware_id,
        "git_sha": git_sha,
        "design_sha": design_sha,
        "spec": spec or {},
    }
    run_id = compute_run_id(identity)
    when = when or datetime.now(timezone.utc)
    run_dir = Path(runs_root) / _slug(target) / f"{_utc_stamp(when)}-{_slug(mode)}-{run_id}"
    (run_dir / "bench").mkdir(parents=True, exist_ok=True)
    (run_dir / "gpu").mkdir(parents=True, exist_ok=True)

    manifest = dict(identity)
    manifest.update({
        "run_id": run_id,
        "status": "created",
        "started_at": when.isoformat(),
        "ended_at": "",
    })
    manifest["manifest_sha"] = manifest_sha(manifest)

    manifest_path = run_dir / "manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
    )
    return RunLayout(run_dir=run_dir, run_id=run_id, manifest_path=manifest_path)


def _read_manifest(run_dir: Path) -> dict:
    mp = run_dir / "manifest.json"
    if not mp.is_file():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def list_runs(
    runs_root: str | Path = DEFAULT_RUNS_ROOT, *, target: str | None = None,
) -> list[dict]:
    """List run bundles (newest first) as summaries read from each manifest.

    Pure read; tolerant of partial/missing manifests. ``target`` filters by subdir."""
    root = Path(runs_root)
    out: list[dict] = []
    if not root.is_dir():
        return out
    targets = [root / target] if target else [d for d in root.iterdir() if d.is_dir()]
    for tdir in targets:
        if not tdir.is_dir():
            continue
        for run_dir in tdir.iterdir():
            if not run_dir.is_dir():
                continue
            man = _read_manifest(run_dir)
            out.append({
                "run_id": man.get("run_id", run_dir.name),
                "target": man.get("target", tdir.name),
                "mode": man.get("mode", ""),
                "status": man.get("status", ""),
                "started_at": man.get("started_at", ""),
                "vllm_version": man.get("vllm_version", ""),
                "hardware_id": man.get("hardware_id", ""),
                "dir": str(run_dir),
            })
    out.sort(key=lambda r: r["dir"], reverse=True)            # dir name is timestamp-prefixed
    return out


def _kept_run_ids(archive_root: str | Path) -> set[str]:
    """run_ids that were promoted to the curated archive (kept winners) — gc-exempt."""
    root = Path(archive_root)
    ids: set[str] = set()
    if not root.is_dir():
        return ids
    for tdir in root.iterdir():
        if tdir.is_dir():
            ids.update(d.name for d in tdir.iterdir() if d.is_dir())
    return ids


def gc_runs(
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    *,
    keep_last: int = 20,
    older_than_days: float | None = None,
    archive_root: str | Path = "archive_policies",
    dry_run: bool = False,
) -> list[dict]:
    """Garbage-collect run bundles, returning the removed (or would-remove) summaries.

    A run is removable if it falls beyond ``keep_last`` (per target, newest kept) OR is
    older than ``older_than_days``. Kept winners (promoted to ``archive_root``) are always
    exempt. ``dry_run`` reports without deleting."""
    import shutil
    import time

    kept = _kept_run_ids(archive_root)
    cutoff = (time.time() - older_than_days * 86400.0) if older_than_days is not None else None
    removed: list[dict] = []
    # group by target, newest first
    by_target: dict[str, list[dict]] = {}
    for r in list_runs(runs_root):
        by_target.setdefault(r["target"], []).append(r)
    for runs in by_target.values():
        for idx, r in enumerate(runs):                       # runs already newest-first
            if r["run_id"] in kept:
                continue
            too_many = idx >= keep_last
            too_old = cutoff is not None and Path(r["dir"]).stat().st_mtime < cutoff
            if too_many or too_old:
                if not dry_run:
                    shutil.rmtree(r["dir"], ignore_errors=True)
                removed.append(r)
    return removed
