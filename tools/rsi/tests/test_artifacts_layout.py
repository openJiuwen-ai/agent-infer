"""P1c: run-centric artifact layout (artifacts/layout.py). Pure filesystem; no GPU."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from vllm_evolve.artifacts.layout import (
    compute_run_id,
    create_run,
    manifest_sha,
)


def test_run_id_stable_across_mutable_fields():
    identity = {"mode": "autopt", "target": "scheduling", "policy_sha": "abc",
                "vllm_version": "0.21.0", "hardware_id": "h100", "git_sha": "g1",
                "design_sha": "d1", "spec": {"metric": "goodput"}}
    base = compute_run_id(identity)
    mutated = dict(identity, run_id="x", manifest_sha="y", status="done",
                   started_at="t0", ended_at="t1")
    assert compute_run_id(mutated) == base          # excluded fields don't move the id


def test_run_id_changes_with_identity():
    a = compute_run_id({"mode": "autopt", "target": "scheduling", "hardware_id": "h100"})
    b = compute_run_id({"mode": "autopt", "target": "scheduling", "hardware_id": "a3"})
    assert a != b                                    # mode-c matrix: hardware splits id


def test_create_run_builds_bundle(tmp_path):
    when = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
    rl = create_run(mode="autopt", target="scheduling", runs_root=tmp_path,
                    policy_sha="deadbeef", vllm_version="0.21.0", hardware_id="h100",
                    spec={"metric": "goodput"}, when=when)
    assert rl.run_dir.is_dir()
    assert rl.bench_dir.is_dir() and rl.gpu_dir.is_dir()
    assert rl.manifest_path.is_file()
    # dir name shape: <ts>-<mode>-<run_id>
    assert rl.run_dir.name == f"20260610-120000-autopt-{rl.run_id}"
    man = rl.read_manifest()
    assert man["run_id"] == rl.run_id == compute_run_id(man)   # id == identity hash
    assert man["policy_sha"] == "deadbeef" and man["status"] == "created"
    assert man["manifest_sha"] == manifest_sha(man)            # integrity self-consistent
    json.loads(rl.manifest_path.read_text(encoding="utf-8"))   # valid JSON on disk


def test_create_run_path_components_are_safe(tmp_path):
    rl = create_run(mode="../evil", target="a/b/../c", runs_root=tmp_path, when=
                    datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc))
    # the bundle must stay under runs_root (no traversal out)
    assert tmp_path.resolve() in rl.run_dir.resolve().parents
    # exactly: runs_root / <target-slug> / <ts>-<mode-slug>-<run_id>
    assert rl.run_dir.parent.parent == tmp_path
