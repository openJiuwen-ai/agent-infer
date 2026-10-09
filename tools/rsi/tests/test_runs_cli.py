"""P5-1: `ve runs ls` / `ve runs gc` over the runs/ working area. No GPU."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from vllm_evolve.artifacts import layout
from vllm_evolve.cli import main as ve_cli


def _mk(runs_root, mode, n, when):
    return layout.create_run(mode=mode, target="scheduling", runs_root=runs_root,
                             policy_sha=f"sha{n}", when=when)


def test_list_and_gc_runs_unit(tmp_path):
    runs = tmp_path / "runs"
    for i in range(3):
        _mk(runs, "autopt", i, datetime(2026, 6, 10, 12, i, 0, tzinfo=timezone.utc))
    rows = layout.list_runs(runs)
    assert len(rows) == 3 and rows[0]["dir"] > rows[-1]["dir"]   # newest first
    # keep_last=1 removes the 2 older; none kept (empty archive root)
    removed = layout.gc_runs(runs, keep_last=1, archive_root=tmp_path / "archive")
    assert len(removed) == 2
    assert len(layout.list_runs(runs)) == 1


def test_gc_exempts_kept_winners(tmp_path):
    runs = tmp_path / "runs"
    rl = _mk(runs, "autopt", 0, datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc))
    arch = tmp_path / "archive" / "scheduling" / rl.run_id
    arch.mkdir(parents=True)                                     # simulate a kept winner
    removed = layout.gc_runs(runs, keep_last=0, archive_root=tmp_path / "archive")
    assert removed == []                                         # kept winner is exempt
    assert len(layout.list_runs(runs)) == 1


def test_ve_runs_ls_and_gc_cli(tmp_path, capsys):
    runs = tmp_path / "runs"
    _mk(runs, "autopt", 0, datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc))
    rc = ve_cli.main(["runs", "ls", "--runs-root", str(runs)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True and out["count"] == 1

    rc = ve_cli.main(["runs", "gc", "--runs-root", str(runs), "--keep-last", "0", "--dry-run"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["dry_run"] is True and out["removed_count"] == 1
    assert len(layout.list_runs(runs)) == 1                      # dry-run did not delete
