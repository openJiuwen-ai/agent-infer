"""P1a: Store must apply the migration registry on open.

Regression guard for the lost wiring (CODEX_LOG R2-M1 / refactor B1): the
migrations module's docstring promises "Store applies pending migrations
automatically on open", but `Store.__init__` only ran `executescript(_SCHEMA)`
and never called `apply_migrations`, so on the live DB the `artifacts` table and
`evaluations.run_id` / `policies.full_sha` columns never existed. This locks the
wiring in. No GPU.
"""
from __future__ import annotations

from vllm_evolve.store.db import Store


def _tables(store: Store) -> set[str]:
    return {r["name"] for r in store.query(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def _cols(store: Store, table: str) -> set[str]:
    return {r["name"] for r in store.query(f"PRAGMA table_info({table})")}


def test_store_applies_migration_registry_on_open(tmp_path):
    store = Store(tmp_path / "fresh.db")
    try:
        applied = {r["id"] for r in store.query("SELECT id FROM schema_migrations")}
        assert {"0001_eval_run_id", "0002_policies_full_sha",
                "0003_artifacts", "0004_runs"} <= applied
        assert "artifacts" in _tables(store)
        assert "runs" in _tables(store)
        assert "run_id" in _cols(store, "evaluations")
        assert "full_sha" in _cols(store, "policies")
        # 0004: matrix-view columns for mode-c (version/hardware iteration)
        assert {"vllm_version", "hardware_id"} <= _cols(store, "evaluations")
    finally:
        store.close()


def test_store_open_is_idempotent(tmp_path):
    db = tmp_path / "again.db"
    Store(db).close()           # first open applies migrations
    store = Store(db)           # second open must be a no-op, not error
    try:
        rows = store.query("SELECT id FROM schema_migrations")
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))            # no duplicate registry rows
        assert "0003_artifacts" in ids
    finally:
        store.close()
