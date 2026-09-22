"""
Store tool — CRUD interface for the unified SQLite store.

Exposes store operations as functions callable from CLI, MCP, or Python.
All operations are idempotent and concurrent-safe (SQLite WAL).
"""
from __future__ import annotations

from pathlib import Path

from vllm_evolve.store.db import Store

# Module-level singleton (lazy init)
_store: Store | None = None


def get_store(db_path: str | Path = "vllm_evolve.db") -> Store:
    """Get or create the global store instance."""
    global _store
    if _store is None:
        _store = Store(db_path)
    return _store


def best(
    item_type: str = "policy",
    target: str | None = None,
    model: str | None = None,
    hardware: str | None = None,
    n: int = 5,
) -> list[dict]:
    """Get top-N items by fitness."""
    store = get_store()
    if item_type == "policy":
        return store.best_policies(target or "scheduling", n)
    else:
        return store.best_configs(model or "", hardware or "", n)


def history(item_type: str, item_id: str) -> list[dict]:
    """Get all evaluations for an item."""
    return get_store().history(item_type, item_id)


def compare(item_type: str, id1: str, id2: str) -> dict:
    """Compare two items."""
    return get_store().compare(item_type, id1, id2)


def list_items(
    item_type: str = "policy",
    filter_value: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List items with their fitness."""
    return get_store().list_all(item_type, filter_value, limit)


def stats() -> dict:
    """Quick stats about the store."""
    return get_store().stats()


# --- Round 1 (Codex review): route public callables through phase guard ---
from vllm_evolve.tools._legacy_guard import _install_guard as _install_legacy_guard  # noqa: E402

_install_legacy_guard(__name__, ("get_store", "best", "history", "compare", "list_items", "stats",))
