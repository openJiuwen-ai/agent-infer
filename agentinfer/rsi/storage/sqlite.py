# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Small SQLite store. No process, model, or deployment side effects."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class ConflictError(ValueError):
    """A revision or idempotency precondition did not hold."""


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Database:
    def __init__(self, path):
        if str(path) == ":memory:":
            raise ValueError("RSI requires a file-backed SQLite database")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            for statement in (
                "CREATE TABLE IF NOT EXISTS rsi_runs (run_id TEXT PRIMARY KEY, state TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS rsi_commands (run_id TEXT, key TEXT, request TEXT NOT NULL, "
                "response TEXT NOT NULL, PRIMARY KEY(run_id, key))",
                "CREATE TABLE IF NOT EXISTS rsi_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "run_id TEXT NOT NULL, event TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS rsi_knowledge (id TEXT PRIMARY KEY, scope TEXT NOT NULL, "
                "backend TEXT NOT NULL, component_version TEXT NOT NULL, workload TEXT NOT NULL, "
                "status TEXT NOT NULL, search_text TEXT NOT NULL, record TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS rsi_knowledge_history (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "knowledge_id TEXT NOT NULL, record TEXT NOT NULL)",
            ):
                db.execute(statement)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
