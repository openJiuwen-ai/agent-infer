"""SQLite migration registry — principled, idempotent, rollback-safe schema evolution.

Replaces the ad-hoc ``ALTER`` pile that used to live in ``Store.__init__`` /
``Store._migrate``. Each migration is a ``detect / up / validate`` triple (plus an
optional ``down``); ``up`` is written to be idempotent and each migration is
recorded in ``schema_migrations`` so it runs at most once. ``Store`` applies
pending migrations automatically on open; ``ar migrate store`` drives the same
registry explicitly with ``status`` / ``--dry-run`` / ``apply``.

Atomicity: each migration runs inside ``with conn:`` (commit on success, roll
back on any exception or a failed ``validate``), so a half-applied migration can
never be recorded. File-level backup + restore for a whole ``ar migrate`` run is
the CLI's responsibility (M0c), layered on top of this.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass


class MigrationError(RuntimeError):
    """A migration's ``validate`` failed, or ``up`` raised."""


@dataclass(frozen=True)
class Migration:
    """One schema step.

    ``detect`` reports whether the change is still pending (for ``status`` /
    dry-run reporting); ``up`` must be idempotent (safe to run even if partially
    applied); ``validate`` confirms the end state; ``down`` is optional and best
    effort (rollback prefers a file-level backup, not ``down``).
    """

    id: str
    description: str
    up: Callable[[sqlite3.Connection], None]
    validate: Callable[[sqlite3.Connection], bool]
    detect: Callable[[sqlite3.Connection], bool]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _has_trigger(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
    ).fetchone() is not None


# --------------------------------------------------------------------------
# 0001 — evaluations.run_id  (formerly Store._migrate; here so the registry owns it)
# --------------------------------------------------------------------------

def _up_run_id(conn: sqlite3.Connection) -> None:
    if "run_id" not in _cols(conn, "evaluations"):
        conn.execute("ALTER TABLE evaluations ADD COLUMN run_id TEXT DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluations(run_id)")


def _detect_run_id(conn: sqlite3.Connection) -> bool:
    return "run_id" not in _cols(conn, "evaluations")


def _validate_run_id(conn: sqlite3.Connection) -> bool:
    return "run_id" in _cols(conn, "evaluations")


# --------------------------------------------------------------------------
# 0002 — policies.full_sha + backfill from existing source_code
# --------------------------------------------------------------------------

def _up_full_sha(conn: sqlite3.Connection) -> None:
    if "full_sha" not in _cols(conn, "policies"):
        conn.execute("ALTER TABLE policies ADD COLUMN full_sha TEXT")
    rows = conn.execute(
        "SELECT policy_id, source_code FROM policies "
        "WHERE full_sha IS NULL OR full_sha=''"
    ).fetchall()
    for pid, src in rows:
        full = hashlib.sha256((src or "").encode("utf-8")).hexdigest()
        conn.execute("UPDATE policies SET full_sha=? WHERE policy_id=?", (full, pid))
    # policy_id = full_sha[:16] (content-addressed), so full_sha is unique per row.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_policies_full_sha ON policies(full_sha)"
    )


def _detect_full_sha(conn: sqlite3.Connection) -> bool:
    if "full_sha" not in _cols(conn, "policies"):
        return True
    return conn.execute(
        "SELECT 1 FROM policies WHERE full_sha IS NULL OR full_sha='' LIMIT 1"
    ).fetchone() is not None


def _validate_full_sha(conn: sqlite3.Connection) -> bool:
    if "full_sha" not in _cols(conn, "policies"):
        return False
    return conn.execute(
        "SELECT 1 FROM policies WHERE full_sha IS NULL OR full_sha='' LIMIT 1"
    ).fetchone() is None


# --------------------------------------------------------------------------
# 0003 — artifacts index table (bundle provenance)
# --------------------------------------------------------------------------

_ARTIFACTS_DDL = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id        TEXT PRIMARY KEY,
    evaluation_id      INTEGER REFERENCES evaluations(id),
    run_id             TEXT NOT NULL,
    item_type          TEXT NOT NULL DEFAULT 'policy',
    policy_id          TEXT NOT NULL REFERENCES policies(policy_id),
    policy_full_sha    TEXT NOT NULL,
    config_full_sha    TEXT DEFAULT '',
    target_name        TEXT NOT NULL,
    profile            TEXT NOT NULL,
    evaluator          TEXT NOT NULL,
    bundle_path        TEXT NOT NULL UNIQUE,
    bundle_sha256      TEXT NOT NULL,
    eval_result_sha256 TEXT,
    outcome_class      TEXT NOT NULL,
    completeness       TEXT NOT NULL CHECK (
        completeness IN ('complete', 'partial', 'metrics_only', 'missing')
    ),
    created_at         TEXT DEFAULT (datetime('now'))
);
"""


def _up_artifacts(conn: sqlite3.Connection) -> None:
    # Single-statement execute() (NOT executescript, which force-commits and would
    # break the migration's explicit transaction).
    conn.execute(_ARTIFACTS_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifacts_policy_full_sha "
        "ON artifacts(policy_full_sha)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifacts_eval ON artifacts(evaluation_id)"
    )


def _detect_artifacts(conn: sqlite3.Connection) -> bool:
    return not _has_table(conn, "artifacts")


def _validate_artifacts(conn: sqlite3.Connection) -> bool:
    return _has_table(conn, "artifacts")


# --------------------------------------------------------------------------
# 0004 — run-level index (the `runs` table) + evaluations version/hardware cols
#
# `runs` records run-LEVEL metadata (one row per `runs/<target>/<...>/` bundle);
# the existing `artifacts` table records the in-run bundle items and their sha,
# joined back via `run_id` (idx_artifacts_run). The two new evaluations columns
# let the matrix view (mode-c version/hardware iteration) filter by vLLM version
# and hardware. New columns are nullable (put_eval does not yet write them).
# --------------------------------------------------------------------------

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    mode         TEXT NOT NULL,
    target       TEXT NOT NULL,
    vllm_version TEXT DEFAULT '',
    hardware_id  TEXT DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'created',
    started_at   TEXT DEFAULT (datetime('now')),
    ended_at     TEXT DEFAULT '',
    manifest_sha TEXT DEFAULT ''
);
"""


def _up_runs(conn: sqlite3.Connection) -> None:
    # Single-statement execute() (NOT executescript, which force-commits and would
    # break the migration's explicit transaction).
    conn.execute(_RUNS_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_target ON runs(target)")
    cols = _cols(conn, "evaluations")
    if "vllm_version" not in cols:
        conn.execute("ALTER TABLE evaluations ADD COLUMN vllm_version TEXT")
    if "hardware_id" not in cols:
        conn.execute("ALTER TABLE evaluations ADD COLUMN hardware_id TEXT")


def _detect_runs(conn: sqlite3.Connection) -> bool:
    if not _has_table(conn, "runs"):
        return True
    cols = _cols(conn, "evaluations")
    return "vllm_version" not in cols or "hardware_id" not in cols


def _validate_runs(conn: sqlite3.Connection) -> bool:
    if not _has_table(conn, "runs"):
        return False
    cols = _cols(conn, "evaluations")
    return "vllm_version" in cols and "hardware_id" in cols


# --------------------------------------------------------------------------
# 0005 — lesson_evidence (Evolution Engine v2)
#
# IMMUTABLE evidence rows for evolution lessons: insert-only, so a cited
# lesson_id forever points at unchanged evidence (citation integrity — a later
# run can never rewrite what an earlier run learned). Dedup/retrieval happens
# through a deterministic aggregate VIEW keyed on (policy_sha, regime, metric);
# `source` names the true origin (e.g. frontier_sim) — a lesson can never
# impersonate a real-vLLM result.
# --------------------------------------------------------------------------

_LESSONS_DDL = """
CREATE TABLE IF NOT EXISTS lesson_evidence (
    lesson_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    source      TEXT NOT NULL,
    policy_sha  TEXT NOT NULL,
    regime      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    score       REAL,
    conclusion  TEXT NOT NULL,
    eval_refs   TEXT DEFAULT '[]',
    created_at  TEXT DEFAULT (datetime('now'))
);
"""

_LESSONS_VIEW_DDL = """
CREATE VIEW IF NOT EXISTS lessons_latest AS
SELECT le.* FROM lesson_evidence le
JOIN (
    SELECT policy_sha, regime, metric, MAX(lesson_id) AS lesson_id
    FROM lesson_evidence GROUP BY policy_sha, regime, metric
) latest ON le.lesson_id = latest.lesson_id
"""


def _up_lessons(conn: sqlite3.Connection) -> None:
    conn.execute(_LESSONS_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lessons_regime "
                 "ON lesson_evidence(regime, metric)")
    conn.execute(_LESSONS_VIEW_DDL)


def _detect_lessons(conn: sqlite3.Connection) -> bool:
    return not _has_table(conn, "lesson_evidence")


def _validate_lessons(conn: sqlite3.Connection) -> bool:
    return _has_table(conn, "lesson_evidence")


# 0006 — hypothesis_events (Research Harness, event-sourced ledger)
#
# APPEND-ONLY events for the hypothesis ledger: insert-only, so a cited event_id forever points at
# unchanged evidence. Three event_types chain a hypothesis: prediction_registered ->
# experiment_executed (requires an existing prediction) -> adjudication_recorded (ref_event_id pins
# the EXACT execution event). The prediction-first + execution-anchor invariants are enforced by the
# db.py helpers (the table only ever sees INSERTs). A hypothesis's current state is a deterministic
# aggregate of its events. ``source`` names the true origin (e.g. frontier_sim); ledger rows are
# PROPOSAL-layer data and never feed the adoption gate.
# --------------------------------------------------------------------------

_HYPOTHESES_DDL = """
CREATE TABLE IF NOT EXISTS hypothesis_events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id  TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    run_id         TEXT,
    payload        TEXT DEFAULT '{}',
    ref_event_id   INTEGER REFERENCES hypothesis_events(event_id),
    source         TEXT,
    created_at     TEXT DEFAULT (datetime('now'))
);
"""


def _up_hypotheses(conn: sqlite3.Connection) -> None:
    conn.execute(_HYPOTHESES_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hypothesis_events_hid "
                 "ON hypothesis_events(hypothesis_id, event_id)")


def _detect_hypotheses(conn: sqlite3.Connection) -> bool:
    return not _has_table(conn, "hypothesis_events")


def _validate_hypotheses(conn: sqlite3.Connection) -> bool:
    return _has_table(conn, "hypothesis_events")


# 0007 — hypothesis_events append-only ENFORCED at the schema level (Research Harness)
#
# 0006's append-only guarantee was helper-only; the table is an ordinary local SQLite file reachable
# by raw SQL, so a cited event_id was not truly immutable. BEFORE UPDATE / BEFORE DELETE triggers
# now RAISE(ABORT) so a cited ``hypothesis:<event_id>`` forever points at byte-for-byte-unchanged
# evidence — even against raw ``UPDATE``/``DELETE``. (New migration id; 0006 is left untouched.)
# --------------------------------------------------------------------------

_HYP_APPEND_ONLY_DDL = """
CREATE TRIGGER IF NOT EXISTS hypothesis_events_no_update
BEFORE UPDATE ON hypothesis_events
BEGIN SELECT RAISE(ABORT, 'hypothesis_events is append-only (no UPDATE)'); END;
"""
_HYP_NO_DELETE_DDL = """
CREATE TRIGGER IF NOT EXISTS hypothesis_events_no_delete
BEFORE DELETE ON hypothesis_events
BEGIN SELECT RAISE(ABORT, 'hypothesis_events is append-only (no DELETE)'); END;
"""


def _up_hyp_append_only(conn: sqlite3.Connection) -> None:
    conn.execute(_HYP_APPEND_ONLY_DDL)
    conn.execute(_HYP_NO_DELETE_DDL)


def _detect_hyp_append_only(conn: sqlite3.Connection) -> bool:
    return not (_has_trigger(conn, "hypothesis_events_no_update")
                and _has_trigger(conn, "hypothesis_events_no_delete"))


def _validate_hyp_append_only(conn: sqlite3.Connection) -> bool:
    return (_has_trigger(conn, "hypothesis_events_no_update")
            and _has_trigger(conn, "hypothesis_events_no_delete"))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

MIGRATIONS: list[Migration] = [
    Migration("0001_eval_run_id", "evaluations.run_id + index",
              _up_run_id, _validate_run_id, _detect_run_id),
    Migration("0002_policies_full_sha", "policies.full_sha column + backfill",
              _up_full_sha, _validate_full_sha, _detect_full_sha),
    Migration("0003_artifacts", "artifacts provenance index table",
              _up_artifacts, _validate_artifacts, _detect_artifacts),
    Migration("0004_runs", "runs table + evaluations vllm_version/hardware_id",
              _up_runs, _validate_runs, _detect_runs),
    Migration("0005_lessons", "immutable lesson_evidence + lessons_latest view",
              _up_lessons, _validate_lessons, _detect_lessons),
    Migration("0006_hypotheses", "append-only hypothesis_events ledger",
              _up_hypotheses, _validate_hypotheses, _detect_hypotheses),
    Migration("0007_hypothesis_events_append_only",
              "hypothesis_events UPDATE/DELETE prevention triggers",
              _up_hyp_append_only, _validate_hyp_append_only, _detect_hyp_append_only),
]


def _ensure_registry(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  id TEXT PRIMARY KEY, checksum TEXT, "
        "  applied_at TEXT DEFAULT (datetime('now')), description TEXT)"
    )
    conn.commit()  # commit the DDL in both default-isolation and autocommit modes


def applied_ids(conn: sqlite3.Connection) -> set[str]:
    _ensure_registry(conn)
    return {r[0] for r in conn.execute("SELECT id FROM schema_migrations").fetchall()}


def status(
    conn: sqlite3.Connection, migrations: list[Migration] | None = None
) -> list[dict]:
    """Per-migration applied / pending-change view for ``ar migrate store status``."""
    migrations = migrations if migrations is not None else MIGRATIONS
    # Read-only: do NOT create schema_migrations here (status must not mutate).
    done: set[str] = set()
    if _has_table(conn, "schema_migrations"):
        done = {
            r[0] for r in conn.execute("SELECT id FROM schema_migrations").fetchall()
        }
    out = []
    for m in migrations:
        is_applied = m.id in done
        out.append({
            "id": m.id,
            "description": m.description,
            "applied": is_applied,
            "pending_change": (not is_applied) and m.detect(conn),
        })
    return out


def apply_migrations(
    conn: sqlite3.Connection,
    migrations: list[Migration] | None = None,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Apply all not-yet-recorded migrations in order. Returns the ids applied.

    ``dry_run=True`` reports what *would* apply without touching the schema.
    Each migration is atomic (``with conn:``); a failed ``validate`` rolls back
    and raises ``MigrationError`` so it is never recorded as applied.
    """
    migrations = migrations if migrations is not None else MIGRATIONS
    # Take manual transaction control (isolation_level=None == autocommit) so each
    # migration is wrapped in an explicit BEGIN IMMEDIATE .. COMMIT/ROLLBACK; this
    # is the only way to guarantee DDL (ALTER/CREATE) is atomic with the registry
    # insert in Python's sqlite3. Restored on the way out.
    prev_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        _ensure_registry(conn)
        done = {
            r[0] for r in conn.execute("SELECT id FROM schema_migrations").fetchall()
        }
        applied: list[str] = []
        for m in migrations:
            if m.id in done:
                continue
            if dry_run:
                applied.append(m.id)
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                m.up(conn)
                if not m.validate(conn):
                    raise MigrationError(f"migration {m.id} failed validate()")
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations "
                    "(id, checksum, applied_at, description) "
                    "VALUES (?, '', datetime('now'), ?)",
                    (m.id, m.description),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")  # no half-applied DDL, not recorded
                raise
            applied.append(m.id)
        return applied
    finally:
        conn.isolation_level = prev_isolation
