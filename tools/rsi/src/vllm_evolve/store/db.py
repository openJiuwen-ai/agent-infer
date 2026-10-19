"""
SQLite store — single database for all vllm-evolve data.

Tables:
  policies        — evolved code (content-addressed, immutable)
  lineage         — parent→child relationships
  configs         — serving configurations (content-addressed)
  evaluations     — all evaluation results (policies + configs, all cascade levels)
  checks          — trust/safety check records
  modeling_samples — training data for lightweight predictors
  gpu_timings     — hardware profiling data (existing)

Design principles:
  - Every tool auto-writes to the DB (no manual store.put needed for evaluations)
  - Content-addressed: same code/config → same ID (idempotent)
  - WAL mode for concurrent reads (multi-agent safe)
  - Single file: vllm_evolve.db in project root
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from vllm_evolve.store.migrations import apply_migrations

_SCHEMA = """\
-- Evolved code policies (need 2: algorithm evolution)
CREATE TABLE IF NOT EXISTS policies (
    policy_id     TEXT PRIMARY KEY,
    source_code   TEXT NOT NULL,
    function_name TEXT NOT NULL,
    target_name   TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now'))
);

-- Code lineage
CREATE TABLE IF NOT EXISTS lineage (
    policy_id  TEXT PRIMARY KEY REFERENCES policies(policy_id),
    parent_id  TEXT,
    generation INTEGER DEFAULT 0,
    strategy   TEXT,
    diff_text  TEXT
);

-- Serving configurations (need 1: day-0 tuning)
CREATE TABLE IF NOT EXISTS configs (
    config_id   TEXT PRIMARY KEY,
    model_name  TEXT NOT NULL,
    hardware    TEXT NOT NULL,
    params_json TEXT NOT NULL,
    source      TEXT DEFAULT 'search',
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Unified evaluation results (both needs, all cascade levels)
CREATE TABLE IF NOT EXISTS evaluations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type    TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    scenario     TEXT NOT NULL,
    evaluator    TEXT NOT NULL,
    seed         INTEGER DEFAULT 42,
    fitness      REAL,
    metrics_json TEXT NOT NULL,
    confidence   REAL,
    wall_time_s  REAL,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_eval_item ON evaluations(item_type, item_id);
CREATE INDEX IF NOT EXISTS idx_eval_tier ON evaluations(evaluator);

-- Trust / legality check records
CREATE TABLE IF NOT EXISTS checks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type   TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    issues_json TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Lightweight model training data (distilled from detailed sim / NPU)
CREATE TABLE IF NOT EXISTS modeling_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name    TEXT NOT NULL,
    hardware      TEXT NOT NULL,
    regime        TEXT NOT NULL,
    features_json TEXT NOT NULL,
    latency_ms    REAL NOT NULL,
    source        TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now'))
);

-- GPU/NPU timing (existing schema, kept compatible)
CREATE TABLE IF NOT EXISTS gpu_timings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model         TEXT NOT NULL,
    gpu           TEXT NOT NULL,
    operation     TEXT NOT NULL,
    batch_size    INTEGER NOT NULL,
    seq_len       INTEGER NOT NULL,
    kv_block_size INTEGER DEFAULT 16,
    mean_ms       REAL NOT NULL,
    p50_ms        REAL NOT NULL,
    p99_ms        REAL NOT NULL,
    std_ms        REAL NOT NULL,
    num_samples   INTEGER DEFAULT 10,
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(model, gpu, operation, batch_size, seq_len, kv_block_size)
);

-- Views for quick ranking
CREATE VIEW IF NOT EXISTS policy_ranking AS
SELECT p.policy_id, p.target_name, l.generation, l.strategy,
       AVG(e.fitness) AS avg_fitness,
       MAX(e.fitness) AS best_fitness,
       COUNT(e.id) AS eval_count,
       p.created_at
FROM policies p
LEFT JOIN lineage l ON p.policy_id = l.policy_id
LEFT JOIN evaluations e ON e.item_type = 'policy' AND e.item_id = p.policy_id
GROUP BY p.policy_id;

CREATE VIEW IF NOT EXISTS config_ranking AS
SELECT c.config_id, c.model_name, c.hardware, c.params_json,
       AVG(CASE WHEN e.evaluator = 'lightweight' THEN e.fitness END) AS l0_fitness,
       AVG(CASE WHEN e.evaluator IN ('live_npu', 'live_vllm', 'real_vllm')
                THEN e.fitness END) AS l2_fitness,
       MIN(e.confidence) AS min_confidence,
       COUNT(e.id) AS eval_count,
       c.created_at
FROM configs c
LEFT JOIN evaluations e ON e.item_type = 'config' AND e.item_id = c.config_id
GROUP BY c.config_id;
"""


def _content_hash(text: str) -> str:
    """SHA-256 content hash, first 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Store:
    """Unified SQLite store for vllm-evolve.

    Usage:
        store = Store()                    # opens/creates vllm_evolve.db
        store = Store("path/to/my.db")     # custom path

        # Policies (need 2)
        pid = store.put_policy(code, "schedule_batch", "scheduling")
        store.put_lineage(pid, parent_id="abc", generation=3, strategy="edit")

        # Configs (need 1)
        cid = store.put_config("Qwen2.5-72B", "ascend_a3_8x", {"max_num_batched_tokens": 8192})

        # Evaluations (real vLLM only)
        store.put_eval("policy", pid, "steady_low", "real_vllm", fitness=0.62, metrics={...})

        # Queries
        store.best_policies("scheduling", n=5)
        store.best_configs("Qwen2.5-72B", "ascend_a3_8x", n=5)
        store.history("policy", pid)
        store.compare("policy", id1, id2)
    """

    def __init__(self, db_path: str | Path = "vllm_evolve.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            apply_migrations(self._conn)  # registry-owned schema evolution (idempotent)
        except sqlite3.DatabaseError:
            # Corrupted DB — remove and recreate
            self._path.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                self._path.with_name(self._path.name + suffix).unlink(missing_ok=True)
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            apply_migrations(self._conn)  # registry-owned schema evolution (idempotent)

    # ── Policies ──────────────────────────────────────────────

    def put_policy(
        self,
        source_code: str,
        function_name: str,
        target_name: str,
    ) -> str:
        """Store a policy. Idempotent (same code → same ID). Returns policy_id."""
        policy_id = _content_hash(source_code)
        self._conn.execute(
            "INSERT OR IGNORE INTO policies (policy_id, source_code, function_name, target_name) "
            "VALUES (?, ?, ?, ?)",
            (policy_id, source_code, function_name, target_name),
        )
        self._conn.commit()
        return policy_id

    def put_lineage(
        self,
        policy_id: str,
        parent_id: str | None = None,
        generation: int = 0,
        strategy: str = "",
        diff_text: str = "",
    ) -> None:
        """Record lineage for a policy."""
        self._conn.execute(
            "INSERT OR REPLACE INTO lineage "
            "(policy_id, parent_id, generation, strategy, diff_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (policy_id, parent_id, generation, strategy, diff_text),
        )
        self._conn.commit()

    def get_policy(self, policy_id: str) -> dict | None:
        """Get a policy by ID."""
        row = self._conn.execute(
            "SELECT * FROM policies WHERE policy_id = ?", (policy_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── Configs ───────────────────────────────────────────────

    def put_config(
        self,
        model_name: str,
        hardware: str,
        params: dict,
        source: str = "search",
    ) -> str:
        """Store a config. Idempotent. Returns config_id."""
        params_json = json.dumps(params, sort_keys=True)
        config_id = _content_hash(params_json + model_name + hardware)
        self._conn.execute(
            "INSERT OR IGNORE INTO configs (config_id, model_name, hardware, params_json, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (config_id, model_name, hardware, params_json, source),
        )
        self._conn.commit()
        return config_id

    def get_config(self, config_id: str) -> dict | None:
        """Get a config by ID."""
        row = self._conn.execute(
            "SELECT * FROM configs WHERE config_id = ?", (config_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── Evaluations ───────────────────────────────────────────

    def put_eval(
        self,
        item_type: str,
        item_id: str,
        scenario: str,
        evaluator: str,
        fitness: float | None = None,
        metrics: dict | None = None,
        confidence: float | None = None,
        wall_time_s: float | None = None,
        seed: int = 42,
    ) -> int:
        """Record an evaluation result. Returns row ID."""
        cur = self._conn.execute(
            "INSERT INTO evaluations "
            "(item_type, item_id, scenario, evaluator, seed, "
            "fitness, metrics_json, confidence, wall_time_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_type, item_id, scenario, evaluator, seed,
                fitness, json.dumps(metrics or {}), confidence, wall_time_s,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    # ── Checks ────────────────────────────────────────────────

    def put_check(
        self,
        item_type: str,
        item_id: str,
        passed: bool,
        issues: list[str] | None = None,
    ) -> None:
        """Record a trust/safety check result."""
        self._conn.execute(
            "INSERT INTO checks (item_type, item_id, passed, issues_json) VALUES (?, ?, ?, ?)",
            (item_type, item_id, int(passed), json.dumps(issues or [])),
        )
        self._conn.commit()

    def put_lesson(self, *, run_id: str, source: str, policy_sha: str, regime: str,
                   metric: str, score: float | None, conclusion: str,
                   eval_refs: list | None = None) -> int:
        """Insert one IMMUTABLE lesson-evidence row (Evolution v2). Insert-only by design —
        a cited lesson_id forever points at unchanged evidence. Returns the lesson_id."""
        cur = self._conn.execute(
            "INSERT INTO lesson_evidence (run_id, source, policy_sha, regime, metric, score, "
            "conclusion, eval_refs) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, source, policy_sha, regime, metric, score, conclusion,
             json.dumps(eval_refs or [])),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_lessons(self, *, regime: str | None = None, metric: str | None = None,
                    limit: int = 10) -> list[dict]:
        """Deduped lessons (latest evidence per (policy_sha, regime, metric) via the
        lessons_latest view), best-score-first, HARD-capped by ``limit``."""
        q = "SELECT * FROM lessons_latest"
        cond, args = [], []
        if regime:
            cond.append("regime = ?")
            args.append(regime)
        if metric:
            cond.append("metric = ?")
            args.append(metric)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY score DESC, lesson_id DESC LIMIT ?"
        args.append(max(1, int(limit)))
        rows = self._conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def get_lesson(self, lesson_id: int) -> dict | None:
        """One evidence row by id (citation verification: the row is immutable)."""
        r = self._conn.execute(
            "SELECT * FROM lesson_evidence WHERE lesson_id = ?", (lesson_id,)).fetchone()
        return dict(r) if r else None

    # ── Hypothesis ledger (Research Harness, R4) ──────────────
    # Append-only event sourcing: every helper only INSERTs, so a cited event_id is forever stable.
    # The chain is enforced here (prediction-first; adjudication anchored to a real execution).

    def register_prediction(self, *, hypothesis_id: str, run_id: str, statement: str,
                            prediction: dict, source: str = "") -> int:
        """Append a ``prediction_registered`` event (prediction FIRST). Exactly ONE prediction per
        ``hypothesis_id`` — a new statement/prediction must use a new id, so the aggregate view can
        never combine a later prediction with an earlier verdict. Returns its event_id."""
        if self._has_event(hypothesis_id, "prediction_registered"):
            raise ValueError(
                f"hypothesis {hypothesis_id!r} already has a prediction; use a new hypothesis_id")
        cur = self._conn.execute(
            "INSERT INTO hypothesis_events (hypothesis_id, event_type, run_id, payload, source) "
            "VALUES (?, 'prediction_registered', ?, ?, ?)",
            (hypothesis_id, run_id, json.dumps({"statement": statement, "prediction": prediction}),
             source))
        self._conn.commit()
        return int(cur.lastrowid)

    def record_execution(self, *, hypothesis_id: str, experiment_ref: dict,
                         run_id: str = "") -> int:
        """Append an ``experiment_executed`` event. REFUSES unless a ``prediction_registered``
        event exists for this hypothesis (prediction-first is structural). Returns the id."""
        if not self._has_event(hypothesis_id, "prediction_registered"):
            raise ValueError(
                f"cannot record execution for {hypothesis_id!r}: no prediction_registered event "
                "(prediction must precede experiment)")
        cur = self._conn.execute(
            "INSERT INTO hypothesis_events (hypothesis_id, event_type, run_id, payload) "
            "VALUES (?, 'experiment_executed', ?, ?)",
            (hypothesis_id, run_id, json.dumps({"experiment_ref": experiment_ref})))
        self._conn.commit()
        return int(cur.lastrowid)

    def record_adjudication(self, *, hypothesis_id: str, experiment_event_id: int, verdict: str,
                            measured: dict) -> int:
        """Append an ``adjudication_recorded`` event whose ``ref_event_id`` pins the EXACT execution
        event it judged. REFUSES unless ``verdict`` is one of supported/falsified/inconclusive AND
        the referenced event exists and is an ``experiment_executed`` for this hypothesis. Returns
        its event_id."""
        if verdict not in ("supported", "falsified", "inconclusive"):
            raise ValueError(
                f"verdict must be supported/falsified/inconclusive, got {verdict!r}")
        ev = self._conn.execute(
            "SELECT hypothesis_id, event_type FROM hypothesis_events WHERE event_id = ?",
            (experiment_event_id,)).fetchone()
        if ev is None or ev["event_type"] != "experiment_executed" \
                or ev["hypothesis_id"] != hypothesis_id:
            raise ValueError(
                f"cannot adjudicate: event {experiment_event_id} is not an experiment_executed "
                f"event for hypothesis {hypothesis_id!r}")
        cur = self._conn.execute(
            "INSERT INTO hypothesis_events (hypothesis_id, event_type, payload, ref_event_id) "
            "VALUES (?, 'adjudication_recorded', ?, ?)",
            (hypothesis_id, json.dumps({"verdict": verdict, "measured": measured}),
             experiment_event_id))
        self._conn.commit()
        return int(cur.lastrowid)

    def get_hypothesis_event(self, event_id: int) -> dict | None:
        """One ledger event by id (citation verification: the row is immutable)."""
        r = self._conn.execute(
            "SELECT * FROM hypothesis_events WHERE event_id = ?", (event_id,)).fetchone()
        return dict(r) if r else None

    def get_hypothesis(self, hypothesis_id: str) -> dict | None:
        """Deterministic aggregate VIEW of a hypothesis from its append-only events: the latest
        prediction, all executions, and the latest verdict. None if the hypothesis is unknown."""
        rows = [dict(r) for r in self._conn.execute(
            "SELECT * FROM hypothesis_events WHERE hypothesis_id = ? ORDER BY event_id",
            (hypothesis_id,)).fetchall()]
        if not rows:
            return None
        return self._aggregate_hypothesis(hypothesis_id, rows)

    def list_hypotheses(self, *, verdict: str | None = None, limit: int = 50) -> list[dict]:
        """Current aggregate view per hypothesis (newest first), optionally filtered by latest
        verdict — research consumes supported / falsified / inconclusive alike (falsified is
        first-class memory)."""
        ids = [r["hypothesis_id"] for r in self._conn.execute(
            "SELECT hypothesis_id, MAX(event_id) AS mx FROM hypothesis_events "
            "GROUP BY hypothesis_id ORDER BY mx DESC LIMIT ?", (max(1, int(limit)),)).fetchall()]
        views = [self.get_hypothesis(h) for h in ids]
        out = [v for v in views if v and (verdict is None or v.get("verdict") == verdict)]
        return out

    def _has_event(self, hypothesis_id: str, event_type: str) -> bool:
        r = self._conn.execute(
            "SELECT 1 FROM hypothesis_events WHERE hypothesis_id = ? AND event_type = ? LIMIT 1",
            (hypothesis_id, event_type)).fetchone()
        return r is not None

    @staticmethod
    def _aggregate_hypothesis(hypothesis_id: str, rows: list) -> dict:
        pred = next((r for r in reversed(rows) if r["event_type"] == "prediction_registered"), None)
        adj = next((r for r in reversed(rows) if r["event_type"] == "adjudication_recorded"), None)
        execs = [r for r in rows if r["event_type"] == "experiment_executed"]
        pred_payload = json.loads(pred["payload"]) if pred else {}
        adj_payload = json.loads(adj["payload"]) if adj else {}
        return {
            "hypothesis_id": hypothesis_id,
            "statement": pred_payload.get("statement", ""),
            "prediction": pred_payload.get("prediction"),
            "prediction_event_id": pred["event_id"] if pred else None,
            "execution_event_ids": [e["event_id"] for e in execs],
            "verdict": adj_payload.get("verdict") if adj else "unadjudicated",
            "measured": adj_payload.get("measured"),
            "adjudication_event_id": adj["event_id"] if adj else None,
            # the verdict is bound to the EXACT execution it judged (chain coherence)
            "adjudicated_execution_event_id": adj["ref_event_id"] if adj else None,
            "event_ids": [r["event_id"] for r in rows],
        }

    # ── Queries ───────────────────────────────────────────────

    def best_policies(self, target_name: str, n: int = 5) -> list[dict]:
        """Top-N policies by avg fitness."""
        rows = self._conn.execute(
            "SELECT * FROM policy_ranking WHERE target_name = ? "
            "ORDER BY avg_fitness DESC LIMIT ?",
            (target_name, n),
        ).fetchall()
        return [dict(r) for r in rows]

    def best_configs(
        self, model_name: str, hardware: str, n: int = 5
    ) -> list[dict]:
        """Top-N configs by highest-tier available fitness."""
        rows = self._conn.execute(
            "SELECT * FROM config_ranking "
            "WHERE model_name = ? AND hardware = ? "
            "ORDER BY COALESCE(l2_fitness, l0_fitness) DESC LIMIT ?",
            (model_name, hardware, n),
        ).fetchall()
        return [dict(r) for r in rows]

    def history(self, item_type: str, item_id: str) -> list[dict]:
        """All evaluations for an item, ordered by time."""
        rows = self._conn.execute(
            "SELECT * FROM evaluations WHERE item_type = ? AND item_id = ? "
            "ORDER BY created_at",
            (item_type, item_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def lineage_chain(self, policy_id: str) -> list[dict]:
        """Walk the lineage chain from a policy back to seed."""
        chain = []
        current = policy_id
        seen = set()
        while current and current not in seen:
            seen.add(current)
            row = self._conn.execute(
                "SELECT l.*, p.source_code FROM lineage l "
                "JOIN policies p ON l.policy_id = p.policy_id "
                "WHERE l.policy_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            chain.append(dict(row))
            current = row["parent_id"]
        return chain

    def compare(self, item_type: str, id1: str, id2: str) -> dict:
        """Compare two items: show metrics side by side."""
        evals1 = self.history(item_type, id1)
        evals2 = self.history(item_type, id2)
        return {"id1": id1, "evals1": evals1, "id2": id2, "evals2": evals2}

    def list_all(
        self, item_type: str, target_or_model: str | None = None, limit: int = 50
    ) -> list[dict]:
        """List items with their best fitness."""
        if item_type == "policy":
            query = "SELECT * FROM policy_ranking"
            params: list = []
            if target_or_model:
                query += " WHERE target_name = ?"
                params.append(target_or_model)
            query += " ORDER BY avg_fitness DESC LIMIT ?"
            params.append(limit)
        else:
            query = "SELECT * FROM config_ranking"
            params = []
            if target_or_model:
                query += " WHERE model_name = ?"
                params.append(target_or_model)
            query += " ORDER BY COALESCE(l2_fitness, l0_fitness) DESC LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    def stats(self) -> dict[str, int]:
        """Quick stats."""
        counts = {}
        for table in ("policies", "configs", "evaluations", "checks"):
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0]
        return counts

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute raw SQL query and return list of dicts."""
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
