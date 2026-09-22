# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exact-scope SQL retrieval; no implicit wildcard or self-certified validation."""

import json
from uuid import uuid4

from agentinfer.rsi.storage import ConflictError, Database, encode

from .seeds import NAMESPACES, demo_records


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value.strip() == "*":
        raise ValueError(f"{name} must be explicit, nonempty text (no wildcard)")


def _evidence(items):
    if not isinstance(items, list):
        raise ValueError("evidence must be a list of reference/outcome objects")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid evidence")
        _text(item.get("ref"), "evidence.ref")
        _text(item.get("outcome"), "evidence.outcome")


class KnowledgeRepository:
    def __init__(self, db_path):
        self.db = Database(db_path)

    @staticmethod
    def _load(db, knowledge_id):
        row = db.execute("SELECT record FROM rsi_knowledge WHERE id=?", (knowledge_id,)).fetchone()
        if row is None:
            raise KeyError(knowledge_id)
        return json.loads(row[0])

    @staticmethod
    def _save(db, record):
        args = tuple(
            record[key] for key in ("knowledge_id", "scope", "backend", "component_version", "workload", "status")
        )
        db.execute(
            "INSERT INTO rsi_knowledge VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "status=excluded.status, record=excluded.record",
            (*args, record["title"] + "\n" + record["claim"], encode(record)),
        )
        db.execute(
            "INSERT INTO rsi_knowledge_history(knowledge_id,record) VALUES (?,?)",
            (record["knowledge_id"], encode(record)),
        )

    def propose(self, record):
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        record = json.loads(encode(record))
        if record.get("status", "draft") != "draft":
            raise ValueError("proposals always start as draft; cannot self-validate")
        for field in ("namespace", "scope", "backend", "component_version", "workload", "title", "claim"):
            _text(record.get(field), field)
        if record["namespace"] not in NAMESPACES:
            raise ValueError("unsupported namespace")
        record.setdefault("knowledge_id", uuid4().hex)
        _text(record["knowledge_id"], "knowledge_id")
        record.setdefault("evidence", [])
        _evidence(record["evidence"])
        record.update(schema_version=1, revision=0, status="draft")
        with self.db.transaction() as db:
            if db.execute("SELECT 1 FROM rsi_knowledge WHERE id=?", (record["knowledge_id"],)).fetchone():
                raise ConflictError("knowledge_id already exists")
            self._save(db, record)
        return record

    def get(self, knowledge_id):
        with self.db.transaction() as db:
            return self._load(db, knowledge_id)

    def search(self, *, scope, backend, component_version, workload, text="", status=None):
        for name, value in (
            ("scope", scope),
            ("backend", backend),
            ("component_version", component_version),
            ("workload", workload),
        ):
            _text(value, name)
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if status is not None and (
            not isinstance(status, str) or status not in {"draft", "reviewed", "validated", "stale", "retracted"}
        ):
            raise ValueError("invalid status")
        sql = (
            "SELECT record FROM rsi_knowledge WHERE scope=? AND backend=? "
            "AND component_version=? AND workload=? AND instr(lower(search_text),lower(?))>0"
        )
        args = [scope, backend, component_version, workload, text]
        if status is None:
            sql += " AND status IN ('draft','reviewed','validated')"
        else:
            sql += " AND status=?"
            args.append(status)
        with self.db.transaction() as db:
            return [json.loads(row[0]) for row in db.execute(sql + " ORDER BY id", args)]

    def transition(self, knowledge_id, *, expected_revision, status, reason, evidence=None):
        _text(reason, "reason")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("invalid expected_revision")
        if status == "validated":
            raise ValueError("validated is unavailable until a trusted experimental evidence verifier is implemented")
        if not isinstance(status, str) or status not in {"reviewed", "stale", "retracted"}:
            raise ValueError("unsupported knowledge transition")
        evidence = [] if evidence is None else json.loads(encode(evidence))
        _evidence(evidence)
        if status == "reviewed" and not any(item["outcome"] == "source_reviewed" for item in evidence):
            raise ValueError(
                "reviewed requires an explicit source_reviewed evidence reference, not experimental validation"
            )
        with self.db.transaction() as db:
            record = self._load(db, knowledge_id)
            if record["revision"] != expected_revision:
                raise ConflictError("stale knowledge revision")
            if record["status"] == "retracted" or record["status"] == status:
                raise ValueError("transition is not allowed")
            record.update(status=status, revision=record["revision"] + 1, reason=reason)
            record["evidence"].extend(evidence)
            self._save(db, record)
        return record

    def history(self, knowledge_id):
        with self.db.transaction() as db:
            self._load(db, knowledge_id)
            rows = db.execute(
                "SELECT record FROM rsi_knowledge_history WHERE knowledge_id=? ORDER BY seq", (knowledge_id,)
            )
            return [json.loads(row[0]) for row in rows]

    def seed_demo(self):
        records = []
        for seed in demo_records():
            try:
                records.append(self.propose(seed))
            except ConflictError:
                records.append(self.get(seed["knowledge_id"]))
        return records
