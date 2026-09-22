# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""A ten-stage controller with explicitly simulated deployment transitions."""

import json

from agentinfer.rsi.storage import ConflictError, Database, encode


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class Controller:
    """All methods are local calls; callers must provide their own authentication.

    No trusted production evaluation or deployment adapter exists in this MVP.
    Only demo_* commands can create release qualifications or deployment facts.
    """

    def __init__(self, db_path):
        self.db = Database(db_path)

    def create(self, run_id, *, baseline="v0.8.2", max_trials=3, demo=False, backend="cuda"):
        _require(isinstance(run_id, str) and bool(run_id.strip()), "run_id is required")
        _require(isinstance(baseline, str) and bool(baseline.strip()), "baseline is required")
        _require(isinstance(backend, str) and backend in {"cuda", "ascend"}, "backend must be cuda or ascend")
        _require(type(demo) is bool, "demo must be a boolean")
        _positive_int(max_trials, "max_trials")
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "revision": 0,
            "stage": 1,
            "status": "running",
            "paused": False,
            "demo": demo,
            "backend": backend,
            "baseline": baseline,
            "candidate": None,
            "accepted": None,
            "active": baseline,
            "last_good": baseline,
            "max_trials": max_trials,
            "trials_used": 0,
            "round": 1,
            "evaluation": None,
            "recovery_needed": False,
            "excluded": [],
            "deployment_mode": "simulated" if demo else "unavailable",
        }
        with self.db.transaction() as db:
            if db.execute("SELECT 1 FROM rsi_runs WHERE run_id=?", (run_id,)).fetchone():
                raise ConflictError("run_id already exists")
            db.execute("INSERT INTO rsi_runs VALUES (?, ?)", (run_id, encode(state)))
            self._event(db, run_id, "created", state)
        return state

    @staticmethod
    def _load(db, run_id):
        row = db.execute("SELECT state FROM rsi_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row["state"])

    @staticmethod
    def _event(db, run_id, action, state):
        event = {"action": action, "revision": state["revision"], "state": state}
        db.execute("INSERT INTO rsi_events(run_id,event) VALUES (?,?)", (run_id, encode(event)))

    def get(self, run_id):
        with self.db.transaction() as db:
            return self._load(db, run_id)

    def list_runs(self):
        with self.db.transaction() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT state FROM rsi_runs ORDER BY run_id")]

    def events(self, run_id):
        with self.db.transaction() as db:
            self._load(db, run_id)
            return self._events(db, run_id)

    @staticmethod
    def _events(db, run_id):
        rows = db.execute("SELECT seq,event FROM rsi_events WHERE run_id=? ORDER BY seq", (run_id,))
        return [{"seq": row["seq"], **json.loads(row["event"])} for row in rows]

    def snapshot(self, run_id):
        """Read state and its event cursor under one transaction."""
        with self.db.transaction() as db:
            state = self._load(db, run_id)
            events = self._events(db, run_id)
            return {"run": state, "events": events, "event_cursor": events[-1]["seq"]}

    def command(self, run_id, *, expected_revision, idempotency_key, action, payload=None):
        _require(type(expected_revision) is int and expected_revision >= 0, "invalid expected_revision")
        _require(isinstance(idempotency_key, str) and bool(idempotency_key.strip()), "idempotency_key is required")
        _require(isinstance(action, str), "action must be a string")
        payload = {} if payload is None else payload
        _require(isinstance(payload, dict), "payload must be an object")
        request = encode({"action": action, "payload": payload, "expected_revision": expected_revision})
        with self.db.transaction() as db:
            saved = db.execute(
                "SELECT request,response FROM rsi_commands WHERE run_id=? AND key=?", (run_id, idempotency_key)
            ).fetchone()
            if saved:
                if saved["request"] != request:
                    raise ConflictError("idempotency key already used for a different request")
                return json.loads(saved["response"])
            state = self._load(db, run_id)
            if state["revision"] != expected_revision:
                raise ConflictError("stale expected_revision")
            self._apply(state, action, payload)
            state["revision"] += 1
            response = encode(state)
            db.execute("UPDATE rsi_runs SET state=? WHERE run_id=?", (response, run_id))
            db.execute("INSERT INTO rsi_commands VALUES (?,?,?,?)", (run_id, idempotency_key, request, response))
            self._event(db, run_id, action, state)
        return state

    @staticmethod
    def _apply(s, action, p):
        allowed = {
            "advance": set(),
            "pause": set(),
            "resume": set(),
            "retry": set(),
            "exclude": set(),
            "complete": set(),
            "next_round": set(),
            "start_trial": {"candidate_id"},
            "update_budget": {"max_trials"},
            "demo_evaluate": {"result"},
            "demo_activate": {"outcome"},
            "demo_soak": {"result"},
            "demo_rollback": {"healthy"},
        }
        _require(action in allowed, "unsupported action; production evaluation/deployment is unavailable")
        _require(not (p.keys() - allowed[action]), "unsupported payload fields")
        _require(s["status"] != "completed", "run is completed")
        if action.startswith("demo_"):
            _require(s["demo"], "simulated actions require an explicitly created demo run")
        if s["status"] == "needs_recovery":
            _require(action == "demo_rollback", "recovery required before further commands")
        if s["paused"]:
            _require(
                action in {"pause", "resume", "demo_evaluate", "demo_soak", "demo_rollback", "complete"},
                "run is paused; new work cannot start",
            )
        stage = s["stage"]
        if action == "pause":
            s["paused"] = True
        elif action == "resume":
            s["paused"] = False
        elif action == "update_budget":
            _positive_int(p.get("max_trials"), "max_trials")
            _require(p["max_trials"] >= s["trials_used"], "budget cannot erase spent trials")
            s["max_trials"] = p["max_trials"]
        elif action == "advance":
            _require(stage in {1, 2, 3, 5, 6}, "use the dedicated trial/evaluation/deployment action")
            s["stage"] += 1
        elif action == "start_trial":
            _require(stage == 4, "trial requires stage 4")
            _require(s["trials_used"] < s["max_trials"], "trial budget exhausted")
            candidate = p.get("candidate_id", f"{s['run_id']}-trial-{s['trials_used'] + 1}")
            _require(isinstance(candidate, str) and bool(candidate.strip()), "candidate_id is required")
            _require(candidate not in s["excluded"] and candidate != s["active"], "candidate is excluded or active")
            s.update(stage=5, candidate=candidate, accepted=None, evaluation=None)
            s["trials_used"] += 1
        elif action == "retry":
            _require(
                stage == 7 and s["evaluation"] in {"FAIL", "INCONCLUSIVE"},
                "retry requires a rejected/inconclusive trial",
            )
            _require(s["trials_used"] < s["max_trials"], "trial budget exhausted")
            s.update(stage=3, candidate=None, accepted=None, evaluation=None)
        elif action == "exclude":
            _require(stage in {5, 6, 7} and s["candidate"] is not None, "no excludable candidate")
            s["excluded"].append(s["candidate"])
            s.update(stage=3, candidate=None, accepted=None, evaluation=None)
            if s["trials_used"] >= s["max_trials"]:
                s["stage"] = 10
        elif action == "demo_evaluate":
            _require(stage == 7 and s["candidate"] is not None, "evaluation requires stage 7")
            result = p.get("result")
            _require(isinstance(result, str) and result in {"PASS", "FAIL", "INCONCLUSIVE"}, "invalid result")
            s["evaluation"] = result
            if result == "PASS":
                s.update(accepted=s["candidate"], stage=8)
            elif s["trials_used"] >= s["max_trials"]:
                s["stage"] = 10
        elif action == "demo_activate":
            _require(
                stage == 8 and s["accepted"] is not None and not s["recovery_needed"],
                "no accepted activation candidate",
            )
            outcome = p.get("outcome", "success")
            _require(
                isinstance(outcome, str) and outcome in {"success", "failed", "unknown"}, "invalid activation outcome"
            )
            if outcome == "success":
                s.update(active=s["accepted"], stage=9)
            else:
                s["recovery_needed"] = True
        elif action == "demo_soak":
            _require(stage == 9, "soak requires stage 9")
            result = p.get("result")
            _require(isinstance(result, str) and result in {"PASS", "FAIL"}, "invalid soak result")
            if result == "PASS":
                s.update(last_good=s["active"], stage=10)
            else:
                s.update(stage=8, recovery_needed=True)
        elif action == "demo_rollback":
            _require(stage in {8, 9}, "rollback requires activation/observation stage")
            healthy = p.get("healthy", True)
            _require(type(healthy) is bool, "healthy must be a boolean")
            if healthy:
                if s["candidate"] not in s["excluded"]:
                    s["excluded"].append(s["candidate"])
                s.update(active=s["last_good"], accepted=None, stage=10, recovery_needed=False, status="running")
            else:
                s.update(recovery_needed=True, status="needs_recovery")
        elif action == "next_round":
            _require(stage == 10 and not s["recovery_needed"], "round has not safely finished")
            _require(s["trials_used"] < s["max_trials"], "trial budget exhausted")
            s.update(stage=1, baseline=s["last_good"], candidate=None, accepted=None, evaluation=None)
            s["round"] += 1
        elif action == "complete":
            _require(stage in {1, 2, 3, 4, 10}, "cannot complete while a trial or release is pending")
            s.update(stage=10, status="completed")
