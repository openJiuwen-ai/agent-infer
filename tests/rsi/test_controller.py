# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agentinfer.rsi.controller import Controller
from agentinfer.rsi.storage import ConflictError


def send(control, action, **payload):
    revision = control.get("run")["revision"]
    return control.command(
        "run", expected_revision=revision, idempotency_key=f"cmd-{revision}", action=action, payload=payload
    )


def at_evaluation(control):
    for _ in range(3):
        send(control, "advance")
    send(control, "start_trial", candidate_id="candidate-1")
    send(control, "advance")
    return send(control, "advance")


class TestController(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.control = Controller(Path(temporary.name) / "rsi.sqlite")
        self.control.create("run", demo=True)

    def test_idempotency_and_revision_survive_reopen(self):
        control = self.control
        request = {"expected_revision": 0, "idempotency_key": "same", "action": "advance", "payload": {}}
        first = control.command("run", **request)
        reopened = Controller(control.db.path)
        self.assertEqual(reopened.command("run", **request), first)
        self.assertEqual(len(reopened.events("run")), 2)
        with self.assertRaisesRegex(ConflictError, "different request"):
            reopened.command("run", **{**request, "payload": {"x": 1}})
        with self.assertRaisesRegex(ConflictError, "stale"):
            reopened.command("run", expected_revision=0, idempotency_key="other", action="pause")
        self.assertEqual(reopened.get("run")["revision"], 1)

    def test_concurrent_commands_have_one_revision_winner(self):
        control = self.control

        def attempt(key):
            try:
                return control.command("run", expected_revision=0, idempotency_key=key, action="advance")
            except ConflictError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, ["a", "b"]))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(control.get("run")["revision"], 1)

    def test_acceptance_activation_soak_and_next_round(self):
        control = self.control
        self.assertEqual(at_evaluation(control)["stage"], 7)
        state = send(control, "demo_evaluate", result="PASS")
        self.assertEqual((state["accepted"], state["active"], state["last_good"]), ("candidate-1", "v0.8.2", "v0.8.2"))
        state = send(control, "demo_activate")
        self.assertEqual((state["active"], state["last_good"], state["stage"]), ("candidate-1", "v0.8.2", 9))
        send(control, "pause")
        state = send(control, "demo_soak", result="PASS")
        self.assertEqual(state["last_good"], "candidate-1")
        with self.assertRaisesRegex(ValueError, "paused"):
            send(control, "next_round")
        send(control, "resume")
        state = send(control, "next_round")
        self.assertEqual((state["stage"], state["round"], state["baseline"]), (1, 2, "candidate-1"))
        self.assertEqual(state["trials_used"], 1)

    def test_rollback_failure_blocks_work_and_recovers_while_paused(self):
        control = self.control
        at_evaluation(control)
        send(control, "demo_evaluate", result="PASS")
        send(control, "demo_activate")
        send(control, "pause")
        send(control, "demo_soak", result="FAIL")
        state = send(control, "demo_rollback", healthy=False)
        self.assertEqual((state["status"], state["last_good"]), ("needs_recovery", "v0.8.2"))
        with self.assertRaisesRegex(ValueError, "recovery required"):
            send(control, "resume")
        state = send(control, "demo_rollback", healthy=True)
        self.assertEqual((state["active"], state["last_good"], state["stage"]), ("v0.8.2", "v0.8.2", 10))
        self.assertIsNone(state["accepted"])
        self.assertIn("candidate-1", state["excluded"])
        self.assertEqual(send(control, "complete")["status"], "completed")

    def test_unknown_activation_cannot_reactivate_before_recovery(self):
        control = self.control
        at_evaluation(control)
        send(control, "demo_evaluate", result="PASS")
        send(control, "demo_activate", outcome="unknown")
        with self.assertRaisesRegex(ValueError, "activation"):
            send(control, "demo_activate")
        self.assertEqual(send(control, "demo_rollback")["stage"], 10)

    def test_budget_retry_pause_and_exclude(self):
        control = self.control
        at_evaluation(control)
        send(control, "demo_evaluate", result="INCONCLUSIVE")
        send(control, "retry")
        send(control, "advance")
        send(control, "pause")
        with self.assertRaisesRegex(ValueError, "paused"):
            send(control, "start_trial")
        send(control, "resume")
        send(control, "start_trial", candidate_id="excluded")
        send(control, "exclude")
        send(control, "advance")
        with self.assertRaisesRegex(ValueError, "excluded"):
            send(control, "start_trial", candidate_id="excluded")
        send(control, "update_budget", max_trials=2)
        with self.assertRaisesRegex(ValueError, "budget"):
            send(control, "start_trial")

    def test_non_demo_cannot_self_certify_or_publish(self):
        control = self.control
        control.create("production")
        for revision, action in enumerate(("advance", "advance", "advance", "start_trial", "advance", "advance")):
            control.command("production", expected_revision=revision, idempotency_key=str(revision), action=action)
        with self.assertRaisesRegex(ValueError, "demo"):
            control.command(
                "production",
                expected_revision=6,
                idempotency_key="bad",
                action="demo_evaluate",
                payload={"result": "PASS"},
            )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            control.command("production", expected_revision=6, idempotency_key="bad", action="publish")
        state = control.get("production")
        self.assertIsNone(state["accepted"])
        self.assertEqual(state["active"], state["last_good"])

    def test_failed_last_trial_reaches_archive(self):
        control = self.control
        send(control, "update_budget", max_trials=1)
        at_evaluation(control)
        state = send(control, "demo_evaluate", result="FAIL")
        self.assertEqual(state["stage"], 10)
        self.assertIsNone(state["accepted"])
        self.assertEqual(send(control, "complete")["status"], "completed")

    def test_invalid_result_is_rejected_without_mutation(self):
        control = self.control
        state = at_evaluation(control)
        with self.assertRaisesRegex(ValueError, "invalid result"):
            send(control, "demo_evaluate", result={"unexpected": "object"})
        self.assertEqual(control.get("run"), state)
        snapshot = control.snapshot("run")
        self.assertEqual(snapshot["events"][-1]["revision"], snapshot["run"]["revision"])
        self.assertEqual(snapshot["event_cursor"], snapshot["events"][-1]["seq"])

    def test_excluding_last_trial_finishes_without_releasing_new_work(self):
        control = self.control
        send(control, "update_budget", max_trials=1)
        at_evaluation(control)
        self.assertEqual(send(control, "exclude")["stage"], 10)
        with self.assertRaisesRegex(ValueError, "budget"):
            send(control, "next_round")

    def test_deliberate_stop_before_trial_archives_run(self):
        state = send(self.control, "complete")
        self.assertEqual((state["stage"], state["status"], state["trials_used"]), (10, "completed", 0))
