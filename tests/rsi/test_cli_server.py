# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""CPU-only integration checks for the CLI and loopback demo boundary."""

import contextlib
import http.client
import io
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from agentinfer.rsi.cli import main
from agentinfer.rsi.controller import Controller
from agentinfer.rsi.dashboard.server import make_handler

ROOT = Path(__file__).resolve().parents[2]


class CliTests(unittest.TestCase):
    def call(self, args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = main(args)
        return code, output.getvalue(), error.getvalue()

    def test_demo_reuses_state_and_preserves_intervention(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["demo", "--run-dir", directory]
            code, output, _ = self.call(args)
            self.assertEqual(code, 0)
            run = json.loads(output)["run"]
            self.assertEqual(run["stage"], 7)
            controller = Controller(Path(directory) / "state.sqlite")
            paused = controller.command(
                run["run_id"], expected_revision=run["revision"], idempotency_key="test-pause", action="pause"
            )
            code, output, _ = self.call(args)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["run"], paused)

    def test_sample_cannot_establish_acceptance(self):
        code, output, _ = self.call(["evaluate", str(ROOT / "examples/rsi/feedback.json")])
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["status"], "INCONCLUSIVE")

    def test_array_manifest_is_a_controlled_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.json"
            path.write_text("[]", encoding="utf-8")
            code, _, error = self.call(["evaluate", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("manifest", error)

    def test_record_synthetic_provenance_is_preserved(self):
        data = json.loads((ROOT / "examples/rsi/feedback.json").read_text(encoding="utf-8"))
        data["synthetic"] = False
        for record in data["records"]:
            record["synthetic"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record-demo.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            code, output, _ = self.call(["evaluate", str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(output)["synthetic"])


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.directory.name) / "state.sqlite"
        self.controller = Controller(self.db_path)
        self.controller.create("test-run", demo=True)
        handler = make_handler(self.db_path, "test-run")

        class QuietHandler(handler):
            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.directory.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8")
        finally:
            connection.close()

    def test_snapshot_and_packaged_dashboard(self):
        status, body = self.request("GET", "/api/rsi/snapshot")
        self.assertEqual(status, 200)
        snapshot = json.loads(body)
        self.assertTrue(snapshot["synthetic"])
        self.assertFalse(snapshot["production_connected"])
        self.assertEqual(snapshot["run"]["revision"], snapshot["events"][-1]["revision"])
        status, body = self.request("GET", "/dashboard.html")
        self.assertEqual(status, 200)
        self.assertIn("<!doctype html>", body.lower())

    def test_origin_and_host_rejection_do_not_mutate(self):
        command = json.dumps({"action": "pause", "expected_revision": 0, "idempotency_key": "blocked"})
        status, _ = self.request(
            "POST", "/api/rsi/commands", command, {"Content-Type": "application/json", "Origin": "https://example.org"}
        )
        self.assertEqual(status, 403)
        status, _ = self.request("GET", "/api/rsi/snapshot", headers={"Host": "example.org"})
        self.assertEqual(status, 403)
        self.assertEqual(self.controller.get("test-run")["revision"], 0)

    def test_command_replay_and_revision_conflict(self):
        command = {"action": "pause", "expected_revision": 0, "idempotency_key": "once"}
        headers = {"Content-Type": "application/json"}
        status, first = self.request("POST", "/api/rsi/commands", json.dumps(command), headers)
        self.assertEqual(status, 200)
        status, replay = self.request("POST", "/api/rsi/commands", json.dumps(command), headers)
        self.assertEqual(status, 200)
        self.assertEqual(first, replay)
        command["idempotency_key"] = "stale"
        status, _ = self.request("POST", "/api/rsi/commands", json.dumps(command), headers)
        self.assertEqual(status, 409)
        self.assertEqual(self.controller.get("test-run")["revision"], 1)

    def test_invalid_content_type_and_non_demo_mutation(self):
        status, _ = self.request("POST", "/api/rsi/commands", "{}", {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.assertEqual(self.controller.get("test-run")["revision"], 0)
        self.controller.create("production", demo=False)
        self.server.RequestHandlerClass = make_handler(self.db_path, "production")
        status, _ = self.request(
            "POST",
            "/api/rsi/commands",
            json.dumps({"action": "pause", "expected_revision": 0, "idempotency_key": "no"}),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(self.controller.get("production")["revision"], 0)


if __name__ == "__main__":
    unittest.main()
