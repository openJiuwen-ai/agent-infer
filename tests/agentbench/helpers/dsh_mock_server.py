# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Script model responses while real DSH owns tools, sessions and workspaces."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def write_chat_completion_stream(handler: BaseHTTPRequestHandler, chunks: list[dict]) -> None:
    """Write scripted OpenAI SSE chunks followed by the terminal marker."""
    payload = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class DshMockServer(ThreadingHTTPServer):
    def __init__(self, profile: str, *, concurrent_tasks: int = 1):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.profile = profile
        self.requests = []
        self.states = {}
        self.probed_roots = set()
        self.barrier = threading.Barrier(concurrent_tasks)
        self.hold_after_patch = False
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.shutdown()
        self.thread.join(5)
        self.server_close()

    def respond(self, body):
        identity = json.loads(body["vllm_xargs"]["agentic_context"])
        key = (identity["task_id"], identity["program_id"])
        stage = self.states.get(key, 0)
        self.requests.append({"identity": identity, "body": body, "time": time.monotonic()})
        if stage == 0 and identity["agent_role"] == "lead":
            self.barrier.wait(timeout=30)
            if key not in self.probed_roots:
                self.probed_roots.add(key)
                if self.profile == "plan-subagent":
                    return "bash", {
                        "command": "touch forbidden-plan-write",
                        "description": "Attempt a write before planning",
                    }
                return "subagent", {
                    "description": "Attempt disabled delegation",
                    "prompt": "Do not execute this disabled tool.",
                }
        if identity["agent_role"] == "subagent":
            names = {tool["function"]["name"] for tool in body["tools"]}
            assert names <= {"glob", "grep", "read"}, names
            if stage == 0:
                self.states[key] = 1
                return "bash", {
                    "command": "touch forbidden-child-write",
                    "description": "Attempt a forbidden child write",
                }
            if stage == 1:
                self.states[key] = 2
                return "read", {"file_path": "calc.py"}
            return None, "EXPLORE_REPORT: calc.py add subtracts; change it to addition."
        if self.profile == "plan-subagent":
            if stage == 0:
                self.states[key] = 1
                return "subagent", {
                    "description": "Inspect the addition bug",
                    "prompt": "Read calc.py and explain the addition bug. Do not modify anything.",
                }
            if stage == 1:
                if "EXPLORE_REPORT" not in json.dumps(body["messages"]):
                    return None, "Waiting for the exploration report."
                self.states[key] = 2
                return "exit_plan_mode", {
                    "plan": "# Fix addition\nUse the exploration report, change subtraction to addition, and run the test."
                }
            stage -= 2
        else:
            names = {tool["function"]["name"] for tool in body["tools"]}
            assert not names & {"subagent", "subagent_fork", "send_message", "jobs"}, names
        if stage == 0:
            self.states[key] = 3 if self.profile == "plan-subagent" else 1
            return "bash", {
                "command": "printf 'def add(a, b):\n    return a + b\n' > calc.py\npython3 -m unittest -v",
                "description": "Fix addition and run tests",
            }
        if self.hold_after_patch:
            self.blocked.set()
            self.release.wait(timeout=30)
        return None, "Fixed addition; tests pass."


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            payload = json.dumps({"object": "list", "data": [{"id": "dsh-test", "max_model_len": 32768}]}).encode()
            mime = "application/json"
        else:
            payload = b"# Mock metrics; no performance claim.\n"
            mime = "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        try:
            tool, value = self.server.respond(body)
        except Exception as exc:
            self.send_error(500, str(exc))
            return
        if tool:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call-{len(self.server.requests)}",
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(value)},
                    }
                ]
            }
        else:
            delta = {"content": value}
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool else "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        ]
        write_chat_completion_stream(
            self,
            [
                {"id": "dsh-mock", "object": "chat.completion.chunk", "model": "dsh-test", "created": 0, **chunk}
                for chunk in chunks
            ],
        )

    def log_message(self, *_args):
        pass


def seed_tasks(root: Path, count: int = 4):
    """Make a local SWE-bench-shaped fixture with a genuinely failing baseline."""
    import subprocess

    repo = root / "cache" / "agentbench-fixture__tiny-calc"
    repo.mkdir(parents=True)
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / ".gitignore").write_text("__pycache__/\n")
    (repo / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\nclass TestAdd(unittest.TestCase):\n"
        "    def test_add(self): self.assertEqual(add(2, 3), 5)\n"
        "    def test_zero(self): self.assertEqual(add(0, 0), 0)\n"
    )
    baseline = subprocess.run(["python3", "-m", "unittest", "-v"], cwd=repo, capture_output=True, text=True)
    assert baseline.returncode != 0 and "FAIL: test_add" in baseline.stderr
    (root / "baseline.log").write_text(baseline.stdout + baseline.stderr)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", ".gitignore", "calc.py", "test_calc.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "baseline"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    rows = [
        {
            "instance_id": f"calc-{i}",
            "repo": "agentbench-fixture/tiny-calc",
            "base_commit": commit,
            "problem_statement": "Fix add(a, b) in calc.py: add(2, 3) should return 5. Run python3 -m unittest -v and leave the fix in the working tree. Keep the tests unchanged.",
        }
        for i in range(count)
    ]
    (root / "instances.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "selection.txt").write_text("".join(row["instance_id"] + "\n" for row in rows))
    return {
        "index_path": str(root / "instances.jsonl"),
        "selection_path": str(root / "selection.txt"),
        "cache_dir": str(root / "cache"),
    }
