"""
E2E verification script for AgentScheduler custom scheduler.

Usage on GPU host:
    python tests/verify_remote.py [--model MODEL] [--timeout SECONDS] [--port PORT]

This script:
1. Runs the unit tests for scheduler resolution
2. Starts vllm-acache serve with a small model in a background process
3. Verifies AgentScheduler is loaded by checking the server logs
4. Sends a test completion request to verify the server works
5. Cleans up
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def green(s):
    return f"\033[92m{s}\033[0m"


def red(s):
    return f"\033[91m{s}\033[0m"


def run_unit_tests():
    print("=" * 60)
    print("Stage 1: Unit tests for AgentScheduler resolution")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(ROOT, "tests", "test_scheduler.py"), "-v"],
        capture_output=False,
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(red("UNIT TESTS FAILED"))
        return False
    print(green("Unit tests passed"))
    return True


def start_server(model, port, timeout):
    print("=" * 60)
    print(f"Stage 2: Starting vllm-acache serve with model={model}")
    print("=" * 60)

    log_file = os.path.join(ROOT, "verify_server.log")
    env = os.environ.copy()
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--scheduler-cls", "agentcache.core.scheduler.AgentScheduler",
        ],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd=ROOT,
        start_new_session=True,
    )

    # Wait for server to be ready.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(red(f"Server exited early with code {proc.returncode}"))
            print("Last 30 lines of log:")
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print("  " + line.rstrip())
            return None, None

        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            urllib.request.urlopen(req, timeout=5)
            print(green("Server is ready"))
            break
        except Exception:
            time.sleep(2)
    else:
        print(red("Server did not become ready in time"))
        proc.terminate()
        return None, None

    return proc, log_file


def verify_scheduler_in_logs(log_file):
    print("=" * 60)
    print("Stage 3: Verifying AgentScheduler in server logs")
    print("=" * 60)

    with open(log_file) as f:
        content = f.read()

    # Check for AgentScheduler reference.
    if "AgentScheduler" in content:
        print(green("AgentScheduler found in server logs"))
        return True
    elif "Using custom scheduler class" in content:
        print(green("Custom scheduler class usage confirmed in logs"))
        return True
    else:
        print(red("AgentScheduler NOT found in server logs"))
        print("Log excerpt (last 2KB):")
        print(content[-2048:])
        return False


def send_test_request(port, timeout):
    print("=" * 60)
    print("Stage 4: Sending test completion request")
    print("=" * 60)

    url = f"http://localhost:{port}/v1/completions"
    payload = json.dumps({
        "model": "default",
        "prompt": "Hello, my name is",
        "max_tokens": 16,
    }).encode()

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            body = json.loads(resp.read().decode())
            if "choices" in body and len(body["choices"]) > 0:
                text = body["choices"][0].get("text", "")
                print(green(f"Completion received: {text!r}"))
                return True
        except Exception as e:
            print(f"Request failed, retrying: {e}")
            time.sleep(3)

    print(red("Test request did not succeed"))
    return False


def main():
    parser = argparse.ArgumentParser(description="E2E verification of AgentScheduler")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model to test with")
    parser.add_argument("--port", type=int, default=18888, help="Server port")
    parser.add_argument("--timeout", type=int, default=300, help="Startup timeout (seconds)")
    parser.add_argument("--skip-serve", action="store_true", help="Skip serve test (unit tests only)")
    args = parser.parse_args()

    sys.path.insert(0, ROOT)

    print(f"AgentCache E2E Scheduler Verification")
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"Working dir: {ROOT}")
    print()

    # Stage 1: Unit tests
    if not run_unit_tests():
        sys.exit(1)

    if args.skip_serve:
        print(green("All checks passed (serve test skipped)"))
        sys.exit(0)

    # Stage 2: Start server
    proc, log_file = start_server(args.model, args.port, args.timeout)
    if proc is None:
        sys.exit(1)

    try:
        # Stage 3: Verify AgentScheduler loaded
        if not verify_scheduler_in_logs(log_file):
            print(red("VERIFICATION FAILED: AgentScheduler not detected"))
            sys.exit(1)

        # Stage 4: Send test request
        if not send_test_request(args.port, args.timeout):
            print(red("VERIFICATION FAILED: Test request failed"))
            sys.exit(1)

        print()
        print(green("=" * 60))
        print(green("ALL E2E VERIFICATION CHECKS PASSED"))
        print(green("=" * 60))
        print()
        print("Summary:")
        print("  - AgentScheduler resolves correctly")
        print("  - vllm-acache serve starts with AgentScheduler")
        print("  - Server responds to completion requests")
    finally:
        print("Shutting down server...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        print("Server stopped")


if __name__ == "__main__":
    main()
