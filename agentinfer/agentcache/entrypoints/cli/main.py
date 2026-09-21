# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

"""Delegate vLLM commands while intercepting AgentInfer benchmarks and serve takeover."""

import sys
from pathlib import Path


def _is_bench_delegation(argv: list[str]) -> bool:
    """Return whether argv requests the explicit AgentInfer benchmark path."""

    offset = 1 if argv and Path(argv[0]).name == "vllm" else 0
    return (
        len(argv) >= offset + 3
        and argv[offset : offset + 2] == ["bench", "serve"]
        and "--agentinfer" in argv[offset + 2 :]
    )


def _serve_takeover_args(argv: list[str]) -> list[str] | None:
    """Return the serve argument list when argv requests the AgentInfer serve takeover."""

    offset = 1 if argv and Path(argv[0]).name == "vllm" else 0
    rest = argv[offset:]
    if not rest or rest[0] != "serve":
        return None
    serve_args = rest[1:]
    if any(arg == "--agentinfer" or arg.startswith("--agentinfer=") for arg in serve_args):
        return serve_args
    return None


def main(argv: list[str] | None = None) -> int:
    """Route AgentInfer benchmarks, serve takeover, or delegation to upstream vLLM."""

    raw_argv = list(sys.argv if argv is None else argv)
    if _is_bench_delegation(raw_argv):
        from agentinfer.agentcache.entrypoints.bench import main as benchmark_main

        return benchmark_main(raw_argv)
    serve_args = _serve_takeover_args(raw_argv)
    if serve_args is not None:
        from agentinfer.agentcache.entrypoints.cli import serve_profile

        try:
            return serve_profile.run_agentinfer_serve(serve_args)
        except serve_profile.AgentInferServeError as exc:
            print(f"[agentinfer] error: {exc}", file=sys.stderr)
            return 2
    return _delegate_vllm(raw_argv)


def _delegate_vllm(raw_argv: list[str]) -> int:
    """Invoke upstream vLLM while preserving the caller's sys.argv object."""

    try:
        from vllm.entrypoints.cli.main import main as vllm_main
    except ImportError as exc:
        raise RuntimeError("vLLM CLI is unavailable; install the pinned vLLM dependency") from exc

    original = sys.argv
    try:
        sys.argv = raw_argv
        result = vllm_main()
        return 0 if result is None else int(result)
    finally:
        sys.argv = original
