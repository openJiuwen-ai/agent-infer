"""Delegate vLLM commands while intercepting explicit AgentInfer benchmarks."""

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


def main(argv: list[str] | None = None) -> int:
    """Route AgentInfer benchmarks or delegate unchanged to upstream vLLM."""

    raw_argv = list(sys.argv if argv is None else argv)
    if _is_bench_delegation(raw_argv):
        from agentinfer.agentcache.entrypoints.bench import main as benchmark_main

        return benchmark_main(raw_argv)
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
