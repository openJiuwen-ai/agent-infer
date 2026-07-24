"""Delegate commands to the upstream vLLM CLI."""

import sys


def main(argv: list[str] | None = None) -> int:
    """Delegate unchanged to upstream vLLM."""

    raw_argv = list(sys.argv if argv is None else argv)
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
