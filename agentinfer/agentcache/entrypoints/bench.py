# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

"""Delegate explicit AgentInfer benchmark invocations to BenchKit."""

import sys
from pathlib import Path

_BENCHKIT_COMMANDS = frozenset({"compare", "prepare", "run", "summarize"})


def _strip_program_name(argv: list[str]) -> list[str]:
    """Remove a supported console-script name from an argument vector."""

    if argv and Path(argv[0]).name in {"agentinfer-bench", "vllm"}:
        return argv[1:]
    return argv


def _normalize_delegated_argv(argv: list[str]) -> tuple[list[str], dict[str, object] | None]:
    """Normalize an explicit vLLM benchmark invocation for BenchKit."""

    normalized = _strip_program_name(list(argv))
    if len(normalized) >= 2 and normalized[:2] == ["bench", "serve"]:
        rest = normalized[2:]
        if "--agentinfer" not in rest:
            return normalized, None
        rest.remove("--agentinfer")
        if not rest or rest[0] not in _BENCHKIT_COMMANDS:
            rest.insert(0, "run")
        return rest, {
            "entrypoint": "vllm bench serve --agentinfer",
            "argv": ["bench", "serve", "--agentinfer", *rest],
        }
    return normalized, None


def main(argv: list[str] | None = None) -> int:
    """Dispatch an explicit AgentInfer benchmark invocation."""

    normalized, metadata = _normalize_delegated_argv(list(sys.argv if argv is None else argv))
    if metadata is None:
        raise ValueError("expected an explicit AgentInfer benchmark invocation")
    from agentinfer.agentbench.benchkit.cli import main as benchkit_main

    return benchkit_main(normalized, cli_metadata=metadata)


if __name__ == "__main__":
    raise SystemExit(main())
