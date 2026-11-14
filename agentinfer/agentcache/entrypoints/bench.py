# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

"""Delegate explicit AgentInfer benchmark invocations to BenchKit."""

import sys

_BENCHKIT_COMMANDS = frozenset({"compare", "compare-correctness", "evaluate", "prepare", "replay", "run", "summarize"})


def _program_name(arg: str) -> str:
    """Return the console-script name ``arg`` refers to, without a launcher suffix.

    ``pip`` writes console scripts as ``<name>.exe`` on Windows, so comparing ``argv[0]`` to the bare script name
    would miss the entry point on that platform. Both separators are handled so the result does not depend on the
    platform the check happens to run on.
    """

    name = arg.replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".exe") else name


def _strip_program_name(argv: list[str]) -> list[str]:
    """Remove a supported console-script name from an argument vector."""

    if argv and _program_name(argv[0]) in {"agentinfer-bench", "vllm"}:
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


def _normalize_direct_argv(argv: list[str]) -> tuple[list[str], dict[str, object]]:
    """Normalize a direct ``agentinfer-bench`` invocation for BenchKit."""

    normalized = _strip_program_name(list(argv))
    # ``agentinfer-bench`` names no upstream command, so every argument belongs to BenchKit; its parser owns
    # ``--help``, the required subcommand, and invalid-command errors.
    return normalized, {
        "entrypoint": "agentinfer-bench",
        "argv": ["agentinfer-bench", *normalized],
    }


def main(argv: list[str] | None = None) -> int:
    """Dispatch an explicit AgentInfer benchmark invocation."""

    raw_argv = list(sys.argv if argv is None else argv)
    direct = bool(raw_argv) and _program_name(raw_argv[0]) == "agentinfer-bench"
    if direct:
        normalized, metadata = _normalize_direct_argv(raw_argv)
    else:
        normalized, metadata = _normalize_delegated_argv(raw_argv)
    if metadata is None:
        raise ValueError("expected an explicit AgentInfer benchmark invocation")
    from agentinfer.agentbench.benchkit.cli import main as benchkit_main

    return benchkit_main(normalized, cli_metadata=metadata)


if __name__ == "__main__":
    raise SystemExit(main())
