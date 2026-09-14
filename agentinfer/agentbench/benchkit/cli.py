# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Command-line interface for preparing, running, summarizing, and comparing benchmarks."""

import argparse
import asyncio
import logging
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from types import UnionType

from pydantic import BaseModel

from ..replay.config import ReplayBenchConfig, load_replay_config
from .config import AgentBenchConfig, load_config

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure concise CLI logging when the application has no handlers."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass(frozen=True)
class _Override:
    """Describe one schema-derived command-line override."""

    flag: str  # e.g. "--task-num"
    dest: str  # e.g. "task_num" in argparse.Namespace
    section: str  # e.g. "experiment" in AgentBenchConfig
    field: str  # e.g. "task_num" in ExperimentConfig
    value_type: type  # e.g. int, bool, or Path
    choices: tuple[object, ...] | None  # Values inferred from Literal annotations
    path: bool  # Whether CLI values require cwd-relative path resolution
    help: str  # Description copied from the Pydantic field


def _resolve_type(annotation: object) -> tuple[type, tuple[object, ...] | None]:
    """Resolve a model annotation into an argparse type and choices."""

    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        choices = typing.get_args(annotation)
        return type(choices[0]), choices
    if origin in (typing.Union, UnionType, types.UnionType):
        members = tuple(member for member in typing.get_args(annotation) if member is not type(None))
        if len(members) == 1:
            return _resolve_type(members[0])
    if annotation in (bool, float, int, str, Path):
        return typing.cast(type, annotation), None
    raise ValueError(f"unsupported CLI override type: {annotation}")


def _discover_overrides(
    schema: type[BaseModel] = AgentBenchConfig,
) -> tuple[_Override, ...]:
    """Discover supported CLI overrides from the configuration schema."""

    overrides = []
    flags: set[str] = set()
    destinations: set[str] = set()
    for section_name, section_info in schema.model_fields.items():
        section_type = section_info.annotation
        if not isinstance(section_type, type) or not hasattr(section_type, "model_fields"):
            continue
        for field_name, field_info in section_type.model_fields.items():
            extra = field_info.json_schema_extra
            if not isinstance(extra, dict) or not extra.get("cli"):
                continue
            metadata = extra["cli"] if isinstance(extra["cli"], dict) else {}
            flag = metadata.get("flag", f"--{field_name.replace('_', '-')}")
            dest = metadata.get("dest", field_name)
            if flag in flags or dest in destinations:
                raise RuntimeError(f"duplicate CLI override: {flag} / {dest}")
            flags.add(flag)
            destinations.add(dest)
            value_type, choices = _resolve_type(field_info.annotation)
            overrides.append(
                _Override(
                    flag,
                    dest,
                    section_name,
                    field_name,
                    value_type,
                    choices,
                    value_type is Path,
                    field_info.description or "",
                )
            )
    return tuple(overrides)


_OVERRIDES = _discover_overrides()
_REPLAY_OVERRIDES = _discover_overrides(ReplayBenchConfig)


def _resolve_cli_path(path: Path, *, allow_bare: bool = False) -> Path:
    """Resolve a CLI path against the invocation directory."""

    expanded = path.expanduser()
    if expanded.is_absolute() or (allow_bare and len(expanded.parts) == 1):
        return expanded
    return expanded.resolve()


def _register_schema_args(
    parser: argparse.ArgumentParser,
    overrides: tuple[_Override, ...],
) -> None:
    """Register schema-derived overrides on one command parser."""

    for override in overrides:
        kwargs: dict[str, object] = {"default": None, "help": override.help}
        if override.value_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = override.value_type
        if override.choices:
            kwargs["choices"] = override.choices
        parser.add_argument(override.flag, dest=override.dest, **kwargs)


def _register_run_args(parser: argparse.ArgumentParser) -> None:
    """Register schema-derived overrides on the run parser."""

    _register_schema_args(parser, _OVERRIDES)


def _apply_schema_overrides(
    args: argparse.Namespace,
    config: BaseModel,
    schema: type[BaseModel],
    overrides: tuple[_Override, ...],
) -> BaseModel:
    """Apply explicit CLI values and revalidate one complete schema."""

    payload = config.model_dump(mode="python")
    for override in overrides:
        value = getattr(args, override.dest, None)
        if value is None:
            continue
        if override.path:
            value = _resolve_cli_path(
                value,
                allow_bare=override.section == "agent" and override.field == "executable",
            )
        payload[override.section][override.field] = value
    return schema.model_validate(payload)


def _apply_cli_overrides(args: argparse.Namespace, config: AgentBenchConfig) -> AgentBenchConfig:
    """Apply explicit CLI values and revalidate the complete configuration."""

    return typing.cast(
        AgentBenchConfig,
        _apply_schema_overrides(
            args,
            config,
            AgentBenchConfig,
            _OVERRIDES,
        ),
    )


def _apply_replay_cli_overrides(
    args: argparse.Namespace,
    config: ReplayBenchConfig,
) -> ReplayBenchConfig:
    """Apply Replay CLI values and revalidate the complete configuration."""

    return typing.cast(
        ReplayBenchConfig,
        _apply_schema_overrides(
            args,
            config,
            ReplayBenchConfig,
            _REPLAY_OVERRIDES,
        ),
    )


def _explicit_schema_overrides(
    args: argparse.Namespace,
    overrides: tuple[_Override, ...],
) -> dict[str, object]:
    """Serialize explicitly supplied overrides for run metadata."""

    values = {}
    for override in overrides:
        value = getattr(args, override.dest, None)
        if value is not None:
            values[override.dest] = str(value) if isinstance(value, Path) else value
    return values


def _parser() -> argparse.ArgumentParser:
    """Build the BenchKit command-line parser."""

    parser = argparse.ArgumentParser(description="AgentInfer benchmark harness")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Prepare benchmark dataset inputs")
    prepare.add_argument("dataset", choices=["swebench"])
    prepare.add_argument("--output-dir", type=Path, default=Path("data/swebench"))

    run = commands.add_parser("run", help="Run one benchmark")
    run.add_argument("--config", type=Path, default=None)
    _register_run_args(run)

    replay = commands.add_parser(
        "replay",
        help="Execute a deterministic Trace Replay workload",
    )
    replay.add_argument("--config", type=Path, required=True)
    _register_schema_args(replay, _REPLAY_OVERRIDES)

    summarize = commands.add_parser("summarize", help="Combine existing run summaries")
    summarize.add_argument("run_dirs", type=Path, nargs="+")
    summarize.add_argument("--output-csv", type=Path, default=Path("combined-summary.csv"))
    summarize.add_argument("--output-figs-dir", type=Path)

    compare = commands.add_parser("compare", help="Compare finalized benchmark runs")
    compare.add_argument("--baseline", type=Path, nargs="+", required=True)
    compare.add_argument("--candidate", type=Path, nargs="+", required=True)
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, cli_metadata: dict[str, object] | None = None) -> int:
    """Parse and dispatch one BenchKit command."""

    _configure_logging()
    args = _parser().parse_args(argv)
    metadata = dict(cli_metadata or {"entrypoint": "agentinfer-bench", "argv": argv or sys.argv[1:]})
    if args.command in {"replay", "run"}:
        if args.config is not None:
            metadata["config_path"] = str(args.config)
        metadata["overrides"] = _explicit_schema_overrides(
            args,
            _REPLAY_OVERRIDES if args.command == "replay" else _OVERRIDES,
        )
        if args.command == "replay":
            _replay(args, metadata)
        else:
            asyncio.run(_run(args, metadata))
    elif args.command == "summarize":
        _summarize(args)
    elif args.command == "compare":
        _compare(args)
    else:
        _prepare(args)
    return 0


async def _run(args: argparse.Namespace, cli_metadata: dict[str, object]) -> None:
    """Delegate benchmark execution to the Runner API."""

    from .runner import run_benchmark

    config = _apply_cli_overrides(args, load_config(args.config))
    run_dir = await run_benchmark(config, cli_metadata=cli_metadata)
    logger.info("Results: %s", run_dir)


def _replay(args: argparse.Namespace, cli_metadata: dict[str, object]) -> None:
    """Build and execute a Trace Replay workload."""

    from ..replay.runner import run_replay

    config = _apply_replay_cli_overrides(
        args,
        load_replay_config(args.config),
    )
    run_dir = run_replay(config, cli_metadata=cli_metadata)
    logger.info("Replay result: %s", run_dir)


def _summarize(args: argparse.Namespace) -> None:
    """Combine existing run summaries into one CSV export."""

    from .summarize import combine_summaries

    output = args.output_csv.resolve()
    figures_dir = args.output_figs_dir.resolve() if args.output_figs_dir is not None else None
    combine_summaries(
        [run_dir.resolve() for run_dir in args.run_dirs],
        output=output,
        figures_dir=figures_dir,
    )
    logger.info("Combined summaries written to %s", output)
    if figures_dir is not None:
        logger.info("Distribution figures written to %s", figures_dir)


def _compare(args: argparse.Namespace) -> None:
    """Delegate comparison and write its report to standard output."""

    from .compare import compare

    report = compare(
        [path.resolve() for path in args.baseline],
        [path.resolve() for path in args.candidate],
        as_json=args.json,
        confidence=args.confidence,
    )
    sys.stdout.write(report + "\n")


def _prepare(args: argparse.Namespace) -> None:
    """Prepare the selected benchmark dataset."""

    from .dataset import prepare_swebench

    if args.dataset == "swebench":
        prepared = prepare_swebench(args.output_dir.resolve())
    else:
        raise ValueError(f"unsupported benchmark dataset: {args.dataset}")
    logger.info("Prepared %s rows at %s", prepared.rows, prepared.index_path)
