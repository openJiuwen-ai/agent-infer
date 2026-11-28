# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Command-line interface for preparing, running, summarizing, and comparing benchmarks."""

import argparse
import asyncio
import json
import logging
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from types import UnionType

from pydantic import BaseModel

from ..replay.config import ReplayBenchConfig, ReplayTraceType, builtin_replay_config_path, load_replay_config
from .config import AgentBenchConfig, load_config

if typing.TYPE_CHECKING:
    from .evaluation.config import EvaluationConfig

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
_REPLAY_REQUIRED_WITHOUT_CONFIG = (
    ("trace_type", "--trace-type"),
    ("task_num", "--task-num"),
    ("base_url", "--base-url"),
    ("model", "--model"),
)


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
    normalized = _schema_override_payload(args, overrides)
    for section, values in normalized.items():
        payload[section].update(values)
    return schema.model_validate(payload)


def _schema_override_payload(
    args: argparse.Namespace,
    overrides: tuple[_Override, ...],
) -> dict[str, dict[str, object]]:
    """Return explicitly supplied CLI values grouped by configuration section."""

    payload: dict[str, dict[str, object]] = {}
    for override in overrides:
        value = getattr(args, override.dest, None)
        if value is None:
            continue
        if override.path:
            value = _resolve_cli_path(
                value,
                allow_bare=override.section == "agent" and override.field == "executable",
            )
        payload.setdefault(override.section, {})[override.field] = value
    return payload


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

    prepare = commands.add_parser("prepare", help="Prepare benchmark dataset inputs", allow_abbrev=False)
    prepare.add_argument("dataset", choices=["swebench", "terminalbench"])
    prepare.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to agentinfer/agentbench/data/terminalbench for Terminal-Bench, or data/swebench for SWE-bench",
    )
    prepare.add_argument(
        "--with-images",
        action="store_true",
        help="Terminal-Bench: prepare native Docker images for new or existing task data",
    )
    prepare.add_argument(
        "--task-nums",
        type=int,
        help="Terminal-Bench: prepare images for the first N tasks in all.txt; requires --with-images",
    )

    run = commands.add_parser("run", help="Run one benchmark")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--with-correctness", action="store_true", help="Run the dataset's official correctness checks")
    run.add_argument("--evaluation-config", type=Path, help="SWE-bench evaluation YAML; requires --with-correctness")
    _register_run_args(run)

    replay = commands.add_parser(
        "replay",
        help="Execute a deterministic Trace Replay workload",
    )
    replay.add_argument("--config", type=Path, default=None)
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

    evaluate = commands.add_parser("evaluate", help="Evaluate completed SWE-bench tasks using the official harness")
    evaluate.add_argument("--result-path", type=Path, required=True)
    evaluate.add_argument(
        "--config", type=Path, help="Optional evaluation YAML; defaults to the current Python environment"
    )
    evaluate.add_argument("--output-dir", type=Path)

    correctness = commands.add_parser("compare-correctness", help="Compare finalized SWE-bench correctness results")
    correctness.add_argument("--baseline", type=Path, required=True)
    correctness.add_argument("--candidate", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None, *, cli_metadata: dict[str, object] | None = None) -> int:
    """Parse and dispatch one BenchKit command."""

    _configure_logging()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.dataset != "terminalbench" and (args.with_images or args.task_nums is not None):
            parser.error("--with-images and --task-nums are only supported for terminalbench prepare")
        if args.task_nums is not None and args.task_nums < 1:
            parser.error("--task-nums must be positive")
        if args.task_nums is not None and not args.with_images:
            parser.error("--task-nums requires --with-images")
    if args.command in {"evaluate", "compare-correctness"}:
        return _correctness(args, parser)
    if args.command == "replay" and args.config is None:
        missing = [flag for dest, flag in _REPLAY_REQUIRED_WITHOUT_CONFIG if getattr(args, dest) is None]
        if missing:
            parser.error(f"replay without --config requires: {', '.join(missing)}")
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
            return _run(args, parser, metadata)
    elif args.command == "summarize":
        _summarize(args)
    elif args.command == "compare":
        _compare(args)
    else:
        _prepare(args)
    return 0


def _correctness(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command == "evaluate":
        return _evaluate(args.result_path, _evaluation_config(args.config, parser), args.output_dir)

    from .evaluation.compare import compare_correctness

    try:
        report = compare_correctness(args.baseline, args.candidate)
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return {"passed": 0, "failed": 1, "insufficient_evidence": 2}[report["status"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("Correctness operation failed: %s", exc)
        return 1


def _evaluation_config(path: Path | None, parser: argparse.ArgumentParser) -> "EvaluationConfig":
    from .evaluation.config import load_evaluation_config

    try:
        return load_evaluation_config(path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def _evaluate(run_dir: Path, config: "EvaluationConfig", output_dir: Path | None = None) -> int:
    from .evaluation.runner import evaluate

    try:
        output, code = evaluate(run_dir, config, output_dir)
        logger.info("Correctness results: %s", output / "correctness.json")
        sys.stdout.write((output / "correctness.json").read_text(encoding="utf-8") + "\n")
        return code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("Correctness operation failed: %s", exc)
        return 1


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser, cli_metadata: dict[str, object]) -> int:
    """Dispatch benchmark execution and dataset-specific correctness checks."""

    from .runner import run_benchmark

    if args.evaluation_config is not None and not args.with_correctness:
        parser.error("--evaluation-config requires --with-correctness")
    config = _apply_cli_overrides(args, load_config(args.config))
    terminalbench = config.dataset.name == "terminalbench"
    if terminalbench and args.evaluation_config is not None:
        parser.error("--evaluation-config is only supported for SWE-bench")
    evaluation = (
        _evaluation_config(args.evaluation_config, parser) if args.with_correctness and not terminalbench else None
    )
    options = {"with_correctness": args.with_correctness} if terminalbench else {}
    run_dir = asyncio.run(run_benchmark(config, cli_metadata=cli_metadata, **options))
    logger.info("Results: %s", run_dir)
    if evaluation is not None:
        return _evaluate(run_dir, evaluation)
    if terminalbench and args.with_correctness:
        from .collectors.correctness import load_correctness_artifact

        scoring = load_correctness_artifact(run_dir / "correctness.json")
        scored = scoring.metadata.get("scored_trials")
        if not isinstance(scored, int) or scored < 1:
            logger.error("Terminal-Bench scoring produced no valid rewards: %s", scoring.reason or "zero scored trials")
            return 1
    return 0


def _replay(args: argparse.Namespace, cli_metadata: dict[str, object]) -> None:
    """Build and execute a Trace Replay workload."""

    from ..replay.runner import run_replay

    config_path = args.config
    if config_path is None:
        config_path = builtin_replay_config_path(typing.cast(ReplayTraceType, args.trace_type))
        cli_metadata["builtin_config"] = config_path.name
    config = load_replay_config(
        config_path,
        overrides=_schema_override_payload(args, _REPLAY_OVERRIDES),
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

    from .dataset import prepare_dataset

    default_dir = (
        Path(__file__).resolve().parent.parent / "data" / "terminalbench"
        if args.dataset == "terminalbench"
        else Path("data") / args.dataset
    )
    output_dir = (args.output_dir if args.output_dir is not None else default_dir).resolve()
    if args.with_images:
        prepared = prepare_dataset(args.dataset, output_dir, with_images=True, task_nums=args.task_nums)
    else:
        prepared = prepare_dataset(args.dataset, output_dir)
    logger.info("Prepared %s rows at %s", prepared.rows, prepared.index_path)
