# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentinfer.agentbench.benchkit.cli import (
    _apply_cli_overrides,
    _parser,
    _prepare,
    _resolve_cli_path,
    _resolve_type,
    main,
)
from agentinfer.agentbench.benchkit.config import AgentBenchConfig, load_config


def _config(path: Path) -> None:
    path.write_text(
        "experiment:\n  result_dir: relative-results\n",
        encoding="utf-8",
    )


def test_schema_overrides_metrics_url(tmp_path: Path) -> None:
    parser = _parser()
    config = AgentBenchConfig()

    args = parser.parse_args(["run", "--config", "c.yaml", "--metrics-url", "http://vllm:8000/metrics"])
    assert _apply_cli_overrides(args, config).backend.metrics_url == "http://vllm:8000/metrics"


@pytest.mark.parametrize("flag", ["--enabled", "--no-enabled", "--router-url"])
def test_removed_router_cli_flags_are_rejected(flag: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", flag, "http://router"] if flag == "--router-url" else ["run", flag])


def test_cli_paths_are_cwd_relative_and_bare_executable_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _resolve_cli_path(Path("results")) == tmp_path / "results"
    assert _resolve_cli_path(Path("claude"), allow_bare=True) == Path("claude")
    assert _resolve_cli_path(Path("bin/claude"), allow_bare=True) == tmp_path / "bin/claude"


def test_yaml_paths_resolve_before_cli_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    _config(config_path)
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    args = _parser().parse_args(["run", "--config", str(config_path), "--result-dir", "cli-results"])

    configured = load_config(config_path)
    assert configured.experiment.result_dir == tmp_path / "relative-results"
    assert _apply_cli_overrides(args, configured).experiment.result_dir == invocation / "cli-results"


def test_run_delegates_to_runner_with_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    _config(config_path)
    runner = types.ModuleType("agentinfer.agentbench.benchkit.runner")
    runner.run_benchmark = AsyncMock(return_value=tmp_path / "run")
    monkeypatch.setitem(sys.modules, runner.__name__, runner)

    assert main(["run", "--config", str(config_path), "--task-num", "2"]) == 0
    call = runner.run_benchmark.await_args
    assert call.args[0].experiment.task_num == 2
    assert "config_path" not in call.kwargs
    assert call.kwargs["cli_metadata"]["config_path"] == str(config_path)
    assert call.kwargs["cli_metadata"]["overrides"] == {"task_num": 2}


def test_run_without_config_uses_defaults_and_omits_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = types.ModuleType("agentinfer.agentbench.benchkit.runner")
    runner.run_benchmark = AsyncMock(return_value=tmp_path / "run")
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.chdir(tmp_path)

    assert main(["run", "--task-num", "2"]) == 0
    call = runner.run_benchmark.await_args
    assert call.args[0].experiment.task_num == 2
    assert call.args[0].dataset.index_path == tmp_path / "data/swebench/instances.jsonl"
    assert call.args[0].dataset.selection_path == tmp_path / "data/swebench/task-lists/default.txt"
    assert call.args[0].dataset.cache_dir == tmp_path / "data/swebench/repo-cache"
    assert "config_path" not in call.kwargs["cli_metadata"]
    assert call.kwargs["cli_metadata"]["overrides"] == {"task_num": 2}


def test_replay_delegates_to_planner_with_replay_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("", encoding="utf-8")
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(
        f"""
experiment:
  result_dir: result
replay:
  trace_path: {source.name}
""",
        encoding="utf-8",
    )
    calls = []
    runner = types.ModuleType("agentinfer.agentbench.replay.runner")

    def plan(config, *, cli_metadata):
        calls.append((config, cli_metadata))
        return tmp_path / "result"

    runner.run_replay = plan
    monkeypatch.setitem(sys.modules, runner.__name__, runner)

    assert (
        main(
            [
                "replay",
                "--config",
                str(config_path),
                "--task-num",
                "2",
                "--trace-path",
                str(source),
            ]
        )
        == 0
    )
    config, metadata = calls[0]
    assert config.experiment.task_num == 2
    assert config.replay.trace_path == source
    assert metadata["config_path"] == str(config_path)
    assert metadata["overrides"] == {
        "task_num": 2,
        "trace_path": str(source),
    }


def test_compare_delegates_to_owned_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    run_dir = tmp_path / "run"
    compare_module = types.ModuleType("agentinfer.agentbench.benchkit.compare")
    compare_module.compare = lambda *_args, **_kwargs: "comparison"
    monkeypatch.setitem(sys.modules, compare_module.__name__, compare_module)

    assert main(["compare", "--baseline", str(run_dir), "--candidate", str(run_dir), "--json"]) == 0
    assert capsys.readouterr().out == "comparison\n"


def test_summarize_combines_positional_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    output = tmp_path / "combined.csv"
    summarize_module = types.ModuleType("agentinfer.agentbench.benchkit.summarize")
    calls = []
    summarize_module.combine_summaries = lambda runs, **outputs: calls.append((runs, outputs))
    monkeypatch.setitem(sys.modules, summarize_module.__name__, summarize_module)

    assert main(["summarize", str(run1), str(run2), "--output-csv", str(output)]) == 0
    assert calls == [([run1.resolve(), run2.resolve()], {"output": output.resolve(), "figures_dir": None})]


def test_summarize_accepts_one_run_and_uses_default_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    summarize_module = types.ModuleType("agentinfer.agentbench.benchkit.summarize")
    calls = []
    summarize_module.combine_summaries = lambda runs, **outputs: calls.append((runs, outputs))
    monkeypatch.setitem(sys.modules, summarize_module.__name__, summarize_module)
    monkeypatch.chdir(tmp_path)

    assert main(["summarize", str(run_dir)]) == 0
    assert calls == [([run_dir.resolve()], {"output": tmp_path / "combined-summary.csv", "figures_dir": None})]


def test_summarize_passes_figures_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    figures = tmp_path / "figures"
    summarize_module = types.ModuleType("agentinfer.agentbench.benchkit.summarize")
    calls = []
    summarize_module.combine_summaries = lambda runs, **outputs: calls.append((runs, outputs))
    monkeypatch.setitem(sys.modules, summarize_module.__name__, summarize_module)

    assert main(["summarize", str(run1), str(run2), "--output-figs-dir", str(figures)]) == 0
    assert calls == [
        (
            [run1.resolve(), run2.resolve()],
            {"output": (Path.cwd() / "combined-summary.csv").resolve(), "figures_dir": figures.resolve()},
        )
    ]


def test_prepare_delegates_to_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = types.ModuleType("agentinfer.agentbench.benchkit.dataset")
    dataset.prepare_swebench = lambda output: types.SimpleNamespace(rows=1, index_path=output / "instances.jsonl")
    monkeypatch.setitem(sys.modules, dataset.__name__, dataset)
    assert main(["prepare", "swebench", "--output-dir", str(tmp_path / "data")]) == 0


def test_prepare_defaults_output_dir_to_data_swebench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Path] = []
    dataset = types.ModuleType("agentinfer.agentbench.benchkit.dataset")
    dataset.prepare_swebench = lambda output: (
        captured.append(output) or types.SimpleNamespace(rows=1, index_path=output / "instances.jsonl")
    )
    monkeypatch.setitem(sys.modules, dataset.__name__, dataset)
    monkeypatch.chdir(tmp_path)

    assert main(["prepare", "swebench"]) == 0
    assert captured == [tmp_path / "data/swebench"]


def test_unsupported_cli_override_type_fails_during_discovery() -> None:
    with pytest.raises(ValueError, match="unsupported CLI override type"):
        _resolve_type(list[str])


def test_prepare_rejects_unsupported_dataset() -> None:
    with pytest.raises(ValueError, match="unsupported benchmark dataset: other"):
        _prepare(types.SimpleNamespace(dataset="other", output_dir=Path("unused")))
