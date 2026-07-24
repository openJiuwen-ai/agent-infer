import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentinfer.agentbench.benchkit.cli import (
    _apply_cli_overrides,
    _explicit_overrides,
    _parser,
    _prepare,
    _resolve_cli_path,
    _resolve_type,
    main,
)
from agentinfer.agentbench.benchkit.config import AgentBenchConfig, load_config


def _config(path: Path) -> None:
    path.write_text(
        "experiment:\n  result_dir: relative-results\nrouter:\n  enabled: false\n  base_url: null\n",
        encoding="utf-8",
    )


def test_schema_overrides_revalidate_router_and_preserve_false_metadata(tmp_path: Path) -> None:
    parser = _parser()
    config = AgentBenchConfig()

    enabled = parser.parse_args(["run", "--config", "c.yaml", "--enabled", "--router-url", "http://router"])
    assert _apply_cli_overrides(enabled, config).router.model_dump() == {
        "enabled": True,
        "base_url": "http://router",
        "control_timeout_seconds": 10.0,
    }

    disabled = parser.parse_args(["run", "--config", "c.yaml", "--no-enabled"])
    assert _explicit_overrides(disabled)["enabled"] is False
    with pytest.raises(ValueError, match="router.enabled"):
        _apply_cli_overrides(parser.parse_args(["run", "--config", "c.yaml", "--router-url", "http://router"]), config)


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

    assert main(["run", "--config", str(config_path), "--task-num", "2", "--no-enabled"]) == 0
    call = runner.run_benchmark.await_args
    assert call.args[0].experiment.task_num == 2
    assert "config_path" not in call.kwargs
    assert call.kwargs["cli_metadata"]["config_path"] == str(config_path)
    assert call.kwargs["cli_metadata"]["overrides"] == {"task_num": 2, "enabled": False}


def test_summarize_and_compare_delegate_to_owned_apis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    _config(config_path)
    run_dir = tmp_path / "run"
    runner = types.ModuleType("agentinfer.agentbench.benchkit.runner")
    summary = types.SimpleNamespace(to_dict=lambda: {"status": "completed"})
    runner.summarize_run = lambda actual, _config: summary if actual == run_dir else pytest.fail("wrong run")
    compare_module = types.ModuleType("agentinfer.agentbench.benchkit.compare")
    compare_module.compare = lambda *_args, **_kwargs: "comparison"
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.setitem(sys.modules, compare_module.__name__, compare_module)

    assert main(["summarize", "--config", str(config_path), "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "summary.json").read_text(encoding="utf-8").strip().startswith("{")
    assert main(["compare", "--baseline", str(run_dir), "--candidate", str(run_dir), "--json"]) == 0
    assert capsys.readouterr().out == "comparison\n"


def test_prepare_delegates_to_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = types.ModuleType("agentinfer.agentbench.benchkit.dataset")
    dataset.prepare_swebench = lambda output: types.SimpleNamespace(rows=1, index_path=output / "instances.jsonl")
    monkeypatch.setitem(sys.modules, dataset.__name__, dataset)
    assert main(["prepare", "swebench", "--output-dir", str(tmp_path / "data")]) == 0


def test_unsupported_cli_override_type_fails_during_discovery() -> None:
    with pytest.raises(ValueError, match="unsupported CLI override type"):
        _resolve_type(list[str])


def test_prepare_rejects_unsupported_dataset() -> None:
    with pytest.raises(ValueError, match="unsupported benchmark dataset: other"):
        _prepare(types.SimpleNamespace(dataset="other", output_dir=Path("unused")))
