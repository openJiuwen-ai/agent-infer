# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import argparse
import json
import os
import sys
import types
from typing import Any

import pytest

from agentinfer.agentcache.entrypoints.cli import serve_profile


def _stub_parser() -> argparse.ArgumentParser:
    """Serve parser stub with the upstream dests injection interacts with."""

    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("model_tag", nargs="?", default=None)
    parser.add_argument("--scheduler-cls", dest="scheduler_cls", default=None)
    parser.add_argument("--middleware", dest="middleware", action="append", type=str, default=[])
    parser.add_argument("--additional-config", dest="additional_config", type=json.loads, default={})
    parser.add_argument(
        "--async-scheduling",
        dest="async_scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve_profile.add_agentinfer_arguments(parser)
    return parser


def _parse(argv: list[str]) -> tuple[argparse.Namespace, set[str]]:
    return serve_profile.parse_args_with_explicit_keys(_stub_parser(), argv)


def _install_stub_vllm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a stub vLLM surface exposing make_arg_parser, cli_env_setup, and ServeSubcommand."""

    recorded: dict[str, Any] = {}

    def make_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("model_tag", nargs="?", default=None)
        parser.add_argument("--port", type=int, default=None)
        parser.add_argument("--scheduler-cls", dest="scheduler_cls", default=None)
        parser.add_argument("--middleware", dest="middleware", action="append", type=str, default=[])
        parser.add_argument("--additional-config", dest="additional_config", type=json.loads, default={})
        parser.add_argument(
            "--async-scheduling",
            dest="async_scheduling",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        return parser

    def cli_env_setup() -> None:
        recorded["cli_env_setup"] = True

    class ServeSubcommand:
        def validate(self, args: argparse.Namespace) -> None:
            recorded["validated"] = True

        def cmd(self, args: argparse.Namespace) -> None:
            recorded["namespace"] = args

    cli_args = types.ModuleType("vllm.entrypoints.openai.cli_args")
    cli_args.make_arg_parser = make_arg_parser
    entrypoints_utils = types.ModuleType("vllm.entrypoints.utils")
    entrypoints_utils.cli_env_setup = cli_env_setup
    serve_mod = types.ModuleType("vllm.entrypoints.cli.serve")
    serve_mod.ServeSubcommand = ServeSubcommand
    for name, module in {
        "vllm": types.ModuleType("vllm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.cli_args": cli_args,
        "vllm.entrypoints.utils": entrypoints_utils,
        "vllm.entrypoints.cli": types.ModuleType("vllm.entrypoints.cli"),
        "vllm.entrypoints.cli.serve": serve_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return recorded


def test_bare_flag_pins_async_and_async_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(serve_profile.LIFECYCLE_SOCKET_ENV, raising=False)
    namespace, _ = _parse(["MODEL", "--agentinfer"])
    injected = serve_profile.inject_profile(namespace, set())

    assert namespace.async_scheduling is True
    assert namespace.scheduler_cls == serve_profile.ASYNC_SCHEDULER_CLS
    assert namespace.middleware == list(serve_profile.DEFAULT_SERVE_PROFILE.middlewares)
    assert namespace.additional_config == {"agentcache": {}}
    assert injected["async_scheduling"] is True


def test_explicit_async_choice_is_preserved_with_matching_bridge() -> None:
    namespace, explicit = _parse(["MODEL", "--agentinfer", "--async-scheduling"])
    serve_profile.inject_profile(namespace, explicit)
    assert namespace.async_scheduling is True
    assert namespace.scheduler_cls == serve_profile.ASYNC_SCHEDULER_CLS

    namespace, explicit = _parse(["MODEL", "--agentinfer", "--no-async-scheduling"])
    serve_profile.inject_profile(namespace, explicit)
    assert namespace.async_scheduling is False
    assert namespace.scheduler_cls == serve_profile.SYNC_SCHEDULER_CLS


def test_upstream_default_async_mode_uses_async_bridge_without_forcing() -> None:
    namespace, explicit = _parse(["MODEL", "--agentinfer"])
    assert "async_scheduling" not in explicit
    assert namespace.async_scheduling is None
    serve_profile.inject_profile(namespace, explicit)
    assert namespace.async_scheduling is True


def test_explicit_scheduler_cls_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace, explicit = _parse(["MODEL", "--agentinfer", "--scheduler-cls", "other.Other"])
    with pytest.raises(serve_profile.AgentInferServeError, match="--scheduler-cls"):
        serve_profile.inject_profile(namespace, explicit)


def test_user_middleware_order_is_preserved_and_duplicates_removed() -> None:
    namespace, explicit = _parse(
        [
            "MODEL",
            "--agentinfer",
            "--middleware",
            "custom.Mid",
            "--middleware",
            serve_profile.IDENTITY_MIDDLEWARE,
        ]
    )
    injected = serve_profile.inject_profile(namespace, explicit)

    assert namespace.middleware == [
        "custom.Mid",
        serve_profile.IDENTITY_MIDDLEWARE,
        serve_profile.LIFECYCLE_MIDDLEWARE,
    ]
    assert injected["middleware"] == [serve_profile.LIFECYCLE_MIDDLEWARE]


def test_user_additional_config_merges_with_user_priority() -> None:
    user_config = {
        "other": {"flag": True},
        "agentcache": {"observability": {"enabled": True}},
    }
    namespace, explicit = _parse(["MODEL", "--agentinfer", "--additional-config", json.dumps(user_config)])
    serve_profile.inject_profile(namespace, explicit)

    assert namespace.additional_config == {
        "other": {"flag": True},
        "agentcache": {"observability": {"enabled": True}},
    }


def test_non_object_agentcache_is_rejected() -> None:
    namespace, explicit = _parse(["MODEL", "--agentinfer", "--additional-config", '{"agentcache": "on"}'])
    with pytest.raises(serve_profile.AgentInferServeError, match="agentcache must be a JSON object"):
        serve_profile.inject_profile(namespace, explicit)


def test_invalid_additional_config_json_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = argparse.Namespace(additional_config="{not json")
    with pytest.raises(serve_profile.AgentInferServeError, match="invalid"):
        serve_profile._merge_additional_config(namespace.additional_config, serve_profile.DEFAULT_SERVE_PROFILE)

    namespace = argparse.Namespace(additional_config="[1, 2]")
    with pytest.raises(serve_profile.AgentInferServeError, match="JSON object"):
        serve_profile._merge_additional_config(namespace.additional_config, serve_profile.DEFAULT_SERVE_PROFILE)


def test_run_agentinfer_serve_dispatches_enriched_namespace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded = _install_stub_vllm(monkeypatch)
    monkeypatch.delenv(serve_profile.LIFECYCLE_SOCKET_ENV, raising=False)

    exit_code = serve_profile.run_agentinfer_serve(
        ["MODEL", "--agentinfer", "--port", "8001", "--middleware", "custom.Mid"]
    )

    assert exit_code == 0
    assert recorded["validated"] is True
    served = recorded["namespace"]
    assert served.model_tag == "MODEL"
    assert served.scheduler_cls == serve_profile.ASYNC_SCHEDULER_CLS
    assert served.async_scheduling is True
    assert served.middleware == ["custom.Mid", *serve_profile.DEFAULT_SERVE_PROFILE.middlewares]
    assert served.additional_config == {"agentcache": {}}
    assert os.environ[serve_profile.LIFECYCLE_SOCKET_ENV] == serve_profile.DEFAULT_LIFECYCLE_SOCKET
    err = capsys.readouterr().err
    assert "[agentinfer] --agentinfer injected:" in err
    assert serve_profile.ASYNC_SCHEDULER_CLS in err


def test_run_agentinfer_serve_preserves_existing_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_vllm(monkeypatch)
    monkeypatch.setenv(serve_profile.LIFECYCLE_SOCKET_ENV, "/tmp/custom.sock")

    serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer"])

    assert os.environ[serve_profile.LIFECYCLE_SOCKET_ENV] == "/tmp/custom.sock"


def test_run_agentinfer_serve_requires_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_vllm(monkeypatch)
    with pytest.raises(serve_profile.AgentInferServeError, match="--agentinfer"):
        serve_profile.run_agentinfer_serve(["MODEL"])


def test_run_agentinfer_serve_applies_upstream_cli_env_setup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded = _install_stub_vllm(monkeypatch)
    monkeypatch.delenv(serve_profile.LIFECYCLE_SOCKET_ENV, raising=False)

    serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer"])

    assert recorded["cli_env_setup"] is True


def test_lifecycle_socket_env_defaults_to_config_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_stub_vllm(monkeypatch)
    monkeypatch.delenv(serve_profile.LIFECYCLE_SOCKET_ENV, raising=False)
    config_path = "/tmp/config-driven.sock"
    user_config = {"agentcache": {"lifecycle_socket_path": config_path}}

    serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer", "--additional-config", json.dumps(user_config)])

    assert os.environ[serve_profile.LIFECYCLE_SOCKET_ENV] == config_path
    err = capsys.readouterr().err
    assert f"defaulting {serve_profile.LIFECYCLE_SOCKET_ENV}={config_path}" in err
    assert "from --additional-config" in err


def test_lifecycle_socket_env_and_config_conflict_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_vllm(monkeypatch)
    monkeypatch.setenv(serve_profile.LIFECYCLE_SOCKET_ENV, "/tmp/env.sock")
    user_config = {"agentcache": {"lifecycle_socket_path": "/tmp/config.sock"}}

    with pytest.raises(serve_profile.AgentInferServeError, match="lifecycle socket"):
        serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer", "--additional-config", json.dumps(user_config)])


def test_transparency_line_omits_empty_agentcache_fragment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_stub_vllm(monkeypatch)
    monkeypatch.delenv(serve_profile.LIFECYCLE_SOCKET_ENV, raising=False)

    serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer"])

    err = capsys.readouterr().err
    assert "[agentinfer] --agentinfer injected:" in err
    assert "--additional-config.agentcache" not in err

    user_config = {"agentcache": {"lifecycle_socket_path": "/tmp/config-driven.sock"}}
    monkeypatch.setenv(serve_profile.LIFECYCLE_SOCKET_ENV, "/tmp/config-driven.sock")
    serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer", "--additional-config", json.dumps(user_config)])

    err = capsys.readouterr().err
    assert '--additional-config.agentcache {"lifecycle_socket_path": "/tmp/config-driven.sock"}' in err


def test_empty_lifecycle_socket_env_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_stub_vllm(monkeypatch)
    monkeypatch.setenv(serve_profile.LIFECYCLE_SOCKET_ENV, "")

    serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer"])

    assert os.environ[serve_profile.LIFECYCLE_SOCKET_ENV] == serve_profile.DEFAULT_LIFECYCLE_SOCKET
    err = capsys.readouterr().err
    assert f"defaulting {serve_profile.LIFECYCLE_SOCKET_ENV}=" in err


def test_dispatch_serve_reports_missing_subcommand_as_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _install_stub_vllm(monkeypatch)
    monkeypatch.delitem(sys.modules, "vllm.entrypoints.cli.serve")
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.cli.serve",
        types.ModuleType("vllm.entrypoints.cli.serve"),
    )

    with pytest.raises(serve_profile.AgentInferServeError, match="long-form"):
        serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer"])
    assert "namespace" not in recorded


def test_transparency_line_omits_user_config_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_stub_vllm(monkeypatch)
    monkeypatch.delenv(serve_profile.LIFECYCLE_SOCKET_ENV, raising=False)
    user_config = {"secret_backend": {"api_token": "sk-super-secret"}}

    serve_profile.run_agentinfer_serve(["MODEL", "--agentinfer", "--additional-config", json.dumps(user_config)])

    err = capsys.readouterr().err
    assert "sk-super-secret" not in err
    assert "secret_backend" not in err
