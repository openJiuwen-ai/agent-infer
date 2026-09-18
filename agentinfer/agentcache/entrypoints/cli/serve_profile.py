# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""``vllm serve MODEL --agentinfer``: flag registration, explicit-option parsing, and namespace injection.

The takeover reuses upstream vLLM's ``make_arg_parser`` so every standard serve option, default, and
validation rule stays upstream-owned, adds an ``AgentInferConfig`` argument group, injects the profile
values into the parsed namespace, and dispatches through the upstream serve subcommand. See
``design/module/scheduling/agentinfer-serve-flag.md`` for the contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

ASYNC_SCHEDULER_CLS = "agentinfer.agentcache.core.scheduler.AgentCacheAsyncSchedulerBridge"
SYNC_SCHEDULER_CLS = "agentinfer.agentcache.core.scheduler.AgentCacheSyncSchedulerBridge"
IDENTITY_MIDDLEWARE = "agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware"
LIFECYCLE_MIDDLEWARE = "agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware"
CONTROLLER_FACTORY = "agentinfer.agentcache.core.factory.build_progress_ttl_controller"

LIFECYCLE_SOCKET_ENV = "AGENTCACHE_VLLM_LIFECYCLE_SOCKET"
DEFAULT_LIFECYCLE_SOCKET = "/tmp/agentinfer-vllm-lifecycle.sock"

SCHEDULER_CLS_DEST = "scheduler_cls"
ASYNC_SCHEDULING_DEST = "async_scheduling"
MIDDLEWARE_DEST = "middleware"
ADDITIONAL_CONFIG_DEST = "additional_config"


class AgentInferServeError(Exception):
    """Serve usage error carrying an operator-facing remediation message."""


@dataclass(frozen=True)
class ServeProfile:
    """AgentInfer values injected into a parsed upstream serve namespace."""

    async_scheduler_cls: str
    sync_scheduler_cls: str
    middlewares: tuple[str, ...]
    agentcache_config: dict[str, Any]


DEFAULT_SERVE_PROFILE = ServeProfile(
    async_scheduler_cls=ASYNC_SCHEDULER_CLS,
    sync_scheduler_cls=SYNC_SCHEDULER_CLS,
    middlewares=(IDENTITY_MIDDLEWARE, LIFECYCLE_MIDDLEWARE),
    agentcache_config={"controller_factory": CONTROLLER_FACTORY},
)


def add_agentinfer_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``AgentInferConfig`` argument group on a serve parser."""

    group = parser.add_argument_group(
        "AgentInferConfig",
        description="AgentInfer agent-aware serving additions (see the AgentInfer docs).",
    )
    group.add_argument(
        "--agentinfer",
        action="store_true",
        default=False,
        help="Enable the AgentInfer Progress-TTL serving path: injects the AgentCache scheduler bridge "
        "(async or sync, following --async-scheduling/--no-async-scheduling), the identity and lifecycle "
        "middleware, and the Progress-TTL controller factory.",
    )


class _UnsetList(list):
    """Append-compatible sentinel: argparse's append action copies it into a new list."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<agentinfer-unset>"


_UNSET: Any = _UnsetList()


def parse_args_with_explicit_keys(
    parser: argparse.ArgumentParser, argv: list[str]
) -> tuple[argparse.Namespace, set[str]]:
    """Parse ``argv`` and report which dests the caller explicitly provided.

    argparse cannot distinguish an explicit value from a default, so a probe pass pre-fills every dest
    with a sentinel namespace before parsing: argparse skips defaults for attributes already present,
    so any dest still holding the sentinel was not explicitly provided. The probe uses
    ``parse_known_args`` so unknown options cannot abort detection; the real pass uses ``parse_args``
    and fails with argparse's native error for unknown options.
    """

    probe = argparse.Namespace()
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS or action.dest == "help":
            continue
        if not hasattr(probe, action.dest):
            setattr(probe, action.dest, _UNSET)
    parser.parse_known_args(argv, probe)
    explicit = {
        action.dest
        for action in parser._actions
        if action.dest != argparse.SUPPRESS and getattr(probe, action.dest, _UNSET) is not _UNSET
    }
    namespace = parser.parse_args(argv)
    return namespace, explicit


def _coerce_additional_config(current: Any) -> dict[str, Any]:
    if current is None:
        return {}
    if isinstance(current, dict):
        return dict(current)
    if isinstance(current, str):
        try:
            parsed = json.loads(current)
        except json.JSONDecodeError as exc:
            raise AgentInferServeError(
                f"--agentinfer cannot merge an invalid --additional-config JSON value: {current!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise AgentInferServeError(f"--additional-config must be a JSON object, got: {current!r}")
        return parsed
    raise AgentInferServeError(f"unsupported --additional-config value type: {type(current)!r}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_additional_config(current: Any, profile: ServeProfile) -> dict[str, Any]:
    user_config = _coerce_additional_config(current)
    user_agentcache = user_config.get("agentcache")
    if user_agentcache is not None and not isinstance(user_agentcache, dict):
        raise AgentInferServeError("--additional-config agentcache must be a JSON object when --agentinfer is set.")
    if isinstance(user_agentcache, dict):
        profile_factory = profile.agentcache_config.get("controller_factory")
        if "controller_factory" in user_agentcache and user_agentcache["controller_factory"] != profile_factory:
            raise AgentInferServeError(
                "--agentinfer cannot be combined with a custom agentcache.controller_factory; "
                "use the explicit long-form command documented in docs/en/how-to/integrate-vllm.md."
            )
    merged = _deep_merge({"agentcache": dict(profile.agentcache_config)}, user_config)
    return merged


def inject_profile(
    namespace: argparse.Namespace,
    explicit_dests: set[str],
    profile: ServeProfile | None = None,
) -> dict[str, Any]:
    """Inject profile values into a parsed serve namespace and return the injected summary.

    The scheduling mode is never forced (CLI-INV-008): an explicit
    ``--async-scheduling``/``--no-async-scheduling`` choice is preserved and selects the matching
    bridge; only an unset mode receives the pinned async default, because vLLM may auto-enable async
    scheduling when the option is unset and the bridge must match the engine mode. Raises
    :class:`AgentInferServeError` on conflicting explicit options.
    """

    resolved = profile if profile is not None else DEFAULT_SERVE_PROFILE
    injected: dict[str, Any] = {}

    if SCHEDULER_CLS_DEST in explicit_dests:
        raise AgentInferServeError(
            "--agentinfer cannot be combined with --scheduler-cls; remove --scheduler-cls or use the "
            "explicit long-form command documented in docs/en/how-to/integrate-vllm.md."
        )

    if ASYNC_SCHEDULING_DEST in explicit_dests:
        async_enabled = bool(getattr(namespace, ASYNC_SCHEDULING_DEST, None))
    else:
        async_enabled = True
        if hasattr(namespace, ASYNC_SCHEDULING_DEST) and not getattr(namespace, ASYNC_SCHEDULING_DEST):
            setattr(namespace, ASYNC_SCHEDULING_DEST, True)
            injected[ASYNC_SCHEDULING_DEST] = True

    scheduler_cls = resolved.async_scheduler_cls if async_enabled else resolved.sync_scheduler_cls
    setattr(namespace, SCHEDULER_CLS_DEST, scheduler_cls)
    injected[SCHEDULER_CLS_DEST] = scheduler_cls

    user_middleware = list(getattr(namespace, MIDDLEWARE_DEST, None) or [])
    merged_middleware = list(user_middleware)
    appended: list[str] = []
    for middleware in resolved.middlewares:
        if middleware not in merged_middleware:
            merged_middleware.append(middleware)
            appended.append(middleware)
    if hasattr(namespace, MIDDLEWARE_DEST):
        setattr(namespace, MIDDLEWARE_DEST, merged_middleware)
        injected[MIDDLEWARE_DEST] = appended

    if hasattr(namespace, ADDITIONAL_CONFIG_DEST):
        merged_config = _merge_additional_config(getattr(namespace, ADDITIONAL_CONFIG_DEST, None), resolved)
        setattr(namespace, ADDITIONAL_CONFIG_DEST, merged_config)
        injected[ADDITIONAL_CONFIG_DEST] = merged_config

    return injected


def _print_transparency(
    injected: dict[str, Any], socket_defaulted: bool, socket_value: str, socket_from_config: bool
) -> None:
    parts = []
    if ASYNC_SCHEDULING_DEST in injected:
        parts.append("--async-scheduling")
    if SCHEDULER_CLS_DEST in injected:
        parts.append(f"--scheduler-cls {injected[SCHEDULER_CLS_DEST]}")
    for middleware in injected.get(MIDDLEWARE_DEST, []):
        parts.append(f"--middleware {middleware}")
    if ADDITIONAL_CONFIG_DEST in injected:
        merged = injected[ADDITIONAL_CONFIG_DEST]
        agentcache = merged.get("agentcache", {}) if isinstance(merged, dict) else {}
        shown = {key: agentcache[key] for key in ("controller_factory", "lifecycle_socket_path") if key in agentcache}
        parts.append(f"--additional-config.agentcache {json.dumps(shown, sort_keys=True)}")
    print(f"[agentinfer] --agentinfer injected: {' '.join(parts)}", file=sys.stderr)
    if socket_defaulted:
        source = " (from --additional-config agentcache.lifecycle_socket_path)" if socket_from_config else ""
        print(f"[agentinfer] defaulting {LIFECYCLE_SOCKET_ENV}={socket_value}{source}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        return argparse.ArgumentParser(prog="vllm serve")
    return FlexibleArgumentParser(prog="vllm serve")


def _dispatch_serve(namespace: argparse.Namespace) -> None:
    """Validate and dispatch through the upstream CLI's own serve subcommand."""

    try:
        from vllm.entrypoints.cli.serve import ServeSubcommand
    except ImportError as exc:
        raise AgentInferServeError(
            "the upstream vLLM serve subcommand is unavailable in this vLLM layout; use the explicit "
            "long-form command documented in docs/en/how-to/integrate-vllm.md."
        ) from exc

    command = ServeSubcommand()
    command.validate(namespace)
    command.cmd(namespace)


def _import_make_arg_parser():
    """Resolve the upstream serve parser factory across vLLM layouts."""

    try:
        from vllm.entrypoints.launchers.cli_args import make_arg_parser
    except ImportError:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
    return make_arg_parser


def _setup_cli_env() -> None:
    """Apply the upstream CLI's environment defaults, as the delegating entrypoint would."""

    try:
        from vllm.entrypoints.utils import cli_env_setup
    except ImportError:
        return
    cli_env_setup()


def _apply_lifecycle_socket_env(namespace: argparse.Namespace) -> tuple[bool, str, bool]:
    """Export the lifecycle socket env the middleware needs, consistent with any config override.

    Returns ``(defaulted, value, from_config)``: ``defaulted`` is False when the environment variable
    was already set. The scheduler reads ``agentcache.lifecycle_socket_path`` first and the
    environment second, while the middleware reads only the environment, so the two must agree.
    """

    config_socket: str | None = None
    config = getattr(namespace, ADDITIONAL_CONFIG_DEST, None)
    if isinstance(config, dict):
        agentcache = config.get("agentcache")
        if isinstance(agentcache, dict):
            raw = agentcache.get("lifecycle_socket_path")
            if raw is not None:
                if not isinstance(raw, str) or not raw:
                    raise AgentInferServeError("agentcache.lifecycle_socket_path must be a non-empty string")
                config_socket = raw
    configured = os.environ.get(LIFECYCLE_SOCKET_ENV) or None
    if configured:
        if config_socket is not None and config_socket != configured:
            raise AgentInferServeError(
                f"{LIFECYCLE_SOCKET_ENV}={configured} conflicts with --additional-config "
                f"agentcache.lifecycle_socket_path={config_socket}; keep exactly one lifecycle socket path."
            )
        return False, configured, False
    value = config_socket if config_socket is not None else DEFAULT_LIFECYCLE_SOCKET
    os.environ[LIFECYCLE_SOCKET_ENV] = value
    return True, value, config_socket is not None


def run_agentinfer_serve(serve_argv: list[str]) -> int:
    """Parse, inject, and serve an ``--agentinfer`` takeover command.

    ``serve_argv`` is the argument list following the ``serve`` subcommand. Raises
    :class:`AgentInferServeError` for usage conflicts; the caller maps it to exit code 2.
    """

    _setup_cli_env()
    make_arg_parser = _import_make_arg_parser()

    parser = make_arg_parser(_build_parser())
    add_agentinfer_arguments(parser)
    namespace, explicit_dests = parse_args_with_explicit_keys(parser, serve_argv)
    if not getattr(namespace, "agentinfer", False):
        raise AgentInferServeError("--agentinfer is required for the AgentInfer serve takeover")

    injected = inject_profile(namespace, explicit_dests)
    socket_defaulted, socket_value, socket_from_config = _apply_lifecycle_socket_env(namespace)
    _print_transparency(injected, socket_defaulted, socket_value, socket_from_config)
    _dispatch_serve(namespace)
    return 0
