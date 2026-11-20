# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Task-local JiuwenSwarm runtime management for AgentBench.

Each JiuwenInstance prepares an isolated home and configuration, starts the
Gateway and AgentServer, and manages their readiness, interruption, and
shutdown for one benchmark task.
"""

import asyncio
import json
import os
import shutil
import signal
import socket
import uuid
from pathlib import Path
from typing import cast

import yaml

_STARTUP_TIMEOUT_SECONDS = 120.0
_STOP_TIMEOUT_SECONDS = 10.0


class JiuwenInstance:
    """Prepare, start, interrupt, and stop one task-local Jiuwen instance."""

    def __init__(
        self,
        *,
        executable: Path,
        artifact_dir: Path,
        workspace: Path,
        session_id: str,
        api_base_url: str,
        model: str,
        completion_timeout_seconds: int,
    ) -> None:
        self.executable = executable
        self.artifact_dir = artifact_dir
        self.workspace = workspace
        self.session_id = session_id
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.completion_timeout_seconds = completion_timeout_seconds
        self.root = artifact_dir / "jiuwenswarm-instance"
        self.home_dir = self.root / "home"
        self.data_dir = self.root / "data"
        self.sandbox_tmp_dir = self.root / "sandbox-tmp"
        self.config_path = self.data_dir / "config" / "config.yaml"
        self.web_port, self.agent_server_port, self.gateway_port = _allocate_ports(3)
        self.gateway_url = f"ws://127.0.0.1:{self.gateway_port}/tui"
        # jiuwenbox-server owns the per-task SysOperation SANDBOX backend.
        # Its listen port is allocated immediately before spawn to keep the
        # unavoidable bind gap out of Jiuwen initialization/configuration.
        self.jiuwenbox_port = 0
        self.jiuwenbox_url = ""
        self.jiuwenbox_policy_path = self.artifact_dir / "jiuwenbox-policy.yaml"
        self.environment = self._environment()
        self._init_process: asyncio.subprocess.Process | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._jiuwenbox_process: asyncio.subprocess.Process | None = None
        self._stdout = None
        self._stderr = None
        self._jiuwenbox_stdout = None
        self._jiuwenbox_stderr = None
        self._port_env_signatures: dict[Path, tuple[int, int] | None] = {}

    async def start(self) -> None:
        """Initialize, configure, start, and wait for the task-local Jiuwen services."""

        try:
            self.home_dir.mkdir(parents=True, exist_ok=False)
            self.sandbox_tmp_dir.mkdir()
            self.sandbox_tmp_dir.chmod(0o1777)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            await self._initialize()
            generated = self.home_dir / ".jiuwenswarm" / "config" / "config.yaml"
            if not generated.exists():
                generated = self.config_path
            if not generated.exists():
                raise RuntimeError("jiuwenswarm-init did not create config.yaml")
            if generated != self.config_path:
                await asyncio.to_thread(shutil.copyfile, generated, self.config_path)
            # Jiuwen creates the per-project sandbox through SysOperation.
            # Run one external jiuwenbox-server with AgentBench's task
            # policy as its base so every SysOperation-created sandbox
            # inherits network=host and the exact writable host paths.
            await asyncio.to_thread(self._write_jiuwenbox_policy)
            await self._start_jiuwenbox_server()
            await asyncio.to_thread(self._configure)

            env_paths = (
                self.data_dir / "config" / ".env",
                self.home_dir / ".jiuwenswarm" / "config" / ".env",
            )
            self._port_env_signatures = {path: _file_signature(path) for path in env_paths}
            start_executable = _companion_executable(self.executable, "jiuwenswarm-start")
            self._stdout = (self.artifact_dir / "jiuwenswarm-services.stdout.log").open("wb")
            self._stderr = (self.artifact_dir / "jiuwenswarm-services.stderr.log").open("wb")
            self._process = await asyncio.create_subprocess_exec(
                start_executable,
                "app",
                cwd=self.workspace,
                env=self.environment,
                stdout=self._stdout,
                stderr=self._stderr,
                start_new_session=os.name != "nt",
            )
            await self._wait_ready()
        except BaseException:
            await self.stop()
            raise

    async def _initialize(self) -> None:
        """Run jiuwenswarm-init noninteractively and capture its output."""

        init_executable = _companion_executable(self.executable, "jiuwenswarm-init")
        self._init_process = await asyncio.create_subprocess_exec(
            init_executable,
            cwd=self.workspace,
            env=self.environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        process = self._init_process
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                _STARTUP_TIMEOUT_SECONDS,
            )
        except BaseException:
            await _terminate_process(process)
            raise
        finally:
            self._init_process = None
        await asyncio.gather(
            asyncio.to_thread(
                (self.artifact_dir / "jiuwenswarm-init.stdout.log").write_bytes,
                stdout,
            ),
            asyncio.to_thread(
                (self.artifact_dir / "jiuwenswarm-init.stderr.log").write_bytes,
                stderr,
            ),
        )
        if process.returncode != 0:
            raise RuntimeError(f"jiuwenswarm-init exited with code {process.returncode}")

    async def interrupt(self, timeout_seconds: float = 10.0) -> bool:
        """Send a bounded explicit cancellation request for this task session."""

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        for _ in range(3):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            if await self._interrupt_once(remaining):
                return True
            await asyncio.sleep(min(0.5, max(remaining, 0)))
        return False

    async def _interrupt_once(self, timeout_seconds: float) -> bool:
        try:
            import websockets

            async with websockets.connect(
                self.gateway_url,
                close_timeout=2,
                max_size=8 * 2**20,
                ping_interval=20,
                ping_timeout=60,
            ) as websocket:
                ack = json.loads(await asyncio.wait_for(websocket.recv(), 3))
                if ack.get("type") != "event" or ack.get("event") != "connection.ack":
                    return False
                request_id = f"agentbench-interrupt-{uuid.uuid4().hex[:8]}"
                await websocket.send(
                    json.dumps(
                        {
                            "type": "req",
                            "id": request_id,
                            "method": "chat.interrupt",
                            "is_stream": False,
                            "params": {
                                "session_id": self.session_id,
                                "intent": "cancel",
                                "mode": "code.normal",
                            },
                        }
                    )
                )
                deadline = asyncio.get_running_loop().time() + timeout_seconds
                while asyncio.get_running_loop().time() < deadline:
                    remaining = deadline - asyncio.get_running_loop().time()
                    frame = json.loads(await asyncio.wait_for(websocket.recv(), remaining))
                    if frame.get("type") == "res" and frame.get("id") == request_id:
                        return bool(frame.get("ok")) and bool(frame.get("payload", {}).get("accepted"))
                    if frame.get("event") == "chat.interrupt_result":
                        return bool(frame.get("payload", {}).get("success"))
        except Exception:
            return False
        return False

    async def stop(self) -> None:
        """Terminate owned Jiuwen processes and close their log streams."""

        init_process = self._init_process
        self._init_process = None
        if init_process is not None:
            await _terminate_process(init_process)
        process = self._process
        self._process = None
        if process is not None:
            await _terminate_process(process)
        jiuwenbox_process = self._jiuwenbox_process
        self._jiuwenbox_process = None
        if jiuwenbox_process is not None:
            await _terminate_process(jiuwenbox_process)
        for handle_name in ("_stdout", "_stderr", "_jiuwenbox_stdout", "_jiuwenbox_stderr"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)

    async def _start_jiuwenbox_server(self) -> None:
        """Launch a task-local jiuwenbox-server bound to the allocated port."""

        executable = _companion_executable(self.executable, "jiuwenbox-server")
        if shutil.which(executable) is None:
            raise RuntimeError(f"jiuwenbox-server is required for isolation but is not executable: {executable}")
        self.jiuwenbox_port = _allocate_ports(1)[0]
        self.jiuwenbox_url = f"http://127.0.0.1:{self.jiuwenbox_port}"
        self._jiuwenbox_stdout = (self.artifact_dir / "jiuwenbox-server.stdout.log").open("wb")
        self._jiuwenbox_stderr = (self.artifact_dir / "jiuwenbox-server.stderr.log").open("wb")
        env = dict(self.environment)
        env["JIUWENBOX_POLICY_PATH"] = str(self.jiuwenbox_policy_path)
        self._jiuwenbox_process = await asyncio.create_subprocess_exec(
            executable,
            "--listen",
            self.jiuwenbox_url,
            "--save-logs",
            str(self.artifact_dir / "jiuwenbox-audit"),
            cwd=self.artifact_dir,
            env=env,
            stdout=self._jiuwenbox_stdout,
            stderr=self._jiuwenbox_stderr,
            start_new_session=os.name != "nt",
        )
        # Wait for the HTTP server to answer /health before creating a sandbox.
        deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            if self._jiuwenbox_process.returncode is not None:
                raise RuntimeError(
                    f"jiuwenbox-server exited during startup with code {self._jiuwenbox_process.returncode}"
                )
            if await asyncio.to_thread(_jiuwenbox_healthy, self.jiuwenbox_url):
                return
            await asyncio.sleep(0.2)
        raise TimeoutError("jiuwenbox-server startup timed out")

    async def _wait_ready(self) -> None:
        """Wait for Gateway and AgentServer while following published fallback ports."""

        assert self._process is not None
        deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            self._refresh_runtime_ports()
            if self._process.returncode is not None:
                raise RuntimeError(f"jiuwenswarm-start exited during startup with code {self._process.returncode}")
            ports_ready = await asyncio.gather(
                *(asyncio.to_thread(_port_open, port) for port in (self.gateway_port, self.agent_server_port))
            )
            if all(ports_ready):
                return
            await asyncio.sleep(0.2)
        raise TimeoutError("Jiuwen Gateway/AgentServer startup timed out")

    def _refresh_runtime_ports(self) -> None:
        """Follow Jiuwen's published fallback ports if startup resolved a conflict."""

        for path in (
            self.data_dir / "config" / ".env",
            self.home_dir / ".jiuwenswarm" / "config" / ".env",
        ):
            if not path.exists():
                continue
            signature = _file_signature(path)
            if signature == self._port_env_signatures.get(path):
                continue
            values = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    values[key.strip()] = value.strip().strip("\"'")
            try:
                self.web_port = int(values.get("WEB_PORT", self.web_port))
                self.agent_server_port = int(values.get("AGENT_SERVER_PORT", self.agent_server_port))
                self.gateway_port = int(values.get("GATEWAY_PORT", self.gateway_port))
            except ValueError:
                continue
            self.gateway_url = f"ws://127.0.0.1:{self.gateway_port}/tui"
            self.environment.update(
                {
                    "WEB_PORT": str(self.web_port),
                    "AGENT_SERVER_PORT": str(self.agent_server_port),
                    "GATEWAY_PORT": str(self.gateway_port),
                    "JIUWENSWARM_WEB_PORT": str(self.web_port),
                    "JIUWENSWARM_AGENT_SERVER_PORT": str(self.agent_server_port),
                    "JIUWENSWARM_GATEWAY_PORT": str(self.gateway_port),
                }
            )
            self._port_env_signatures[path] = signature
            return

    def _environment(self) -> dict[str, str]:
        """Build the isolated child-process environment for this benchmark task."""

        values = {
            "HOME": str(self.home_dir),
            "JIUWENSWARM_HOME": str(self.home_dir),
            "JIUWENSWARM_DATA_DIR": str(self.data_dir),
            "AGENTBENCH_ROOT_SESSION_ID": self.session_id,
            "WEB_PORT": str(self.web_port),
            "AGENT_SERVER_PORT": str(self.agent_server_port),
            "GATEWAY_PORT": str(self.gateway_port),
            "JIUWENSWARM_WEB_PORT": str(self.web_port),
            "JIUWENSWARM_AGENT_SERVER_PORT": str(self.agent_server_port),
            "JIUWENSWARM_GATEWAY_PORT": str(self.gateway_port),
        }
        return {**os.environ, **values}

    def _write_jiuwenbox_policy(self) -> None:
        """Write the base policy inherited by Jiuwen-created sandboxes."""

        self.jiuwenbox_policy_path.write_text(
            yaml.safe_dump(
                _jiuwenbox_policy(
                    self.workspace,
                    self.home_dir,
                    self.data_dir,
                    self.sandbox_tmp_dir,
                ),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _configure(self) -> None:
        """Configure Jiuwen's model client, extensions, permissions, and task isolation."""

        raw_data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        data = _require_mapping(raw_data, "config")
        models = _require_mapping(data.get("models"), "models")
        defaults = _require_list(models.get("defaults"), "models.defaults")
        if not defaults:
            raise RuntimeError("Jiuwen config models.defaults must contain a model")
        default_model = _require_mapping(defaults[0], "models.defaults[0]")
        model_config = _require_mapping(
            default_model.get("model_client_config"),
            "models.defaults[0].model_client_config",
        )
        model_config.update(
            {
                "api_base": self.api_base_url,
                "api_key": "agentbench",
                "model_name": self.model,
                "client_provider": "AgentBenchAffinity",
                "max_retries": 1,
                "timeout": 1800,  # Total HTTP/socket-read deadline; >= both stream timeouts.
                "stream_first_chunk_timeout": 1500,  # Max wait for the first parsed stream chunk.
                "stream_idle_timeout": 1200,  # Max gap between subsequent parsed stream chunks.
            }
        )
        react = _require_mapping(data.get("react"), "react")
        react["model_name"] = self.model
        react["completion_timeout"] = self.completion_timeout_seconds
        react["kv_cache_affinity_config"] = {
            # JiuwenSwarm only enables affinity for provider AscendAffinity.
            # AgentBenchAffinity is its identity-injecting subclass/provider,
            # so enablement would otherwise be normalized back to false.
            "enable_kv_cache_affinity": False,
            "enable_kv_cache_release": False,
        }
        extensions = _require_mapping(data.setdefault("extensions", {}), "extensions")
        # Jiuwen scans immediate child dirs of extension_dirs for extension.py.
        # Keep in sync with test_extension_py_is_reserved_for_jiuwenswarm.
        extensions["extension_dirs"] = str(Path(__file__).resolve().parents[1])
        _require_mapping(data.get("setup_guide"), "setup_guide")["enabled"] = False
        _require_mapping(data.get("auto_recap"), "auto_recap")["enabled"] = False
        data["auto_memory_enabled"] = False
        _require_mapping(data.get("updater"), "updater")["enabled"] = False
        data["permissions"] = {"enabled": False}
        modes = _require_mapping(data.get("modes"), "modes")
        code_mode = _require_mapping(modes.get("code"), "modes.code")
        _require_mapping(code_mode.get("memory"), "modes.code.memory")["enabled"] = False
        code_mode["rails"] = []
        code_mode["tools"] = []
        subagents = _require_mapping(react.get("subagents"), "react.subagents")
        browser_agent = _require_mapping(subagents.get("browser_agent"), "react.subagents.browser_agent")
        browser_agent["enabled"] = False
        _require_mapping(data.get("hooks"), "hooks")["disable_all_hooks"] = True
        # SysOperation SANDBOX mode: route bash/write_file/edit_file/... IO
        # through jiuwenbox (bubblewrap + Landlock + seccomp). The top-level
        # sandbox key is read independently of permissions.enabled=False.
        data["sandbox"] = {
            "enabled": True,
            "url": self.jiuwenbox_url,
            "type": "jiuwenbox",
            "startup_mode": "external",
        }
        self.config_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def _require_mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"Jiuwen config {path} must be an object")
    return cast(dict[str, object], value)


def _require_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"Jiuwen config {path} must be a list")
    return cast(list[object], value)


def _companion_executable(executable: Path, name: str) -> str:
    """Resolve a companion Jiuwen executable beside the configured CLI or on PATH."""

    if executable.parent != Path("."):
        candidate = executable.with_name(name)
        if candidate.exists():
            return str(candidate)
    return shutil.which(name) or name


def _allocate_ports(count: int) -> tuple[int, ...]:
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return tuple(int(sock.getsockname()[1]) for sock in sockets)
    finally:
        for sock in sockets:
            sock.close()


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Terminate the process group, escalating to a kill after the stop timeout."""

    if process.returncode is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), _STOP_TIMEOUT_SECONDS)
        return
    except asyncio.TimeoutError:
        pass
    if os.name == "nt":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


def _jiuwenbox_healthy(base_url: str) -> bool:
    """Return True when the jiuwenbox-server answers /health."""

    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _jiuwenbox_policy(
    workspace: Path,
    home_dir: Path,
    data_dir: Path,
    sandbox_tmp_dir: Path,
) -> dict[str, object]:
    """Build the server base policy: runtime RO and four task-owned RW mounts."""

    runtime_ro = [
        ("/bin", "/bin"),
        ("/sbin", "/sbin"),
        ("/usr", "/usr"),
        ("/lib", "/lib"),
        ("/lib64", "/lib64"),
        ("/etc/resolv.conf", "/etc/resolv.conf"),
        ("/etc/hosts", "/etc/hosts"),
        ("/etc/nsswitch.conf", "/etc/nsswitch.conf"),
        ("/etc/host.conf", "/etc/host.conf"),
        ("/etc/ssl/certs", "/etc/ssl/certs"),
        ("/etc/pki", "/etc/pki"),
        ("/opt", "/opt"),
    ]
    return {
        "version": 1,
        "name": "agentbench-task-isolation",
        "network": {"mode": "host"},
        "filesystem_policy": {
            "read_only": ["/", "/bin", "/sbin", "/usr", "/lib", "/lib64", "/etc", "/opt"],
            "read_write": ["/tmp", str(workspace), str(home_dir), str(data_dir)],
            "bind_mounts": [
                *(
                    {"host_path": host, "sandbox_path": sandbox, "mode": "ro"}
                    for host, sandbox in runtime_ro
                    if Path(host).exists()
                ),
                {
                    "host_path": str(sandbox_tmp_dir),
                    "sandbox_path": "/tmp",
                    "mode": "rw",
                },
                {"host_path": str(workspace), "sandbox_path": str(workspace), "mode": "rw"},
                {"host_path": str(home_dir), "sandbox_path": str(home_dir), "mode": "rw"},
                {"host_path": str(data_dir), "sandbox_path": str(data_dir), "mode": "rw"},
            ],
            "device": [
                {"host_path": "/dev/urandom", "sandbox_path": "/dev/urandom"},
                {"host_path": "/dev/null", "sandbox_path": "/dev/null"},
            ],
        },
    }
