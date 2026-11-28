# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify the JiuwenSwarm container boundary and request identity."""

import base64
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from agentinfer.agentbench.agents.jiuwenswarm.terminalbench_entry import HarborJiuwenInstance
from agentinfer.agentbench.benchkit import jiuwenswarm_runtime, terminalbench
from agentinfer.agentbench.benchkit.config import AgentBenchConfig
from tests.helpers.marks import test_marks

from .agents.jiuwenswarm.test_instance import _write_minimal_config

pytestmark = test_marks(hardware="cpu", ci="ready")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    archive = root / "runtime.tar.gz"
    archive.write_bytes(b"fake runtime for command construction")
    manifest = {
        "workswarm": "0.2.8",
        "openjiuwen": "0.1.20",
        "python": "3.12",
        "platform": "Linux",
        "machine": platform.machine(),
        "sha256": jiuwenswarm_runtime.archive_digest(archive),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv(jiuwenswarm_runtime.RUNTIME_ENV, str(root))
    return root


def test_runtime_integrity_is_checked(runtime):
    (runtime / "runtime.tar.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        jiuwenswarm_runtime.validate_runtime(Path("sh"))


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="runtime repair requires the same Python 3.12 ABI")
def test_runtime_repairs_only_a_verified_distribution(tmp_path, monkeypatch):
    source = tmp_path / "installed"
    metadata = source / "demo-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: demo\nVersion: 1.0\n")
    payload = source / "demo.py"
    payload.write_text("content = True\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload.read_bytes()).digest()).decode().rstrip("=")
    (metadata / "RECORD").write_text(f"demo.py,sha256={digest},\ndemo-1.0.dist-info/METADATA,,\n")
    distribution = next(jiuwenswarm_runtime.metadata.distributions(path=[str(source)]))
    monkeypatch.setattr(jiuwenswarm_runtime.metadata, "distribution", lambda _name: distribution)
    site = tmp_path / "runtime/python/lib/python3.12/site-packages"
    assert jiuwenswarm_runtime.repair_from_installed("demo==1.0", site)
    assert not jiuwenswarm_runtime.corrupt_distributions(site)

    payload.write_text("damaged\n")
    assert jiuwenswarm_runtime.corrupt_distributions(source) == ["demo==1.0"]
    assert not jiuwenswarm_runtime.repair_from_installed("demo==1.0", site)
    assert (site / "demo.py").read_text() == "content = True\n"


def test_build_runtime_bundles_managed_python_and_source(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    interpreter = managed / "bin/python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    (managed / "lib/python3.12/site-packages").mkdir(parents=True)
    launcher = managed / "bin/jiuwenswarm"
    launcher.write_text("#!/original/python3\npass\n")
    monkeypatch.setattr(jiuwenswarm_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        jiuwenswarm_runtime.metadata,
        "version",
        {"workswarm": jiuwenswarm_runtime.WORKSWARM_VERSION, "openjiuwen": jiuwenswarm_runtime.OPENJIUWEN_VERSION}.get,
    )
    monkeypatch.setattr(jiuwenswarm_runtime.shutil, "which", lambda _name: "/fake/uv")

    def uv(command, **_options):
        args = command[1:3]
        if args == ["python", "find"]:
            stdout = str(interpreter)
        elif args == ["pip", "freeze"]:
            stdout = "workswarm==0.2.8\n"
        else:
            assert args in (["python", "install"], ["pip", "install"])
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout)

    monkeypatch.setattr(jiuwenswarm_runtime.subprocess, "run", uv)
    output = tmp_path / "bundle"
    jiuwenswarm_runtime.build_runtime(output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sha256"] == jiuwenswarm_runtime.archive_digest(output / "runtime.tar.gz")
    with tarfile.open(output / "runtime.tar.gz") as archive:
        assert archive.getmember(
            "python/lib/python3.12/site-packages/agentinfer/agentbench/agents/jiuwenswarm/terminalbench_entry.py"
        ).isfile()
        entry = archive.extractfile("python/bin/jiuwenswarm")
        assert entry is not None
        assert entry.readline().decode().strip() == f"#!{jiuwenswarm_runtime.REMOTE_RUNTIME}/python/bin/python3"


def test_preflight_records_selected_runtime(runtime, monkeypatch):
    from agentinfer.agentbench.benchkit.terminalbench_images import native_platform

    monkeypatch.setattr(
        "agentinfer.agentbench.benchkit.terminalbench_images.docker_platform",
        lambda: native_platform("Linux", platform.machine()),
    )
    monkeypatch.setattr(terminalbench, "_docker_gateway", lambda: "gateway.example.test")
    monkeypatch.setattr(terminalbench, "_executable_version", lambda *_: (Path("/usr/bin/harbor"), "0.25.0"))
    config = AgentBenchConfig.model_validate(
        {
            "dataset": {"name": "terminalbench"},
            "agent": {"type": "jiuwenswarm", "profile": "agent.code.normal", "executable": "sh"},
            "backend": {"endpoint": "/v1/chat/completions"},
        }
    )
    report = terminalbench.check_terminalbench_preflight(config)
    assert report.installation["runtime"] == str(runtime)
    assert report.versions["workswarm"] == "0.2.8"
    assert report.installation["isolation"] == "harbor-docker"
    assert report.capabilities["container_setup_verified"] is False


def test_container_configuration_does_not_change_swe_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTBENCH_HARBOR_CONTAINER", "1")
    monkeypatch.setenv("HTTPS_PROXY", "")
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: str(path) == "/.dockerenv" or original_exists(path))
    instance = HarborJiuwenInstance(
        executable=Path("jiuwenswarm"),
        artifact_dir=tmp_path / "logs",
        workspace=tmp_path,
        session_id="root",
        api_base_url="http://proxy",
        model="model",
        completion_timeout_seconds=60,
    )
    _write_minimal_config(instance)
    assert "HTTPS_PROXY" not in instance.environment
    instance._configure()
    config = yaml.safe_load(instance.config_path.read_text())
    assert config["sandbox"] == {"enabled": False, "type": "", "url": ""}
    assert config["models"]["defaults"][0]["model_client_config"]["client_provider"] == "AgentBenchAffinity"
    monkeypatch.delenv("AGENTBENCH_HARBOR_CONTAINER")
    with pytest.raises(RuntimeError, match="only run inside"):
        HarborJiuwenInstance(
            executable=Path("jiuwenswarm"),
            artifact_dir=tmp_path,
            workspace=tmp_path,
            session_id="root",
            api_base_url="http://proxy",
            model="model",
            completion_timeout_seconds=60,
        )
