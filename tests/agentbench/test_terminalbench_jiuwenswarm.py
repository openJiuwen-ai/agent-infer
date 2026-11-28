# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify the JiuwenSwarm container boundary and request identity."""

import json
import platform
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
