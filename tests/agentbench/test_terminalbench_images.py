# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Image preparation, resumability and fail-fast behavior without network I/O."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentinfer.agentbench.benchkit import terminalbench, terminalbench_images
from agentinfer.agentbench.benchkit.cli import main
from agentinfer.agentbench.benchkit.dataset import Task
from tests.helpers.marks import test_marks

pytestmark = test_marks(hardware="cpu", ci="ready")


def make_task(root: Path, name: str, *, gpu: int = 0) -> Task:
    root.mkdir(parents=True)
    (root / "instruction.md").write_text("Solve the task.\n")
    (root / "task.toml").write_text(
        f'[task]\nname = "terminal-bench/{name}"\n'
        f"[environment]\ngpus = {gpu}\ndocker_image = 'example/{name}@sha256:env'\n"
        f"[verifier]\nenvironment_mode = 'separate'\n"
        f"[verifier.environment]\ndocker_image = 'example/{name}@sha256:verifier'\n"
    )
    return Task(f"terminal-bench__{name}", "", "", "", task_path=root)


@pytest.fixture
def engine(monkeypatch):
    state = SimpleNamespace(
        cached={}, calls=[], fail=None, downloaded=0, architecture="x86_64", pulled_platform="linux/amd64"
    )
    monkeypatch.delenv("DOCKER_DEFAULT_PLATFORM", raising=False)
    which = terminalbench.shutil.which
    monkeypatch.setattr(terminalbench.shutil, "which", lambda exe: "/fake/harbor" if exe == "harbor" else which(exe))

    def run(command, **kwargs):
        state.calls.append(command)
        if command[:3] == ["/fake/harbor", "datasets", "download"]:
            root = Path(command[command.index("--output-dir") + 1])
            make_task(root / "a", "a")
            make_task(root / "b", "b")
            make_task(root / "gpu", "gpu", gpu=1)
            state.downloaded += 1
            return SimpleNamespace(returncode=0)
        if command[:2] == ["docker", "info"]:
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"OSType": "linux", "Architecture": state.architecture})
            )
        if command[:3] == ["docker", "image", "inspect"]:
            local = state.cached.get(command[-1])
            return SimpleNamespace(returncode=0 if local else 1, stdout=json.dumps([local]), stderr="")
        if command[:2] == ["docker", "build"]:
            image = command[command.index("--tag") + 1]
            state.cached[image] = {"Os": "linux", "Architecture": "amd64", "Id": "sha256:built"}
            return SimpleNamespace(returncode=0)
        if command[:2] == ["docker", "pull"]:
            if command[-1] == state.fail:
                raise subprocess.CalledProcessError(1, command)
            os_name, arch = state.pulled_platform.split("/")
            state.cached[command[-1]] = {"Os": os_name, "Architecture": arch, "Id": "sha256:image"}
            return SimpleNamespace(returncode=0)
        raise AssertionError(command)

    monkeypatch.setattr(terminalbench_images.subprocess, "run", run)
    return state


def test_with_images_extends_prefix_of_existing_export(tmp_path, engine):
    prepare = ["prepare", "terminalbench", "--output-dir", str(tmp_path)]
    assert main(prepare) == 0
    original_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="already exists and is not empty"):
        main(prepare)

    assert main([*prepare, "--with-images", "--task-nums", "1"]) == 0
    assert set(engine.cached) == {"example/a@sha256:env", "example/a@sha256:verifier"}

    start = len(engine.calls)
    assert main([*prepare, "--with-images", "--task-nums", "2"]) == 0
    assert {cmd[-1] for cmd in engine.calls[start:] if cmd[:2] == ["docker", "pull"]} == {
        "example/b@sha256:env",
        "example/b@sha256:verifier",
    }
    report = json.loads((tmp_path / "images.json").read_text())
    assert report["task_ids"] == ["terminal-bench__a", "terminal-bench__b"]
    assert [image["source"] for image in report["images"]] == ["cache", "cache", "pull", "pull"]

    start = len(engine.calls)
    assert main([*prepare, "--with-images", "--task-nums", "2"]) == 0
    assert not any(cmd[:2] == ["docker", "pull"] for cmd in engine.calls[start:])

    assert main([*prepare, "--with-images"]) == 0
    assert len(engine.cached) == 6
    assert "example/gpu@sha256:env" in engine.cached
    assert "example/gpu@sha256:verifier" in engine.cached
    assert engine.downloaded == 1
    assert all(path.read_bytes() == contents for path, contents in original_files.items())


def test_image_preparation_retry_skips_cached_image(tmp_path, engine):
    engine.fail = "example/a@sha256:verifier"
    with pytest.raises(subprocess.CalledProcessError):
        main(["prepare", "terminalbench", "--output-dir", str(tmp_path), "--with-images", "--task-nums", "1"])
    report = json.loads((tmp_path / "images.json").read_text())
    assert report["status"] == "failed"
    assert report["images"][0]["status"] == "ready"
    assert report["images"][1]["status"] == "failed"
    paths = [
        tmp_path / "instances.jsonl",
        tmp_path / "manifest.json",
        tmp_path / "task-lists/all.txt",
    ]
    originals = {path: path.read_bytes() for path in paths}
    engine.fail = None
    assert main(["prepare", "terminalbench", "--output-dir", str(tmp_path), "--with-images", "--task-nums", "1"]) == 0
    assert engine.downloaded == 1
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    assert engine.calls.count(["docker", "pull", "--platform", "linux/amd64", "example/a@sha256:env"]) == 1
    assert json.loads((tmp_path / "images.json").read_text())["status"] == "ready"


@pytest.mark.parametrize(
    "problem, error",
    [
        ("version", ValueError),
        ("missing_index", FileExistsError),
        ("missing_task", ValueError),
        ("missing_list", FileExistsError),
        ("truncated_list", ValueError),
    ],
)
def test_prepare_requires_complete_matching_dataset_to_append(tmp_path, engine, problem, error):
    main(["prepare", "terminalbench", "--output-dir", str(tmp_path)])
    if problem == "version":
        manifest = tmp_path / "manifest.json"
        manifest.write_text(manifest.read_text().replace("4.1.0", "4.0.0"))
    elif problem == "missing_index":
        (tmp_path / "instances.jsonl").unlink()
    elif problem == "missing_list":
        (tmp_path / "task-lists/all.txt").unlink()
    elif problem == "truncated_list":
        (tmp_path / "task-lists/all.txt").write_text("terminal-bench__a\n")
    else:
        (tmp_path / "tasks/b/instruction.md").unlink()
    before = list(engine.calls)
    with pytest.raises(error, match="Terminal-Bench|Expected prepared"):
        main(["prepare", "terminalbench", "--output-dir", str(tmp_path), "--with-images", "--task-nums", "1"])
    assert engine.calls == before


def test_remote_arm_engine_pulls_native_images(tmp_path, engine):
    engine.architecture = "aarch64"
    engine.pulled_platform = "linux/arm64"
    main(["prepare", "terminalbench", "--output-dir", str(tmp_path), "--with-images", "--task-nums", "1"])
    report = json.loads((tmp_path / "images.json").read_text())
    assert report["target_platform"] == "linux/arm64"
    assert all(item["platform"] == "linux/arm64" for item in report["images"])
    assert ["docker", "pull", "--platform", "linux/arm64", "example/a@sha256:env"] in engine.calls


def test_wrong_platform_after_pull_is_not_ready(tmp_path, engine):
    engine.pulled_platform = "linux/arm64"
    with pytest.raises(RuntimeError, match="incompatible with"):
        main(["prepare", "terminalbench", "--output-dir", str(tmp_path), "--with-images", "--task-nums", "1"])
    assert json.loads((tmp_path / "images.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("separate,implicit", [(False, False), (True, False), (True, True)])
def test_dockerfile_images_are_built_in_prepare_only(tmp_path, engine, separate, implicit):
    main(["prepare", "terminalbench", "--output-dir", str(tmp_path)])
    root = tmp_path / "tasks/a"
    config = root / "task.toml"
    config.write_text('[task]\nname="terminal-bench/a"\n[environment]\n')
    (root / "environment").mkdir()
    (root / "environment/Dockerfile").write_text("FROM example:base\n")
    if separate:
        (root / "tests").mkdir()
        (root / "tests/Dockerfile").write_text("FROM example:verifier\n")
        with config.open("a") as stream:
            stream.write(
                "[verifier]\n" + ("" if implicit else 'environment_mode="separate"\n') + "[verifier.environment]\n"
            )
    terminalbench_images.prepare_images(tmp_path, 1)
    report = json.loads((tmp_path / "images.json").read_text())
    assert len(report["images"]) == (2 if separate else 1)
    assert all(item["source"] == "build" for item in report["images"])
    assert not any(command[:2] == ["docker", "pull"] for command in engine.calls)


def test_restricted_task_prepares_sidecar_and_probe(tmp_path, engine, monkeypatch):
    main(["prepare", "terminalbench", "--output-dir", str(tmp_path)])
    config = tmp_path / "tasks/a/task.toml"
    config.write_text(config.read_text().replace("[environment]", '[environment]\nnetwork_mode="none"'))
    context = tmp_path / "sidecar"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM example:base\n")
    monkeypatch.setattr(
        terminalbench_images,
        "sidecar_specs",
        lambda *_: [
            terminalbench_images.ImageSpec("sidecar:hash", context),
            terminalbench_images.ImageSpec("probe:pin"),
        ],
    )
    terminalbench_images.prepare_images(tmp_path, 1)
    assert "sidecar:hash" in engine.cached
    assert "probe:pin" in engine.cached


def test_compose_service_images_are_prepared_in_the_correct_scoring_mode(tmp_path, engine, monkeypatch):
    task = make_task(tmp_path / "task", "ctr")
    assert task.task_path is not None
    registries = {"environment": "api", "tests": "scorer"}
    for directory in registries:
        context = task.task_path / directory
        context.mkdir()
        (context / "docker-compose.yaml").write_text(
            f"services:\n  {registries[directory]}:\n    image: example/{registries[directory]}@sha256:pin\n"
        )
    run = terminalbench_images.subprocess.run

    def resolve_compose(command, **kwargs):
        if command[:2] != ["docker", "compose"]:
            return run(command, **kwargs)
        service = registries[kwargs["cwd"].name]
        assert command[-4:] == ["config", "--no-env-resolution", "--format", "json"]
        spec = json.loads(kwargs["input"])
        spec["services"][service] = {"image": f"example/{service}@sha256:pin"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(spec), stderr="")

    monkeypatch.setattr(terminalbench_images.subprocess, "run", resolve_compose)
    report = tmp_path / "images.json"
    terminalbench_images.prepare_task_images([task], report, with_correctness=False)
    assert set(engine.cached) == {"example/ctr@sha256:env", "example/api@sha256:pin"}
    terminalbench_images.prepare_task_images([task], report, with_correctness=True)
    assert set(engine.cached) == {
        "example/ctr@sha256:env",
        "example/api@sha256:pin",
        "example/ctr@sha256:verifier",
        "example/scorer@sha256:pin",
    }


def test_run_excludes_verifier_only_images_when_scoring_is_disabled(tmp_path, engine, monkeypatch):
    task = make_task(tmp_path / "a", "a")
    config = tmp_path / "a/task.toml"
    config.write_text(config.read_text() + 'network_mode = "none"\n')
    monkeypatch.setattr(terminalbench_images, "sidecar_specs", lambda *_: [terminalbench_images.ImageSpec("probe:pin")])
    report = tmp_path / "images.json"
    terminalbench_images.prepare_task_images([task], report, with_correctness=False)
    assert set(engine.cached) == {"example/a@sha256:env"}

    terminalbench_images.prepare_task_images([task], report, with_correctness=True)
    assert set(engine.cached) == {"example/a@sha256:env", "example/a@sha256:verifier", "probe:pin"}
