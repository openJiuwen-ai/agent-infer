# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prepare exact, native prebuilt task/verifier images before execution."""

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .common import atomic_write_json, utc_now
from .dataset import Task, load_tasks


def read_task_config(root: Path) -> dict:
    return tomllib.loads((root / "task.toml").read_text(encoding="utf-8"))


def native_platform(os_name: str, architecture: str) -> str:
    aliases = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
    arch = aliases.get(architecture.lower())
    if os_name.lower() != "linux" or arch is None:
        raise ValueError(f"Unsupported Terminal-Bench Docker platform: {os_name}/{architecture}")
    return f"linux/{arch}"


def _docker_output(
    arguments: list[str],
    *,
    timeout: int = 30,
    stdin: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            ["docker", *arguments],
            input=stdin,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Docker {' '.join(arguments)} failed: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"Docker {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


def docker_platform() -> str:
    """Read the actual engine, including a remote engine selected by Docker context."""

    engine = json.loads(_docker_output(["info", "--format", "{{json .}}"], timeout=10))
    target = native_platform(engine["OSType"], engine["Architecture"])
    default = os.environ.get("DOCKER_DEFAULT_PLATFORM")
    if default:
        parts = default.lower().split("/")
        if (
            len(parts) not in (2, 3)
            or native_platform(parts[0], parts[1]) != target
            or (len(parts) == 3 and not (target == "linux/arm64" and parts[2] == "v8"))
        ):
            raise ValueError(f"DOCKER_DEFAULT_PLATFORM={default} conflicts with native Docker platform {target}")
    return target


@dataclass(frozen=True)
class ImageSpec:
    image: str
    context: Path | None = None


def local_build_image(context: Path, target: str) -> str:
    """Name a prepared Dockerfile image by its content and native platform."""

    if not (context / "Dockerfile").is_file():
        raise ValueError(f"No prebuilt image or Dockerfile at {context}")
    digest = hashlib.sha256(target.encode())
    for path in sorted(context.rglob("*")):
        digest.update(str(path.relative_to(context)).encode())
        digest.update((path.lstat().st_mode & 0o777).to_bytes(2, "big"))
        if path.is_symlink():
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return f"agentbench-terminalbench:{digest.hexdigest()}"


def environment_image_specs(directory: Path, image: str | None, target: str | None) -> list[ImageSpec]:
    """Resolve the main image followed by the other enabled Compose service images."""

    compose = directory / "docker-compose.yaml"
    local = not image and (directory / "Dockerfile").is_file()
    if local:
        image = local_build_image(directory, target or docker_platform())
    if not compose.is_file():
        if not image:
            raise ValueError(f"No prebuilt image or Dockerfile at {directory}")
        return [ImageSpec(image, directory if local else None)]

    # Harbor layers its main service underneath the task overlay. Use Compose
    # itself to resolve merges, interpolation and enabled profiles.
    missing_main = "agentbench-terminalbench:missing-main-image"
    base = json.dumps({"services": {"main": {"image": image or missing_main}}})
    env = {**os.environ, "PREBUILT_IMAGE_NAME": image or missing_main}
    config = json.loads(
        _docker_output(
            [
                "compose",
                "--project-directory",
                str(directory.resolve()),
                "-f",
                "-",
                "-f",
                str(compose.resolve()),
                "config",
                "--no-env-resolution",
                "--format",
                "json",
            ],
            stdin=base,
            cwd=directory,
            env=env,
        )
    )
    services = config["services"]
    specs = []
    for name in ("main", *(name for name in services if name != "main")):
        service_image = services[name].get("image")
        if not service_image or service_image == missing_main:
            raise ValueError(f"Compose service {name!r} in {compose} needs a prebuilt image")
        specs.append(ImageSpec(service_image, directory if local and service_image == image else None))
    return specs


def task_image_specs(task: Task, target: str | None = None, *, with_correctness: bool = True) -> list[ImageSpec]:
    if task.task_path is None:
        raise ValueError(f"Terminal-Bench task has no task directory: {task.instance_id}")
    root = task.task_path
    config = read_task_config(root)
    environment = config.get("environment", {})
    definitions = [(environment, root / "environment")]
    verifier = config.get("verifier", {})
    mode = verifier.get("environment_mode", "separate" if "environment" in verifier else "shared")
    if with_correctness and mode == "separate":
        verifier_env = verifier.get("environment", {})
        tests = root / "tests"
        if verifier_env.get("docker_image") or any(
            (tests / name).is_file() for name in ("Dockerfile", "docker-compose.yaml")
        ):
            definitions.append((verifier_env, tests))
        else:
            definitions.append((environment, root / "environment"))
    specs = []
    for definition, directory in definitions:
        specs.extend(environment_image_specs(directory, definition.get("docker_image"), target))
    return specs


def _needs_sidecar(tasks: Iterable[Task], *, with_correctness: bool) -> bool:
    def restricted(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        return any(
            (key == "network_mode" and item != "public")
            or (key == "allow_internet" and item is False)
            or restricted(item)
            for key, item in value.items()
            if with_correctness or key != "verifier"
        )

    return any(task.task_path is not None and restricted(read_task_config(task.task_path)) for task in tasks)


def sidecar_specs(target: str, harbor_executable: str | None = None) -> list[ImageSpec]:
    # Harbor may be installed by uv in a different Python environment. Query
    # that installation so prepare and run use exactly the same sidecar tag.
    harbor = shutil.which(harbor_executable or os.environ.get("HARBOR_EXECUTABLE", "harbor"))
    if harbor is None:
        raise FileNotFoundError("Harbor executable not found")
    python = Path(harbor).resolve().parent / "python"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [str(python), "-m", "agentinfer.agentbench.benchkit.harbor_environment", "--describe-sidecar", target],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=env,
    )
    metadata = json.loads(result.stdout)
    return [ImageSpec(metadata["image"], Path(metadata["context"])), ImageSpec(metadata["probe_image"])]


def inspect_local_image(image: str) -> dict | None:
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def _matches_platform(image: dict, target: str) -> bool:
    # Docker image inspect uses canonical OCI architecture names.
    return f"{image.get('Os')}/{image.get('Architecture')}" == target and image.get("Variant", "") in (
        ("", "v8") if target == "linux/arm64" else ("",)
    )


def prepare_images(output_dir: Path, task_nums: int | None = None, *, harbor_executable: str | None = None) -> None:
    tasks = load_tasks(output_dir / "instances.jsonl", output_dir / "task-lists" / "all.txt", task_nums)
    prepare_task_images(tasks, output_dir / "images.json", harbor_executable=harbor_executable)


def prepare_task_images(
    tasks: Iterable[Task], destination: Path, *, harbor_executable: str | None = None, with_correctness: bool = True
) -> None:
    """Fill image caches for the exact task selection, before benchmark timing."""

    tasks = tuple(tasks)
    entries: list[dict[str, object]] = []
    report: dict[str, object] = {
        "scope": "prebuilt_images",
        "status": "preparing",
        "updated_at": utc_now(),
        "task_ids": [task.instance_id for task in tasks],
        "images": entries,
    }

    def save() -> None:
        report["updated_at"] = utc_now()
        atomic_write_json(destination, report)

    save()
    try:
        target = docker_platform()
        report["target_platform"] = target
        specs: dict[str, ImageSpec] = {}
        owners: dict[str, list[str]] = {}
        for task in tasks:
            for spec in task_image_specs(task, target, with_correctness=with_correctness):
                specs.setdefault(spec.image, spec)
                if task.instance_id not in owners.setdefault(spec.image, []):
                    owners[spec.image].append(task.instance_id)
        if _needs_sidecar(tasks, with_correctness=with_correctness):
            for spec in sidecar_specs(target, harbor_executable):
                specs.setdefault(spec.image, spec)
                owners[spec.image] = [task.instance_id for task in tasks]
        entries.extend({"image": image, "task_ids": owners[image], "status": "pending"} for image in specs)
        save()
        for item in entries:
            image = str(item["image"])
            try:
                local = inspect_local_image(image)
                if local is not None and _matches_platform(local, target):
                    item.update(status="ready", source="cache")
                else:
                    context = specs[image].context
                    if context is None:
                        item["status"] = "pulling"
                        save()
                        subprocess.run(["docker", "pull", "--platform", target, image], check=True)
                    else:
                        item.update(status="building", build_context=str(context))
                        save()
                        subprocess.run(
                            ["docker", "build", "--platform", target, "--tag", image, str(context)], check=True
                        )
                    local = inspect_local_image(image)
                    if local is None or not _matches_platform(local, target):
                        raise RuntimeError(f"Prepared image unavailable or incompatible with {target}: {image}")
                    item.update(status="ready", source="pull" if context is None else "build")
                item.update(image_id=local["Id"], platform=target)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
                item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            save()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    report["status"] = "ready"
    save()
