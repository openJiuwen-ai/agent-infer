# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Build and validate an isolated JiuwenSwarm runtime for Harbor containers."""

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from importlib import metadata
from pathlib import Path

WORKSWARM_VERSION = "0.2.8"
OPENJIUWEN_VERSION = "0.1.20"
REMOTE_RUNTIME = "/opt/agentbench-jiuwenswarm"
RUNTIME_ENV = "AGENTBENCH_JIUWENSWARM_RUNTIME"
RUNTIME_IMPORT_CHECK = (
    "from openjiuwen.core.single_agent.agents.react_agent import ReActAgent; "
    "from agentinfer.agentbench.agents.jiuwenswarm.terminalbench_entry import HarborJiuwenInstance"
)


def archive_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def corrupt_distributions(site: Path, names: set[str] | None = None) -> list[str]:
    """Check wheel RECORD hashes so a damaged shared cache is not bundled."""
    corrupt = []
    for distribution in metadata.distributions(path=[str(site)]):
        if names is not None and distribution.metadata["Name"].lower() not in names:
            continue
        for file in distribution.files or ():
            if file.hash is None:
                continue
            path = Path(str(distribution.locate_file(file)))
            if not path.is_file():
                corrupt.append(f"{distribution.metadata['Name']}=={distribution.version}")
                break
            digest = hashlib.new(file.hash.mode)
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
            if actual != file.hash.value:
                corrupt.append(f"{distribution.metadata['Name']}=={distribution.version}")
                break
    return corrupt


def repair_from_installed(requirement: str, site: Path) -> bool:
    """Reuse a verified same-ABI wheel installation, never the entire host venv."""
    name, version = requirement.split("==", 1)
    if sys.version_info[:2] != (3, 12):
        return False
    try:
        source = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    if source.version != version or corrupt_distributions(Path(str(source.locate_file(""))), {name.lower()}):
        return False
    for file in source.files or ():
        path = Path(str(source.locate_file(file)))
        target = (site / str(file)).resolve()
        if not target.is_relative_to(site.parents[2]):
            raise ValueError(f"Runtime package file is outside its Python prefix: {file}")
        if path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return True


def validate_runtime(executable: Path) -> tuple[Path, dict]:
    if shutil.which(str(executable)) is None:
        raise FileNotFoundError(f"JiuwenSwarm executable not found: {executable}")
    configured = os.environ.get(RUNTIME_ENV)
    if not configured:
        raise ValueError(f"Set {RUNTIME_ENV} to a prepared runtime directory; see jiuwenswarm_runtime --help")
    root = Path(configured).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["workswarm"] != WORKSWARM_VERSION
        or manifest["openjiuwen"] != OPENJIUWEN_VERSION
        or manifest["platform"] != "Linux"
        or manifest["machine"] != platform.machine()
        or manifest["python"] != "3.12"
    ):
        raise ValueError(f"Incompatible JiuwenSwarm runtime: {root}")
    if archive_digest(root / "runtime.tar.gz") != manifest["sha256"]:
        raise ValueError(f"JiuwenSwarm runtime checksum mismatch: {root}")
    return root, manifest


def build_runtime(output_dir: Path) -> None:
    """Use managed, relocatable CPython, not the caller's virtual environment."""
    if platform.system() != "Linux":
        raise ValueError("JiuwenSwarm Harbor runtime requires Linux")
    if metadata.version("workswarm") != WORKSWARM_VERSION or metadata.version("openjiuwen") != OPENJIUWEN_VERSION:
        raise ValueError("Prepare the runtime from an environment with workswarm==0.2.8 and openjiuwen==0.1.20")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Runtime output is not empty: {output_dir}")
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required to prepare the JiuwenSwarm runtime")
    subprocess.run([uv, "python", "install", "3.12"], check=True)
    interpreter = Path(
        subprocess.run(
            [uv, "python", "find", "--managed-python", "3.12"], check=True, capture_output=True, text=True
        ).stdout.strip()
    ).resolve()
    with tempfile.TemporaryDirectory(prefix="ab-jw-", dir="/tmp") as directory:
        root = Path(directory)
        shutil.copytree(interpreter.parent.parent, root / "python", symlinks=True)
        python = root / "python" / "bin" / "python3"
        site = root / "python" / "lib" / "python3.12" / "site-packages"
        # uv installs dependencies itself. Managed CPython's bootstrap pip has
        # installation-specific RECORD entries and is not part of this runtime.
        for path in site.iterdir():
            if path.name == "pip" or (path.name.startswith("pip-") and path.name.endswith(".dist-info")):
                shutil.rmtree(path)
        for name in ("pip", "pip3", "pip3.12"):
            (python.parent / name).unlink(missing_ok=True)
        constraints = root / "host-constraints.txt"
        constraints.write_text(
            "\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()})) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--system",
                "--break-system-packages",
                "--link-mode",
                "copy",
                "--constraint",
                str(constraints),
                f"workswarm=={WORKSWARM_VERSION}",
                f"openjiuwen=={OPENJIUWEN_VERSION}",
                "numpy",
                "zstandard",
                "tqdm",
                "prometheus-client",
            ],
            check=True,
        )
        corrupt = corrupt_distributions(site)
        corrupt = [requirement for requirement in corrupt if not repair_from_installed(requirement, site)]
        if corrupt:
            # Repair only this isolated build, without deleting the user's cache.
            subprocess.run(
                [
                    uv,
                    "--no-cache",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--system",
                    "--break-system-packages",
                    "--no-deps",
                    "--reinstall",
                    *corrupt,
                ],
                check=True,
            )
        if remaining := corrupt_distributions(site):
            raise RuntimeError(f"Runtime dependency integrity check failed: {remaining}")
        shutil.copytree(
            Path(__file__).resolve().parents[2],
            site / "agentinfer",
            ignore=shutil.ignore_patterns("__pycache__", "data"),
        )
        # Console entry points must resolve the interpreter at the container path.
        for path in (root / "python" / "bin").iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as handle:
                first = handle.readline()
            if first.startswith(b"#!") and b"python" in first.lower():
                content = path.read_bytes().split(b"\n", 1)[1]
                path.write_bytes(f"#!{REMOTE_RUNTIME}/python/bin/python3\n".encode() + content)
        lock = subprocess.run(
            [uv, "pip", "freeze", "--python", str(python)], check=True, capture_output=True, text=True
        )
        (root / "requirements.lock").write_text(lock.stdout, encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "runtime.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
            for path in sorted(root.iterdir()):
                handle.add(path, arcname=path.name)
        manifest = {
            "workswarm": WORKSWARM_VERSION,
            "openjiuwen": OPENJIUWEN_VERSION,
            "python": "3.12",
            "platform": platform.system(),
            "machine": platform.machine(),
            "sha256": archive_digest(archive),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared JiuwenSwarm runtime at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build_runtime(parser.parse_args().output_dir.resolve())
