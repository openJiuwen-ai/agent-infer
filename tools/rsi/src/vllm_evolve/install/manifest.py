"""Install manifest — records what ``ve init`` wrote so it can be reversed.

Backs up the *entire* original ``settings.json`` and ``CLAUDE.md`` (rather than
trying to surgically un-merge), so uninstall is an exact restore.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = ".vllm-evolve-install.json"


@dataclass
class Manifest:
    version: str = "1"
    created_files: list[str] = field(default_factory=list)  # relative to target dir, fwd slashes
    settings_existed: bool = False
    settings_backup: str | None = None
    claude_md_existed: bool = False
    claude_md_backup: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls(**json.loads(text))


def manifest_path(target_dir: str | Path) -> Path:
    return Path(target_dir) / ".claude" / MANIFEST_NAME


def read_manifest(target_dir: str | Path) -> Manifest | None:
    p = manifest_path(target_dir)
    if not p.exists():
        return None
    try:
        return Manifest.from_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_manifest(target_dir: str | Path, m: Manifest) -> None:
    p = manifest_path(target_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(m.to_json(), encoding="utf-8")
