"""``ve init`` implementation — materialize the packaged Claude Code assets into
a target project's ``.claude/`` directory.

* Copies the agent, the ``ve-*`` skills, and the PreToolUse hook into ``.claude/``.
* Deep-merges ``settings.json`` (preserving all existing keys) to register the
  hook + the ``Bash(ar:*)`` permission.
* Appends a managed block to ``CLAUDE.md``.
* Records a manifest so the install is idempotent and ``--uninstall`` is an exact
  restore of the original ``settings.json`` / ``CLAUDE.md``.

All file I/O uses ``encoding="utf-8"``.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import vllm_evolve
from vllm_evolve.install.manifest import (
    Manifest,
    manifest_path,
    read_manifest,
    write_manifest,
)

_CLAUDE_START = "<!-- vllm-evolve:start -->"
_CLAUDE_END = "<!-- vllm-evolve:end -->"
_ASSET_SUBDIRS = ("agents", "skills", "hooks")


def _assets_dir() -> Path:
    return Path(vllm_evolve.__file__).resolve().parent / "assets" / "claude"


@dataclass
class InstallResult:
    status: str  # "installed" | "already_installed" | "reinstalled"
    target: str
    created_files: list[str]
    hook_path: str


def _render_fragment(assets: Path, python: str, hook_path: str) -> dict:
    """Load settings.fragment.json and fill the hook command (built in Python so
    Windows backslashes are JSON-escaped correctly on dump)."""
    frag = json.loads((assets / "settings.fragment.json").read_text(encoding="utf-8"))
    frag["hooks"]["PreToolUse"][0]["hooks"][0]["command"] = f'{python} "{hook_path}"'
    return frag


def _merge_settings(existing: dict, fragment: dict) -> dict:
    """Append our hook + permission into existing settings, preserving all other
    keys and not duplicating entries."""
    out = copy.deepcopy(existing)
    pre = out.setdefault("hooks", {}).setdefault("PreToolUse", [])
    for entry in fragment.get("hooks", {}).get("PreToolUse", []):
        if entry not in pre:
            pre.append(entry)
    allow = out.setdefault("permissions", {}).setdefault("allow", [])
    for a in fragment.get("permissions", {}).get("allow", []):
        if a not in allow:
            allow.append(a)
    return out


def _strip_block(content: str) -> str:
    if _CLAUDE_START in content and _CLAUDE_END in content:
        pre = content.split(_CLAUDE_START)[0]
        post = content.split(_CLAUDE_END, 1)[1]
        return pre.rstrip() + "\n" + post.lstrip()
    return content


def _write_claude_md(claude_md: Path, snippet: str) -> None:
    block = snippet.strip()
    if claude_md.exists():
        content = _strip_block(claude_md.read_text(encoding="utf-8")).rstrip()
        new = (content + "\n\n" + block + "\n") if content else (block + "\n")
    else:
        new = block + "\n"
    claude_md.write_text(new, encoding="utf-8")


def install(
    target_dir: str | Path, *, force: bool = False, python: str | None = None
) -> InstallResult:
    target = Path(target_dir).resolve()
    claude = target / ".claude"

    existing = read_manifest(target)
    if existing is not None and not force:
        return InstallResult("already_installed", str(target), existing.created_files,
                             str(claude / "hooks" / "phase_guard_hook.py"))
    if existing is not None and force:
        uninstall(target)  # restore originals first so new backups are pristine

    assets = _assets_dir()
    created: list[str] = []

    # 1. copy agents/, skills/, hooks/ into .claude/
    for sub in _ASSET_SUBDIRS:
        src_root = assets / sub
        if not src_root.exists():
            continue
        for src in sorted(src_root.rglob("*")):
            if src.is_dir():
                continue
            rel = Path(".claude") / sub / src.relative_to(src_root)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            created.append(str(rel).replace("\\", "/"))

    hook_path = str((claude / "hooks" / "phase_guard_hook.py").resolve())

    # 2. deep-merge settings.json
    settings_path = claude / "settings.json"
    settings_existed = settings_path.exists()
    settings_backup = settings_path.read_text(encoding="utf-8") if settings_existed else None
    try:
        existing_settings = json.loads(settings_backup) if (settings_backup or "").strip() else {}
    except json.JSONDecodeError:
        existing_settings = {}
    fragment = _render_fragment(assets, python or sys.executable, hook_path)
    merged = _merge_settings(existing_settings, fragment)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    if not settings_existed:
        created.append(".claude/settings.json")

    # 3. CLAUDE.md managed block
    claude_md = target / "CLAUDE.md"
    claude_md_existed = claude_md.exists()
    claude_md_backup = claude_md.read_text(encoding="utf-8") if claude_md_existed else None
    _write_claude_md(claude_md, (assets / "CLAUDE.snippet.md").read_text(encoding="utf-8"))
    if not claude_md_existed:
        created.append("CLAUDE.md")

    # 4. manifest
    write_manifest(target, Manifest(
        created_files=created,
        settings_existed=settings_existed, settings_backup=settings_backup,
        claude_md_existed=claude_md_existed, claude_md_backup=claude_md_backup,
    ))

    return InstallResult("reinstalled" if existing is not None else "installed",
                         str(target), created, hook_path)


def uninstall(target_dir: str | Path) -> dict:
    target = Path(target_dir).resolve()
    m = read_manifest(target)
    if m is None:
        return {"status": "not_installed", "target": str(target)}

    # restore settings.json
    settings_path = target / ".claude" / "settings.json"
    if m.settings_existed and m.settings_backup is not None:
        settings_path.write_text(m.settings_backup, encoding="utf-8")
    elif settings_path.exists():
        settings_path.unlink()

    # restore CLAUDE.md
    claude_md = target / "CLAUDE.md"
    if m.claude_md_existed and m.claude_md_backup is not None:
        claude_md.write_text(m.claude_md_backup, encoding="utf-8")
    elif claude_md.exists():
        claude_md.unlink()

    # delete copied asset files (settings.json / CLAUDE.md handled above)
    for rel in m.created_files:
        if rel in (".claude/settings.json", "CLAUDE.md"):
            continue
        p = target / rel
        if p.exists():
            p.unlink()

    mp = manifest_path(target)
    if mp.exists():
        mp.unlink()
    _prune_empty_dirs(target / ".claude")
    return {"status": "uninstalled", "target": str(target)}


def _prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            d.rmdir()  # only succeeds if empty
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def check(target_dir: str | Path) -> dict:
    target = Path(target_dir).resolve()
    m = read_manifest(target)
    hook = target / ".claude" / "hooks" / "phase_guard_hook.py"
    settings = target / ".claude" / "settings.json"
    return {
        "installed": m is not None,
        "target": str(target),
        "hook_present": hook.exists(),
        "settings_present": settings.exists(),
        "files": m.created_files if m else [],
    }
