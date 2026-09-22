"""Install the repository-local Codex integration for vllm-evolve.

Codex and Claude installs are independent. Codex receives open-format skills under ``.agents/``,
its optimizer definition and phase hook under ``.codex/``, and a managed ``AGENTS.md`` block.
Every touched text file is backed up in a Codex-specific manifest so uninstall is an exact restore.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import vllm_evolve

_AGENTS_START = "<!-- vllm-evolve-codex:start -->"
_AGENTS_END = "<!-- vllm-evolve-codex:end -->"
_MANIFEST = ".vllm-evolve-install.json"


def _package_root() -> Path:
    return Path(vllm_evolve.__file__).resolve().parent


def _codex_assets() -> Path:
    return _package_root() / "assets" / "codex"


def _manifest_path(target: Path) -> Path:
    return target / ".codex" / _MANIFEST


@dataclass
class CodexManifest:
    version: str = "3"
    originals: dict[str, str | None] = field(default_factory=dict)
    managed_hashes: dict[str, str] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path):
        if not path.is_file():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


@dataclass
class CodexInstallResult:
    status: str
    target: str
    created_files: list[str]
    hook_path: str


def _relative(target: Path, path: Path) -> str:
    return path.relative_to(target).as_posix()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remember(target: Path, path: Path, manifest: CodexManifest) -> None:
    relative = _relative(target, path)
    if relative not in manifest.originals:
        manifest.originals[relative] = (
            path.read_text(encoding="utf-8") if path.exists() else None
        )


def _copy_text(
    target: Path,
    source: Path,
    destination: Path,
    manifest: CodexManifest,
) -> None:
    _remember(target, destination, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _strip_agents_block(text: str) -> str:
    if _AGENTS_START not in text or _AGENTS_END not in text:
        return text
    before = text.split(_AGENTS_START, 1)[0]
    after = text.split(_AGENTS_END, 1)[1]
    return before.rstrip() + "\n" + after.lstrip()


def _merge_pre_tool_hooks(existing: dict, fragment: dict) -> dict:
    out = json.loads(json.dumps(existing))
    target = out.setdefault("hooks", {}).setdefault("PreToolUse", [])
    for entry in fragment.get("hooks", {}).get("PreToolUse", []):
        if entry not in target:
            target.append(entry)
    if not out.get("description") and fragment.get("description"):
        out["description"] = fragment["description"]
    return out


def _strip_managed_hook_entries(existing: dict) -> dict:
    """Remove only this package's rendered hook before an in-place refresh."""
    out = json.loads(json.dumps(existing))
    hooks = out.get("hooks", {})
    entries = hooks.get("PreToolUse", [])
    hooks["PreToolUse"] = [
        entry
        for entry in entries
        if not any(
            "phase_guard_hook.py" in str(hook.get("command", ""))
            for hook in entry.get("hooks", [])
        )
    ]
    return out


def _render_hooks(python: str, hook_path: Path) -> dict:
    fragment = json.loads(
        (_codex_assets() / "hooks.fragment.json").read_text(encoding="utf-8")
    )
    fragment["hooks"]["PreToolUse"][0]["hooks"][0]["command"] = (
        f'{python} "{hook_path.resolve()}"'
    )
    return fragment


def install(
    target_dir: str | Path,
    *,
    force: bool = False,
    python: str | None = None,
) -> CodexInstallResult:
    target = Path(target_dir).resolve()
    manifest_path = _manifest_path(target)
    existing = CodexManifest.read(manifest_path)
    if existing is not None and not force:
        health = check(target)
        return CodexInstallResult(
            "already_installed" if health["health"] == "healthy" else health["health"],
            str(target),
            sorted(existing.originals),
            str(target / ".codex" / "hooks" / "phase_guard_hook.py"),
        )
    manifest = existing or CodexManifest()
    codex = _codex_assets()
    hooks_path = target / ".codex" / "hooks.json"
    if hooks_path.exists() and hooks_path.read_text(encoding="utf-8").strip():
        try:
            current_hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid existing Codex hooks file: {hooks_path}") from exc
    else:
        current_hooks = {}
    if existing is not None:
        current_hooks = _strip_managed_hook_entries(current_hooks)

    for source in sorted((codex / "skills").glob("*/SKILL.md")):
        destination = target / ".agents" / "skills" / source.parent.name / "SKILL.md"
        _copy_text(target, source, destination, manifest)
    for source in sorted((codex / "agents").glob("*.toml")):
        destination = target / ".codex" / "agents" / source.name
        _copy_text(target, source, destination, manifest)

    hook_path = target / ".codex" / "hooks" / "phase_guard_hook.py"
    _copy_text(target, codex / "hooks" / "phase_guard_hook.py", hook_path, manifest)
    _remember(target, hooks_path, manifest)
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(
        json.dumps(
            _merge_pre_tool_hooks(
                current_hooks, _render_hooks(python or sys.executable, hook_path)
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    agents_md = target / "AGENTS.md"
    _remember(target, agents_md, manifest)
    current_agents = agents_md.read_text(encoding="utf-8") if agents_md.exists() else ""
    snippet = (codex / "AGENTS.snippet.md").read_text(encoding="utf-8").strip()
    base = _strip_agents_block(current_agents).rstrip()
    agents_md.write_text((base + "\n\n" if base else "") + snippet + "\n", encoding="utf-8")

    manifest.managed_hashes = {
        relative: _file_hash(target / relative)
        for relative in manifest.originals
        if (target / relative).is_file()
    }
    manifest.write(manifest_path)
    return CodexInstallResult(
        "reinstalled" if existing is not None else "installed",
        str(target),
        sorted(manifest.originals),
        str(hook_path),
    )


def uninstall(target_dir: str | Path) -> dict:
    target = Path(target_dir).resolve()
    path = _manifest_path(target)
    manifest = CodexManifest.read(path)
    if manifest is None:
        return {"status": "not_installed", "target": str(target)}
    for relative, original in manifest.originals.items():
        destination = target / relative
        if original is None:
            if destination.exists():
                destination.unlink()
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(original, encoding="utf-8")
    if path.exists():
        path.unlink()
    _prune_empty(target / ".agents")
    _prune_empty(target / ".codex")
    return {"status": "uninstalled", "target": str(target)}


def _prune_empty(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def check(target_dir: str | Path) -> dict:
    target = Path(target_dir).resolve()
    manifest = CodexManifest.read(_manifest_path(target))
    hook_path = target / ".codex" / "hooks" / "phase_guard_hook.py"
    hooks_path = target / ".codex" / "hooks.json"
    issues: list[str] = []
    content_checks: list[dict] = []
    expected_assets: list[tuple[Path, Path]] = []
    codex = _codex_assets()
    for source in sorted((codex / "skills").glob("*/SKILL.md")):
        expected_assets.append(
            (source, target / ".agents" / "skills" / source.parent.name / "SKILL.md")
        )
    for source in sorted((codex / "agents").glob("*.toml")):
        expected_assets.append((source, target / ".codex" / "agents" / source.name))
    expected_assets.append((codex / "hooks" / "phase_guard_hook.py", hook_path))
    for source, destination in expected_assets:
        source_hash = _file_hash(source)
        actual_hash = _file_hash(destination) if destination.is_file() else None
        ok = actual_hash == source_hash
        content_checks.append(
            {
                "path": _relative(target, destination),
                "expected_sha256": source_hash,
                "actual_sha256": actual_hash,
                "ok": ok,
            }
        )
        if not ok:
            issues.append(f"managed asset drift: {_relative(target, destination)}")

    if manifest is not None:
        if not manifest.managed_hashes:
            issues.append("legacy install manifest has no managed content hashes")
        for relative, recorded_hash in manifest.managed_hashes.items():
            path = target / relative
            actual_hash = _file_hash(path) if path.is_file() else None
            if actual_hash != recorded_hash:
                issues.append(f"installed managed file changed: {relative}")

    snippet = (codex / "AGENTS.snippet.md").read_text(encoding="utf-8").strip()
    agents_text = (target / "AGENTS.md").read_text(encoding="utf-8") if (
        target / "AGENTS.md"
    ).is_file() else ""
    agents_block_ok = snippet in agents_text
    if not agents_block_ok:
        issues.append("managed AGENTS.md block is missing or drifted")

    hook_smoke = {"ok": False, "allow_returncode": None, "deny_returncode": None}
    if hooks_path.is_file():
        try:
            hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
            commands = [
                hook["command"]
                for entry in hooks.get("hooks", {}).get("PreToolUse", [])
                for hook in entry.get("hooks", [])
                if "phase_guard_hook.py" in str(hook.get("command", ""))
            ]
            if len(commands) != 1:
                raise ValueError(f"expected one vllm-evolve hook command, found {len(commands)}")
            argv = shlex.split(commands[0])
            allow = subprocess.run(
                argv,
                input=json.dumps(
                    {"tool_name": "Edit", "tool_input": {"file_path": "notes/scratch.md"}}
                ),
                text=True,
                capture_output=True,
                cwd=target,
                timeout=15,
                check=False,
            )
            deny = subprocess.run(
                argv,
                input=json.dumps(
                    {
                        "tool_name": "apply_patch",
                        "tool_input": {
                            "patch": "*** Begin Patch\n*** Update File: "
                            "targets/scheduling/seed.py\n*** End Patch\n"
                        },
                    }
                ),
                text=True,
                capture_output=True,
                cwd=target,
                timeout=15,
                check=False,
            )
            hook_smoke = {
                "ok": allow.returncode == 0 and deny.returncode == 2,
                "allow_returncode": allow.returncode,
                "deny_returncode": deny.returncode,
                "deny_stderr": deny.stderr[-500:],
                "command": commands[0],
            }
            if not hook_smoke["ok"]:
                issues.append("installed hook command failed functional allow/deny smoke")
        except Exception as exc:
            issues.append(f"installed hook command is broken: {type(exc).__name__}: {exc}")

    installed = manifest is not None
    health = "missing" if not installed else "healthy" if not issues else "drifted"
    if installed and not hook_smoke["ok"]:
        health = "broken"
    return {
        "installed": manifest is not None,
        "target": str(target),
        "health": health,
        "issues": issues,
        "hook_present": hook_path.is_file(),
        "hooks_present": hooks_path.is_file(),
        "agent_present": (
            target / ".codex" / "agents" / "vllm-policy-optimizer.toml"
        ).is_file(),
        "skills_present": (
            target / ".agents" / "skills" / "ve-evolve" / "SKILL.md"
        ).is_file(),
        "files": sorted(manifest.originals) if manifest else [],
        "content_checks": content_checks,
        "agents_block_ok": agents_block_ok,
        "hook_smoke": hook_smoke,
    }
