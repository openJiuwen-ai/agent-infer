"""
Target plugin system — dynamic discovery and loading.

Targets are discovered from config/targets/*.yaml files. No hardcoding of
target names anywhere in the framework.
"""

from __future__ import annotations

from pathlib import Path

from vllm_evolve.targets.base import DefaultTargetPlugin, TargetPlugin


class TargetRegistry:
    """Discovers and manages optimization targets.

    Scans config/targets/ for YAML files and creates TargetPlugin instances.
    New targets require ONLY a YAML + targets/{name}/ directory — zero code changes.
    """

    def __init__(self, config_dir: Path = Path("config/targets")):
        self._config_dir = config_dir
        self._plugins: dict[str, TargetPlugin] = {}

    def discover(self) -> list[str]:
        """Scan config dir for available targets. Returns list of target names."""
        names: list[str] = []
        if self._config_dir.exists():
            for f in sorted(self._config_dir.glob("*.yaml")):
                names.append(f.stem)
        return names

    def get(self, name: str) -> TargetPlugin:
        """Get or create a TargetPlugin for the named target."""
        if name not in self._plugins:
            self._plugins[name] = self._load(name)
        return self._plugins[name]

    def _load(self, name: str) -> TargetPlugin:
        """Load a target from its YAML spec."""
        from vllm_evolve.config import load_target_spec
        spec = load_target_spec(name, self._config_dir)
        return DefaultTargetPlugin(spec)

    def list_with_maturity(self) -> list[tuple[str, str]]:
        """Return [(name, maturity)] for all discovered targets."""
        result = []
        for name in self.discover():
            plugin = self.get(name)
            result.append((name, plugin.spec.maturity.value))
        return result
