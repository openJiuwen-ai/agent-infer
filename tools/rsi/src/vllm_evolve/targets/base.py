"""
TargetPlugin ABC + DefaultTargetPlugin (YAML-driven).

A TargetPlugin defines "what to optimize" without any evolution logic.
The DefaultTargetPlugin reads everything from the YAML spec — adding
a new target requires zero Python code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_evolve.config import TargetSpec


class TargetPlugin(ABC):
    """Abstract base for optimization targets.

    Subclass this for custom targets that need logic beyond what
    YAML can express. For most cases, DefaultTargetPlugin suffices.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def spec(self) -> TargetSpec: ...

    @abstractmethod
    def compute_fitness(
        self,
        metrics: dict[str, float],
        seed_metrics: dict[str, float],
    ) -> float:
        """Compute scalar fitness from raw metrics.

        Uses the YAML-defined metrics config (direction, weight, normalization).
        """
        ...

    def prompt_hints(self) -> str:
        """Load domain knowledge from prompt_hints.md (if exists)."""
        return ""

    def validate_decision(self, code: str) -> list[str]:
        """L2 constraint validation. Override for target-specific checks.

        Returns list of violation messages. Empty = OK.
        """
        return []

    def load_skeleton(self, project_root: Path = Path(".")) -> str:
        """Read the skeleton.py source."""
        path = project_root / self.spec.skeleton_path
        return path.read_text(encoding="utf-8")

    def load_seed(self, project_root: Path = Path(".")) -> str:
        """Read the seed.py source."""
        path = project_root / self.spec.seed_path
        return path.read_text(encoding="utf-8")


class DefaultTargetPlugin(TargetPlugin):
    """YAML-driven target — no custom Python needed.

    Reads fitness weights, prompt hints, and constraints entirely from
    the TargetSpec YAML. This is the default for all targets.
    """

    def __init__(self, target_spec: TargetSpec):
        self._spec = target_spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def spec(self) -> TargetSpec:
        return self._spec

    def compute_fitness(
        self,
        metrics: dict[str, float],
        seed_metrics: dict[str, float],
    ) -> float:
        """YAML-driven fitness: weighted ratio-to-seed normalization."""
        from vllm_evolve.targets.fitness import WeightedRatioFitness
        calculator = WeightedRatioFitness(self._spec.metrics)
        return calculator.compute(metrics, seed_metrics)

    def prompt_hints(self) -> str:
        if self._spec.prompt_hints_path:
            path = Path(self._spec.prompt_hints_path)
            if path.exists():
                return path.read_text(encoding="utf-8")
        return ""
