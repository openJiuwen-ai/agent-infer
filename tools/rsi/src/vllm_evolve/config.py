"""
Configuration models for vllm-evolve.

TargetSpec is loaded from config/targets/scheduling.yaml. Evaluation is real vLLM only
(the DES simulator and the legacy strategy/skill scaffold were deleted).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Maturity level for targets and features
# ---------------------------------------------------------------------------

class Maturity(str, Enum):
    SUPPORTED = "supported"         # Phase A: CLI, multi-scenario, CI
    EXPERIMENTAL = "experimental"   # Usable but incomplete
    PROTOTYPE = "prototype"         # skeleton + seed exist
    PLANNED = "planned"             # YAML placeholder only


# ---------------------------------------------------------------------------
# Metric configuration (YAML-driven fitness)
# ---------------------------------------------------------------------------

class MetricConfig(BaseModel):
    name: str
    direction: Literal["maximize", "minimize"]
    weight: float = 1.0
    normalization: Literal["ratio_to_seed", "raw"] = "ratio_to_seed"
    # Optional bounds for ratio normalization (default: 2x for max, 3x for min)
    ceiling_factor: float = 2.0


class MapElitesDimension(BaseModel):
    name: str
    bins: list  # list[float] or list[str] for categorical dimensions


class MapElitesConfig(BaseModel):
    dimensions: list[MapElitesDimension] = Field(default_factory=lambda: [
        MapElitesDimension(name="avg_prompt_length", bins=[128, 512, 1024, 4096]),
        MapElitesDimension(name="request_rate_qps", bins=[1, 5, 20, 100]),
    ])
    max_per_cell: int = 3


# ---------------------------------------------------------------------------
# TargetSpec — loaded from config/targets/*.yaml
# ---------------------------------------------------------------------------

class TargetSpec(BaseModel):
    """Specification of the supported optimization target.

    Loaded from config/targets/scheduling.yaml. Only `scheduling` is shipped/supported — the round
    verbs (ve context / verify / bench) reject any other target name. Evaluation is real vLLM only.
    """
    api_version: str = "vllm-evolve/v1"
    kind: str = "Target"
    name: str
    maturity: Maturity = Maturity.EXPERIMENTAL

    # Serving mode context
    serving_mode: Literal["co_located", "disaggregated", "both"] = "co_located"
    engine: str = "vllm"                    # architecture reservation
    vllm_version_constraint: str = ">=0.14.0"

    # Target definition paths (relative to project root)
    # Optional for config-type targets (serving_config) that don't evolve code
    skeleton_path: str = ""
    seed_path: str = ""
    prompt_hints_path: str = ""             # e.g. "targets/scheduling/prompt_hints.md"

    # Search space (for config-type targets)
    search_space_path: str = ""
    constraints_path: str = ""
    description: str = ""

    # Evolvable functions (empty for config-type targets)
    evolvable_functions: list[dict] = Field(default_factory=list)

    # Fitness: YAML-driven, NOT hardcoded
    metrics: list[MetricConfig] = Field(default_factory=list)

    # MAP-Elites diversity
    map_elites: MapElitesConfig = Field(default_factory=MapElitesConfig)

    # Evaluation — real vLLM only (the DES simulator was deleted)
    evaluator: str = "live"
    evaluation_scenarios: list[str] = Field(
        default_factory=lambda: ["steady_low", "bursty_short"]
    )


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

class EvolutionConfig(BaseModel):
    max_generations: int = 20
    batch_size: int = 4
    checkpoint_interval: int = 5
    eval_mode: Literal["live"] = "live"   # real vLLM only (DES simulator deleted)
    seed: int = 42


class PopulationConfig(BaseModel):
    n_islands: int = 4
    migration_interval: int = 10


class Config(BaseModel):
    # Model and hardware context
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    hardware: str = "nvidia_h100"            # see config/hardware/*.yaml

    # Subsystem configs (evaluation is real vLLM only; the DES simulator was deleted)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)

    # Defaults
    target: str = "scheduling"
    provider: str = "mock"
    provider_kwargs: dict = Field(default_factory=dict)
    output_dir: str = "."
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Config:
    import yaml
    if not path.exists():
        return Config()  # fall back to defaults
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)


def load_target_spec(
    target_name: str,
    config_dir: Path = Path("config/targets"),
) -> TargetSpec:
    import yaml
    target_file = config_dir / f"{target_name}.yaml"
    with open(target_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    spec_data = data.get("spec", data)
    spec_data["name"] = data.get("metadata", {}).get("name", target_name)
    spec_data["api_version"] = data.get("api_version", "vllm-evolve/v1")
    spec_data["kind"] = data.get("kind", "Target")
    return TargetSpec.model_validate(spec_data)
