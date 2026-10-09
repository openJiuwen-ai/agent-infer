"""
L3: Anti-overfitting check.

Rejects candidates where any single scenario's fitness drops below
seed × threshold (default 80%). This prevents overfitting to one scenario
at the expense of others.
"""

from __future__ import annotations


def check_anti_overfit(
    per_scenario_fitness: dict[str, float],
    seed_fitness: float = 1.0,
    threshold: float = 0.80,
) -> list[str]:
    """Check that no scenario drops below seed × threshold.

    Args:
        per_scenario_fitness: {scenario_name: fitness_value}
        seed_fitness: The seed's baseline fitness (default 1.0).
        threshold: Minimum ratio to seed (default 0.80 = 80%).

    Returns:
        List of violation messages. Empty = all OK.
    """
    floor = seed_fitness * threshold
    issues: list[str] = []
    for scenario, fitness in per_scenario_fitness.items():
        if fitness < floor:
            issues.append(
                f"Scenario '{scenario}' fitness {fitness:.3f} < "
                f"floor {floor:.3f} (seed × {threshold})"
            )
    return issues
