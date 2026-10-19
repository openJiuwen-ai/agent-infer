"""Run-centric artifact layout (P1c).

Pure filesystem layer for the working area `runs/<target>/<ts>-<mode>-<run_id>/`.
No DB, no policy logic, no agent imports — see DESIGN.md for the boundary rules.
"""
from vllm_evolve.artifacts.layout import (
    RunLayout,
    compute_run_id,
    create_run,
    manifest_sha,
)

__all__ = ["RunLayout", "compute_run_id", "create_run", "manifest_sha"]
