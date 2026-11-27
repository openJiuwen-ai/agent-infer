# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""EngineCore-local Program cache operations on the scheduler's two HiRadix tiers."""

from agentinfer.agentcache.mori.admission import BackendObservation, ProgramOperation
from agentinfer.agentcache.mori.hiradix import HiRadix
from agentinfer.scheduling.domain import ProgramRef


class HiRadixProgramCache:
    def __init__(self, tree: HiRadix):
        self.tree = tree

    def observe_backend(self) -> BackendObservation:
        tree = self.tree
        # Each native pool reserves one null page.
        return BackendObservation(
            True,
            (tree.device.pool.num_gpu_blocks - 1) * tree.page_size,
            (tree.host.pool.num_gpu_blocks - 1) * tree.page_size,
        )

    def apply_program_ops(self, operations: tuple[ProgramOperation, ...]) -> None:
        placement = self.tree.placement
        for operation in operations:
            if operation.placement is None:
                placement.release(operation.program)
            else:
                placement.mark(operation.program, operation.placement.value)

    def snapshot(self) -> dict:
        """Inspect physical page ownership on demand, outside the scheduling hot path."""
        owners: dict[ProgramRef, dict[str, int]] = {}
        for page in self.tree.pages.values():
            ref = page.program
            if isinstance(ref, ProgramRef):
                row = owners.setdefault(ref, {"hbm_pages": 0, "dram_pages": 0, "shared_pages": 0})
                row["hbm_pages"] += len(page.device)
                row["dram_pages"] += len(page.host)
                row["shared_pages"] += int(page.shared)
        return {
            "placements": self.tree.placement.placements(),
            "owners": [
                {"program_id": ref.program_id, "generation": ref.generation, **row}
                for ref, row in sorted(owners.items())
            ],
            "page_size": self.tree.page_size,
        }
