# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Server-owned scheduling branches; no inference about application context adoption."""

from __future__ import annotations

import uuid
from dataclasses import replace

from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.identity import AgentIdentity


class RequestBranchRegistry:
    """Map external families to independent cache owners without inventing blocking edges."""

    def __init__(self) -> None:
        self._families: dict[str, set[ProgramRef]] = {}
        self._external: dict[ProgramRef, str] = {}
        self._temporary: set[ProgramRef] = set()
        self._internal_ids: set[str] = set()
        self._free_temporary: list[str] = []

    def resolve(self, metadata: AgentIdentity, *, default_busy: bool) -> tuple[AgentIdentity, bool]:
        """Reuse an idle default owner; isolate each overlapping accepted request."""
        if metadata.program_id in self._internal_ids:
            raise ValueError("external Program identity collides with an internal request branch")
        if not default_busy:
            return metadata, False
        internal = self._free_temporary.pop() if self._free_temporary else f"mori-overlap:{uuid.uuid4().hex}"
        self._internal_ids.add(internal)
        return replace(metadata, program_id=internal, parent_program_id=None, blocks_parent=False), True

    def bind(self, external: str, ref: ProgramRef, *, temporary: bool) -> None:
        """Index a materialized exact generation under its external family."""
        self._families.setdefault(external, set()).add(ref)
        self._external[ref] = external
        if temporary:
            self._temporary.add(ref)

    def temporary(self, ref: ProgramRef) -> bool:
        """Return whether protection ends when this generation has no accepted work."""
        return ref in self._temporary

    def family(self, external: str) -> tuple[ProgramRef, ...]:
        """Return live branch generations; callers must not infer an ordering."""
        return tuple(self._families.get(external, ()))

    def release(self, ref: ProgramRef) -> None:
        """Recycle completed slots, retaining generation tombstones only for peak occupancy."""
        external = self._external.pop(ref, None)
        if ref in self._temporary:
            self._free_temporary.append(ref.program_id)
        self._temporary.discard(ref)
        if external is not None:
            family = self._families[external]
            family.discard(ref)
            if not family:
                del self._families[external]
