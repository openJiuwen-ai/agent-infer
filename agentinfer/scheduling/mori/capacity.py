# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Per-plan union accounting. Physical allocation remains owned by the backend."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from agentinfer.scheduling.domain import ProgramRef


class FootprintProvider(Protocol):
    """Supply current-prompt native block IDs; never promise physical readiness.

    Reads are side-effect free and stable for the duration of one serialized
    scheduling hook. Providers must not depend on repeated reads of one tier or
    vary an exception by call count within that hook.
    """

    def footprint(self, program: ProgramRef, tier: str) -> frozenset[int]: ...

    def invalidate_footprint(self, program: ProgramRef) -> None: ...

    def replace_context(self, program: ProgramRef) -> None: ...


@dataclass(frozen=True)
class CapacityClaim:
    """Logical token promise containing a proven subset of unique native blocks."""

    tokens: int
    blocks: frozenset[int] = frozenset()


class CapacityLedger:
    """Count known native blocks once and charge all unproven context conservatively.

    A claim contains only valid blocks from the current prompt version. Unknown
    blocks and speculative decode headroom are private promises. Marginal removal
    counts a shared block only after its last protection claim disappears.
    """

    def __init__(self, quantum: int, *, fast: bool = False) -> None:
        self.quantum = quantum
        self.fast = fast
        self.claims: dict[ProgramRef, CapacityClaim] = {}
        self.owners: Counter[int] = Counter()
        self._shared: set[int] = set()
        self.used = 0

    def incremental(self, claim: CapacityClaim) -> int:
        """Return the additional tokens charged if a new owner takes this claim."""
        if self.fast:
            return claim.tokens - self.quantum * len(claim.blocks & self.owners.keys())
        return claim.tokens - self.quantum * sum(key in self.owners for key in claim.blocks)

    def add(self, ref: ProgramRef, claim: CapacityClaim) -> None:
        """Insert or replace one owner claim, updating shared block multiplicities."""
        if ref in self.claims:
            self.remove(ref)
        if self.fast:
            overlap = claim.blocks & self.owners.keys()
            self.used += claim.tokens - self.quantum * len(overlap)
            self._shared.update(overlap)
        else:
            self.used += self.incremental(claim)
        self.claims[ref] = claim
        self.owners.update(claim.blocks)

    def reclaimable(self, ref: ProgramRef) -> int:
        """Return marginal logical relief, excluding blocks still claimed by peers."""
        claim = self.claims[ref]
        if self.fast:
            return claim.tokens - self.quantum * len(claim.blocks & self._shared)
        return claim.tokens - self.quantum * sum(self.owners[key] > 1 for key in claim.blocks)

    def remove(self, ref: ProgramRef) -> int:
        """Remove a claim and return marginal relief; unknown owners release zero."""
        if ref not in self.claims:
            return 0
        freed = self.reclaimable(ref)
        claim = self.claims.pop(ref)
        if self.fast:
            # Only previously shared keys can cross 2 -> 1. Unique keys never
            # enter this index, and 3+ owners must remain shared after removal.
            newly_unique = {key for key in claim.blocks & self._shared if self.owners[key] == 2}
            self._shared.difference_update(newly_unique)
        self.used -= freed
        for key in claim.blocks:
            self.owners[key] -= 1
            if not self.owners[key]:
                del self.owners[key]
        return freed

    def used_after_swap(self, restore_claim: CapacityClaim, remove_ref: ProgramRef) -> int:
        """Return exact usage after restoring one claim and removing one owner.

        ``restore_claim`` belongs to an owner that is absent from this planning
        ledger, while ``remove_ref`` must still be present.  The computation is
        deliberately read-only: it is the hot-path equivalent of
        ``copy(); add(restore); remove(remove_ref)`` without copying the owner
        map for every replacement trial.
        """
        incoming_claim = self.claims[remove_ref]
        if self.fast:
            restore_incremental = restore_claim.tokens - self.quantum * len(restore_claim.blocks & self.owners.keys())
            retained = (incoming_claim.blocks & self._shared) | (incoming_claim.blocks & restore_claim.blocks)
            return self.used + restore_incremental - incoming_claim.tokens + self.quantum * len(retained)
        restore_incremental = restore_claim.tokens - self.quantum * sum(
            block in self.owners for block in restore_claim.blocks
        )
        remove_relief_after_restore = incoming_claim.tokens - self.quantum * sum(
            self.owners[block] + int(block in restore_claim.blocks) > 1 for block in incoming_claim.blocks
        )
        return self.used + restore_incremental - remove_relief_after_restore

    def copy(self) -> CapacityLedger:
        """Copy accounting for a speculative plan without modifying committed claims."""
        other = CapacityLedger(self.quantum, fast=self.fast)
        other.claims = self.claims.copy()
        other.owners = self.owners.copy()
        other._shared = self._shared.copy()
        other.used = self.used
        return other


class ReliefProjection:
    """Exact union capacity after removing a prefix of ranked Acting owners.

    Build once per scheduling round. A range-add/point-query Fenwick tree
    accounts for new admissions without rescanning every victim per candidate.
    Shared blocks stop providing relief once a newly admitted owner needs them.
    These are hypothetical claims; only accepted transitions release capacity.
    """

    def __init__(self, ledger: CapacityLedger, victims: tuple[ProgramRef, ...]) -> None:
        self.quantum = ledger.quantum
        positions = {ref: i + 1 for i, ref in enumerate(victims)}
        self._last_owner: dict[int, int] = {}
        self._permanent: set[int] = set()
        for ref, claim in ledger.claims.items():
            if ref not in positions:
                self._permanent.update(claim.blocks)
            else:
                for block in claim.blocks:
                    self._last_owner[block] = max(self._last_owner.get(block, 0), positions[ref])
        projected = ledger.copy()
        self._used = [projected.used]
        for ref in victims:
            projected.remove(ref)
            self._used.append(projected.used)
        self._deltas = [0] * (len(self._used) + 1)

    def _add_from(self, start: int, tokens: int) -> None:
        index = start + 1
        while index < len(self._deltas):
            self._deltas[index] += tokens
            index += index & -index

    def _present(self, block: int, removed: int) -> bool:
        return block in self._permanent or self._last_owner.get(block, 0) > removed

    def required(self, removed: int, claim: CapacityClaim) -> int:
        """Compute total union commitment including a candidate after a prefix."""
        total = self._used[removed] + claim.tokens
        index = removed + 1
        while index:
            total += self._deltas[index]
            index -= index & -index
        return total - self.quantum * sum(self._present(block, removed) for block in claim.blocks)

    def admit(self, claim: CapacityClaim) -> None:
        """Protect an actually admitted claim in every later hypothetical prefix."""
        self._add_from(0, claim.tokens)
        for block in claim.blocks:
            if block in self._permanent:
                self._add_from(0, -self.quantum)
            elif (last := self._last_owner.get(block, 0)) > 0:
                self._add_from(0, -self.quantum)
                self._add_from(last, self.quantum)
            self._permanent.add(block)
