# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Current-context native block evidence with removal-safe reverse indexes."""

from __future__ import annotations

from agentinfer.scheduling.domain import ProgramRef


class FootprintBook:
    """A pool removes proof before reusing an ID; prompt replacement clears old proof."""

    def __init__(self) -> None:
        self.enabled = False
        self._by_owner: dict[tuple[str, ProgramRef], set[int]] = {}
        self._by_block: dict[tuple[str, int], set[ProgramRef]] = {}
        self._cache_snapshots = False
        self._snapshots: dict[tuple[str, ProgramRef], frozenset[int]] = {}
        self.snapshot_hits = 0
        self.snapshot_misses = 0

    def configure_snapshot_cache(self, enabled: bool) -> None:
        """Cache immutable evidence only until the next native ownership delta."""
        self._cache_snapshots = enabled
        self._snapshots.clear()
        self.snapshot_hits = self.snapshot_misses = 0

    def add(self, tier: str, block: int, owner: ProgramRef) -> None:
        """Record a native owner delta when single-group accounting is enabled."""
        if not self.enabled or block == 0:
            return
        key = (tier, owner)
        blocks = self._by_owner.setdefault(key, set())
        if self._cache_snapshots:
            if block not in blocks:
                blocks.add(block)
                self._snapshots.pop(key, None)
        else:
            blocks.add(block)
        self._by_block.setdefault((tier, block), set()).add(owner)

    def remove_block(self, tier: str, block: int) -> None:
        """Invalidate all evidence for a block before native reuse of its ID."""
        for owner in self._by_block.pop((tier, block), ()):
            self._snapshots.pop((tier, owner), None)
            blocks = self._by_owner[(tier, owner)]
            blocks.discard(block)
            if not blocks:
                del self._by_owner[(tier, owner)]

    def invalidate(self, owner: ProgramRef) -> None:
        """Drop both tiers of proof on prompt replacement or generation release."""
        for tier in ("gpu", "cpu"):
            self._snapshots.pop((tier, owner), None)
            for block in self._by_owner.pop((tier, owner), ()):
                owners = self._by_block[(tier, block)]
                owners.discard(owner)
                if not owners:
                    del self._by_block[(tier, block)]

    def get(self, owner: ProgramRef, tier: str) -> frozenset[int]:
        """Return immutable evidence; absence requires conservative accounting."""
        if self._cache_snapshots:
            key = (tier, owner)
            if (snapshot := self._snapshots.get(key)) is not None:
                self.snapshot_hits += 1
                return snapshot
            self.snapshot_misses += 1
            snapshot = frozenset(self._by_owner.get(key, ()))
            # Unknown or released owners need no retained empty cache entry.
            if snapshot:
                self._snapshots[key] = snapshot
            return snapshot
        return frozenset(self._by_owner.get((tier, owner), ()))

    def clear(self, tier: str | None = None) -> None:
        """Drop proof for one reset native pool, or all pools at shutdown."""
        if tier is None:
            self._snapshots.clear()
            self._by_owner.clear()
            self._by_block.clear()
        else:
            for key in tuple(self._snapshots):
                if key[0] == tier:
                    self._snapshots.pop(key)
            for pool, block in tuple(self._by_block):
                if pool == tier:
                    self.remove_block(pool, block)
