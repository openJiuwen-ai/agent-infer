# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Incremental class-partitioned LRU index used by both MORI cache tiers."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from agentinfer.scheduling.cache_placement import CacheClass
from agentinfer.scheduling.domain import ProgramRef

EntryT = TypeVar("EntryT", bound=Hashable)


class StaleRepairLimitExceeded(RuntimeError):
    """Legacy exception retained for imports; stale heaps now rebuild in place."""


@dataclass
class _Entry(Generic[EntryT]):
    key: EntryT
    owners: frozenset[ProgramRef]
    cache_class: CacheClass
    native_lru_seq: int
    version: int
    ready: bool


class TypedLRUIndex(Generic[EntryT]):
    """Exact class order plus native LRU order without proposal-time scans.

    Tier changes eagerly visit only the reverse-indexed entries of the changed
    Program. That costs ``O(B_p log N)`` for a Program owning ``B_p`` entries;
    claiming both exact ordering and ``O(1)`` class change would be incorrect.
    Proposal work is proportional to selected entries, protected heads, and a
    bounded number of stale heap pops, followed by a class-local rebuild when
    necessary. All mutations and reservations must run on the engine thread.
    """

    def __init__(
        self,
        *,
        eviction_order: tuple[CacheClass, ...],
        protection_rank: Mapping[CacheClass, int],
        class_for_program: Callable[[ProgramRef], CacheClass],
        max_stale_repairs_per_proposal: int = 4096,
    ) -> None:
        if set(eviction_order) != set(CacheClass):
            raise ValueError("typed LRU eviction_order must contain every cache class exactly once")
        if set(protection_rank) != set(CacheClass):
            raise ValueError("typed LRU protection_rank must define every cache class")
        if max_stale_repairs_per_proposal <= 0:
            raise ValueError("max_stale_repairs_per_proposal must be positive")
        self.eviction_order = eviction_order
        self._protection_rank = dict(protection_rank)
        self._class_for_program = class_for_program
        self._max_stale_repairs = max_stale_repairs_per_proposal
        self._entries: dict[EntryT, _Entry[EntryT]] = {}
        self._program_entries: dict[ProgramRef, set[EntryT]] = defaultdict(set)
        self._class_entries: dict[CacheClass, set[EntryT]] = {cache_class: set() for cache_class in CacheClass}
        self._ready_counts: Counter[CacheClass] = Counter()
        self._heaps: dict[CacheClass, list[tuple[int, int, int, EntryT]]] = {
            cache_class: [] for cache_class in CacheClass
        }
        self._ticket = 0
        self.stats: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def program_entry_count(self) -> int:
        return sum(len(entries) for entries in self._program_entries.values())

    def upsert(
        self,
        key: EntryT,
        *,
        owners: Iterable[ProgramRef],
        native_lru_seq: int,
        ready: bool,
    ) -> None:
        """Insert or replace one native entry observation."""
        normalized_owners = frozenset(owners)
        current = self._entries.get(key)
        version = 0 if current is None else current.version + 1
        if current is not None:
            self._class_entries[current.cache_class].discard(key)
            if current.ready:
                self._ready_counts[current.cache_class] -= 1
            self._detach_owners(key, current.owners - normalized_owners)
        self._attach_owners(key, normalized_owners - (current.owners if current is not None else frozenset()))
        entry = _Entry(
            key=key,
            owners=normalized_owners,
            cache_class=self._strongest_class(normalized_owners),
            native_lru_seq=native_lru_seq,
            version=version,
            ready=ready,
        )
        self._entries[key] = entry
        self._class_entries[entry.cache_class].add(key)
        if ready:
            self._ready_counts[entry.cache_class] += 1
            self._push(entry)
        self.stats["upserts"] += 1

    def update_owners(self, key: EntryT, owners: Iterable[ProgramRef]) -> None:
        """Replace owners while preserving native recency and readiness."""
        current = self._entries.get(key)
        if current is None:
            raise KeyError(key)
        self.upsert(
            key,
            owners=owners,
            native_lru_seq=current.native_lru_seq,
            ready=current.ready,
        )

    def touch(self, key: EntryT, native_lru_seq: int) -> None:
        """Commit a native recency change without changing class."""
        current = self._entries.get(key)
        if current is None:
            return
        if native_lru_seq < current.native_lru_seq:
            raise ValueError("native LRU sequence must be monotonic per entry")
        current.native_lru_seq = native_lru_seq
        current.version += 1
        if current.ready:
            self._push(current)
        self.stats["touches"] += 1

    def mark_ready(self, key: EntryT, ready: bool) -> None:
        """Commit a native evictability transition."""
        current = self._entries.get(key)
        if current is None or current.ready is ready:
            return
        self._ready_counts[current.cache_class] += 1 if ready else -1
        current.ready = ready
        current.version += 1
        if ready:
            self._push(current)
        self.stats["ready_changes"] += 1

    def remove(self, key: EntryT) -> None:
        """Remove one entry after native eviction or cache reset."""
        current = self._entries.pop(key, None)
        if current is None:
            return
        self._class_entries[current.cache_class].discard(key)
        if current.ready:
            self._ready_counts[current.cache_class] -= 1
        self._detach_owners(key, current.owners)
        self.stats["removes"] += 1

    def clear(self) -> None:
        """Clear live state; stale heap records are discarded eagerly on reset."""
        self._entries.clear()
        self._program_entries.clear()
        for entries in self._class_entries.values():
            entries.clear()
        self._ready_counts.clear()
        for heap in self._heaps.values():
            heap.clear()
        self.stats["clears"] += 1

    def reclassify_program(self, program: ProgramRef) -> int:
        """Eagerly move only entries reverse-indexed to ``program``.

        Returns the number of live entries whose strongest-owner class changed.
        """
        changed = 0
        for key in tuple(self._program_entries.get(program, ())):
            current = self._entries.get(key)
            if current is None:
                continue
            cache_class = self._strongest_class(current.owners)
            if cache_class is current.cache_class:
                continue
            previous_class = current.cache_class
            self._class_entries[previous_class].discard(key)
            self._class_entries[cache_class].add(key)
            if current.ready:
                self._ready_counts[previous_class] -= 1
                self._ready_counts[cache_class] += 1
            current.cache_class = cache_class
            current.version += 1
            if current.ready:
                self._push(current)
            changed += 1
        self.stats["program_reclassifications"] += 1
        self.stats["entries_reclassified"] += changed
        return changed

    def release_program(self, program: ProgramRef) -> int:
        """Remove one generation as owner and reclassify its shared entries."""
        keys = tuple(self._program_entries.pop(program, ()))
        changed = 0
        for key in keys:
            current = self._entries.get(key)
            if current is None or program not in current.owners:
                continue
            remaining = current.owners - {program}
            self.upsert(
                key,
                owners=remaining,
                native_lru_seq=current.native_lru_seq,
                ready=current.ready,
            )
            changed += 1
        self.stats["program_releases"] += 1
        return changed

    def ready_count(self, classes: Iterable[CacheClass]) -> int:
        """Return maintained evictable counts without scanning cache entries."""
        return sum(self._ready_counts[cache_class] for cache_class in classes)

    def propose_exact(
        self, count: int, protected: Iterable[EntryT] = (), *, allowed_classes: Iterable[CacheClass] | None = None
    ) -> tuple[EntryT, ...] | None:
        """Return exactly ``count`` ordered victims without mutating live entries."""
        with self.reserve_exact(count, protected, allowed_classes=allowed_classes) as proposed:
            return proposed

    @contextmanager
    def reserve_exact(
        self,
        count: int,
        protected: Iterable[EntryT] = (),
        *,
        allowed_classes: Iterable[CacheClass] | None = None,
    ) -> Iterator[tuple[EntryT, ...] | None]:
        """Reserve victims until the caller removes them or leaves the scope.

        Entries that remain live are restored on exit. This keeps read-only
        HBM proposals and rejected CPU commits non-mutating while avoiding a
        pop/push cycle for CPU victims removed inside the reservation.
        """
        if count < 0:
            raise ValueError("proposal count must be non-negative")
        if count == 0:
            yield ()
            return
        allowed = set(self.eviction_order if allowed_classes is None else allowed_classes)
        protected_set = set(protected)
        selected: list[tuple[int, int, int, EntryT]] = []
        selected_keys: set[EntryT] = set()
        deferred: list[tuple[CacheClass, tuple[int, int, int, EntryT]]] = []
        stale_repairs = 0
        proposed: tuple[EntryT, ...] | None = None
        try:
            for cache_class in self.eviction_order:
                if cache_class not in allowed:
                    continue
                heap = self._heaps[cache_class]
                while heap and len(selected) < count:
                    item = heapq.heappop(heap)
                    if not self._is_current(cache_class, item):
                        stale_repairs += 1
                        if stale_repairs > self._max_stale_repairs:
                            self.stats["stale_repair_limit"] += 1
                            # Keep already popped live records out of the rebuilt
                            # heap: the reservation restores those exactly once.
                            excluded = selected_keys | {record[3] for _, record in deferred}
                            self._rebuild_heap(cache_class, excluded)
                            heap = self._heaps[cache_class]
                        continue
                    key = item[3]
                    if key in selected_keys:
                        continue
                    if key in protected_set:
                        deferred.append((cache_class, item))
                        continue
                    selected.append(item)
                    selected_keys.add(key)
                if len(selected) == count:
                    break
            self.stats["proposals"] += 1
            self.stats["stale_repairs"] += stale_repairs
            if len(selected) != count:
                self.stats["insufficient_proposals"] += 1
            else:
                self.stats["proposed_entries"] += count
                proposed = tuple(item[3] for item in selected)
            yield proposed
        finally:
            for cache_class, item in deferred:
                heapq.heappush(self._heaps[cache_class], item)
            for item in selected:
                entry = self._entries.get(item[3])
                if entry is not None and self._is_current(entry.cache_class, item):
                    heapq.heappush(self._heaps[entry.cache_class], item)

    def owners(self, key: EntryT) -> frozenset[ProgramRef]:
        """Return immutable owners for tests and adapter diagnostics."""
        current = self._entries.get(key)
        return current.owners if current is not None else frozenset()

    def entries_for_program(self, program: ProgramRef) -> tuple[EntryT, ...]:
        """Return the reverse-indexed live entries for one exact generation."""
        return tuple(self._program_entries.get(program, ()))

    def cache_class(self, key: EntryT) -> CacheClass | None:
        """Return the current strongest-owner class."""
        current = self._entries.get(key)
        return current.cache_class if current is not None else None

    def _strongest_class(self, owners: Iterable[ProgramRef]) -> CacheClass:
        classes = (self._class_for_program(owner) for owner in owners)
        return max(classes, key=self._protection_rank.__getitem__, default=CacheClass.INACTIVE)

    def _push(self, entry: _Entry[EntryT]) -> None:
        self._ticket += 1
        heapq.heappush(
            self._heaps[entry.cache_class],
            (entry.native_lru_seq, self._ticket, entry.version, entry.key),
        )
        self._maybe_compact(entry.cache_class)

    def _maybe_compact(self, cache_class: CacheClass) -> None:
        """Bound stale heap growth with an amortized, class-local rebuild."""
        heap = self._heaps[cache_class]
        ready = self._ready_counts[cache_class]
        if len(heap) < max(64, 2 * ready):
            return
        self._rebuild_heap(cache_class)

    def _rebuild_heap(self, cache_class: CacheClass, excluded: Iterable[EntryT] = ()) -> None:
        """Discard stale records, preserving any live records held by a reservation."""
        excluded_keys = set(excluded)
        rebuilt = [
            (entry.native_lru_seq, self._new_ticket(), entry.version, entry.key)
            for key in self._class_entries[cache_class]
            if key not in excluded_keys and (entry := self._entries.get(key)) is not None and entry.ready
        ]
        heapq.heapify(rebuilt)
        self._heaps[cache_class] = rebuilt
        self.stats["heap_compactions"] += 1
        self.stats["heap_compaction_entries"] += len(rebuilt)

    def _new_ticket(self) -> int:
        self._ticket += 1
        return self._ticket

    def _is_current(self, cache_class: CacheClass, item: tuple[int, int, int, EntryT]) -> bool:
        sequence, _, version, key = item
        current = self._entries.get(key)
        return (
            current is not None
            and current.ready
            and current.cache_class is cache_class
            and current.native_lru_seq == sequence
            and current.version == version
        )

    def _attach_owners(self, key: EntryT, owners: Iterable[ProgramRef]) -> None:
        for owner in owners:
            self._program_entries[owner].add(key)

    def _detach_owners(self, key: EntryT, owners: Iterable[ProgramRef]) -> None:
        for owner in owners:
            entries = self._program_entries.get(owner)
            if entries is None:
                continue
            entries.discard(key)
            if not entries:
                self._program_entries.pop(owner, None)
