# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Logical cache classes shared by placement strategies and native adapters."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from agentinfer.scheduling.domain import ProgramRef


class CacheClass(str, Enum):
    """Committed protection class for cache entries owned by a Program."""

    INACTIVE = "inactive"
    IDLE = "idle"
    BUSY = "busy"


class ProgramCacheClassObserver(Protocol):
    """Observer notified after a strategy commits a Program class change."""

    def on_cache_class_changed(
        self,
        program: ProgramRef,
        previous: CacheClass,
        current: CacheClass,
    ) -> None:
        """Reclassify entries owned by ``program`` without changing recency."""

    def on_program_released(self, program: ProgramRef) -> None:
        """Remove one released generation from every ownership index."""


class ProgramCacheClassProvider(Protocol):
    """Read-only Program class surface consumed by cache-management adapters."""

    @property
    def cache_class_epoch(self) -> int:
        """Return the monotonic committed-class epoch."""

    def cache_class_for_program(self, program: ProgramRef) -> CacheClass:
        """Return the committed class for one exact Program generation."""

    def subscribe_cache_class_observer(self, observer: ProgramCacheClassObserver) -> None:
        """Subscribe an engine-thread observer to committed class changes."""

    def unsubscribe_cache_class_observer(self, observer: ProgramCacheClassObserver) -> None:
        """Remove a previously subscribed observer during scheduler shutdown."""
