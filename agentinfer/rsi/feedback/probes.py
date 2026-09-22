# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Future collection boundary; no profiler or hardware integration is connected."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .models import Category, FeedbackRecord, Verdict
from .taxonomy import Backend, Layer


class Probe(Protocol):
    """Collect external evidence; implementations must declare unavailable data."""

    def collect(self, scope: Mapping[str, str]) -> Sequence[FeedbackRecord]: ...


@dataclass(frozen=True)
class UnavailableProbe:
    """Represent an unimplemented probe without fabricating successful metrics."""

    check_id: str
    hypothesis_id: str
    layer: Layer
    category: Category
    metric: str
    unit: str

    def collect(self, scope: Mapping[str, str]) -> Sequence[FeedbackRecord]:
        dimensions = {
            key: value for key, value in scope.items() if key not in ("candidate_id", "baseline_id", "backend")
        }
        return [
            FeedbackRecord(
                check_id=self.check_id,
                hypothesis_id=self.hypothesis_id,
                candidate_id=scope["candidate_id"],
                baseline_id=scope["baseline_id"],
                backend=Backend(scope["backend"]),
                layer=self.layer,
                category=self.category,
                verdict=Verdict.UNAVAILABLE,
                metric=self.metric,
                value=None,
                unit=self.unit,
                scope=dimensions,
                diagnostic=False,
                evidence_refs=(),
                next_test="Connect and validate this probe on the pinned backend before collecting real evidence.",
            )
        ]
