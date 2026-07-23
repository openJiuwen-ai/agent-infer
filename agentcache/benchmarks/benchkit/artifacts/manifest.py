"""Canonical run manifest artifacts."""

from dataclasses import asdict, replace
from datetime import datetime
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from ..common import utc_now


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RunManifest:
    schema_version: Literal["1"]
    run_id: str
    status: Literal["running", "completed", "failed"]
    created_at: datetime
    finished_at: datetime | None
    config: dict[str, object]
    evidence: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["created_at"] = self.created_at.isoformat()
        value["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return value


def build_run_manifest(run_id: str, config: dict[str, object]) -> RunManifest:
    return RunManifest("1", run_id, "running", utc_now(), None, config, ())


def finalize_run_manifest(
    manifest: RunManifest,
    evidence: tuple[dict[str, object], ...],
    *,
    status: Literal["completed", "failed"] = "completed",
) -> RunManifest:
    return replace(manifest, status=status, finished_at=utc_now(), evidence=evidence)
