"""Collect source-control evidence without writing artifacts."""

import subprocess
from pathlib import Path

from ..metrics.schema import EvidenceCapture


def collect_source_control(repo: Path) -> EvidenceCapture:
    """Capture the repository commit and dirty state within bounded Git calls."""

    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        )
        metadata = {"commit": commit, "dirty": dirty}
        return EvidenceCapture("source_control", None, True, None, metadata)
    except (subprocess.SubprocessError, OSError) as exc:
        return EvidenceCapture("source_control", None, False, str(exc), {})
