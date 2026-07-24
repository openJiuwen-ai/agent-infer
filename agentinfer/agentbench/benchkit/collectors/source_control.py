"""Collect source-control evidence without writing artifacts."""

import subprocess
from pathlib import Path

from ..metrics.schema import EvidenceCapture


def collect_source_control(source_path: Path) -> EvidenceCapture:
    """Capture the imported source checkout commit and dirty state."""

    location = source_path.parent if source_path.is_file() else source_path
    try:
        root = Path(
            subprocess.run(
                ["git", "-C", str(location), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        )
        return EvidenceCapture(
            "source_control",
            None,
            True,
            None,
            {"root": str(root), "commit": commit, "dirty": dirty},
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return EvidenceCapture("source_control", None, False, str(exc), {})
