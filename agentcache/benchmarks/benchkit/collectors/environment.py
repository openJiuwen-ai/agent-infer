"""Collect host and Python environment evidence without writing artifacts."""

import platform
import sys

from ..metrics.schema import EvidenceCapture


def collect_environment() -> EvidenceCapture:
    """Capture host and Python metadata as in-memory evidence."""

    metadata = {"platform": platform.platform(), "hostname": platform.node(), "python_version": sys.version}
    return EvidenceCapture("environment", None, True, None, metadata)
