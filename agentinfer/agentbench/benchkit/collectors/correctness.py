# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Import external correctness evidence without writing artifacts."""

import json
from pathlib import Path

from ..metrics.schema import EvidenceCapture


def load_correctness_artifact(path: Path) -> EvidenceCapture:
    """Load an external evaluator JSON artifact and expose its raw metadata."""

    if not path.is_file():
        return EvidenceCapture("correctness", None, False, "artifact missing", {})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return EvidenceCapture("correctness", path, False, str(exc), {})
    if not isinstance(payload, dict):
        return EvidenceCapture("correctness", path, False, "artifact must contain a JSON object", {})
    return EvidenceCapture("correctness", path, True, None, payload)
