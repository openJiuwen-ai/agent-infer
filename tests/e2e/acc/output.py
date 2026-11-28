# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Manage accuracy output directory names and export score summaries."""

import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def output_dir_name(feature: str) -> str:
    """Return a feature, timestamp and unique suffix for a run/result directory pair."""
    return f"{feature}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"


def export_correctness(evaluation: Path, result_dir: Path) -> None:
    """Copy the score summary; the same-named run directory retains the complete evidence."""
    result_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(evaluation, result_dir / "correctness.json")
    print(f"[correctness] result={result_dir / 'correctness.json'}", flush=True)
