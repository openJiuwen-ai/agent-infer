# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Configuration for the supported, isolated SWE-bench adapter."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from ..config import StrictModel


class EvaluationConfig(StrictModel):
    harness_python: Path
    expected_harness_version: Literal["4.1.0"]
    max_workers: int = Field(2, ge=1)
    timeout_seconds: int = Field(1800, ge=1)
    overall_timeout_seconds: int = Field(7200, ge=1)
    cleanup_timeout_seconds: int = Field(30, ge=1)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid evaluation YAML: {exc}") from exc
    config = EvaluationConfig.model_validate(payload)
    # Do not resolve the executable's symlink: that would bypass its virtualenv.
    if not config.harness_python.is_absolute():
        config.harness_python = path.absolute().parent / config.harness_python
    return config
