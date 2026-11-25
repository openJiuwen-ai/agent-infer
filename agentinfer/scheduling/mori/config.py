# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Configuration for ThunderAgent MORI Program scheduling."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MoriConfig:
    interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("MORI schedule interval must be finite and positive")
