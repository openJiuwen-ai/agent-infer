# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Versioned contracts for independently finalized correctness evaluations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CorrectnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskCorrectness(CorrectnessModel):
    instance_id: str
    status: Literal["resolved", "unresolved", "empty_patch", "error"]
    patch_sha256: str | None
    reason: str | None = None
    report_path: str | None = None
    log_path: str | None = None
    environment_failure: bool = False


class CorrectnessCounts(CorrectnessModel):
    total: int = Field(gt=0)
    resolved: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    empty_patch: int = Field(ge=0)
    error: int = Field(ge=0)

    @model_validator(mode="after")
    def check_total(self):
        if self.total != self.resolved + self.unresolved + self.empty_patch + self.error:
            raise ValueError("correctness counts do not sum to total")
        return self


class EvaluationSummary(CorrectnessModel):
    schema_version: Literal["1"] = "1"
    source_run_id: str
    evaluation_id: str
    scope: Literal["swebench_verified/test/completed_tasks"] = "swebench_verified/test/completed_tasks"
    counts: CorrectnessCounts
    resolved_rate: float = Field(ge=0, le=1)
    manifest_path: Literal["evaluation_manifest.json"] = "evaluation_manifest.json"
    task_results_path: Literal["task_results.jsonl"] = "task_results.jsonl"

    @model_validator(mode="after")
    def check_rate(self):
        if self.resolved_rate != self.counts.resolved / self.counts.total:
            raise ValueError("resolved_rate does not match counts")
        return self
