# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Aggregate correctness without dropping empty patches or evaluation errors."""

from collections import Counter


def correctness_metrics(results: list[dict], instance_ids: list[str]) -> dict:
    ids = [result["instance_id"] for result in results]
    if not instance_ids or len(set(ids)) != len(ids) or set(ids) != set(instance_ids):
        raise ValueError("correctness results must cover each selected task exactly once")
    counts = Counter(result["status"] for result in results)
    statuses = ("resolved", "unresolved", "empty_patch", "error")
    if set(counts) - set(statuses):
        raise ValueError("unknown correctness status")
    return {
        "counts": {"total": len(ids), **{status: counts[status] for status in statuses}},
        "resolved_rate": counts["resolved"] / len(ids),
    }
