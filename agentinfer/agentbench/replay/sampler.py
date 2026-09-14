# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Stable shuffled-cycle sampling for Replay source sessions."""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict

from .schema import ReplaySession, SampledSession

_REPLAY_NAMESPACE = uuid.UUID("08bcf83c-e28d-5cb6-8677-020bdfb74775")


def _cycle_key(seed: int, cycle_index: int, session_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{cycle_index}\0{session_id}\0sampler/v1".encode()).hexdigest()
    return digest, session_id


def sample_replay_sessions(
    sessions: tuple[ReplaySession, ...],
    *,
    total_tasks: int,
    seed: int,
    plan_namespace: str,
) -> tuple[SampledSession, ...]:
    """Sample complete hash-shuffled cycles with isolated Runtime sessions."""

    replayable = sorted(
        (session for session in sessions if session.replayable),
        key=lambda session: session.source_session_id,
    )
    if not replayable:
        raise ValueError("the Replay trace has no usable sessions")

    selected: list[ReplaySession] = []
    cycle_index = 0
    while len(selected) < total_tasks:
        cycle = sorted(
            replayable,
            key=lambda session: _cycle_key(
                seed,
                cycle_index,
                session.source_session_id,
            ),
        )
        selected.extend(cycle)
        cycle_index += 1

    ordinals: dict[str, int] = defaultdict(int)
    sampled = []
    for task_index, source in enumerate(selected[:total_tasks]):
        sample_ordinal = ordinals[source.source_session_id]
        ordinals[source.source_session_id] += 1
        runtime_session_id = str(
            uuid.uuid5(
                _REPLAY_NAMESPACE,
                (f"{plan_namespace}\0{source.source_session_id}\0{sample_ordinal}"),
            )
        )
        sampled.append(
            SampledSession(
                task_index=task_index,
                sample_ordinal=sample_ordinal,
                runtime_session_id=runtime_session_id,
                source=source,
            )
        )
    return tuple(sampled)
