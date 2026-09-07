# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Host-neutral scheduling contracts shared by AgentCache and AgentRouter."""

from agentinfer.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DispatchTarget, DpRankInfo
from agentinfer.scheduling.domain import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    ProgramTokenObservation,
    ProgramView,
    SharedPrefixAttribution,
    TokenObservationSource,
)
from agentinfer.scheduling.events import SchedulingEvent, SchedulingEventKind, StrategyDiagnostic
from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.headers import AgentRequestIdentity, select_router_headers
from agentinfer.scheduling.identity import AgentIdentity, MetadataError, encode_agent_identity, parse_agent_identity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.observability import SchedulerObservabilityConfig
from agentinfer.scheduling.program_registry import ProgramRegistry, StaleProgramReferenceError
from agentinfer.scheduling.program_runtime import RuntimeProgram
from agentinfer.scheduling.request_pool import RequestPool, RequestPoolEntry, RequestPoolStatus
from agentinfer.scheduling.runtime import ProgramScheduler
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from agentinfer.scheduling.strategy import SchedulingStrategy
from agentinfer.scheduling.transitions import (
    TransitionController,
    TransitionKind,
    TransitionRequest,
    TransitionResult,
)

__all__ = [
    "AdmissionDisposition",
    "AdmissionOutcome",
    "AgentIdentity",
    "AgentRequestIdentity",
    "BackendInfo",
    "BackendPoolInfo",
    "DispatchTarget",
    "DpRankInfo",
    "MetadataError",
    "ProgramLifecycle",
    "ProgramRegistry",
    "ProgramRef",
    "ProgramState",
    "ProgramStatus",
    "ProgramTokenObservation",
    "ProgramView",
    "ProgramScheduler",
    "RequestPool",
    "RequestPoolEntry",
    "RequestPoolStatus",
    "RuntimeProgram",
    "SchedulingEvent",
    "SchedulingEventKind",
    "SchedulingSnapshot",
    "SchedulingStrategy",
    "SchedulerObservabilityConfig",
    "SharedPrefixAttribution",
    "StrategyDiagnostic",
    "StrategyFactors",
    "StaleProgramReferenceError",
    "TokenObservationSource",
    "TransitionController",
    "TransitionKind",
    "TransitionRequest",
    "TransitionResult",
    "encode_agent_identity",
    "parse_agent_identity",
    "select_router_headers",
]
