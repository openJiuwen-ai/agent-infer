"""Host-neutral Program scheduling contracts shared by AgentCache and AgentRouter."""

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
from agentinfer.scheduling.identity import AgentIdentity, MetadataError, encode_agent_identity, parse_agent_identity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
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
    "BackendInfo",
    "BackendPoolInfo",
    "DispatchTarget",
    "DpRankInfo",
    "MetadataError",
    "ProgramLifecycle",
    "ProgramRef",
    "ProgramState",
    "ProgramStatus",
    "ProgramTokenObservation",
    "ProgramView",
    "SchedulingEvent",
    "SchedulingEventKind",
    "SchedulingSnapshot",
    "SchedulingStrategy",
    "SharedPrefixAttribution",
    "StrategyDiagnostic",
    "StrategyFactors",
    "TokenObservationSource",
    "TransitionController",
    "TransitionKind",
    "TransitionRequest",
    "TransitionResult",
    "encode_agent_identity",
    "parse_agent_identity",
]
