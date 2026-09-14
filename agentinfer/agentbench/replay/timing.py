# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Deterministic same-agent interval models for Trace Replay."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

from .config import ReplayConfig
from .schema import ReplayRequest

_ANCHOR_PROBABILITIES = (0.50, 0.95, 0.99)
_ANCHOR_NAMES = ("p50_seconds", "p95_seconds", "p99_seconds")
_ANCHOR_Z = tuple(NormalDist().inv_cdf(probability) for probability in _ANCHOR_PROBABILITIES)
_U_CLAMP = 1e-12
_CDF_VERSION = "python-statistics-normaldist/v1"
_MAX_ANCHOR_RESIDUAL_RATIO = 0.02


@dataclass(frozen=True)
class IntervalModel:
    """Auditable metadata for trace or three-quantile Lognormal intervals."""

    mode: str
    fit_version: str
    mu: float | None = None
    sigma: float | None = None
    anchors: dict[str, float] | None = None
    anchor_predictions: dict[str, float] | None = None
    anchor_residual_ratios: dict[str, float] | None = None
    log_space_rmse: float | None = None
    mean_predicted: float | None = None
    u_clamp: float | None = None
    cdf_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize fit inputs, predictions, and reproducibility metadata."""

        return asdict(self)


def _lognormal_quantile(probability: float, mu: float, sigma: float) -> float:
    bounded = min(1 - _U_CLAMP, max(_U_CLAMP, probability))
    return math.exp(mu + sigma * NormalDist().inv_cdf(bounded))


def _fit_three_quantiles(quantiles: tuple[float, float, float]) -> tuple[float, float]:
    """Fit ``log(q_p) = mu + sigma * Phi^-1(p)`` by ordinary least squares."""

    log_quantiles = tuple(math.log(value) for value in quantiles)
    mean_z = sum(_ANCHOR_Z) / len(_ANCHOR_Z)
    mean_log_q = sum(log_quantiles) / len(log_quantiles)
    numerator = sum(
        (z_value - mean_z) * (log_q - mean_log_q) for z_value, log_q in zip(_ANCHOR_Z, log_quantiles, strict=True)
    )
    denominator = sum((z_value - mean_z) ** 2 for z_value in _ANCHOR_Z)
    sigma = numerator / denominator
    mu = mean_log_q - sigma * mean_z
    return mu, sigma


def build_interval_model(config: ReplayConfig | None = None) -> IntervalModel:
    """Fit and return the selected auditable interval model."""

    if config is None or config.interval_mode == "trace":
        return IntervalModel(mode="trace", fit_version="agentinfer-replay-trace/v2")
    configured = config.interval_lognormal
    if configured is None:  # Also guarded by ReplayConfig; retain direct-call fail-closed behavior.
        raise ValueError("interval_mode=lognormal requires p50/p95/p99 anchors")
    quantiles = (configured.p50_seconds, configured.p95_seconds, configured.p99_seconds)
    mu, sigma = _fit_three_quantiles(quantiles)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("p50/p95/p99 anchors do not define a positive Lognormal sigma")
    predictions = tuple(_lognormal_quantile(probability, mu, sigma) for probability in _ANCHOR_PROBABILITIES)
    anchors = dict(zip(_ANCHOR_NAMES, quantiles, strict=True))
    anchor_predictions = dict(zip(_ANCHOR_NAMES, predictions, strict=True))
    residuals = {
        name: predicted / observed - 1
        for name, predicted, observed in zip(_ANCHOR_NAMES, predictions, quantiles, strict=True)
    }
    max_residual = max(abs(value) for value in residuals.values())
    if max_residual >= _MAX_ANCHOR_RESIDUAL_RATIO:
        raise ValueError(
            "p50/p95/p99 anchors are inconsistent with one Lognormal distribution: "
            f"maximum fitted residual is {max_residual:.3%}, limit is {_MAX_ANCHOR_RESIDUAL_RATIO:.1%}"
        )
    squared_log_residuals = [
        (math.log(predicted) - math.log(observed)) ** 2
        for predicted, observed in zip(predictions, quantiles, strict=True)
    ]
    return IntervalModel(
        mode="lognormal",
        fit_version="lognormal-three-quantile-ols/v1",
        mu=mu,
        sigma=sigma,
        anchors=anchors,
        anchor_predictions=anchor_predictions,
        anchor_residual_ratios=residuals,
        log_space_rmse=math.sqrt(sum(squared_log_residuals) / len(squared_log_residuals)),
        mean_predicted=math.exp(mu + sigma * sigma / 2),
        u_clamp=_U_CLAMP,
        cdf_version=_CDF_VERSION,
    )


def deterministic_uniform(sample_seed: int, runtime_session_id: str, request_key: str) -> float:
    """Map stable request identity to a reproducible open-interval uniform."""

    material = f"{sample_seed}\0{runtime_session_id}\0{request_key}\0interval/v1"
    integer = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    uniform = (integer + 0.5) / 2**64
    return min(1 - _U_CLAMP, max(_U_CLAMP, uniform))


def backend_sampling_seed(
    config: ReplayConfig,
    request: ReplayRequest,
    runtime_session_id: str,
) -> int:
    """Return a deterministic signed-31-bit Backend seed."""

    material = f"{config.sample_seed}\0{runtime_session_id}\0{request.key}\0backend-seed/v1"
    digest = hashlib.sha256(material.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def effective_interval_seconds(
    config: ReplayConfig,
    request: ReplayRequest,
    model: IntervalModel | None = None,
    runtime_session_id: str | None = None,
) -> float:
    """Return a deterministic completion-to-next-send interval."""

    if request.send_after is None and request.delay_seconds == 0:
        return 0.0
    if request.dependency_kind != "same_agent" or request.same_agent_gap_seconds is None:
        return request.delay_seconds
    if config.interval_mode == "trace":
        return max(
            0.0,
            request.same_agent_gap_seconds * config.trace_same_agent_gap_scale
            + config.trace_same_agent_gap_offset_seconds,
        )
    if model is None or model.mode != "lognormal" or runtime_session_id is None:
        raise ValueError("lognormal interval sampling requires a model and runtime_session_id")
    assert model.mu is not None and model.sigma is not None
    uniform = deterministic_uniform(config.sample_seed, runtime_session_id, request.key)
    return _lognormal_quantile(uniform, model.mu, model.sigma)
