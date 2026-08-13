"""Sampling-phase-marginalized query-conditioned video token selection.

PhaseFuse consumes one dense, time-ordered candidate stream.  It splits that
stream into interleaved physical sampling phases, applies a pluggable temporal
transform to every phase, aligns the resulting saliency signals in absolute
time, and fuses them with a robust consensus and an explicit phase-uncertainty
penalty.  Boundary selection and frame allocation are jointly constrained by
the exact output budget, so every retained segment receives a configurable
minimum number of frames.

The module is deliberately independent of video decoding and model inference.
Upstream code may therefore cache dense frame/query features and pass its
dense-to-decoded scatter map through ``source_frame_indices`` without coupling
PhaseFuse to a particular backbone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np

from .transforms import TemporalTransform


@dataclass(frozen=True)
class PhaseFuseConfig:
    """Configuration for the training-free PhaseFuse selector."""

    num_phases: int = 4
    frame_budget: int = 16
    min_frames_per_segment: int = 1
    min_boundary_distance_sec: float = 5.0
    boundary_threshold_mad: float = 0.5
    uncertainty_penalty: float = 0.25
    relevance_weight: float = 0.5
    phase_vote_weight: float = 0.25
    consensus: Literal["median", "mean"] = "median"
    segment_duration_weight: float = 0.2
    segment_relevance_weight: float = 0.4
    segment_event_weight: float = 0.3
    segment_diversity_weight: float = 0.1
    segment_uncertainty_penalty: float = 0.1
    allocation_temperature: float = 1.0
    mmr_lambda: float = 0.7
    mmr_visual_weight: float = 0.75
    temporal_redundancy_scale_sec: float = 2.0
    short_phase_policy: Literal["error", "single_stream"] = "single_stream"
    selection_strategy: Literal["segmented", "global_coverage"] = "segmented"
    uniform_reserve: int = 0
    selection_event_weight: float = 1.0
    component_scaling: Literal["robust_z", "percentile"] = "robust_z"
    min_selection_distance_sec: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "num_phases",
            "frame_budget",
            "min_frames_per_segment",
            "uniform_reserve",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if name == "uniform_reserve":
                if int(value) < 0:
                    raise ValueError("uniform_reserve must be non-negative")
            elif int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_frames_per_segment > self.frame_budget:
            raise ValueError("min_frames_per_segment cannot exceed frame_budget")
        if self.uniform_reserve > self.frame_budget:
            raise ValueError("uniform_reserve cannot exceed frame_budget")
        nonnegative = (
            "min_boundary_distance_sec",
            "boundary_threshold_mad",
            "uncertainty_penalty",
            "relevance_weight",
            "phase_vote_weight",
            "segment_duration_weight",
            "segment_relevance_weight",
            "segment_event_weight",
            "segment_diversity_weight",
            "segment_uncertainty_penalty",
            "selection_event_weight",
            "min_selection_distance_sec",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.allocation_temperature)
            or self.allocation_temperature <= 0
        ):
            raise ValueError("allocation_temperature must be finite and positive")
        if not np.isfinite(self.mmr_lambda) or not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must lie in [0, 1]")
        if (
            not np.isfinite(self.mmr_visual_weight)
            or not 0 <= self.mmr_visual_weight <= 1
        ):
            raise ValueError("mmr_visual_weight must lie in [0, 1]")
        if (
            not np.isfinite(self.temporal_redundancy_scale_sec)
            or self.temporal_redundancy_scale_sec <= 0
        ):
            raise ValueError(
                "temporal_redundancy_scale_sec must be finite and positive"
            )
        if self.consensus not in {"median", "mean"}:
            raise ValueError("consensus must be 'median' or 'mean'")
        if self.short_phase_policy not in {"error", "single_stream"}:
            raise ValueError("short_phase_policy must be 'error' or 'single_stream'")
        if self.selection_strategy not in {"segmented", "global_coverage"}:
            raise ValueError(
                "selection_strategy must be 'segmented' or 'global_coverage'"
            )
        if self.component_scaling not in {"robust_z", "percentile"}:
            raise ValueError("component_scaling must be 'robust_z' or 'percentile'")


@dataclass
class PhaseFuseTrace:
    """Complete, serializable trace of one PhaseFuse selection."""

    config: PhaseFuseConfig
    timestamps_sec: np.ndarray
    source_frame_indices: np.ndarray | None
    input_valid_mask: np.ndarray
    selectable_mask: np.ndarray
    input_phase_ids: np.ndarray
    phase_ids: np.ndarray
    phase_indices: tuple[np.ndarray, ...]
    transform_methods: tuple[str, ...]
    aligned_phase_saliency: np.ndarray
    normalized_aligned_phase_saliency: np.ndarray
    phase_support_mask: np.ndarray
    full_support_mask: np.ndarray
    fusion_domain_mask: np.ndarray
    consensus: np.ndarray
    uncertainty: np.ndarray
    phase_vote: np.ndarray
    normalized_relevance: np.ndarray
    event_scores: np.ndarray
    boundary_threshold: float
    boundary_candidate_mask: np.ndarray
    boundary_indices: np.ndarray
    segments: tuple[tuple[int, int], ...]
    segment_utilities: np.ndarray
    allocation: np.ndarray
    anchor_indices: np.ndarray
    selected_indices: np.ndarray
    used_fallback: bool
    fallback_reason: str | None

    @property
    def selected_timestamps_sec(self) -> np.ndarray:
        return self.timestamps_sec[self.selected_indices]

    @property
    def selected_dense_indices(self) -> np.ndarray:
        """Alias spelling out that selection indices address the dense input."""

        return self.selected_indices

    @property
    def selected_source_frame_indices(self) -> np.ndarray | None:
        if self.source_frame_indices is None:
            return None
        return self.source_frame_indices[self.selected_indices]

    @property
    def boundary_timestamps_sec(self) -> np.ndarray:
        return self.timestamps_sec[self.boundary_indices]

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return strict-JSON-compatible metadata and, optionally, dense arrays."""

        config_payload = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in asdict(self.config).items()
        }
        payload: dict[str, Any] = {
            "config": config_payload,
            "num_candidates": int(self.timestamps_sec.size),
            "num_input_valid_candidates": int(np.sum(self.input_valid_mask)),
            "num_selectable_candidates": int(np.sum(self.selectable_mask)),
            "num_phases": len(self.phase_indices),
            "phase_indices": [row.astype(int).tolist() for row in self.phase_indices],
            "transform_methods": list(self.transform_methods),
            "boundary_threshold": float(self.boundary_threshold),
            "boundary_indices": self.boundary_indices.astype(int).tolist(),
            "boundary_timestamps_sec": self.boundary_timestamps_sec.astype(
                float
            ).tolist(),
            "segments": [list(segment) for segment in self.segments],
            "segment_utilities": self.segment_utilities.astype(float).tolist(),
            "allocation": self.allocation.astype(int).tolist(),
            "anchor_indices": self.anchor_indices.astype(int).tolist(),
            "anchor_timestamps_sec": self.timestamps_sec[self.anchor_indices]
            .astype(float)
            .tolist(),
            "selected_indices": self.selected_indices.astype(int).tolist(),
            "selected_dense_indices": self.selected_indices.astype(int).tolist(),
            "selected_timestamps_sec": self.selected_timestamps_sec.astype(
                float
            ).tolist(),
            "selected_source_frame_indices": (
                None
                if self.selected_source_frame_indices is None
                else self.selected_source_frame_indices.astype(int).tolist()
            ),
            "used_fallback": bool(self.used_fallback),
            "fallback_reason": self.fallback_reason,
        }
        if include_arrays:
            aligned = self.aligned_phase_saliency.astype(object)
            aligned[~np.isfinite(self.aligned_phase_saliency)] = None
            normalized_aligned = self.normalized_aligned_phase_saliency.astype(object)
            normalized_aligned[~np.isfinite(self.normalized_aligned_phase_saliency)] = (
                None
            )
            payload.update(
                {
                    "timestamps_sec": self.timestamps_sec.astype(float).tolist(),
                    "source_frame_indices": (
                        None
                        if self.source_frame_indices is None
                        else self.source_frame_indices.astype(int).tolist()
                    ),
                    "input_valid_mask": self.input_valid_mask.astype(bool).tolist(),
                    "selectable_mask": self.selectable_mask.astype(bool).tolist(),
                    "input_phase_ids": self.input_phase_ids.astype(int).tolist(),
                    "phase_ids": self.phase_ids.astype(int).tolist(),
                    "aligned_phase_saliency": aligned.tolist(),
                    "normalized_aligned_phase_saliency": (normalized_aligned.tolist()),
                    "phase_support_mask": self.phase_support_mask.astype(bool).tolist(),
                    "full_support_mask": self.full_support_mask.astype(bool).tolist(),
                    "fusion_domain_mask": self.fusion_domain_mask.astype(bool).tolist(),
                    "consensus": self.consensus.astype(float).tolist(),
                    "uncertainty": self.uncertainty.astype(float).tolist(),
                    "phase_vote": self.phase_vote.astype(float).tolist(),
                    "normalized_relevance": self.normalized_relevance.astype(
                        float
                    ).tolist(),
                    "event_scores": self.event_scores.astype(float).tolist(),
                    "boundary_candidate_mask": self.boundary_candidate_mask.astype(
                        bool
                    ).tolist(),
                }
            )
        return payload


def interleaved_phase_ids(num_samples: int, num_phases: int) -> np.ndarray:
    """Assign dense sample ``j`` to phase ``j mod num_phases``."""

    for name, value in (("num_samples", num_samples), ("num_phases", num_phases)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    return np.arange(int(num_samples), dtype=int) % int(num_phases)


def _validated_phase_ids(
    phase_ids: Sequence[int] | None, num_samples: int, num_phases: int
) -> np.ndarray:
    if phase_ids is None:
        return interleaved_phase_ids(num_samples, num_phases)
    raw = np.asarray(phase_ids)
    if raw.ndim != 1 or raw.shape != (num_samples,):
        raise ValueError("phase_ids must be one-dimensional and frame-aligned")
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError("phase_ids must contain integers")
    result = raw.astype(int, copy=False)
    if np.any(result < 0) or np.any(result >= num_phases):
        raise ValueError("phase_ids must lie in [0, config.num_phases)")
    if set(result.tolist()) != set(range(num_phases)):
        raise ValueError("phase_ids must contain every configured phase")
    return result


def align_phase_values(
    timestamps_sec: Sequence[float],
    phase_indices: Sequence[Sequence[int]],
    phase_values: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly align phase-local values onto an absolute-time dense grid.

    Values outside a phase's observed time range are left unsupported (NaN),
    rather than silently extrapolated.
    """

    timestamps = np.asarray(timestamps_sec, dtype=float)
    if (
        timestamps.ndim != 1
        or timestamps.size < 2
        or not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("timestamps_sec must be finite and strictly increasing")
    if len(phase_indices) != len(phase_values) or not phase_indices:
        raise ValueError("phase_indices and phase_values must have equal nonzero size")
    aligned = np.full((len(phase_indices), timestamps.size), np.nan, dtype=float)
    support = np.zeros_like(aligned, dtype=bool)
    for phase, (raw_indices, raw_values) in enumerate(zip(phase_indices, phase_values)):
        indices = np.asarray(raw_indices)
        values = np.asarray(raw_values, dtype=float)
        if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
            raise TypeError("every phase index vector must contain integers")
        indices = indices.astype(int, copy=False)
        if (
            indices.size < 2
            or np.any(np.diff(indices) <= 0)
            or indices[0] < 0
            or indices[-1] >= timestamps.size
        ):
            raise ValueError(
                "phase indices must be sorted, unique, in range, and length >= 2"
            )
        if values.shape != (indices.size,) or not np.all(np.isfinite(values)):
            raise ValueError("phase values must be finite and phase-aligned")
        phase_times = timestamps[indices]
        mask = (timestamps >= phase_times[0]) & (timestamps <= phase_times[-1])
        aligned[phase, mask] = np.interp(timestamps[mask], phase_times, values)
        support[phase, mask] = True
    return aligned, support


def _positive_robust_normalize(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    result[~finite] = np.nan
    observed = values[finite]
    if observed.size == 0:
        raise ValueError("at least one finite value is required")
    location = float(np.median(observed))
    scale = 1.4826 * float(np.median(np.abs(observed - location)))
    if scale <= np.finfo(float).eps:
        scale = float(np.std(observed))
    if scale <= np.finfo(float).eps:
        return result
    result[finite] = np.maximum((observed - location) / scale, 0.0)
    return result


def _percentile_rank_scale(values: np.ndarray) -> np.ndarray:
    """Map finite values to tie-aware empirical ranks in ``[0, 1]``.

    Only ordering affects the result, so changing the amplitude of an extreme
    observation cannot rescale every other candidate.  Equal-valued constant
    components carry no discriminative evidence and map to zero.
    """

    result = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    result[~finite] = np.nan
    observed = values[finite]
    if observed.size == 0:
        raise ValueError("at least one finite value is required")
    unique, inverse, counts = np.unique(
        observed, return_inverse=True, return_counts=True
    )
    if unique.size == 1:
        return result
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))
    midranks = starts + (counts - 1) / 2.0
    result[finite] = midranks[inverse] / float(observed.size - 1)
    return result


def _scale_component(
    values: np.ndarray, method: Literal["robust_z", "percentile"]
) -> np.ndarray:
    if method == "robust_z":
        return _positive_robust_normalize(values)
    if method == "percentile":
        return _percentile_rank_scale(values)
    raise ValueError("unknown component scaling")  # pragma: no cover - config guards


def robust_phase_consensus(
    aligned_phase_values: np.ndarray,
    *,
    aggregation: Literal["median", "mean"] = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Return phase-symmetric consensus and scaled MAD uncertainty."""

    values = np.asarray(aligned_phase_values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("aligned_phase_values must have shape [phase, time]")
    if np.any(np.isinf(values)) or np.any(np.all(np.isnan(values), axis=0)):
        raise ValueError("every time must have finite support from at least one phase")
    if aggregation == "median":
        consensus = np.nanmedian(values, axis=0)
    elif aggregation == "mean":
        consensus = np.nanmean(values, axis=0)
    else:
        raise ValueError("aggregation must be 'median' or 'mean'")
    uncertainty = 1.4826 * np.nanmedian(np.abs(values - consensus[None, :]), axis=0)
    return np.asarray(consensus, dtype=float), np.asarray(uncertainty, dtype=float)


def _better_state(
    first: tuple[float, tuple[int, ...]] | None,
    second: tuple[float, tuple[int, ...]] | None,
) -> tuple[float, tuple[int, ...]] | None:
    if first is None:
        return second
    if second is None:
        return first
    if first[0] != second[0]:
        return first if first[0] > second[0] else second
    if len(first[1]) != len(second[1]):
        return first if len(first[1]) < len(second[1]) else second
    return first if first[1] < second[1] else second


def select_weighted_interval_indices(
    timestamps_sec: Sequence[float],
    weights: Sequence[float],
    max_count: int,
    min_distance_sec: float,
    *,
    valid_mask: Sequence[bool] | None = None,
    min_index_distance: int = 1,
    edge_margin: int = 1,
    segment_capacity_mask: Sequence[bool] | None = None,
    min_segment_capacity: int = 0,
) -> np.ndarray:
    """Select an at-most-``max_count`` maximum-weight boundary set.

    Only strictly positive weights are eligible.  Ties prefer fewer boundaries
    and then lexicographically earlier dense indices.
    """

    timestamps = np.asarray(timestamps_sec, dtype=float)
    scores = np.asarray(weights, dtype=float)
    if (
        timestamps.ndim != 1
        or timestamps.size < 2
        or not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("timestamps_sec must be finite and strictly increasing")
    if scores.shape != timestamps.shape or not np.all(np.isfinite(scores)):
        raise ValueError("weights must be finite and timestamp-aligned")
    for name, value, minimum in (
        ("max_count", max_count, 0),
        ("min_index_distance", min_index_distance, 1),
        ("edge_margin", edge_margin, 1),
        ("min_segment_capacity", min_segment_capacity, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if not np.isfinite(min_distance_sec) or min_distance_sec < 0:
        raise ValueError("min_distance_sec must be finite and non-negative")
    if max_count == 0:
        return np.asarray([], dtype=int)
    if valid_mask is None:
        mask = np.ones(timestamps.size, dtype=bool)
    else:
        raw_mask = np.asarray(valid_mask)
        if raw_mask.dtype.kind != "b" or raw_mask.shape != timestamps.shape:
            raise ValueError("valid_mask must be a frame-aligned boolean vector")
        mask = raw_mask.astype(bool, copy=True)
    mask[:edge_margin] = False
    mask[max(timestamps.size - edge_margin, 0) :] = False
    mask[0] = False
    mask[-1] = False
    capacity_prefix = None
    if segment_capacity_mask is not None:
        raw_capacity_mask = np.asarray(segment_capacity_mask)
        if (
            raw_capacity_mask.dtype.kind != "b"
            or raw_capacity_mask.shape != timestamps.shape
        ):
            raise ValueError(
                "segment_capacity_mask must be a frame-aligned boolean vector"
            )
        capacity_prefix = np.concatenate(
            ([0], np.cumsum(raw_capacity_mask.astype(np.int64)))
        )
        total_capacity = int(capacity_prefix[-1])
        indices = np.arange(timestamps.size)
        mask &= capacity_prefix[indices] >= int(min_segment_capacity)
        mask &= (total_capacity - capacity_prefix[indices]) >= int(min_segment_capacity)
    elif min_segment_capacity:
        raise ValueError(
            "segment_capacity_mask is required when min_segment_capacity is positive"
        )
    candidates = np.flatnonzero(mask & (scores > 0.0))
    if candidates.size == 0:
        return np.asarray([], dtype=int)

    count = min(int(max_count), int(candidates.size))
    states: list[list[tuple[float, tuple[int, ...]] | None]] = [
        [None] * (count + 1) for _ in range(candidates.size + 1)
    ]
    states[0][0] = (0.0, ())
    for row, raw_index in enumerate(candidates, start=1):
        index = int(raw_index)
        compatible = 0
        for prior_position, raw_prior in enumerate(candidates[: row - 1], start=1):
            prior = int(raw_prior)
            if (
                index - prior >= int(min_index_distance)
                and timestamps[index] - timestamps[prior] >= float(min_distance_sec)
                and (
                    capacity_prefix is None
                    or capacity_prefix[index] - capacity_prefix[prior]
                    >= int(min_segment_capacity)
                )
            ):
                compatible = prior_position
        states[row][0] = (0.0, ())
        for selected_count in range(1, count + 1):
            excluded = states[row - 1][selected_count]
            prior_state = states[compatible][selected_count - 1]
            included = None
            if prior_state is not None:
                included = (
                    prior_state[0] + float(scores[index]),
                    (*prior_state[1], index),
                )
            states[row][selected_count] = _better_state(excluded, included)
    best: tuple[float, tuple[int, ...]] | None = None
    for state in states[-1]:
        best = _better_state(best, state)
    return np.asarray(() if best is None else best[1], dtype=int)


def allocate_exact_budget(
    capacities: Sequence[int],
    utilities: Sequence[float],
    total_budget: int,
    *,
    min_per_segment: int = 1,
    temperature: float = 1.0,
) -> np.ndarray:
    """Capacity-safe softmax/largest-remainder allocation summing exactly K."""

    raw_capacities = np.asarray(capacities)
    scores = np.asarray(utilities, dtype=float)
    if raw_capacities.ndim != 1 or raw_capacities.size == 0:
        raise ValueError("capacities must be a non-empty one-dimensional vector")
    if raw_capacities.dtype.kind not in {"i", "u"}:
        raise TypeError("capacities must contain integers")
    capacity = raw_capacities.astype(int, copy=False)
    if np.any(capacity < 0):
        raise ValueError("capacities must be non-negative")
    if scores.shape != capacity.shape or not np.all(np.isfinite(scores)):
        raise ValueError("utilities must be finite and capacity-aligned")
    for name, value, minimum in (
        ("total_budget", total_budget, 0),
        ("min_per_segment", min_per_segment, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    minimum_total = int(min_per_segment) * capacity.size
    if np.any(capacity < int(min_per_segment)):
        raise ValueError("every segment must support min_per_segment")
    if total_budget < minimum_total or total_budget > int(np.sum(capacity)):
        raise ValueError("total_budget is infeasible for the segment capacities")

    allocation = np.full(capacity.size, int(min_per_segment), dtype=int)
    remaining = int(total_budget - np.sum(allocation))
    while remaining:
        active = np.flatnonzero(allocation < capacity)
        if active.size == 0:
            raise RuntimeError("capacity allocation stalled before reaching budget")
        logits = scores[active] / float(temperature)
        logits -= float(np.max(logits))
        proportions = np.exp(logits)
        proportions /= float(np.sum(proportions))
        ideal = remaining * proportions
        floors = np.floor(ideal).astype(int)
        floors = np.minimum(floors, capacity[active] - allocation[active])
        if int(np.sum(floors)):
            allocation[active] += floors
            remaining -= int(np.sum(floors))
        fractions = ideal - np.floor(ideal)
        remainder_order = sorted(
            range(active.size), key=lambda pos: (-fractions[pos], int(active[pos]))
        )
        made_progress = False
        for position in remainder_order:
            segment = int(active[position])
            if remaining == 0:
                break
            if allocation[segment] >= capacity[segment]:
                continue
            allocation[segment] += 1
            remaining -= 1
            made_progress = True
        if not made_progress and remaining:
            raise RuntimeError("capacity allocation made no remainder progress")
    return allocation


def select_mmr_indices(
    candidate_indices: Sequence[int],
    relevance_scores: Sequence[float],
    count: int,
    *,
    features: np.ndarray | None = None,
    timestamps_sec: Sequence[float] | None = None,
    lambda_param: float = 0.7,
    visual_weight: float = 0.75,
    temporal_scale_sec: float = 2.0,
) -> np.ndarray:
    """Select exactly ``count`` candidates using deterministic MMR."""

    raw_candidates = np.asarray(candidate_indices)
    relevance = np.asarray(relevance_scores, dtype=float)
    if raw_candidates.ndim != 1 or raw_candidates.dtype.kind not in {"i", "u"}:
        raise TypeError("candidate_indices must be a one-dimensional integer vector")
    candidates = raw_candidates.astype(int, copy=False)
    if np.any(np.diff(candidates) <= 0):
        raise ValueError("candidate_indices must be sorted and unique")
    if relevance.ndim != 1 or not np.all(np.isfinite(relevance)):
        raise ValueError("relevance_scores must be a finite one-dimensional vector")
    if candidates.size and (candidates[0] < 0 or candidates[-1] >= relevance.size):
        raise ValueError("candidate_indices are outside relevance_scores")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    if count < 0 or count > candidates.size:
        raise ValueError("count must lie in [0, number of candidates]")
    if not np.isfinite(lambda_param) or not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must lie in [0, 1]")
    if not np.isfinite(visual_weight) or not 0 <= visual_weight <= 1:
        raise ValueError("visual_weight must lie in [0, 1]")
    if not np.isfinite(temporal_scale_sec) or temporal_scale_sec <= 0:
        raise ValueError("temporal_scale_sec must be finite and positive")
    if count == 0:
        return np.asarray([], dtype=int)

    if timestamps_sec is None:
        timestamps = np.arange(relevance.size, dtype=float)
    else:
        timestamps = np.asarray(timestamps_sec, dtype=float)
        if (
            timestamps.shape != relevance.shape
            or not np.all(np.isfinite(timestamps))
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise ValueError("timestamps_sec must be finite, increasing, and aligned")
    normalized_features = None
    if features is not None:
        matrix = np.asarray(features, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != relevance.size
            or matrix.shape[1] == 0
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("features must be finite with shape [time, dimension]")
        normalized_features = np.zeros_like(matrix, dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms > np.finfo(float).eps
        normalized_features[nonzero] = matrix[nonzero] / norms[nonzero, None]

    candidate_relevance = relevance[candidates]
    span = float(np.max(candidate_relevance) - np.min(candidate_relevance))
    if span <= np.finfo(float).eps:
        normalized_relevance = np.ones(candidates.size, dtype=float)
    else:
        normalized_relevance = (
            candidate_relevance - float(np.min(candidate_relevance))
        ) / span
    selected_positions: list[int] = []
    available = set(range(candidates.size))
    while len(selected_positions) < int(count):
        best_position = None
        best_score = -np.inf
        for position in sorted(available, key=lambda pos: int(candidates[pos])):
            redundancy = 0.0
            if selected_positions:
                temporal = max(
                    np.exp(
                        -abs(
                            timestamps[candidates[position]]
                            - timestamps[candidates[prior]]
                        )
                        / temporal_scale_sec
                    )
                    for prior in selected_positions
                )
                if normalized_features is None:
                    redundancy = float(temporal)
                else:
                    visual = max(
                        max(
                            0.0,
                            float(
                                np.dot(
                                    normalized_features[candidates[position]],
                                    normalized_features[candidates[prior]],
                                )
                            ),
                        )
                        for prior in selected_positions
                    )
                    redundancy = float(
                        visual_weight * visual + (1.0 - visual_weight) * temporal
                    )
            score = float(
                lambda_param * normalized_relevance[position]
                - (1.0 - lambda_param) * redundancy
            )
            if score > best_score:
                best_score = score
                best_position = position
        if best_position is None:  # pragma: no cover - guarded by count validation
            raise RuntimeError("MMR selection stalled")
        selected_positions.append(best_position)
        available.remove(best_position)
    return np.asarray(
        sorted(int(candidates[position]) for position in selected_positions),
        dtype=int,
    )


def _uniform_coverage_indices(
    candidate_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    count: int,
) -> np.ndarray:
    """Choose exact deterministic temporal anchors at uniform bin centers."""

    if count == 0:
        return np.asarray([], dtype=int)
    if count > candidate_indices.size:  # pragma: no cover - caller validates budget
        raise ValueError("uniform anchor count exceeds candidate count")
    first_time = float(timestamps_sec[candidate_indices[0]])
    last_time = float(timestamps_sec[candidate_indices[-1]])
    fractions = (np.arange(count, dtype=float) + 0.5) / float(count)
    targets = first_time + fractions * (last_time - first_time)
    available = set(int(index) for index in candidate_indices)
    selected: list[int] = []
    for target in targets:
        best = min(
            available,
            key=lambda index: (abs(float(timestamps_sec[index]) - target), index),
        )
        selected.append(best)
        available.remove(best)
    return np.asarray(sorted(selected), dtype=int)


def _seeded_global_mmr_indices(
    candidate_indices: np.ndarray,
    token_scores: np.ndarray,
    total_count: int,
    *,
    seed_indices: np.ndarray,
    features: np.ndarray | None,
    timestamps_sec: np.ndarray,
    lambda_param: float,
    visual_weight: float,
    temporal_scale_sec: float,
    min_distance_sec: float,
) -> np.ndarray | None:
    """Run one global MMR pass whose redundancy state starts with anchors.

    ``None`` means the requested minimum temporal distance made the exact
    budget infeasible for this deterministic greedy pass.  Callers may then
    retry with a relaxed distance while keeping the same anchors.
    """

    selected = seed_indices.astype(int).tolist()
    if len(selected) > total_count:
        raise ValueError("seed count exceeds total_count")
    if len(selected) > 1 and np.any(
        np.diff(timestamps_sec[np.asarray(selected, dtype=int)]) < min_distance_sec
    ):
        return None
    available = set(int(index) for index in candidate_indices) - set(selected)
    candidate_scores = token_scores[candidate_indices]
    span = float(np.max(candidate_scores) - np.min(candidate_scores))
    if span <= np.finfo(float).eps:
        normalized_scores = np.ones(token_scores.size, dtype=float)
    else:
        normalized_scores = np.zeros(token_scores.size, dtype=float)
        normalized_scores[candidate_indices] = (
            candidate_scores - float(np.min(candidate_scores))
        ) / span

    normalized_features = None
    if features is not None:
        normalized_features = np.zeros_like(features, dtype=float)
        norms = np.linalg.norm(features, axis=1)
        nonzero = norms > np.finfo(float).eps
        normalized_features[nonzero] = features[nonzero] / norms[nonzero, None]

    while len(selected) < total_count:
        eligible = [
            index
            for index in sorted(available)
            if all(
                abs(float(timestamps_sec[index] - timestamps_sec[prior]))
                >= min_distance_sec
                for prior in selected
            )
        ]
        if not eligible:
            return None
        best_index = None
        best_score = -np.inf
        for index in eligible:
            redundancy = 0.0
            if selected:
                temporal = max(
                    np.exp(
                        -abs(float(timestamps_sec[index] - timestamps_sec[prior]))
                        / temporal_scale_sec
                    )
                    for prior in selected
                )
                if normalized_features is None:
                    redundancy = float(temporal)
                else:
                    visual = max(
                        max(
                            0.0,
                            float(
                                np.dot(
                                    normalized_features[index],
                                    normalized_features[prior],
                                )
                            ),
                        )
                        for prior in selected
                    )
                    redundancy = float(
                        visual_weight * visual + (1.0 - visual_weight) * temporal
                    )
            score = float(
                lambda_param * normalized_scores[index]
                - (1.0 - lambda_param) * redundancy
            )
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:  # pragma: no cover - eligible is non-empty
            raise RuntimeError("global MMR selection stalled")
        selected.append(best_index)
        available.remove(best_index)
    return np.asarray(sorted(selected), dtype=int)


def _local_maxima_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(values.size, dtype=bool)
    if values.size >= 3:
        mask[1:-1] = (
            (values[1:-1] >= values[:-2])
            & (values[1:-1] >= values[2:])
            & ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:]))
        )
    return mask


def _segments_from_boundaries(
    boundaries: np.ndarray, num_samples: int
) -> tuple[tuple[int, int], ...]:
    edges = (0, *boundaries.astype(int).tolist(), int(num_samples))
    return tuple((int(start), int(stop)) for start, stop in pairwise(edges))


def _repair_boundaries_for_capacity(
    boundaries: np.ndarray,
    weights: np.ndarray,
    selectable_mask: np.ndarray,
    min_frames: int,
) -> tuple[np.ndarray, bool]:
    result = boundaries.astype(int).tolist()
    repaired = False
    while result:
        segments = _segments_from_boundaries(
            np.asarray(result, dtype=int), weights.size
        )
        capacities = [
            int(np.sum(selectable_mask[start:stop])) for start, stop in segments
        ]
        bad = next(
            (index for index, value in enumerate(capacities) if value < min_frames),
            None,
        )
        if bad is None:
            break
        if bad == 0:
            remove_position = 0
        elif bad == len(segments) - 1:
            remove_position = len(result) - 1
        else:
            left_position = bad - 1
            right_position = bad
            left = result[left_position]
            right = result[right_position]
            remove_position = (
                left_position if weights[left] < weights[right] else right_position
            )
        result.pop(remove_position)
        repaired = True
    return np.asarray(result, dtype=int), repaired


def _segment_utilities(
    segments: tuple[tuple[int, int], ...],
    timestamps: np.ndarray,
    relevance: np.ndarray,
    event: np.ndarray,
    uncertainty: np.ndarray,
    features: np.ndarray | None,
    selectable_mask: np.ndarray,
    config: PhaseFuseConfig,
) -> np.ndarray:
    utilities = []
    median_step = float(np.median(np.diff(timestamps)))
    for start, stop in segments:
        local_indices = np.flatnonzero(selectable_mask[start:stop]) + start
        if local_indices.size == 0:  # pragma: no cover - repaired before this call
            raise RuntimeError("segment utility received an empty valid segment")
        duration = float(
            timestamps[local_indices[-1]] - timestamps[local_indices[0]] + median_step
        )
        diversity = 0.0
        if features is not None and local_indices.size > 1:
            local = features[local_indices]
            center = np.mean(local, axis=0)
            diversity = float(np.mean(np.sum(np.square(local - center), axis=1)))
        utilities.append(
            config.segment_duration_weight * np.log1p(duration)
            + config.segment_relevance_weight * float(np.max(relevance[local_indices]))
            + config.segment_event_weight * float(np.max(event[local_indices]))
            + config.segment_diversity_weight * diversity
            - config.segment_uncertainty_penalty
            * float(np.mean(uncertainty[local_indices]))
        )
    return np.asarray(utilities, dtype=float)


def _resolve_transform(
    phase: int,
    transform: TemporalTransform | None,
    transform_factory: Callable[[int], TemporalTransform] | None,
) -> TemporalTransform:
    resolved = transform if transform_factory is None else transform_factory(phase)
    if resolved is None or not callable(getattr(resolved, "transform", None)):
        raise TypeError("transform/factory must provide transform(signal)")
    return resolved


def run_phasefuse(
    timestamps_sec: Sequence[float],
    relevance_scores: Sequence[float],
    features: np.ndarray | None,
    *,
    config: PhaseFuseConfig,
    transform: TemporalTransform | None = None,
    transform_factory: Callable[[int], TemporalTransform] | None = None,
    phase_ids: Sequence[int] | None = None,
    valid_mask: Sequence[bool] | None = None,
    source_frame_indices: Sequence[int] | None = None,
) -> PhaseFuseTrace:
    """Run PhaseFuse on one dense outer-origin candidate record.

    Exactly one of ``transform`` and ``transform_factory`` is required.  A
    factory receives the integer phase ID, allowing stateful transform objects
    to be isolated when necessary.  When ``source_frame_indices`` is supplied,
    at most one token can be selected for each decoded source frame.  The
    earliest valid dense target is the deterministic representative of a
    duplicated source frame.
    """

    if not isinstance(config, PhaseFuseConfig):
        raise TypeError("config must be a PhaseFuseConfig")
    if (transform is None) == (transform_factory is None):
        raise ValueError("supply exactly one of transform or transform_factory")
    timestamps = np.asarray(timestamps_sec, dtype=float)
    relevance = np.asarray(relevance_scores, dtype=float)
    if (
        timestamps.ndim != 1
        or timestamps.size < 2
        or not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("timestamps_sec must be finite and strictly increasing")
    if relevance.shape != timestamps.shape or not np.all(np.isfinite(relevance)):
        raise ValueError("relevance_scores must be finite and timestamp-aligned")
    if config.frame_budget > timestamps.size:
        raise ValueError("frame_budget cannot exceed the dense candidate count")
    if valid_mask is None:
        input_valid = np.ones(timestamps.size, dtype=bool)
    else:
        raw_valid = np.asarray(valid_mask)
        if raw_valid.dtype.kind != "b" or raw_valid.shape != timestamps.shape:
            raise ValueError("valid_mask must be a frame-aligned boolean vector")
        input_valid = raw_valid.astype(bool, copy=True)
    source_indices = None
    if source_frame_indices is not None:
        raw_source = np.asarray(source_frame_indices)
        if raw_source.shape != timestamps.shape or raw_source.dtype.kind not in "iu":
            raise ValueError(
                "source_frame_indices must be a frame-aligned integer vector"
            )
        if np.any(raw_source < 0):
            raise ValueError("source_frame_indices must be non-negative")
        if raw_source.dtype.kind == "u" and np.any(raw_source > np.iinfo(np.int64).max):
            raise ValueError("source_frame_indices exceed the supported integer range")
        source_indices = raw_source.astype(np.int64, copy=True)
        if np.any(np.diff(source_indices) < 0):
            raise ValueError("source_frame_indices must be non-decreasing")
    feature_matrix = None
    if features is not None:
        feature_matrix = np.asarray(features, dtype=float)
        if (
            feature_matrix.ndim != 2
            or feature_matrix.shape[0] != timestamps.size
            or feature_matrix.shape[1] == 0
            or not np.all(np.isfinite(feature_matrix))
        ):
            raise ValueError("features must be finite with shape [time, dimension]")

    input_phase_ids = _validated_phase_ids(
        phase_ids, timestamps.size, config.num_phases
    )
    phase_index_rows = tuple(
        np.flatnonzero(input_phase_ids == phase) for phase in range(config.num_phases)
    )
    effective_phase_ids = input_phase_ids.copy()
    used_fallback = False
    fallback_reason = None
    if any(indices.size < 2 for indices in phase_index_rows):
        if config.short_phase_policy == "error":
            raise ValueError("every sampling phase must contain at least two samples")
        used_fallback = True
        fallback_reason = "insufficient_phase_samples:single_stream"
        phase_index_rows = (np.arange(timestamps.size, dtype=int),)
        effective_phase_ids = np.zeros(timestamps.size, dtype=int)

    phase_saliency: list[np.ndarray] = []
    methods: list[str] = []
    for phase, indices in enumerate(phase_index_rows):
        phase_transform = _resolve_transform(phase, transform, transform_factory)
        result = phase_transform.transform(relevance[indices])
        saliency = np.asarray(getattr(result, "saliency", None), dtype=float)
        if (
            saliency.shape != (indices.size,)
            or not np.all(np.isfinite(saliency))
            or np.any(saliency < 0)
        ):
            raise ValueError(
                "each transform saliency must be finite, non-negative, and phase-aligned"
            )
        phase_saliency.append(saliency)
        methods.append(str(getattr(result, "method", type(phase_transform).__name__)))

    aligned, support = align_phase_values(timestamps, phase_index_rows, phase_saliency)
    full_support = np.all(support, axis=0)
    fusion_domain = full_support & input_valid
    if not np.any(fusion_domain):
        raise ValueError("valid_mask has no common support across sampling phases")
    selectable = fusion_domain.copy()
    if source_indices is not None:
        seen_sources: set[int] = set()
        for dense_index in np.flatnonzero(selectable):
            source_index = int(source_indices[dense_index])
            if source_index in seen_sources:
                selectable[dense_index] = False
            else:
                seen_sources.add(source_index)
    if int(np.sum(selectable)) < config.frame_budget:
        raise ValueError("valid/deduplicated candidates are fewer than frame_budget")
    masked_aligned = aligned.copy()
    masked_aligned[:, ~fusion_domain] = np.nan
    # Normalize each phase independently so a transform-amplitude difference
    # cannot give one sampling phase more influence in the consensus.
    normalized_aligned = np.vstack(
        [_scale_component(row, config.component_scaling) for row in masked_aligned]
    )
    support_count = np.sum(support, axis=0)
    consensus = np.zeros(timestamps.size, dtype=float)
    uncertainty = np.zeros(timestamps.size, dtype=float)
    consensus[fusion_domain], uncertainty[fusion_domain] = robust_phase_consensus(
        normalized_aligned[:, fusion_domain], aggregation=config.consensus
    )
    phase_vote = np.zeros(timestamps.size, dtype=float)
    phase_vote[fusion_domain] = (
        np.sum(
            support[:, fusion_domain]
            & (np.nan_to_num(normalized_aligned[:, fusion_domain], nan=0.0) > 0.0),
            axis=0,
        )
        / support_count[fusion_domain]
    )
    masked_relevance = np.full(timestamps.size, np.nan, dtype=float)
    masked_relevance[fusion_domain] = relevance[fusion_domain]
    normalized_relevance = np.nan_to_num(
        _scale_component(masked_relevance, config.component_scaling), nan=0.0
    )
    event = consensus - config.uncertainty_penalty * uncertainty
    event += config.phase_vote_weight * phase_vote
    if config.selection_strategy == "segmented":
        # Preserve the original PhaseFuse definition and selection behavior.
        event += config.relevance_weight * normalized_relevance
    if not np.all(np.isfinite(event)):
        raise RuntimeError("phase fusion produced non-finite event scores")
    event_location = float(np.median(event[fusion_domain]))
    event_mad = 1.4826 * float(np.median(np.abs(event[fusion_domain] - event_location)))
    threshold = event_location + config.boundary_threshold_mad * event_mad
    anchor_indices = np.asarray([], dtype=int)
    if config.selection_strategy == "global_coverage":
        # PhaseFuse v2 deliberately removes event-derived hard segmentation.
        candidate_mask = np.zeros(timestamps.size, dtype=bool)
        boundaries = np.asarray([], dtype=int)
        segments = ((0, int(timestamps.size)),)
        utilities = _segment_utilities(
            segments,
            timestamps,
            normalized_relevance,
            event,
            uncertainty,
            feature_matrix,
            selectable,
            config,
        )
        allocation = np.asarray([config.frame_budget], dtype=int)
        candidates = np.flatnonzero(selectable)
        anchor_indices = _uniform_coverage_indices(
            candidates, timestamps, config.uniform_reserve
        )
        # Relevance appears exactly once in the token score.  Event saliency is
        # independently tunable and uncertainty may be kept diagnostic-only by
        # setting uncertainty_penalty=0.
        token_scores = normalized_relevance + config.selection_event_weight * event
        selected = _seeded_global_mmr_indices(
            candidates,
            token_scores,
            config.frame_budget,
            seed_indices=anchor_indices,
            features=feature_matrix,
            timestamps_sec=timestamps,
            lambda_param=config.mmr_lambda,
            visual_weight=config.mmr_visual_weight,
            temporal_scale_sec=config.temporal_redundancy_scale_sec,
            min_distance_sec=config.min_selection_distance_sec,
        )
        if selected is None:
            selected = _seeded_global_mmr_indices(
                candidates,
                token_scores,
                config.frame_budget,
                seed_indices=anchor_indices,
                features=feature_matrix,
                timestamps_sec=timestamps,
                lambda_param=config.mmr_lambda,
                visual_weight=config.mmr_visual_weight,
                temporal_scale_sec=config.temporal_redundancy_scale_sec,
                min_distance_sec=0.0,
            )
            if selected is None:  # pragma: no cover - zero distance is feasible
                raise RuntimeError("global MMR failed after distance relaxation")
            used_fallback = True
            reason = "min_selection_distance_relaxed"
            fallback_reason = (
                reason if fallback_reason is None else f"{fallback_reason};{reason}"
            )
    else:
        candidate_mask = full_support & input_valid & _local_maxima_mask(event)
        weights = np.maximum(event - threshold, 0.0)
        max_boundaries = max(
            0,
            config.frame_budget // config.min_frames_per_segment - 1,
        )
        if max_boundaries == 0 or timestamps.size < 3:
            boundaries = np.asarray([], dtype=int)
        else:
            boundaries = select_weighted_interval_indices(
                timestamps,
                weights,
                max_boundaries,
                config.min_boundary_distance_sec,
                valid_mask=candidate_mask,
                min_index_distance=1,
                edge_margin=1,
                segment_capacity_mask=selectable,
                min_segment_capacity=config.min_frames_per_segment,
            )
        boundaries, repaired = _repair_boundaries_for_capacity(
            boundaries,
            weights,
            selectable,
            config.min_frames_per_segment,
        )
        if repaired:
            used_fallback = True
            repair_reason = "segment_capacity_repair"
            fallback_reason = (
                repair_reason
                if fallback_reason is None
                else f"{fallback_reason};{repair_reason}"
            )
        if boundaries.size == 0 and not used_fallback:
            used_fallback = True
            fallback_reason = (
                "boundary_budget_zero"
                if max_boundaries == 0
                else "no_positive_boundary_evidence"
            )
        segments = _segments_from_boundaries(boundaries, timestamps.size)
        utilities = _segment_utilities(
            segments,
            timestamps,
            normalized_relevance,
            event,
            uncertainty,
            feature_matrix,
            selectable,
            config,
        )
        capacities = np.asarray(
            [int(np.sum(selectable[start:stop])) for start, stop in segments], dtype=int
        )
        allocation = allocate_exact_budget(
            capacities,
            utilities,
            config.frame_budget,
            min_per_segment=config.min_frames_per_segment,
            temperature=config.allocation_temperature,
        )
        selection_relevance = normalized_relevance + event
        selected_rows: list[int] = []
        for (start, stop), count in zip(segments, allocation):
            candidates = np.flatnonzero(selectable[start:stop]) + start
            selected_rows.extend(
                select_mmr_indices(
                    candidates,
                    selection_relevance,
                    int(count),
                    features=feature_matrix,
                    timestamps_sec=timestamps,
                    lambda_param=config.mmr_lambda,
                    visual_weight=config.mmr_visual_weight,
                    temporal_scale_sec=config.temporal_redundancy_scale_sec,
                ).tolist()
            )
        selected = np.asarray(sorted(selected_rows), dtype=int)
    if (
        selected.size != config.frame_budget
        or np.unique(selected).size != selected.size
        or not np.all(selectable[selected])
        or (
            source_indices is not None
            and np.unique(source_indices[selected]).size != selected.size
        )
    ):
        raise RuntimeError("PhaseFuse failed to produce an exact valid frame budget")
    return PhaseFuseTrace(
        config=config,
        timestamps_sec=timestamps.copy(),
        source_frame_indices=source_indices,
        input_valid_mask=input_valid,
        selectable_mask=selectable.copy(),
        input_phase_ids=input_phase_ids.copy(),
        phase_ids=effective_phase_ids.copy(),
        phase_indices=tuple(row.copy() for row in phase_index_rows),
        transform_methods=tuple(methods),
        aligned_phase_saliency=aligned,
        normalized_aligned_phase_saliency=normalized_aligned,
        phase_support_mask=support,
        full_support_mask=full_support,
        fusion_domain_mask=fusion_domain,
        consensus=consensus,
        uncertainty=uncertainty,
        phase_vote=phase_vote,
        normalized_relevance=normalized_relevance,
        event_scores=event,
        boundary_threshold=float(threshold),
        boundary_candidate_mask=candidate_mask,
        boundary_indices=boundaries,
        segments=segments,
        segment_utilities=utilities,
        allocation=allocation,
        anchor_indices=anchor_indices,
        selected_indices=selected,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
    )


class PhaseFuse:
    """Reusable object wrapper around :func:`run_phasefuse`."""

    def __init__(
        self,
        transform: TemporalTransform,
        config: PhaseFuseConfig | None = None,
    ) -> None:
        if not callable(getattr(transform, "transform", None)):
            raise TypeError("transform must provide transform(signal)")
        self.transform = transform
        self.config = config or PhaseFuseConfig()

    def run(
        self,
        timestamps_sec: Sequence[float],
        relevance_scores: Sequence[float],
        features: np.ndarray | None = None,
        *,
        phase_ids: Sequence[int] | None = None,
        valid_mask: Sequence[bool] | None = None,
        source_frame_indices: Sequence[int] | None = None,
    ) -> PhaseFuseTrace:
        return run_phasefuse(
            timestamps_sec,
            relevance_scores,
            features,
            config=self.config,
            transform=self.transform,
            phase_ids=phase_ids,
            valid_mask=valid_mask,
            source_frame_indices=source_frame_indices,
        )


__all__ = [
    "PhaseFuse",
    "PhaseFuseConfig",
    "PhaseFuseTrace",
    "align_phase_values",
    "allocate_exact_budget",
    "interleaved_phase_ids",
    "robust_phase_consensus",
    "run_phasefuse",
    "select_mmr_indices",
    "select_weighted_interval_indices",
]
