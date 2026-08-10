"""Traceable WFS-SB pipeline with a replaceable temporal transform."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wfs.core import WFSBudgetAllocator, WFSEventDetector, WFSFrameSelector

from .transforms import TemporalTransform, TransformResult


@dataclass
class SelectionTrace:
    """All intermediate values needed by the phase-stability experiments."""

    transform: TransformResult
    peaks: np.ndarray
    segments: List[Tuple[int, int]]
    importance_scores: List[float]
    valid_segments: List[Tuple[int, int]]
    valid_importance_scores: List[float]
    allocation: Dict[int, int]
    selected_indices: List[int]
    used_fallback: bool

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Convert the trace to JSON-compatible metadata."""

        payload: Dict[str, Any] = {
            "transform": self.transform.summary(),
            "peaks": self.peaks.astype(int).tolist(),
            "segments": [list(segment) for segment in self.segments],
            "importance_scores": [float(value) for value in self.importance_scores],
            "valid_segments": [list(segment) for segment in self.valid_segments],
            "valid_importance_scores": [
                float(value) for value in self.valid_importance_scores
            ],
            "allocation": {str(key): int(value) for key, value in self.allocation.items()},
            "selected_indices": [int(index) for index in self.selected_indices],
            "used_fallback": bool(self.used_fallback),
        }
        if include_arrays:
            payload["representation"] = self.transform.representation.tolist()
            payload["coarse_detail"] = self.transform.coarse_detail.tolist()
            payload["saliency"] = self.transform.saliency.tolist()
            payload["saliency_norm"] = self.transform.normalized_saliency.tolist()
        return payload


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for the unchanged post-transform WFS-SB stages."""

    height_factor: float = 0.5
    prominence_factor: float = 0.05
    w_duration: float = 0.4
    w_mean: float = 0.2
    w_max: float = 0.3
    w_var: float = 0.1
    strictness_factor: float = 1.2
    temperature: float = 1.0
    lambda_param: float = 0.5


class PhaseStableWFS:
    """Run WFS-SB while exposing every intermediate experimental artifact."""

    def __init__(
        self,
        transform: TemporalTransform,
        config: Optional[SelectionConfig] = None,
    ) -> None:
        self.transform = transform
        self.config = config or SelectionConfig()
        self.event_detector = WFSEventDetector(
            wavelet=self.transform.config.wavelet,
            height_factor=self.config.height_factor,
            prominence_factor=self.config.prominence_factor,
        )
        self.budget_allocator = WFSBudgetAllocator(
            w_duration=self.config.w_duration,
            w_mean=self.config.w_mean,
            w_max=self.config.w_max,
            w_var=self.config.w_var,
            strictness_factor=self.config.strictness_factor,
            temperature=self.config.temperature,
        )
        self.frame_selector = WFSFrameSelector(lambda_param=self.config.lambda_param)

    def run(
        self,
        relevance_scores: Sequence[float],
        num_frames: int,
        min_peak_distance: int,
        features: Optional[np.ndarray] = None,
    ) -> SelectionTrace:
        """Run transform, segmentation, budget allocation, and frame selection."""

        scores = np.asarray(relevance_scores, dtype=float)
        if scores.ndim != 1 or scores.size < 2:
            raise ValueError("relevance_scores must be a 1-D sequence of length >= 2")
        if not np.all(np.isfinite(scores)):
            raise ValueError("relevance_scores must contain only finite values")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if min_peak_distance <= 0:
            raise ValueError("min_peak_distance must be positive")
        if features is not None:
            features = np.asarray(features)
            if features.ndim < 2 or features.shape[0] != scores.size:
                raise ValueError("features must be frame-aligned with relevance_scores")

        transform_result = self.transform.transform(scores)
        peaks = self.event_detector.detect_peaks(
            transform_result.coarse_detail, min_peak_distance
        )
        segments = self.event_detector.create_segments(peaks, scores.size)

        if scores.size <= num_frames:
            return SelectionTrace(
                transform=transform_result,
                peaks=peaks,
                segments=segments,
                importance_scores=[],
                valid_segments=segments,
                valid_importance_scores=[],
                allocation={},
                selected_indices=list(range(scores.size)),
                used_fallback=True,
            )

        if len(segments) <= 1 or peaks.size == 0:
            selected = self._fallback_selection(scores, num_frames)
            return SelectionTrace(
                transform=transform_result,
                peaks=peaks,
                segments=segments,
                importance_scores=[],
                valid_segments=segments,
                valid_importance_scores=[],
                allocation={},
                selected_indices=selected,
                used_fallback=True,
            )

        importance = [
            self.budget_allocator.compute_importance(segment, scores, scores.size)
            for segment in segments
        ]
        valid_segments, valid_scores = self.budget_allocator.filter_segments(
            segments, importance
        )
        allocation = self.budget_allocator.allocate_budget(valid_scores, num_frames)

        normalized_scores = 2.0 * scores - 1.0
        selected: List[int] = []
        for segment_index, allocated in allocation.items():
            selected.extend(
                self.frame_selector.select_from_segment(
                    segment=valid_segments[segment_index],
                    n_frames=allocated,
                    relevance_scores=normalized_scores,
                    features=features,
                )
            )
        selected = self.frame_selector.adjust_to_budget(
            selected_frames=selected,
            target_budget=num_frames,
            relevance_scores=normalized_scores,
            features=features,
        )

        return SelectionTrace(
            transform=transform_result,
            peaks=peaks,
            segments=list(segments),
            importance_scores=[float(value) for value in importance],
            valid_segments=list(valid_segments),
            valid_importance_scores=[float(value) for value in valid_scores],
            allocation={int(key): int(value) for key, value in allocation.items()},
            selected_indices=[int(index) for index in selected],
            used_fallback=False,
        )

    @staticmethod
    def _fallback_selection(scores: np.ndarray, num_frames: int) -> List[int]:
        n_uniform = num_frames // 2
        n_top = num_frames - n_uniform
        uniform = np.linspace(0, scores.size - 1, n_uniform, dtype=int)
        top_candidates = np.argsort(scores)[::-1]
        top: List[int] = []
        uniform_set = set(int(index) for index in uniform)
        for index in top_candidates:
            if int(index) not in uniform_set:
                top.append(int(index))
                if len(top) >= n_top:
                    break
        return sorted([int(index) for index in uniform] + top)[:num_frames]

    def config_dict(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the selection configuration."""

        return asdict(self.config)
