"""Canonical risk-controlled residual video-token decisions.

The selector separates a phase-dependent scout pass from the final decode.
Scout relevance is interpolated onto an origin-zero physical time lattice.
Twelve deterministic coverage anchors are retained and four query-adaptive
residual targets are added. Returned timestamps are decision targets, not
indices into any scout decode.

The frozen RC12 primary does not use TI-DWT, visual features, or MMR in its
decision score. TI-DWT remains explicitly diagnostic after a cached ablation
showed that giving it decision weight was dominated by relevance-only RC12.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class RC12Config:
    """Frozen, training-free RC12 decision policy."""

    frame_budget: int = 16
    anchor_count: int = 12
    max_lattice_hz: float = 2.0
    smoothing_sigma_sec: float = 4.0
    score_quantum: float = 0.1
    min_residual_distance_sec: float = 2.0
    coverage_tiebreak_cap_sec: float = 8.0
    lattice_origin_sec: float = 0.0

    def __post_init__(self) -> None:
        for name in ("frame_budget", "anchor_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if self.frame_budget <= 0:
            raise ValueError("frame_budget must be positive")
        if not 0 <= self.anchor_count <= self.frame_budget:
            raise ValueError("anchor_count must lie in [0, frame_budget]")
        for name in (
            "max_lattice_hz",
            "smoothing_sigma_sec",
            "score_quantum",
            "coverage_tiebreak_cap_sec",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.min_residual_distance_sec)
            or self.min_residual_distance_sec < 0.0
        ):
            raise ValueError(
                "min_residual_distance_sec must be finite and non-negative"
            )
        if not np.isfinite(self.lattice_origin_sec):
            raise ValueError("lattice_origin_sec must be finite")


@dataclass(frozen=True)
class RC12Decision:
    """Auditable canonical decision returned by select_rc12_targets."""

    config: RC12Config
    support_start_sec: float
    support_stop_sec: float
    scout_step_sec: float
    decision_step_sec: float
    lattice_timestamps_sec: np.ndarray
    interpolated_relevance: np.ndarray
    percentile_relevance: np.ndarray
    smoothed_relevance: np.ndarray
    quantized_scores: np.ndarray
    anchor_indices: np.ndarray
    residual_indices: np.ndarray
    selected_indices: np.ndarray
    distance_relaxation_used: bool

    @property
    def target_timestamps_sec(self) -> np.ndarray:
        return self.lattice_timestamps_sec[self.selected_indices]

    def to_dict(
        self, *, include_arrays: bool = False, method: str = "rc12"
    ) -> dict[str, Any]:
        if method not in {"rc12", "rc14"}:
            raise ValueError("canonical residual method must be rc12 or rc14")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "method": method,
            "config": asdict(self.config),
            "support_start_sec": float(self.support_start_sec),
            "support_stop_sec": float(self.support_stop_sec),
            "scout_step_sec": float(self.scout_step_sec),
            "decision_step_sec": float(self.decision_step_sec),
            "anchor_indices": self.anchor_indices.astype(int).tolist(),
            "anchor_timestamps_sec": self.lattice_timestamps_sec[
                self.anchor_indices
            ]
            .astype(float)
            .tolist(),
            "residual_indices": self.residual_indices.astype(int).tolist(),
            "residual_timestamps_sec": self.lattice_timestamps_sec[
                self.residual_indices
            ]
            .astype(float)
            .tolist(),
            "selected_indices": self.selected_indices.astype(int).tolist(),
            "target_timestamps_sec": self.target_timestamps_sec.astype(float).tolist(),
            "distance_relaxation_used": bool(self.distance_relaxation_used),
            "decision_signal": "smoothed_query_relevance_only",
            "ti_dwt_role": "diagnostic_only",
        }
        if include_arrays:
            payload["lattice_timestamps_sec"] = self.lattice_timestamps_sec.astype(
                float
            ).tolist()
            payload["interpolated_relevance"] = self.interpolated_relevance.astype(
                float
            ).tolist()
            payload["percentile_relevance"] = self.percentile_relevance.astype(
                float
            ).tolist()
            payload["smoothed_relevance"] = self.smoothed_relevance.astype(
                float
            ).tolist()
            payload["quantized_scores"] = self.quantized_scores.astype(float).tolist()
        return payload


def _finite_vector(name: str, values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with at least two values")
    return result


def percentile_rank(values: Sequence[float]) -> np.ndarray:
    """Return tie-aware empirical percentile ranks in [0, 1]."""

    array = _finite_vector("values", values)
    unique, inverse, counts = np.unique(
        array, return_inverse=True, return_counts=True
    )
    if unique.size == 1:
        return np.zeros(array.size, dtype=float)
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))
    midranks = starts + (counts - 1) / 2.0
    return np.asarray(midranks[inverse] / float(array.size - 1), dtype=float)


def canonical_lattice(
    scout_timestamps_sec: Sequence[float],
    *,
    support_start_sec: float,
    support_stop_sec: float,
    config: RC12Config,
) -> tuple[np.ndarray, float, float]:
    """Build the origin-zero decision lattice used by RC12.

    The lattice never upsamples beyond the median scout cadence and is capped
    at max_lattice_hz. This gives a 0.5-second lattice for the cached VideoMME
    4 Hz scouts and a 1-second lattice for QVHighlights 1 Hz scouts.
    """

    scout = _finite_vector("scout_timestamps_sec", scout_timestamps_sec)
    if np.any(np.diff(scout) <= 0.0):
        raise ValueError("scout_timestamps_sec must be strictly increasing")
    for name, value in (
        ("support_start_sec", support_start_sec),
        ("support_stop_sec", support_stop_sec),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if support_start_sec >= support_stop_sec:
        raise ValueError("support_start_sec must be smaller than support_stop_sec")
    tolerance = 1e-9
    if (
        support_start_sec < scout[0] - tolerance
        or support_stop_sec > scout[-1] + tolerance
    ):
        raise ValueError("canonical support must lie inside the scout support")

    scout_step = float(np.median(np.diff(scout)))
    decision_step = max(scout_step, 1.0 / float(config.max_lattice_hz))
    first_tick = int(
        np.ceil(
            (float(support_start_sec) - config.lattice_origin_sec - tolerance)
            / decision_step
        )
    )
    last_tick = int(
        np.floor(
            (float(support_stop_sec) - config.lattice_origin_sec + tolerance)
            / decision_step
        )
    )
    lattice = config.lattice_origin_sec + decision_step * np.arange(
        first_tick, last_tick + 1, dtype=float
    )
    if lattice.size < config.frame_budget:
        raise ValueError("canonical support has fewer lattice points than frame_budget")
    return lattice, scout_step, decision_step


def _uniform_anchor_indices(lattice: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.asarray([], dtype=int)
    fractions = (np.arange(count, dtype=float) + 0.5) / float(count)
    targets = lattice[0] + fractions * (lattice[-1] - lattice[0])
    available = np.ones(lattice.size, dtype=bool)
    selected: list[int] = []
    for target in targets:
        distances = np.where(available, np.abs(lattice - target), np.inf)
        index = int(np.argmin(distances))
        selected.append(index)
        available[index] = False
    return np.asarray(sorted(selected), dtype=int)


def select_canonical_uniform_targets(
    scout_timestamps_sec: Sequence[float],
    *,
    support_start_sec: float,
    support_stop_sec: float,
    config: RC12Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full canonical lattice and K=16 uniform target indices.

    Fresh-decode duplicate repair must use the full lattice dynamically against
    the set of accepted decoded frames; it must not consume a static backup list.
    """

    resolved = config or RC12Config()
    lattice, _, _ = canonical_lattice(
        scout_timestamps_sec,
        support_start_sec=support_start_sec,
        support_stop_sec=support_stop_sec,
        config=resolved,
    )
    selected = _uniform_anchor_indices(lattice, resolved.frame_budget)
    return lattice, selected


def select_rc12_targets(
    scout_timestamps_sec: Sequence[float],
    relevance_scores: Sequence[float],
    *,
    support_start_sec: float,
    support_stop_sec: float,
    config: RC12Config | None = None,
) -> RC12Decision:
    """Select exact canonical targets from one phase-dependent scout pass."""

    resolved = config or RC12Config()
    scout = _finite_vector("scout_timestamps_sec", scout_timestamps_sec)
    relevance = _finite_vector("relevance_scores", relevance_scores)
    if relevance.shape != scout.shape:
        raise ValueError("relevance_scores must align with scout_timestamps_sec")
    if np.any(np.diff(scout) <= 0.0):
        raise ValueError("scout_timestamps_sec must be strictly increasing")
    lattice, scout_step, decision_step = canonical_lattice(
        scout,
        support_start_sec=support_start_sec,
        support_stop_sec=support_stop_sec,
        config=resolved,
    )
    interpolated = np.interp(lattice, scout, relevance)
    ranked = percentile_rank(interpolated)
    smoothed = gaussian_filter1d(
        ranked,
        sigma=float(resolved.smoothing_sigma_sec) / decision_step,
        mode="nearest",
    )
    quantized = (
        np.floor(smoothed / float(resolved.score_quantum))
        * float(resolved.score_quantum)
    )

    anchors = _uniform_anchor_indices(lattice, resolved.anchor_count)
    selected = set(int(value) for value in anchors)
    residuals: list[int] = []
    relaxed = False
    while len(selected) < resolved.frame_budget:
        remaining = [index for index in range(lattice.size) if index not in selected]
        eligible = [
            index
            for index in remaining
            if all(
                abs(float(lattice[index] - lattice[prior]))
                >= resolved.min_residual_distance_sec
                for prior in selected
            )
        ]
        if not eligible:
            eligible = remaining
            relaxed = True
        best = max(
            eligible,
            key=lambda index: (
                float(quantized[index]),
                min(
                    min(
                        abs(float(lattice[index] - lattice[prior]))
                        for prior in selected
                    ),
                    float(resolved.coverage_tiebreak_cap_sec),
                ),
                -index,
            ),
        )
        selected.add(best)
        residuals.append(best)

    selected_indices = np.asarray(sorted(selected), dtype=int)
    return RC12Decision(
        config=resolved,
        support_start_sec=float(support_start_sec),
        support_stop_sec=float(support_stop_sec),
        scout_step_sec=scout_step,
        decision_step_sec=decision_step,
        lattice_timestamps_sec=lattice,
        interpolated_relevance=interpolated,
        percentile_relevance=ranked,
        smoothed_relevance=smoothed,
        quantized_scores=quantized,
        anchor_indices=anchors,
        residual_indices=np.asarray(sorted(residuals), dtype=int),
        selected_indices=selected_indices,
        distance_relaxation_used=relaxed,
    )


__all__ = [
    "RC12Config",
    "RC12Decision",
    "canonical_lattice",
    "percentile_rank",
    "select_canonical_uniform_targets",
    "select_rc12_targets",
]
