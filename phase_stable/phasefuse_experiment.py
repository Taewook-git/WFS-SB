"""Compute-matched PhaseFuse selection experiment on cached dense signals."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from wfs.core import compute_dwt_level

from .artifacts import OriginSignalRecord, artifact_id, write_jsonl
from .phasefuse import (
    PhaseFuseConfig,
    PhaseFuseTrace,
    run_phasefuse,
    select_mmr_indices,
)
from .pipeline import PhaseStableWFS, SelectionConfig, SelectionTrace
from .repro import sha256_file
from .transforms import TransformConfig, build_transform

PHASEFUSE_DEFAULT_METHODS = (
    "uniform_dense",
    "dense_topk_mmr",
    "single_dwt",
    "single_swt",
    "multiphase_dwt",
    "multiphase_swt_mean",
    "dense_swt",
    "phasefuse",
)
PHASEFUSE_METHODS = (*PHASEFUSE_DEFAULT_METHODS, "phasefuse_v2")


@dataclass(frozen=True)
class PhaseFuseFileConfig:
    experiment: PhaseFuseExperimentConfig
    sampling: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PhaseFuseExperimentConfig:
    """Frozen selection protocol shared by every compute-matched arm."""

    # V2 is supported but opt-in: upgrading must not add an expensive MLLM arm.
    methods: tuple[str, ...] = PHASEFUSE_DEFAULT_METHODS
    frame_budget: int = 16
    num_phases: int = 4
    min_frames_per_segment: int = 2
    wavelet: str = "db4"
    level: int | None = None
    drift_level: int = 3
    dwt_mode: str = "symmetric"
    shared_padding: bool = True
    padding_mode: str = "reflect"
    swt_norm: bool = True
    min_boundary_distance_sec: float = 5.0
    boundary_threshold_mad: float = 0.5
    uncertainty_penalty: float = 0.25
    relevance_weight: float = 0.5
    phase_vote_weight: float = 0.25
    segment_duration_weight: float = 0.2
    segment_relevance_weight: float = 0.4
    segment_event_weight: float = 0.3
    segment_diversity_weight: float = 0.1
    segment_uncertainty_penalty: float = 0.1
    allocation_temperature: float = 1.0
    mmr_lambda: float = 0.7
    mmr_visual_weight: float = 0.75
    temporal_redundancy_scale_sec: float = 2.0
    # PhaseFuse-v2 selection controls. Original arms deliberately ignore these.
    selection_strategy: Literal["segmented", "global_coverage"] = "global_coverage"
    uniform_reserve: int = 8
    selection_event_weight: float = 0.25
    component_scaling: Literal["robust_z", "percentile"] = "percentile"
    min_selection_distance_sec: float = 0.5
    legacy_selection: SelectionConfig = field(default_factory=SelectionConfig)

    def __post_init__(self) -> None:
        methods = tuple(str(method) for method in self.methods)
        if not methods or len(methods) != len(set(methods)):
            raise ValueError("methods must be non-empty and unique")
        unknown = sorted(set(methods) - set(PHASEFUSE_METHODS))
        if unknown:
            raise ValueError(f"unsupported PhaseFuse methods: {unknown}")
        for name in (
            "frame_budget",
            "num_phases",
            "min_frames_per_segment",
            "drift_level",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.level is not None and (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level <= 0
        ):
            raise ValueError("level must be a positive integer or null")
        if self.min_frames_per_segment > self.frame_budget:
            raise ValueError("min_frames_per_segment cannot exceed frame_budget")
        if (
            isinstance(self.uniform_reserve, bool)
            or not isinstance(self.uniform_reserve, int)
            or self.uniform_reserve < 0
        ):
            raise ValueError("uniform_reserve must be a non-negative integer")
        if self.selection_strategy not in {"segmented", "global_coverage"}:
            raise ValueError(
                "selection_strategy must be 'segmented' or 'global_coverage'"
            )
        if self.component_scaling not in {"robust_z", "percentile"}:
            raise ValueError("component_scaling must be 'robust_z' or 'percentile'")
        if "phasefuse_v2" in methods:
            if self.num_phases != 4:
                raise ValueError("phasefuse_v2 requires exactly four physical phases")
            if self.selection_strategy != "global_coverage":
                raise ValueError("phasefuse_v2 requires global_coverage selection")
            if self.uniform_reserve > self.frame_budget:
                raise ValueError(
                    "phasefuse_v2 uniform_reserve cannot exceed frame_budget"
                )
        numeric = (
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
            "allocation_temperature",
            "mmr_lambda",
            "mmr_visual_weight",
            "temporal_redundancy_scale_sec",
            "selection_event_weight",
            "min_selection_distance_sec",
        )
        for name in numeric:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.selection_event_weight < 0:
            raise ValueError("selection_event_weight must be non-negative")
        if self.min_selection_distance_sec < 0:
            raise ValueError("min_selection_distance_sec must be non-negative")
        object.__setattr__(self, "methods", methods)

    def selector_config(
        self,
        *,
        num_phases: int | None = None,
        consensus: str = "median",
        uncertainty_penalty: float | None = None,
        relevance_weight: float | None = None,
        phase_vote_weight: float | None = None,
        use_v2_selection: bool = False,
    ) -> PhaseFuseConfig:
        strategy_options: dict[str, Any] = {}
        if use_v2_selection:
            strategy_options = {
                "selection_strategy": self.selection_strategy,
                "uniform_reserve": self.uniform_reserve,
                "selection_event_weight": self.selection_event_weight,
                "component_scaling": self.component_scaling,
                "min_selection_distance_sec": self.min_selection_distance_sec,
            }
        return PhaseFuseConfig(
            num_phases=self.num_phases if num_phases is None else num_phases,
            frame_budget=self.frame_budget,
            min_frames_per_segment=self.min_frames_per_segment,
            min_boundary_distance_sec=self.min_boundary_distance_sec,
            boundary_threshold_mad=self.boundary_threshold_mad,
            uncertainty_penalty=(
                self.uncertainty_penalty
                if uncertainty_penalty is None
                else uncertainty_penalty
            ),
            relevance_weight=(
                self.relevance_weight if relevance_weight is None else relevance_weight
            ),
            phase_vote_weight=(
                self.phase_vote_weight
                if phase_vote_weight is None
                else phase_vote_weight
            ),
            consensus=consensus,  # type: ignore[arg-type]
            segment_duration_weight=self.segment_duration_weight,
            segment_relevance_weight=self.segment_relevance_weight,
            segment_event_weight=self.segment_event_weight,
            segment_diversity_weight=self.segment_diversity_weight,
            segment_uncertainty_penalty=self.segment_uncertainty_penalty,
            allocation_temperature=self.allocation_temperature,
            mmr_lambda=self.mmr_lambda,
            mmr_visual_weight=self.mmr_visual_weight,
            temporal_redundancy_scale_sec=self.temporal_redundancy_scale_sec,
            **strategy_options,
        )


@dataclass(frozen=True)
class _SelectionResult:
    selected_indices: np.ndarray
    boundary_indices: np.ndarray
    segments: tuple[tuple[int, int], ...]
    allocation: tuple[int, ...]
    used_fallback: bool
    fallback_reason: str | None
    metadata: Mapping[str, Any]
    dense_arrays: Mapping[str, np.ndarray]


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"record metadata is missing {name!r}")
    return value


def _record_contract(
    record: OriginSignalRecord, config: PhaseFuseExperimentConfig
) -> tuple[np.ndarray, np.ndarray, float, float]:
    contract = _mapping(record.metadata, "multiphase")
    if int(contract.get("num_phases", -1)) != config.num_phases:
        raise ValueError("record num_phases does not match the experiment config")
    phase_ids = np.asarray(contract.get("phase_ids"), dtype=int)
    if phase_ids.shape != (len(record.timestamps_sec),):
        raise ValueError("record phase_ids are not dense-target aligned")
    support = np.asarray(contract.get("manifest_common_valid_support_sec"), dtype=float)
    if (
        support.shape != (2,)
        or not np.all(np.isfinite(support))
        or support[1] < support[0]
    ):
        raise ValueError("record has invalid manifest_common_valid_support_sec")
    base_fps = float(contract.get("base_sample_fps", 0.0))
    dense_fps = float(contract.get("dense_sample_fps", 0.0))
    if (
        not math.isfinite(base_fps)
        or not math.isfinite(dense_fps)
        or base_fps <= 0
        or not math.isclose(dense_fps, base_fps * config.num_phases)
    ):
        raise ValueError("record has an invalid physical sampling-rate contract")
    return phase_ids, support, base_fps, dense_fps


def _load_features(record: OriginSignalRecord) -> np.ndarray:
    if record.visual_features_path is None:
        raise ValueError("PhaseFuse requires dense visual features")
    path = Path(record.visual_features_path)
    if not path.is_file():
        raise FileNotFoundError(f"visual feature file does not exist: {path}")
    contract = _mapping(record.metadata, "multiphase")
    expected_sha = contract.get("visual_features_sha256")
    expected_size = contract.get("visual_features_size_bytes")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("dense feature contract has no valid SHA-256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_sha
    ):
        raise ValueError(f"dense visual feature provenance mismatch: {path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(record.timestamps_sec)
        or matrix.shape[1] == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError(f"dense visual features are invalid: {path}")
    return matrix


def _valid_unique_indices(
    record: OriginSignalRecord,
    support: np.ndarray,
    *,
    phase_ids: np.ndarray | None = None,
    phase_id: int | None = None,
) -> np.ndarray:
    timestamps = np.asarray(record.timestamps_sec, dtype=float)
    sources = np.asarray(record.source_frame_indices, dtype=int)
    mask = (timestamps >= support[0] - 1e-12) & (timestamps <= support[1] + 1e-12)
    if phase_id is not None:
        if phase_ids is None:
            raise ValueError("phase_ids are required for phase filtering")
        mask &= phase_ids == phase_id
    selected: list[int] = []
    seen: set[int] = set()
    for index in np.flatnonzero(mask):
        source = int(sources[index])
        if source not in seen:
            selected.append(int(index))
            seen.add(source)
    return np.asarray(selected, dtype=int)


def _transform_config(
    method: str, level: int, config: PhaseFuseExperimentConfig
) -> TransformConfig:
    return TransformConfig(
        method=method,  # type: ignore[arg-type]
        wavelet=config.wavelet,
        level=level,
        dwt_mode=config.dwt_mode,
        shared_padding=config.shared_padding,
        padding_mode=config.padding_mode,
        swt_norm=config.swt_norm,
    )


def _phase_level(phase_count: int, config: PhaseFuseExperimentConfig) -> int:
    return config.level or compute_dwt_level(
        phase_count, config.wavelet, config.drift_level
    )


def _phasefuse_result(trace: PhaseFuseTrace, *, kind: str) -> _SelectionResult:
    selected_uncertainty = trace.uncertainty[trace.selected_indices]
    supported_uncertainty = trace.uncertainty[trace.fusion_domain_mask]
    if supported_uncertainty.size == 0:
        raise RuntimeError("PhaseFuse trace has no common-support uncertainty values")
    metadata = {
        "selector_kind": kind,
        "phasefuse": trace.to_dict(include_arrays=False),
        "phase_uncertainty_mean": float(np.mean(supported_uncertainty)),
        "selected_phase_uncertainty_mean": float(np.mean(selected_uncertainty)),
        "selected_phase_uncertainty_max": float(np.max(selected_uncertainty)),
        "zero_allocation_segments": int(np.sum(trace.allocation == 0)),
        "max_boundary_count": int(
            trace.config.frame_budget // trace.config.min_frames_per_segment - 1
        ),
        "selection_strategy": trace.config.selection_strategy,
        "component_scaling": trace.config.component_scaling,
        "uniform_reserve": int(trace.config.uniform_reserve),
        "selection_event_weight": float(trace.config.selection_event_weight),
        "min_selection_distance_sec": float(trace.config.min_selection_distance_sec),
    }
    arrays = {
        "consensus": trace.consensus,
        "uncertainty": trace.uncertainty,
        "phase_vote": trace.phase_vote,
        "event_scores": trace.event_scores,
        "normalized_relevance": trace.normalized_relevance,
        "full_support_mask": trace.full_support_mask,
        "fusion_domain_mask": trace.fusion_domain_mask,
        "boundary_candidate_mask": trace.boundary_candidate_mask,
    }
    return _SelectionResult(
        selected_indices=trace.selected_indices,
        boundary_indices=trace.boundary_indices,
        segments=trace.segments,
        allocation=tuple(int(value) for value in trace.allocation),
        used_fallback=trace.used_fallback,
        fallback_reason=trace.fallback_reason,
        metadata=metadata,
        dense_arrays=arrays,
    )


def _run_fused(
    record: OriginSignalRecord,
    features: np.ndarray,
    phase_ids: np.ndarray,
    support: np.ndarray,
    config: PhaseFuseExperimentConfig,
    *,
    transform_method: str,
    consensus: str = "median",
    uncertainty_penalty: float | None = None,
    relevance_weight: float | None = None,
    phase_vote_weight: float | None = None,
    dense_single_stream: bool = False,
    use_v2_selection: bool = False,
) -> _SelectionResult:
    timestamps = np.asarray(record.timestamps_sec, dtype=float)
    valid = (timestamps >= support[0] - 1e-12) & (timestamps <= support[1] + 1e-12)
    phase_count = int(np.sum(phase_ids == 0))
    level = _phase_level(phase_count, config)
    selector_config = config.selector_config(
        num_phases=1 if dense_single_stream else config.num_phases,
        consensus=consensus,
        uncertainty_penalty=uncertainty_penalty,
        relevance_weight=relevance_weight,
        phase_vote_weight=phase_vote_weight,
        use_v2_selection=use_v2_selection,
    )
    if dense_single_stream:
        level += round(math.log2(config.num_phases))
        selected_phase_ids = np.zeros_like(phase_ids)
    else:
        selected_phase_ids = phase_ids
    transform = build_transform(_transform_config(transform_method, level, config))
    trace = run_phasefuse(
        record.timestamps_sec,
        record.relevance_scores,
        features,
        config=selector_config,
        transform=transform,
        phase_ids=selected_phase_ids,
        valid_mask=valid,
        source_frame_indices=record.source_frame_indices,
    )
    return _phasefuse_result(
        trace,
        kind=(
            "direct_dense_single_stream" if dense_single_stream else "multiphase_fusion"
        ),
    )


def _legacy_segments(
    trace: SelectionTrace, phase_indices: np.ndarray
) -> tuple[tuple[int, int], ...]:
    mapped: list[tuple[int, int]] = []
    for start, stop in trace.segments:
        dense_start = int(phase_indices[start])
        dense_stop = int(phase_indices[stop - 1]) + 1
        mapped.append((dense_start, dense_stop))
    return tuple(mapped)


def _run_legacy(
    record: OriginSignalRecord,
    features: np.ndarray,
    phase_ids: np.ndarray,
    support: np.ndarray,
    base_fps: float,
    config: PhaseFuseExperimentConfig,
    method: str,
) -> _SelectionResult:
    candidates = _valid_unique_indices(record, support, phase_ids=phase_ids, phase_id=0)
    if candidates.size < config.frame_budget:
        raise ValueError(
            f"single-phase {method} has fewer unique candidates than frame_budget"
        )
    level = _phase_level(candidates.size, config)
    transform = build_transform(_transform_config(method, level, config))
    selector = PhaseStableWFS(transform, config.legacy_selection)
    trace = selector.run(
        np.asarray(record.relevance_scores, dtype=float)[candidates],
        config.frame_budget,
        max(1, math.ceil(config.min_boundary_distance_sec * base_fps)),
        features[candidates],
    )
    selected = candidates[np.asarray(trace.selected_indices, dtype=int)]
    boundaries = candidates[np.asarray(trace.peaks, dtype=int)]
    allocation = tuple(int(value) for _, value in sorted(trace.allocation.items()))
    return _SelectionResult(
        selected_indices=np.asarray(sorted(selected.tolist()), dtype=int),
        boundary_indices=np.asarray(sorted(boundaries.tolist()), dtype=int),
        segments=_legacy_segments(trace, candidates),
        allocation=allocation,
        used_fallback=trace.used_fallback,
        fallback_reason="legacy_wfs_fallback" if trace.used_fallback else None,
        metadata={
            "selector_kind": "legacy_single_phase_wfs",
            "phase_id": 0,
            "candidate_count": int(candidates.size),
            "transform": trace.transform.summary(),
        },
        dense_arrays={
            "legacy_representation": trace.transform.representation,
            "legacy_saliency": trace.transform.saliency,
        },
    )


def _run_baseline(
    record: OriginSignalRecord,
    features: np.ndarray,
    support: np.ndarray,
    config: PhaseFuseExperimentConfig,
    method: str,
) -> _SelectionResult:
    candidates = _valid_unique_indices(record, support)
    if candidates.size < config.frame_budget:
        raise ValueError(f"{method} has fewer unique candidates than frame_budget")
    if method == "uniform_dense":
        positions = np.linspace(0, candidates.size - 1, config.frame_budget, dtype=int)
        selected = candidates[positions]
    else:
        selected = select_mmr_indices(
            candidates,
            record.relevance_scores,
            config.frame_budget,
            features=features,
            timestamps_sec=record.timestamps_sec,
            lambda_param=config.mmr_lambda,
            visual_weight=config.mmr_visual_weight,
            temporal_scale_sec=config.temporal_redundancy_scale_sec,
        )
    return _SelectionResult(
        selected_indices=np.asarray(sorted(selected.tolist()), dtype=int),
        boundary_indices=np.asarray([], dtype=int),
        segments=((0, len(record.timestamps_sec)),),
        allocation=(config.frame_budget,),
        used_fallback=False,
        fallback_reason=None,
        metadata={
            "selector_kind": method,
            "candidate_count": int(candidates.size),
        },
        dense_arrays={},
    )


def run_phasefuse_method(
    record: OriginSignalRecord,
    method: str,
    config: PhaseFuseExperimentConfig,
    *,
    features: np.ndarray | None = None,
) -> _SelectionResult:
    """Run one registered arm while preserving a shared dense candidate grid."""

    if method not in PHASEFUSE_METHODS:
        raise ValueError(f"unsupported PhaseFuse method: {method}")
    phase_ids, support, base_fps, _ = _record_contract(record, config)
    stable_features = _load_features(record) if features is None else features
    if method in {"uniform_dense", "dense_topk_mmr"}:
        return _run_baseline(record, stable_features, support, config, method)
    if method in {"single_dwt", "single_swt"}:
        return _run_legacy(
            record,
            stable_features,
            phase_ids,
            support,
            base_fps,
            config,
            method.removeprefix("single_"),
        )
    if method == "multiphase_dwt":
        return _run_fused(
            record,
            stable_features,
            phase_ids,
            support,
            config,
            transform_method="dwt",
        )
    if method == "multiphase_swt_mean":
        return _run_fused(
            record,
            stable_features,
            phase_ids,
            support,
            config,
            transform_method="swt",
            consensus="mean",
            uncertainty_penalty=0.0,
        )
    if method == "dense_swt":
        return _run_fused(
            record,
            stable_features,
            phase_ids,
            support,
            config,
            transform_method="swt",
            uncertainty_penalty=0.0,
            dense_single_stream=True,
        )
    if method == "phasefuse_v2":
        return _run_fused(
            record,
            stable_features,
            phase_ids,
            support,
            config,
            transform_method="swt",
            consensus="median",
            uncertainty_penalty=0.0,
            relevance_weight=0.0,
            phase_vote_weight=0.0,
            use_v2_selection=True,
        )
    return _run_fused(
        record,
        stable_features,
        phase_ids,
        support,
        config,
        transform_method="swt",
    )


def _validate_selection(
    record: OriginSignalRecord,
    selection: _SelectionResult,
    config: PhaseFuseExperimentConfig,
) -> None:
    selected = selection.selected_indices
    if (
        selected.shape != (config.frame_budget,)
        or selected.dtype.kind not in "iu"
        or np.any(np.diff(selected) <= 0)
        or selected[0] < 0
        or selected[-1] >= len(record.timestamps_sec)
    ):
        raise RuntimeError("method did not produce an exact sorted frame budget")
    sources = np.asarray(record.source_frame_indices, dtype=int)[selected]
    if np.unique(sources).size != config.frame_budget or np.any(np.diff(sources) <= 0):
        raise RuntimeError("selected source frames must be unique and increasing")


def save_phasefuse_trace(
    output_dir: str | Path,
    record: OriginSignalRecord,
    method: str,
    selection: _SelectionResult,
    config: PhaseFuseExperimentConfig,
) -> dict[str, Any]:
    """Persist one method trace in the generic lmms-export contract."""

    _validate_selection(record, selection, config)
    root = Path(output_dir)
    arrays_dir = root / "trace_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    array_path = (
        arrays_dir / f"{artifact_id(*record.item_key, record.origin_id, method)}.npz"
    )
    np.savez_compressed(
        array_path,
        relevance_scores=np.asarray(record.relevance_scores, dtype=float),
        **{name: np.asarray(value) for name, value in selection.dense_arrays.items()},
    )
    selected = selection.selected_indices
    boundaries = selection.boundary_indices
    timestamps = np.asarray(record.timestamps_sec, dtype=float)
    actual = np.asarray(record.actual_pts_sec, dtype=float)
    sources = np.asarray(record.source_frame_indices, dtype=int)
    hashes = np.asarray(record.pixel_hashes, dtype=str) if record.pixel_hashes else None
    metadata = dict(selection.metadata)
    row: dict[str, Any] = {
        "schema_version": 1,
        "dataset": record.dataset,
        "video_id": record.video_id,
        "question_id": record.question_id,
        "origin_id": record.origin_id,
        "origin_sec": record.origin_sec,
        "method": method,
        "timestamps_sec": list(record.timestamps_sec),
        "actual_pts_sec": list(record.actual_pts_sec),
        "source_frame_indices": list(record.source_frame_indices),
        "pixel_hashes": list(record.pixel_hashes),
        "visual_features_path": record.visual_features_path,
        "record_metadata": dict(record.metadata),
        "array_path": str(array_path.resolve()),
        "peaks": boundaries.astype(int).tolist(),
        "peaks_sec": timestamps[boundaries].astype(float).tolist(),
        "segments": [list(segment) for segment in selection.segments],
        "valid_segments": [list(segment) for segment in selection.segments],
        "importance_scores": [],
        "valid_importance_scores": [],
        "allocation": {
            str(index): int(value) for index, value in enumerate(selection.allocation)
        },
        "selected_indices": selected.astype(int).tolist(),
        "selected_timestamps_sec": timestamps[selected].astype(float).tolist(),
        "selected_actual_pts_sec": actual[selected].astype(float).tolist(),
        "selected_source_frame_indices": sources[selected].astype(int).tolist(),
        "selected_pixel_hashes": (
            [] if hashes is None else hashes[selected].astype(str).tolist()
        ),
        "used_fallback": bool(selection.used_fallback),
        "fallback_reason": selection.fallback_reason,
        "method_metadata": metadata,
        "transform": metadata.get("transform"),
    }
    for name in (
        "phase_uncertainty_mean",
        "selected_phase_uncertainty_mean",
        "selected_phase_uncertainty_max",
        "zero_allocation_segments",
        "max_boundary_count",
        "selection_strategy",
        "component_scaling",
        "uniform_reserve",
        "selection_event_weight",
        "min_selection_distance_sec",
    ):
        if name in metadata:
            row[name] = metadata[name]
    return row


def run_phasefuse_experiment(
    records: Sequence[OriginSignalRecord],
    output_dir: str | Path,
    *,
    config: PhaseFuseExperimentConfig | None = None,
    methods: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run every arm for a rectangular dense-record grid and write traces."""

    experiment = config or PhaseFuseExperimentConfig()
    selected_methods = experiment.methods if methods is None else tuple(methods)
    if not selected_methods or len(selected_methods) != len(set(selected_methods)):
        raise ValueError("methods must be non-empty and unique")
    if any(method not in experiment.methods for method in selected_methods):
        raise ValueError("requested methods must be enabled by the experiment config")
    stable_records = tuple(records)
    if not stable_records or any(
        not isinstance(record, OriginSignalRecord) for record in stable_records
    ):
        raise ValueError("records must contain OriginSignalRecord objects")
    keys: set[tuple[str, str, str, int]] = set()
    for record in stable_records:
        key = (*record.item_key, record.origin_id)
        if key in keys:
            raise ValueError(f"duplicate dense record: {key}")
        keys.add(key)

    rows: list[dict[str, Any]] = []
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for record in sorted(
        stable_records,
        key=lambda value: (*value.item_key, value.origin_id),
    ):
        features = _load_features(record)
        for method in selected_methods:
            selection = run_phasefuse_method(
                record, method, experiment, features=features
            )
            rows.append(
                save_phasefuse_trace(root, record, method, selection, experiment)
            )
    write_jsonl(root / "traces.jsonl", rows)
    summary = {
        "command": "run-phasefuse",
        "config": {
            **asdict(experiment),
            "legacy_selection": asdict(experiment.legacy_selection),
        },
        "methods": list(selected_methods),
        "num_signal_records": len(stable_records),
        "num_trace_rows": len(rows),
        "frame_budget": experiment.frame_budget,
        "protocol": {
            "outer_origins_are_evaluation_perturbations": True,
            "inner_phases_are_method_inputs": True,
            "outer_origins_never_fused": True,
            "shared_dense_candidate_grid": True,
            "exact_source_frame_budget": True,
        },
    }
    temporary = root / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(root / "summary.json")
    return rows


def config_from_mapping(payload: Mapping[str, Any]) -> PhaseFuseExperimentConfig:
    """Build a strict config from the ``phasefuse`` YAML section."""

    if not isinstance(payload, Mapping):
        raise TypeError("phasefuse config must be a mapping")
    values = dict(payload)
    if "methods" in values:
        raw_methods = values["methods"]
        if isinstance(raw_methods, (str, bytes)) or not isinstance(
            raw_methods, Sequence
        ):
            raise ValueError("phasefuse.methods must be a sequence")
        values["methods"] = tuple(raw_methods)
    legacy = values.get("legacy_selection")
    if legacy is not None:
        if not isinstance(legacy, Mapping):
            raise ValueError("phasefuse.legacy_selection must be a mapping")
        values["legacy_selection"] = SelectionConfig(**dict(legacy))
    try:
        return PhaseFuseExperimentConfig(**values)
    except TypeError as exc:
        raise ValueError(f"unknown or invalid PhaseFuse config field: {exc}") from exc


def load_phasefuse_config(path: str | Path) -> PhaseFuseFileConfig:
    """Load the strict ``phasefuse`` and physical-sampling YAML sections."""

    try:
        yaml = importlib.import_module("yaml")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyYAML is required to read PhaseFuse config files") from exc
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("PhaseFuse config root must be a mapping")
    experiment_values = payload.get("phasefuse")
    if not isinstance(experiment_values, Mapping):
        raise TypeError("PhaseFuse config requires a 'phasefuse' mapping")
    sampling = payload.get("sampling", {})
    metadata = payload.get("metadata", {})
    if not isinstance(sampling, Mapping) or not isinstance(metadata, Mapping):
        raise TypeError("sampling and metadata config sections must be mappings")
    return PhaseFuseFileConfig(
        experiment=config_from_mapping(experiment_values),
        sampling=dict(sampling),
        metadata=dict(metadata),
    )


__all__ = [
    "PHASEFUSE_DEFAULT_METHODS",
    "PHASEFUSE_METHODS",
    "PhaseFuseExperimentConfig",
    "PhaseFuseFileConfig",
    "config_from_mapping",
    "load_phasefuse_config",
    "run_phasefuse_experiment",
    "run_phasefuse_method",
    "save_phasefuse_trace",
]
