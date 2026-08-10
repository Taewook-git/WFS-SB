"""Experiment orchestration and aggregation for phase-stable WFS-SB."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pywt

from wfs.core import compute_dwt_level, compute_min_peak_distance

from .artifacts import (
    OriginSignalRecord,
    load_trace_arrays,
    save_trace_npz,
    write_jsonl,
)
from .metrics import (
    boundary_count_cv,
    js_divergence,
    mllm_stability_metrics,
    normalized_l1_distance,
    pairwise_cosine_consistency,
    pairwise_normalized_l1,
    scale_energy_drift,
    segmentation_consistency,
    selected_timestamp_metrics,
    tolerant_boundary_metrics,
    video_cluster_paired_bootstrap,
)
from .pipeline import PhaseStableWFS, SelectionConfig
from .transforms import TransformConfig, build_transform


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings used for transform, selection, and stability measurement."""

    methods: tuple[str, ...] = ("dwt", "swt")
    wavelet: str = "db4"
    level: Optional[int] = None
    drift_level: int = 3
    frame_budget: int = 16
    min_distance_ratio: float = 0.02
    min_distance_absolute: int = 5
    shared_padding: bool = True
    padding_mode: str = "reflect"
    dwt_mode: str = "symmetric"
    swt_norm: bool = True
    cycle_shifts: tuple[int, ...] = tuple(range(16))
    cycle_aggregation: str = "mean"
    gaussian_sigma: Optional[float] = None
    boundary_tolerance_sec: float = 1.0
    report_boundary_tolerances_sec: tuple[float, ...] = (0.5, 1.0, 2.0)
    selected_tolerance_sec: float = 1.0
    edge_margin_sec: float = 0.0

    def __post_init__(self) -> None:
        supported = {"dwt", "swt", "cycle_spin", "gaussian"}
        if not self.methods or any(method not in supported for method in self.methods):
            raise ValueError(f"methods must be chosen from {sorted(supported)}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates")
        if self.level is not None and self.level < 1:
            raise ValueError("level must be positive when supplied")
        if self.frame_budget <= 0:
            raise ValueError("frame_budget must be positive")
        if self.min_distance_ratio < 0 or self.min_distance_absolute < 1:
            raise ValueError("invalid minimum peak-distance setting")
        if self.boundary_tolerance_sec < 0 or self.selected_tolerance_sec < 0:
            raise ValueError("matching tolerances must be non-negative")
        if (
            not self.report_boundary_tolerances_sec
            or any(value < 0 for value in self.report_boundary_tolerances_sec)
        ):
            raise ValueError("report boundary tolerances must be non-empty and non-negative")
        if self.edge_margin_sec < 0:
            raise ValueError("edge_margin_sec must be non-negative")
        if self.gaussian_sigma is not None and self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive when supplied")


def group_signal_records(
    records: Iterable[OriginSignalRecord],
) -> Dict[tuple[str, str, str], list[OriginSignalRecord]]:
    """Group records by dataset/video/question and validate origin uniqueness."""

    groups: Dict[tuple[str, str, str], list[OriginSignalRecord]] = defaultdict(list)
    for record in records:
        if not isinstance(record, OriginSignalRecord):
            raise TypeError("records must contain OriginSignalRecord objects")
        groups[record.item_key].append(record)
    if not groups:
        raise ValueError("at least one signal record is required")
    for key, group in groups.items():
        group.sort(key=lambda record: record.origin_id)
        origin_ids = [record.origin_id for record in group]
        if len(origin_ids) != len(set(origin_ids)):
            raise ValueError(f"duplicate origin_id in item {key}")
        if len(group) < 2:
            raise ValueError(f"item {key} must contain at least two origins")
        lengths = {len(record.relevance_scores) for record in group}
        if len(lengths) != 1:
            raise ValueError(f"item {key} origins must use a common candidate count")
    return dict(groups)


def align_time_series(
    timestamps: Sequence[Sequence[float]],
    arrays: Sequence[np.ndarray],
    *,
    edge_margin_sec: float = 0.0,
    step_sec: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate origin-wise arrays onto their common absolute-time grid.

    Each array may have any leading dimensions, but its final dimension must
    match the corresponding timestamp vector.  The output shape is
    ``(n_origins, *leading_shape, n_common_times)``.
    """

    if len(timestamps) != len(arrays) or len(arrays) < 2:
        raise ValueError("timestamps and arrays must contain the same >=2 origins")
    if not np.isfinite(edge_margin_sec) or edge_margin_sec < 0:
        raise ValueError("edge_margin_sec must be finite and non-negative")

    times = [np.asarray(value, dtype=float) for value in timestamps]
    values = [np.asarray(value, dtype=float) for value in arrays]
    leading_shape = values[0].shape[:-1]
    inferred_steps = []
    for index, (time, value) in enumerate(zip(times, values)):
        if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
            raise ValueError(f"timestamps[{index}] must be a finite 1-D array of length >=2")
        if np.any(np.diff(time) <= 0):
            raise ValueError(f"timestamps[{index}] must be strictly increasing")
        if value.shape[:-1] != leading_shape or value.shape[-1] != time.size:
            raise ValueError("all arrays must share leading shape and align with timestamps")
        if not np.all(np.isfinite(value)):
            raise ValueError("arrays must contain only finite values")
        inferred_steps.append(float(np.median(np.diff(time))))

    start = max(time[0] for time in times) + edge_margin_sec
    end = min(time[-1] for time in times) - edge_margin_sec
    if end <= start:
        raise ValueError("origins have no common interval after edge trimming")
    if step_sec is None:
        step = max(inferred_steps)
    else:
        if not np.isfinite(step_sec) or step_sec <= 0:
            raise ValueError("step_sec must be finite and positive")
        step = float(step_sec)
    count = int(np.floor((end - start) / step + 1e-12)) + 1
    grid = start + np.arange(count, dtype=float) * step
    grid = grid[grid <= end + 1e-10]
    if grid.size < 2:
        raise ValueError("common interpolation grid must contain at least two timestamps")

    aligned = []
    for time, value in zip(times, values):
        flat = value.reshape(-1, value.shape[-1])
        interpolated = np.stack(
            [np.interp(grid, time, row) for row in flat], axis=0
        ).reshape(*leading_shape, grid.size)
        aligned.append(interpolated)
    return grid, np.stack(aligned, axis=0)


def _load_features(record: OriginSignalRecord) -> Optional[np.ndarray]:
    if record.visual_features_path is None:
        return None
    path = Path(record.visual_features_path)
    if not path.is_file():
        raise FileNotFoundError(f"visual feature file does not exist: {path}")
    features = np.load(path, allow_pickle=False)
    if features.shape[0] != len(record.relevance_scores):
        raise ValueError(f"visual feature length mismatch: {path}")
    return np.asarray(features)


def _method_transform_config(
    method: str,
    level: int,
    config: ExperimentConfig,
) -> TransformConfig:
    return TransformConfig(
        method=method,  # type: ignore[arg-type]
        wavelet=config.wavelet,
        level=level,
        dwt_mode=config.dwt_mode,
        shared_padding=config.shared_padding,
        padding_mode=config.padding_mode,
        swt_norm=config.swt_norm,
        cycle_shifts=config.cycle_shifts,
        cycle_aggregation=config.cycle_aggregation,  # type: ignore[arg-type]
        gaussian_sigma=config.gaussian_sigma,
    )


def run_real_origin_experiment(
    records: Sequence[OriginSignalRecord],
    output_dir: str | Path,
    *,
    config: Optional[ExperimentConfig] = None,
    selection_config: Optional[SelectionConfig] = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Run every configured transform and write traces plus item metrics."""

    experiment = config or ExperimentConfig()
    groups = group_signal_records(records)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trace_rows: list[Dict[str, Any]] = []

    for item_key in sorted(groups):
        group = groups[item_key]
        signal_length = len(group[0].relevance_scores)
        level = experiment.level
        if level is None:
            level = compute_dwt_level(
                signal_length,
                wavelet=experiment.wavelet,
                drift=experiment.drift_level,
            )
        min_distance = compute_min_peak_distance(
            signal_length,
            ratio=experiment.min_distance_ratio,
            absolute_min=experiment.min_distance_absolute,
        )
        for method in experiment.methods:
            transform_config = _method_transform_config(method, level, experiment)
            pipeline = PhaseStableWFS(
                build_transform(transform_config),
                selection_config,
            )
            for record in group:
                trace = pipeline.run(
                    record.relevance_scores,
                    experiment.frame_budget,
                    min_distance,
                    features=_load_features(record),
                )
                _, row = save_trace_npz(output_root, record, method, trace)
                row["level"] = level
                row["min_peak_distance"] = min_distance
                trace_rows.append(row)

    trace_path = output_root / "traces.jsonl"
    write_jsonl(trace_path, trace_rows)
    metric_rows = compute_trace_metrics(trace_rows, config=experiment)
    write_jsonl(output_root / "item_metrics.jsonl", metric_rows)
    return trace_rows, metric_rows


def _safe_js(first: np.ndarray, second: np.ndarray) -> float:
    first_mass = float(np.sum(first))
    second_mass = float(np.sum(second))
    if first_mass <= 0 and second_mass <= 0:
        return 0.0
    if first_mass <= 0 or second_mass <= 0:
        return 1.0
    return js_divergence(first, second)


def _safe_pairwise_cosine(values: Sequence[np.ndarray]) -> Dict[str, float | int]:
    """Cosine summary that records, rather than crashes on, flat outputs."""

    matrix = np.asarray(values, dtype=float).reshape(len(values), -1)
    norms = np.linalg.norm(matrix, axis=1)
    zero = norms <= np.finfo(float).eps * max(1.0, np.sqrt(matrix.shape[1]))
    if not np.any(zero):
        result = dict(pairwise_cosine_consistency(matrix))
        result["zero_origins"] = 0
        return result
    similarities: list[float] = []
    for first, second in combinations(range(matrix.shape[0]), 2):
        if zero[first] and zero[second]:
            similarity = 1.0
        elif zero[first] or zero[second]:
            similarity = 0.0
        else:
            similarity = float(
                np.dot(matrix[first], matrix[second]) / (norms[first] * norms[second])
            )
        similarities.append(float(np.clip(similarity, -1.0, 1.0)))
    return {
        "mean": float(np.mean(similarities)),
        "worst": float(np.min(similarities)),
        "n_pairs": len(similarities),
        "zero_origins": int(np.sum(zero)),
    }


def _safe_centered_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = np.asarray(first, dtype=float).ravel()
    second_flat = np.asarray(second, dtype=float).ravel()
    first_centered = first_flat - np.mean(first_flat)
    second_centered = second_flat - np.mean(second_flat)
    first_norm = float(np.linalg.norm(first_centered))
    second_norm = float(np.linalg.norm(second_centered))
    epsilon = np.finfo(float).eps * max(1.0, np.sqrt(first_flat.size))
    if first_norm <= epsilon and second_norm <= epsilon:
        return 1.0 if np.allclose(first_flat, second_flat) else 0.0
    if first_norm <= epsilon or second_norm <= epsilon:
        return 0.0
    return float(
        np.clip(
            np.dot(first_centered, second_centered) / (first_norm * second_norm),
            -1.0,
            1.0,
        )
    )


def _safe_energy_drift(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    first_mass = float(np.sum(first))
    second_mass = float(np.sum(second))
    if first_mass <= 0 and second_mass <= 0:
        return {"l1": 0.0, "js_distance": 0.0}
    if first_mass <= 0 or second_mass <= 0:
        return {"l1": 1.0, "js_distance": 1.0}
    drift = scale_energy_drift(first, second, inputs_are_proportions=True)
    return {"l1": drift["l1"], "js_distance": drift["js_distance"]}


def _mean_worst(values: Sequence[float], *, higher_is_better: bool) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("pair metric values must be non-empty and finite")
    worst = np.min(array) if higher_is_better else np.max(array)
    return float(np.mean(array)), float(worst)


def compute_trace_metrics(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    config: Optional[ExperimentConfig] = None,
) -> list[Dict[str, Any]]:
    """Compute item-level all-origin-pair metrics from persisted traces."""

    experiment = config or ExperimentConfig()
    groups: Dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        key = (
            str(row["dataset"]),
            str(row["video_id"]),
            str(row["question_id"]),
            str(row["method"]),
        )
        groups[key].append(row)

    results: list[Dict[str, Any]] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: int(row["origin_id"]))
        if len(rows) < 2:
            raise ValueError(f"trace group {key} must contain at least two origins")
        arrays = [load_trace_arrays(row) for row in rows]
        timestamps = [row["timestamps_sec"] for row in rows]
        grid, representations = align_time_series(
            timestamps,
            [array["representation"] for array in arrays],
            edge_margin_sec=experiment.edge_margin_sec,
        )
        _, saliencies = align_time_series(
            timestamps,
            [array["saliency"] for array in arrays],
            edge_margin_sec=experiment.edge_margin_sec,
            step_sec=float(np.median(np.diff(grid))),
        )
        _, input_signals = align_time_series(
            timestamps,
            [array["relevance_scores"] for array in arrays],
            edge_margin_sec=experiment.edge_margin_sec,
            step_sec=float(np.median(np.diff(grid))),
        )

        rep_cos = _safe_pairwise_cosine(list(representations))
        rep_l1 = pairwise_normalized_l1(representations)
        sal_cos = _safe_pairwise_cosine(list(saliencies))
        sal_l1 = pairwise_normalized_l1(saliencies)

        sal_js_values: list[float] = []
        rep_correlation_values: list[float] = []
        sal_correlation_values: list[float] = []
        input_l1_values: list[float] = []
        saliency_gain_values: list[float] = []
        energy_l1_values: list[float] = []
        energy_js_values: list[float] = []
        boundary_f1_values: list[float] = []
        boundary_distance_values: list[float] = []
        boundary_f1_by_tolerance: Dict[float, list[float]] = {
            float(tolerance): []
            for tolerance in experiment.report_boundary_tolerances_sec
        }
        segment_ari_values: list[float] = []
        segment_vi_values: list[float] = []
        segment_duration_drift_values: list[float] = []
        selected_f1_values: list[float] = []
        selected_distance_values: list[float] = []
        selected_embedding_values: list[float] = []
        common_start, common_end = float(grid[0]), float(grid[-1])

        feature_matrices: list[Optional[np.ndarray]] = []
        for row in rows:
            feature_path = row.get("visual_features_path")
            if feature_path is None:
                feature_matrices.append(None)
                continue
            matrix = np.asarray(np.load(str(feature_path), allow_pickle=False), dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != len(row["timestamps_sec"]):
                raise ValueError(
                    f"visual features are not frame-aligned for {key}/origin={row['origin_id']}"
                )
            feature_matrices.append(matrix)

        mean_segment_durations: list[float] = []
        saliency_variances: list[float] = []
        peak_to_background: list[float] = []
        boundary_rates: list[float] = []
        for row, array in zip(rows, arrays):
            row_times = np.asarray(row["timestamps_sec"], dtype=float)
            step = float(np.median(np.diff(row_times)))
            segment_lengths = [max(0, int(end) - int(start)) * step for start, end in row["segments"]]
            mean_segment_durations.append(float(np.mean(segment_lengths)))
            saliency = np.asarray(array["saliency"], dtype=float)
            saliency_variances.append(float(np.var(saliency)))
            peak_indices = np.asarray(row["peaks"], dtype=int)
            peak_mask = np.zeros(saliency.size, dtype=bool)
            peak_mask[peak_indices] = True
            if peak_indices.size and np.any(~peak_mask):
                peak_mean = float(np.mean(saliency[peak_mask]))
                background_mean = float(np.mean(saliency[~peak_mask]))
                peak_to_background.append(
                    peak_mean / max(background_mean, np.finfo(float).eps)
                )
            duration_min = max(step, row_times[-1] - row_times[0] + step) / 60.0
            boundary_rates.append(len(peak_indices) / duration_min)

        for first_index, second_index in combinations(range(len(rows)), 2):
            rep_correlation_values.append(
                _safe_centered_correlation(
                    representations[first_index], representations[second_index]
                )
            )
            sal_correlation_values.append(
                _safe_centered_correlation(
                    saliencies[first_index], saliencies[second_index]
                )
            )
            sal_js_values.append(
                _safe_js(saliencies[first_index].ravel(), saliencies[second_index].ravel())
            )
            input_l1 = normalized_l1_distance(
                input_signals[first_index], input_signals[second_index]
            )
            saliency_pair_l1 = normalized_l1_distance(
                saliencies[first_index], saliencies[second_index]
            )
            input_l1_values.append(input_l1)
            saliency_gain_values.append(
                saliency_pair_l1 / max(input_l1, np.finfo(float).eps)
                if saliency_pair_l1 > 0
                else 0.0
            )
            drift = _safe_energy_drift(
                arrays[first_index]["scale_energy_proportions"],
                arrays[second_index]["scale_energy_proportions"],
            )
            energy_l1_values.append(drift["l1"])
            energy_js_values.append(drift["js_distance"])

            first_boundaries = np.asarray(rows[first_index]["peaks_sec"], dtype=float)
            second_boundaries = np.asarray(rows[second_index]["peaks_sec"], dtype=float)
            first_boundaries = first_boundaries[
                (first_boundaries >= common_start) & (first_boundaries <= common_end)
            ]
            second_boundaries = second_boundaries[
                (second_boundaries >= common_start) & (second_boundaries <= common_end)
            ]
            boundary = tolerant_boundary_metrics(
                first_boundaries,
                second_boundaries,
                tolerance=experiment.boundary_tolerance_sec,
            )
            boundary_f1_values.append(boundary["f1"])
            for tolerance, values in boundary_f1_by_tolerance.items():
                values.append(
                    tolerant_boundary_metrics(
                        first_boundaries,
                        second_boundaries,
                        tolerance=tolerance,
                    )["f1"]
                )
            if boundary["mean_distance"] is not None:
                boundary_distance_values.append(boundary["mean_distance"])
            segment = segmentation_consistency(first_boundaries, second_boundaries, grid)
            segment_ari_values.append(segment["ari"])
            segment_vi_values.append(segment["vi"])

            duration_first = mean_segment_durations[first_index]
            duration_second = mean_segment_durations[second_index]
            duration_denominator = duration_first + duration_second
            segment_duration_drift_values.append(
                0.0
                if duration_denominator <= 0
                else abs(duration_first - duration_second) / duration_denominator
            )

            selected_kwargs: Dict[str, Any] = {}
            first_features = feature_matrices[first_index]
            second_features = feature_matrices[second_index]
            if first_features is not None and second_features is not None:
                first_selected_indices = np.asarray(
                    rows[first_index]["selected_indices"], dtype=int
                )
                second_selected_indices = np.asarray(
                    rows[second_index]["selected_indices"], dtype=int
                )
                selected_kwargs = {
                    "first_embeddings": first_features[first_selected_indices],
                    "second_embeddings": second_features[second_selected_indices],
                }
            selected = selected_timestamp_metrics(
                rows[first_index]["selected_actual_pts_sec"],
                rows[second_index]["selected_actual_pts_sec"],
                tolerance=experiment.selected_tolerance_sec,
                **selected_kwargs,
            )
            selected_f1_values.append(selected["f1"])
            if selected["optimal_mean_distance"] is not None:
                selected_distance_values.append(selected["optimal_mean_distance"])
            if selected.get("matched_embedding_cosine") is not None:
                selected_embedding_values.append(selected["matched_embedding_cosine"])

        sal_js_mean, sal_js_worst = _mean_worst(sal_js_values, higher_is_better=False)
        energy_l1_mean, energy_l1_worst = _mean_worst(
            energy_l1_values, higher_is_better=False
        )
        energy_js_mean, energy_js_worst = _mean_worst(
            energy_js_values, higher_is_better=False
        )
        boundary_f1_mean, boundary_f1_worst = _mean_worst(
            boundary_f1_values, higher_is_better=True
        )
        segment_ari_mean, segment_ari_worst = _mean_worst(
            segment_ari_values, higher_is_better=True
        )
        segment_vi_mean, segment_vi_worst = _mean_worst(
            segment_vi_values, higher_is_better=False
        )
        selected_f1_mean, selected_f1_worst = _mean_worst(
            selected_f1_values, higher_is_better=True
        )
        rep_correlation_mean, rep_correlation_worst = _mean_worst(
            rep_correlation_values, higher_is_better=True
        )
        sal_correlation_mean, sal_correlation_worst = _mean_worst(
            sal_correlation_values, higher_is_better=True
        )
        input_l1_mean, input_l1_worst = _mean_worst(
            input_l1_values, higher_is_better=False
        )
        saliency_gain_mean, saliency_gain_worst = _mean_worst(
            saliency_gain_values, higher_is_better=False
        )

        result: Dict[str, Any] = {
            "dataset": key[0],
            "video_id": key[1],
            "question_id": key[2],
            "method": key[3],
            "num_origins": len(rows),
            "num_origin_pairs": len(rows) * (len(rows) - 1) // 2,
            "decode_error_ms_mean": float(
                np.mean([float(row["decode_error_ms_mean"]) for row in rows])
            ),
            "decode_error_ms_worst": float(
                np.max([float(row["decode_error_ms_max"]) for row in rows])
            ),
            "representation_consistency_mean": rep_cos["mean"],
            "representation_consistency_worst": rep_cos["worst"],
            "representation_l1_mean": rep_l1["mean"],
            "representation_l1_worst": rep_l1["worst"],
            "representation_centered_correlation_mean": rep_correlation_mean,
            "representation_centered_correlation_worst": rep_correlation_worst,
            "representation_zero_origin_rate": float(rep_cos["zero_origins"]) / len(rows),
            "saliency_consistency_mean": sal_cos["mean"],
            "saliency_consistency_worst": sal_cos["worst"],
            "saliency_l1_mean": sal_l1["mean"],
            "saliency_l1_worst": sal_l1["worst"],
            "saliency_centered_correlation_mean": sal_correlation_mean,
            "saliency_centered_correlation_worst": sal_correlation_worst,
            "saliency_zero_origin_rate": float(sal_cos["zero_origins"]) / len(rows),
            "saliency_js_mean": sal_js_mean,
            "saliency_js_worst": sal_js_worst,
            "input_signal_l1_mean": input_l1_mean,
            "input_signal_l1_worst": input_l1_worst,
            "saliency_perturbation_gain_mean": saliency_gain_mean,
            "saliency_perturbation_gain_worst": saliency_gain_worst,
            "saliency_temporal_variance_mean": float(np.mean(saliency_variances)),
            "peak_to_background_ratio_mean": (
                None if not peak_to_background else float(np.mean(peak_to_background))
            ),
            "energy_l1_drift_mean": energy_l1_mean,
            "energy_l1_drift_worst": energy_l1_worst,
            "energy_js_drift_mean": energy_js_mean,
            "energy_js_drift_worst": energy_js_worst,
            "boundary_f1_mean": boundary_f1_mean,
            "boundary_f1_worst": boundary_f1_worst,
            "boundary_distance_mean": (
                None
                if not boundary_distance_values
                else float(np.mean(boundary_distance_values))
            ),
            "boundary_count_mean": float(
                np.mean([len(row["peaks_sec"]) for row in rows])
            ),
            "boundary_count_cv": boundary_count_cv(
                [len(row["peaks_sec"]) for row in rows]
            ),
            "boundaries_per_minute_mean": float(np.mean(boundary_rates)),
            "segment_count_mean": float(np.mean([len(row["segments"]) for row in rows])),
            "mean_segment_duration_sec": float(np.mean(mean_segment_durations)),
            "mean_segment_duration_drift": float(
                np.mean(segment_duration_drift_values)
            ),
            "segment_ari_mean": segment_ari_mean,
            "segment_ari_worst": segment_ari_worst,
            "segment_vi_mean": segment_vi_mean,
            "segment_vi_worst": segment_vi_worst,
            "selected_f1_mean": selected_f1_mean,
            "selected_f1_worst": selected_f1_worst,
            "selected_distance_mean": (
                None
                if not selected_distance_values
                else float(np.mean(selected_distance_values))
            ),
            "selected_embedding_cosine_mean": (
                None
                if not selected_embedding_values
                else float(np.mean(selected_embedding_values))
            ),
            "fallback_rate": float(np.mean([bool(row["used_fallback"]) for row in rows])),
        }
        for tolerance, values in boundary_f1_by_tolerance.items():
            label = f"{tolerance:g}".replace(".", "p")
            mean, worst = _mean_worst(values, higher_is_better=True)
            result[f"boundary_f1_at_{label}s_mean"] = mean
            result[f"boundary_f1_at_{label}s_worst"] = worst
        results.append(result)
    return results


def aggregate_item_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute descriptive method means for every finite numeric item metric."""

    by_method: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_method[str(row["method"])].append(row)
    result: Dict[str, Dict[str, float]] = {}
    ignored = {"dataset", "video_id", "question_id", "method"}
    for method, rows in sorted(by_method.items()):
        keys = sorted(set.intersection(*(set(row.keys()) for row in rows)) - ignored)
        summary: Dict[str, float] = {"num_items": float(len(rows))}
        for key in keys:
            values = [row[key] for row in rows]
            if all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, bool)
                and np.isfinite(value)
                for value in values
            ):
                summary[key] = float(np.mean(values))
        result[method] = summary
    return result


def _top_nms_indices(
    saliency: np.ndarray,
    count: int,
    min_distance: int,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    values = np.asarray(saliency, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("saliency must be a finite 1-D array of length >=3")
    if count <= 0 or min_distance <= 0:
        raise ValueError("count and min_distance must be positive")
    # Boundaries exclude endpoints. Ranking is stable: earlier time wins ties.
    candidate_indices = np.arange(1, values.size - 1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must align with saliency")
        candidate_indices = candidate_indices[mask[candidate_indices]]
    candidates = candidate_indices[
        np.lexsort((candidate_indices, -values[candidate_indices]))
    ]
    selected: list[int] = []
    for candidate in candidates:
        index = int(candidate)
        if all(abs(index - prior) >= min_distance for prior in selected):
            selected.append(index)
            if len(selected) == count:
                return np.asarray(sorted(selected), dtype=int)
    raise ValueError(
        f"cannot select {count} boundaries with min_distance={min_distance} "
        f"from {values.size} samples"
    )


def compute_matched_cardinality_metrics(
    trace_rows: Sequence[Mapping[str, Any]],
    boundary_counts: int | Mapping[str, int],
    *,
    tolerance_sec: float = 1.0,
    edge_margin_sec: float = 0.0,
) -> list[Dict[str, Any]]:
    """Evaluate top-B boundaries using pre-calibrated, video-level counts.

    Mapping keys may be ``video_id`` or ``dataset/video_id``. Counts are never
    estimated from the evaluated origin, preventing test-offset retuning.
    """

    if tolerance_sec < 0 or edge_margin_sec < 0:
        raise ValueError("tolerance and edge margin must be non-negative")
    groups: Dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        key = (
            str(row["dataset"]),
            str(row["video_id"]),
            str(row["question_id"]),
            str(row["method"]),
        )
        groups[key].append(row)

    def resolve_count(key: tuple[str, str, str, str]) -> int:
        if isinstance(boundary_counts, (int, np.integer)) and not isinstance(
            boundary_counts, bool
        ):
            count = int(boundary_counts)
        elif isinstance(boundary_counts, Mapping):
            compound = f"{key[0]}/{key[1]}"
            if compound in boundary_counts:
                count = int(boundary_counts[compound])
            elif key[1] in boundary_counts:
                count = int(boundary_counts[key[1]])
            else:
                raise ValueError(f"no calibrated boundary count for {compound}")
        else:
            raise TypeError("boundary_counts must be an integer or mapping")
        if count <= 0:
            raise ValueError("calibrated boundary counts must be positive")
        return count

    results: list[Dict[str, Any]] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: int(row["origin_id"]))
        if len(rows) < 2:
            raise ValueError(f"trace group {key} must contain at least two origins")
        count = resolve_count(key)
        boundaries: list[np.ndarray] = []
        timestamps: list[Sequence[float]] = []
        arrays = [load_trace_arrays(row) for row in rows]
        timestamps = [row["timestamps_sec"] for row in rows]
        grid, _ = align_time_series(
            timestamps,
            [array["saliency"] for array in arrays],
            edge_margin_sec=edge_margin_sec,
        )
        for row, array in zip(rows, arrays):
            times = np.asarray(row["timestamps_sec"], dtype=float)
            indices = _top_nms_indices(
                array["saliency"],
                count,
                int(row["min_peak_distance"]),
                valid_mask=(times >= grid[0]) & (times <= grid[-1]),
            )
            boundaries.append(times[indices])
        f1_values: list[float] = []
        distance_values: list[float] = []
        ari_values: list[float] = []
        vi_values: list[float] = []
        for first, second in combinations(range(len(rows)), 2):
            metric = tolerant_boundary_metrics(
                boundaries[first],
                boundaries[second],
                tolerance=tolerance_sec,
            )
            f1_values.append(metric["f1"])
            if metric["mean_distance"] is not None:
                distance_values.append(metric["mean_distance"])
            segment = segmentation_consistency(
                boundaries[first], boundaries[second], grid
            )
            ari_values.append(segment["ari"])
            vi_values.append(segment["vi"])
        results.append(
            {
                "dataset": key[0],
                "video_id": key[1],
                "question_id": key[2],
                "method": key[3],
                "matched_boundary_count": count,
                "num_origins": len(rows),
                "matched_boundary_f1_mean": float(np.mean(f1_values)),
                "matched_boundary_f1_worst": float(np.min(f1_values)),
                "matched_boundary_distance_mean": (
                    None
                    if not distance_values
                    else float(np.mean(distance_values))
                ),
                "matched_segment_ari_mean": float(np.mean(ari_values)),
                "matched_segment_vi_mean": float(np.mean(vi_values)),
            }
        )
    return results


def paired_metric_bootstrap(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str = "dwt",
    treatment_method: str = "swt",
    metric_names: Sequence[str],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Video-cluster paired bootstrap for item metrics."""

    by_method: Dict[str, Dict[tuple[str, str, str], Mapping[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        key = (str(row["dataset"]), str(row["video_id"]), str(row["question_id"]))
        by_method[str(row["method"])][key] = row
    baseline = by_method.get(baseline_method, {})
    treatment = by_method.get(treatment_method, {})
    common = sorted(set(baseline) & set(treatment))
    if not common:
        raise ValueError("baseline and treatment have no paired items")

    result: Dict[str, Any] = {
        "baseline_method": baseline_method,
        "treatment_method": treatment_method,
        "num_paired_items": len(common),
        "metrics": {},
    }
    cluster_ids = [f"{key[0]}\x1f{key[1]}" for key in common]
    for metric_name in metric_names:
        baseline_values = np.asarray([baseline[key][metric_name] for key in common], dtype=float)
        treatment_values = np.asarray([treatment[key][metric_name] for key in common], dtype=float)
        result["metrics"][metric_name] = video_cluster_paired_bootstrap(
            cluster_ids,
            baseline_values,
            treatment_values,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
    return result


def controlled_shift_metrics(
    signal: Sequence[float],
    *,
    methods: Sequence[str] = ("dwt", "swt"),
    shifts: Sequence[int] = tuple(range(16)),
    wavelet: str = "db4",
    level: Optional[int] = None,
    drift_level: int = 3,
    shared_padding: bool = True,
    edge_samples: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate inverse-aligned transforms of one fixed circularly shifted signal."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a finite 1-D array of length >=2")
    shift_values = tuple(dict.fromkeys(int(shift) for shift in shifts))
    if len(shift_values) < 2:
        raise ValueError("at least two distinct shifts are required")
    resolved_level = level or compute_dwt_level(values.size, wavelet, drift_level)
    if edge_samples is None:
        filter_length = pywt.Wavelet(wavelet).dec_len
        effective_support = 1 + (filter_length - 1) * (2**resolved_level - 1)
        resolved_edge = int(np.ceil(effective_support / 2)) + max(
            abs(shift) for shift in shift_values
        )
        resolved_edge = min(resolved_edge, max(0, (values.size - 2) // 2))
    else:
        resolved_edge = int(edge_samples)
        if resolved_edge < 0 or 2 * resolved_edge >= values.size:
            raise ValueError("edge_samples must leave at least one interior sample")
    interior = (
        slice(resolved_edge, -resolved_edge)
        if resolved_edge > 0
        else slice(None)
    )
    results: Dict[str, Dict[str, float]] = {}
    for method in methods:
        transform = build_transform(
            TransformConfig(
                method=method,  # type: ignore[arg-type]
                wavelet=wavelet,
                level=resolved_level,
                shared_padding=shared_padding,
                cycle_shifts=tuple(range(2 ** min(resolved_level, 4))),
            )
        )
        representations = []
        saliencies = []
        energies = []
        for shift in shift_values:
            transformed = transform.transform(np.roll(values, shift))
            aligned_representation = np.roll(
                transformed.representation, -shift, axis=1
            )[:, interior]
            aligned_saliency = np.roll(transformed.saliency, -shift)[interior]
            representations.append(aligned_representation)
            saliencies.append(aligned_saliency)
            scale_energy = np.sum(np.square(aligned_representation), axis=1)
            energy_total = float(np.sum(scale_energy))
            energies.append(
                np.zeros_like(scale_energy)
                if energy_total <= np.finfo(float).eps
                else scale_energy / energy_total
            )
        rep_cos = _safe_pairwise_cosine(representations)
        rep_l1 = pairwise_normalized_l1(representations)
        sal_cos = _safe_pairwise_cosine(saliencies)
        sal_l1 = pairwise_normalized_l1(saliencies)
        energy_drifts = [
            _safe_energy_drift(energies[a], energies[b])
            for a, b in combinations(range(len(energies)), 2)
        ]
        results[method] = {
            "edge_samples": float(resolved_edge),
            "representation_consistency_mean": float(rep_cos["mean"]),
            "representation_consistency_worst": float(rep_cos["worst"]),
            "representation_l1_mean": float(rep_l1["mean"]),
            "representation_l1_worst": float(rep_l1["worst"]),
            "saliency_consistency_mean": float(sal_cos["mean"]),
            "saliency_consistency_worst": float(sal_cos["worst"]),
            "saliency_l1_mean": float(sal_l1["mean"]),
            "saliency_l1_worst": float(sal_l1["worst"]),
            "energy_l1_drift_mean": float(
                np.mean([value["l1"] for value in energy_drifts])
            ),
            "energy_l1_drift_worst": float(
                np.max([value["l1"] for value in energy_drifts])
            ),
        }
    return results


def evaluate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str = "dwt",
    treatment_method: str = "swt",
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Aggregate MLLM prediction JSONL and compute paired video-cluster CIs."""

    grouped: Dict[str, Dict[tuple[str, str, str], Dict[int, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        required = ("dataset", "video_id", "question_id", "origin_id", "method", "prediction", "gold")
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"prediction row missing fields: {', '.join(missing)}")
        method = str(row["method"])
        item = (str(row["dataset"]), str(row["video_id"]), str(row["question_id"]))
        origin_id = int(row["origin_id"])
        if origin_id in grouped[method][item]:
            raise ValueError(f"duplicate prediction for {method}/{item}/origin={origin_id}")
        grouped[method][item][origin_id] = row

    method_metrics: Dict[str, Any] = {}
    matrices: Dict[
        str,
        tuple[list[tuple[str, str, str]], tuple[int, ...], np.ndarray, np.ndarray],
    ] = {}
    for method, items in sorted(grouped.items()):
        item_keys = sorted(items)
        origin_sets = [tuple(sorted(items[key])) for key in item_keys]
        if not origin_sets or any(value != origin_sets[0] for value in origin_sets[1:]):
            raise ValueError(f"method {method} does not have a rectangular origin grid")
        predictions = np.asarray(
            [
                [items[key][origin]["prediction"] for origin in origin_sets[0]]
                for key in item_keys
            ]
        )
        gold = np.asarray([items[key][origin_sets[0][0]]["gold"] for key in item_keys])
        for key in item_keys:
            if any(items[key][origin]["gold"] != items[key][origin_sets[0][0]]["gold"] for origin in origin_sets[0]):
                raise ValueError(f"gold answer changes across origins for {method}/{key}")
        method_metrics[method] = mllm_stability_metrics(predictions, gold)
        matrices[method] = (item_keys, origin_sets[0], predictions, gold)

    if baseline_method not in matrices or treatment_method not in matrices:
        raise ValueError("both baseline_method and treatment_method are required")
    baseline_keys, baseline_origins, baseline_predictions, baseline_gold = matrices[
        baseline_method
    ]
    treatment_keys, treatment_origins, treatment_predictions, treatment_gold = matrices[
        treatment_method
    ]
    if baseline_keys != treatment_keys:
        raise ValueError("baseline and treatment prediction items do not align")
    if baseline_origins != treatment_origins:
        raise ValueError("baseline and treatment origin grids do not align")
    if not np.array_equal(baseline_gold, treatment_gold):
        raise ValueError("baseline and treatment gold answers do not align")

    def downstream_effect(baseline_sample: np.ndarray, treatment_sample: np.ndarray) -> np.ndarray:
        # Gold values are encoded in an extra final column to keep cluster
        # bootstrap rows self-contained.
        base_pred, gold_values = baseline_sample[:, :-1], baseline_sample[:, -1]
        treatment_pred = treatment_sample[:, :-1]
        base = mllm_stability_metrics(base_pred, gold_values)
        treatment = mllm_stability_metrics(treatment_pred, gold_values)
        return np.asarray(
            [
                treatment["mean_accuracy"] - base["mean_accuracy"],
                treatment["robust_accuracy"] - base["robust_accuracy"],
                treatment["pairwise_answer_disagreement"]
                - base["pairwise_answer_disagreement"],
            ],
            dtype=float,
        )

    # Object arrays allow arbitrary multiple-choice labels while preserving
    # the row-wise cluster resampling contract.
    baseline_bootstrap = np.column_stack([baseline_predictions, baseline_gold])
    treatment_bootstrap = np.column_stack([treatment_predictions, treatment_gold])
    comparison = video_cluster_paired_bootstrap(
        [f"{key[0]}\x1f{key[1]}" for key in baseline_keys],
        baseline_bootstrap,
        treatment_bootstrap,
        statistic=downstream_effect,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    comparison["effect_order"] = [
        "delta_mean_accuracy",
        "delta_robust_accuracy",
        "delta_pairwise_answer_disagreement",
    ]
    return {
        "methods": method_metrics,
        "comparison": comparison,
        "baseline_method": baseline_method,
        "treatment_method": treatment_method,
    }


__all__ = [
    "ExperimentConfig",
    "aggregate_item_metrics",
    "align_time_series",
    "compute_trace_metrics",
    "compute_matched_cardinality_metrics",
    "controlled_shift_metrics",
    "evaluate_prediction_rows",
    "group_signal_records",
    "paired_metric_bootstrap",
    "run_real_origin_experiment",
]
