"""Boundary-policy and cardinality-separation experiments.

The experiment deliberately separates three interventions:

* ``adaptive``: the native height/prominence/distance detector;
* ``topc``: a quota selector over strict local maxima that preserves the
  native adaptive count for the same method/item/origin;
* ``nested_bXX``: a common, predeclared boundary dose whose boundary sets are
  nested prefixes of one maximum-B frontier.

This module consumes existing :class:`OriginSignalRecord` artifacts.  Dense
feature extraction is therefore not repeated.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pywt
from scipy.signal import find_peaks, peak_prominences, peak_widths

from wfs.core import compute_dwt_level, compute_min_peak_distance

from .analysis import (
    ExperimentConfig,
    aggregate_item_metrics,
    compute_trace_metrics,
    group_signal_records,
    paired_metric_bootstrap,
)
from .artifacts import OriginSignalRecord, save_trace_npz, write_jsonl
from .metrics import video_cluster_paired_bootstrap
from .pipeline import (
    PhaseStableWFS,
    SelectionConfig,
    select_nested_nms_frontier,
    select_top_nms_indices,
)
from .transforms import TransformConfig, build_transform

PEAK_METRICS = (
    "local_max_count",
    "height_pass_count",
    "height_prominence_pass_count",
    "boundary_count",
    "local_max_per_minute",
    "boundaries_per_minute",
    "boundary_local_retention",
    "interior_boundary_count",
    "edge_boundary_fraction",
    "segment_pressure",
    "zero_allocation_fraction",
)

POLICY_STABILITY_METRICS = (
    "boundary_f1_mean",
    "segment_ari_mean",
    "selected_f1_mean",
    "selected_embedding_cosine_mean",
)

FIDELITY_METRICS = (
    "selected_relevant_fraction",
    "relevant_window_recall",
    "relevant_clip_recall",
    "mean_selected_saliency_vote",
    "mean_gt_clip_nearest_selected_sec",
)


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


def _load_features(record: OriginSignalRecord) -> np.ndarray | None:
    if record.visual_features_path is None:
        return None
    path = Path(record.visual_features_path)
    if not path.is_file():
        raise FileNotFoundError(f"visual feature file does not exist: {path}")
    features = np.load(path, allow_pickle=False)
    if features.shape[0] != len(record.relevance_scores):
        raise ValueError(f"visual feature length mismatch: {path}")
    return np.asarray(features)


def _finite_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _median_step(timestamps: Sequence[float]) -> float:
    values = np.asarray(timestamps, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("timestamps must be a strictly increasing 1-D sequence")
    step = float(np.median(np.diff(values)))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("timestamps have no finite positive sampling step")
    return step


def peak_filter_funnel(
    saliency: Sequence[float],
    *,
    height_factor: float,
    prominence_factor: float,
    min_distance: int,
    timestamps_sec: Sequence[float],
    wavelet: str,
    level: int,
) -> dict[str, Any]:
    """Decompose the adaptive detector into local-max/filter stages."""

    values = np.asarray(saliency, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("saliency must be a finite 1-D array of length >= 3")
    if np.any(values < 0):
        raise ValueError("saliency must be non-negative")
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")

    height = float(np.mean(values) + height_factor * np.std(values))
    prominence = float(prominence_factor * (np.max(values) - np.min(values)))
    local_peaks, _ = find_peaks(values)
    height_peaks, _ = find_peaks(values, height=height)
    distance_peaks, _ = find_peaks(
        values,
        height=height,
        distance=max(1, int(min_distance)),
    )
    height_prominence_peaks, _ = find_peaks(
        values,
        height=height,
        prominence=prominence,
    )
    final_peaks, _ = find_peaks(
        values,
        height=height,
        prominence=prominence,
        distance=max(1, int(min_distance)),
    )

    step_sec = _median_step(timestamps_sec)
    exposure_sec = step_sec * values.size
    sample_fps = 1.0 / step_sec
    wavelet_object = pywt.Wavelet(wavelet)
    effective_support = 1 + (wavelet_object.dec_len - 1) * (2**level - 1)
    support_margin = min((effective_support - 1) // 2, (values.size - 1) // 2)
    interior_start = int(support_margin)
    interior_stop = int(values.size - support_margin)
    interior_defined = interior_stop > interior_start
    interior_final = (
        final_peaks[(final_peaks >= interior_start) & (final_peaks < interior_stop)]
        if interior_defined
        else np.asarray([], dtype=int)
    )

    final_prominence = (
        peak_prominences(values, final_peaks)[0]
        if final_peaks.size
        else np.asarray([], dtype=float)
    )
    final_width = (
        peak_widths(values, final_peaks, rel_height=0.5)[0]
        if final_peaks.size
        else np.asarray([], dtype=float)
    )
    spacing = np.diff(final_peaks)
    peak_mask = np.zeros(values.size, dtype=bool)
    peak_mask[final_peaks] = True
    peak_mean = _finite_mean(values[peak_mask])
    background_mean = _finite_mean(values[~peak_mask])

    return {
        "num_samples": int(values.size),
        "sample_fps": sample_fps,
        "sample_step_sec": step_sec,
        "exposure_sec": exposure_sec,
        "wavelet_level": int(level),
        "effective_scale_sec": float((2**level) * step_sec),
        "wavelet_effective_support_samples": int(effective_support),
        "wavelet_support_margin_samples": int(support_margin),
        "interior_defined": bool(interior_defined),
        "height_threshold": height,
        "prominence_threshold": prominence,
        "height_over_max": height / max(float(np.max(values)), np.finfo(float).eps),
        "prominence_over_max": prominence
        / max(float(np.max(values)), np.finfo(float).eps),
        "min_peak_distance_samples": int(min_distance),
        "min_peak_distance_sec": float(min_distance * step_sec),
        "local_max_count": int(local_peaks.size),
        "height_pass_count": int(height_peaks.size),
        "height_distance_pass_count": int(distance_peaks.size),
        "height_prominence_pass_count": int(height_prominence_peaks.size),
        "boundary_count": int(final_peaks.size),
        "interior_boundary_count": (
            int(interior_final.size) if interior_defined else None
        ),
        "local_max_per_minute": float(local_peaks.size * 60.0 / exposure_sec),
        "boundaries_per_minute": float(final_peaks.size * 60.0 / exposure_sec),
        "boundary_local_retention": (
            float(final_peaks.size / local_peaks.size) if local_peaks.size else None
        ),
        "height_local_retention": (
            float(height_peaks.size / local_peaks.size) if local_peaks.size else None
        ),
        "distance_height_retention": (
            float(distance_peaks.size / height_peaks.size)
            if height_peaks.size
            else None
        ),
        "prominence_distance_retention": (
            float(final_peaks.size / distance_peaks.size)
            if distance_peaks.size
            else None
        ),
        "prominence_height_retention_diagnostic": (
            float(height_prominence_peaks.size / height_peaks.size)
            if height_peaks.size
            else None
        ),
        "saliency_mean": float(np.mean(values)),
        "saliency_std": float(np.std(values)),
        "saliency_max": float(np.max(values)),
        "saliency_sum": float(np.sum(values)),
        "peak_mean": peak_mean,
        "background_mean": background_mean,
        "peak_background_ratio": (
            peak_mean / max(background_mean, np.finfo(float).eps)
            if peak_mean is not None and background_mean is not None
            else None
        ),
        "mean_final_prominence": _finite_mean(final_prominence),
        "median_final_prominence": (
            float(np.median(final_prominence)) if final_prominence.size else None
        ),
        "median_final_width_samples": (
            float(np.median(final_width)) if final_width.size else None
        ),
        "median_final_spacing_samples": (
            float(np.median(spacing)) if spacing.size else None
        ),
        "local_peak_indices": local_peaks.astype(int).tolist(),
        "final_peak_indices": final_peaks.astype(int).tolist(),
    }


def _allocation_diagnostics(trace: Any, frame_budget: int) -> dict[str, float | int]:
    allocation_values = np.asarray(list(trace.allocation.values()), dtype=int)
    zero_count = int(np.sum(allocation_values == 0)) if allocation_values.size else 0
    return {
        "segment_count": int(len(trace.segments)),
        "valid_segment_count": int(len(trace.valid_segments)),
        "discarded_segment_count": int(len(trace.segments) - len(trace.valid_segments)),
        "zero_allocation_segments": zero_count,
        "zero_allocation_fraction": (
            float(zero_count / allocation_values.size)
            if allocation_values.size
            else 0.0
        ),
        "segment_pressure": float(len(trace.segments) / frame_budget),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write diagnostic rows with nested values encoded as strict JSON."""

    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, allow_nan=False)
                        if isinstance(value, (Mapping, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def qvhighlights_selection_fidelity(
    record: OriginSignalRecord,
    selected_indices: Sequence[int],
) -> dict[str, float | int] | None:
    """Evaluate sparse selected frames against independent QVHighlights labels."""

    query_metadata = record.metadata.get("query_metadata")
    if not isinstance(query_metadata, Mapping):
        return None
    windows = query_metadata.get("relevant_windows_sec")
    clip_ids = query_metadata.get("relevant_clip_ids")
    votes = query_metadata.get("saliency_votes")
    if windows is None or clip_ids is None or votes is None:
        return None

    selected = np.asarray(selected_indices, dtype=int)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("selected_indices must contain at least one frame")
    actual_times = np.asarray(record.actual_pts_sec, dtype=float)[selected]
    normalized_windows = [(float(value[0]), float(value[1])) for value in windows]
    relevant_mask = np.asarray(
        [
            any(start <= timestamp < stop for start, stop in normalized_windows)
            for timestamp in actual_times
        ],
        dtype=bool,
    )
    window_hits = [
        any(start <= timestamp < stop for timestamp in actual_times)
        for start, stop in normalized_windows
    ]
    selected_clip_ids = np.floor(actual_times / 2.0).astype(int)
    relevant_clip_ids = np.asarray([int(value) for value in clip_ids], dtype=int)
    unique_selected = set(selected_clip_ids.tolist())
    clip_recall = float(
        sum(int(value) in unique_selected for value in relevant_clip_ids)
        / max(1, relevant_clip_ids.size)
    )

    vote_by_clip: dict[int, float] = {}
    for clip_id, raw_votes in zip(relevant_clip_ids.tolist(), votes):
        vote_array = np.asarray(raw_votes, dtype=float)
        if (
            vote_array.ndim != 1
            or vote_array.size == 0
            or not np.all(np.isfinite(vote_array))
        ):
            raise ValueError("QVHighlights saliency votes must be finite vectors")
        vote_by_clip[int(clip_id)] = float(np.mean(vote_array))
    selected_votes = np.asarray(
        [vote_by_clip.get(int(clip_id), 0.0) for clip_id in selected_clip_ids],
        dtype=float,
    )
    clip_centers = 2.0 * relevant_clip_ids.astype(float) + 1.0
    nearest = np.min(np.abs(clip_centers[:, None] - actual_times[None, :]), axis=1)
    return {
        "selected_count": int(selected.size),
        "selected_relevant_count": int(np.sum(relevant_mask)),
        "selected_relevant_fraction": float(np.mean(relevant_mask)),
        "relevant_window_count": int(len(normalized_windows)),
        "relevant_window_recall": float(np.mean(window_hits)),
        "relevant_clip_count": int(relevant_clip_ids.size),
        "relevant_clip_recall": clip_recall,
        "mean_selected_saliency_vote": float(np.mean(selected_votes)),
        "mean_gt_clip_nearest_selected_sec": float(np.mean(nearest)),
    }


def _paired_rows(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> tuple[
    list[tuple[str, str, str, int]], list[Mapping[str, Any]], list[Mapping[str, Any]]
]:
    key_fields = ("dataset", "video_id", "question_id", "origin_id")

    def index(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, str, str, int], Mapping[str, Any]]:
        result: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
        for row in rows:
            key = (
                str(row[key_fields[0]]),
                str(row[key_fields[1]]),
                str(row[key_fields[2]]),
                int(row.get(key_fields[3], -1)),
            )
            if key in result:
                raise ValueError(f"duplicate policy row: {key}")
            result[key] = row
        return result

    first_index = index(first)
    second_index = index(second)
    if set(first_index) != set(second_index):
        raise ValueError("policy comparison rows do not align exactly")
    keys = sorted(first_index)
    return keys, [first_index[key] for key in keys], [second_index[key] for key in keys]


def _method_policy_interaction(
    first_dwt: Sequence[Mapping[str, Any]],
    first_swt: Sequence[Mapping[str, Any]],
    second_dwt: Sequence[Mapping[str, Any]],
    second_swt: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Jointly bootstrap ``(SWT-DWT)_second - (SWT-DWT)_first``."""

    keys, first_dwt_rows, first_swt_rows = _paired_rows(first_dwt, first_swt)
    second_keys, second_dwt_rows, second_swt_rows = _paired_rows(second_dwt, second_swt)
    if keys != second_keys:
        raise ValueError("method-policy interaction rows do not align exactly")

    def matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    np.nan if row.get(name) is None else float(row[name])
                    for name in metrics
                ]
                for row in rows
            ],
            dtype=float,
        )

    arms = tuple(
        matrix(rows)
        for rows in (first_dwt_rows, first_swt_rows, second_dwt_rows, second_swt_rows)
    )
    defined = np.all(np.stack([np.isfinite(arm) for arm in arms]), axis=(0, 1))
    active_metrics = [name for name, keep in zip(metrics, defined) if keep]
    unavailable = [name for name, keep in zip(metrics, defined) if not keep]
    if not active_metrics:
        return {
            "definition": "(SWT-DWT)_second_policy-(SWT-DWT)_first_policy",
            "effect_order": [],
            "unavailable_metrics": unavailable,
            "reason": "no four-arm metric is finite in every paired row",
        }
    first = np.column_stack((arms[0][:, defined], arms[1][:, defined]))
    second = np.column_stack((arms[2][:, defined], arms[3][:, defined]))
    width = len(active_metrics)
    clusters = np.empty(len(keys), dtype=object)
    clusters[:] = [(key[0], key[1]) for key in keys]
    result = video_cluster_paired_bootstrap(
        clusters,
        first,
        second,
        statistic=lambda first_sample, second_sample: np.mean(
            (second_sample[:, width:] - second_sample[:, :width])
            - (first_sample[:, width:] - first_sample[:, :width]),
            axis=0,
        ),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return {
        "definition": "(SWT-DWT)_second_policy-(SWT-DWT)_first_policy",
        "effect_order": active_metrics,
        "unavailable_metrics": unavailable,
        **result,
    }


def _comparison(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    keys, aligned_first, aligned_second = _paired_rows(first, second)
    baseline = np.asarray(
        [
            [np.nan if row.get(name) is None else float(row[name]) for name in metrics]
            for row in aligned_first
        ],
        dtype=float,
    )
    treatment = np.asarray(
        [
            [np.nan if row.get(name) is None else float(row[name]) for name in metrics]
            for row in aligned_second
        ],
        dtype=float,
    )
    defined = np.all(np.isfinite(baseline) & np.isfinite(treatment), axis=0)
    if not np.any(defined):
        return {
            "effect_order": [],
            "unavailable_metrics": list(metrics),
            "reason": "no paired metric is finite in every row",
        }
    active_metrics = [name for name, keep in zip(metrics, defined) if keep]
    baseline = baseline[:, defined]
    treatment = treatment[:, defined]
    clusters = np.empty(len(keys), dtype=object)
    clusters[:] = [(key[0], key[1]) for key in keys]
    result = video_cluster_paired_bootstrap(
        clusters,
        baseline,
        treatment,
        statistic=lambda a, b: np.mean(b - a, axis=0),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return {
        "effect_order": active_metrics,
        "unavailable_metrics": [
            name for name, keep in zip(metrics, defined) if not keep
        ],
        **result,
    }


def summarize_policy_rows(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    if not rows:
        return {
            "num_rows": 0,
            "descriptive": {},
            "method_contrasts": {},
            "policy_contrasts": {},
            "method_policy_interactions": {},
        }
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["base_method"]), str(row["policy_id"]))].append(row)

    descriptive: dict[str, Any] = {}
    for (method, policy_id), group in sorted(grouped.items()):
        label = f"{method}/{policy_id}"
        method_values: dict[str, float | None] = {}
        for name in metrics:
            values = [float(row[name]) for row in group if row.get(name) is not None]
            method_values[name] = float(np.mean(values)) if values else None
        descriptive[label] = {
            "num_rows": len(group),
            "num_videos": len({(row["dataset"], row["video_id"]) for row in group}),
            **method_values,
        }

    policies = sorted({policy for _, policy in grouped})
    method_contrasts: dict[str, Any] = {}
    if {"dwt", "swt"}.issubset({method for method, _ in grouped}):
        for policy_id in policies:
            if ("dwt", policy_id) not in grouped or ("swt", policy_id) not in grouped:
                continue
            method_contrasts[policy_id] = _comparison(
                grouped[("dwt", policy_id)],
                grouped[("swt", policy_id)],
                metrics,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
            )

    policy_contrasts: dict[str, Any] = {}
    methods = sorted({method for method, _ in grouped})
    for method in methods:
        if (method, "adaptive") in grouped and (method, "topc") in grouped:
            policy_contrasts[f"{method}:topc-minus-adaptive"] = _comparison(
                grouped[(method, "adaptive")],
                grouped[(method, "topc")],
                metrics,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
            )
        nested = sorted(
            policy
            for candidate_method, policy in grouped
            if candidate_method == method and policy.startswith("nested_b")
        )
        for first_policy, second_policy in zip(nested, nested[1:]):
            policy_contrasts[f"{method}:{second_policy}-minus-{first_policy}"] = (
                _comparison(
                    grouped[(method, first_policy)],
                    grouped[(method, second_policy)],
                    metrics,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=seed,
                )
            )
    method_policy_interactions: dict[str, Any] = {}
    comparison_pairs: list[tuple[str, str]] = []
    if "adaptive" in policies:
        comparison_pairs.extend(
            ("adaptive", policy) for policy in policies if policy != "adaptive"
        )
    nested_policies = [policy for policy in policies if policy.startswith("nested_b")]
    comparison_pairs.extend(zip(nested_policies, nested_policies[1:]))
    for first_policy, second_policy in dict.fromkeys(comparison_pairs):
        required = {
            ("dwt", first_policy),
            ("swt", first_policy),
            ("dwt", second_policy),
            ("swt", second_policy),
        }
        if not required.issubset(grouped):
            continue
        method_policy_interactions[f"{second_policy}-minus-{first_policy}"] = (
            _method_policy_interaction(
                grouped[("dwt", first_policy)],
                grouped[("swt", first_policy)],
                grouped[("dwt", second_policy)],
                grouped[("swt", second_policy)],
                metrics,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
            )
        )
    return {
        "num_rows": len(rows),
        "descriptive": descriptive,
        "method_contrasts": method_contrasts,
        "policy_contrasts": policy_contrasts,
        "method_policy_interactions": method_policy_interactions,
    }


def run_policy_separation_experiment(
    records: Sequence[OriginSignalRecord],
    output_dir: str | Path,
    *,
    b_values: Sequence[int],
    config: ExperimentConfig | None = None,
    selection_config: SelectionConfig | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Run adaptive, count-preserving Top-C, and nested common-B policies."""

    experiment = config or ExperimentConfig()
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in b_values
    ):
        raise TypeError("b_values must contain integers")
    counts = tuple(int(value) for value in b_values)
    if (
        not counts
        or any(value <= 0 for value in counts)
        or tuple(sorted(set(counts))) != counts
    ):
        raise ValueError(
            "b_values must be unique positive integers in increasing order"
        )
    groups = group_signal_records(records)
    for item_key, group in groups.items():
        lengths = {len(record.relevance_scores) for record in group}
        if len(lengths) != 1:
            raise ValueError(
                f"item {item_key} origins must use a common candidate count"
            )
    missing_features = [
        record.item_key + (record.origin_id,)
        for record in records
        if record.visual_features_path is None
    ]
    if missing_features:
        raise ValueError(
            "policy-separation requires visual features for selection stability; "
            f"first missing record: {missing_features[0]}"
        )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []

    for item_key in sorted(groups):
        group = groups[item_key]
        signal_length = len(group[0].relevance_scores)
        level = experiment.level or compute_dwt_level(
            signal_length,
            wavelet=experiment.wavelet,
            drift=experiment.drift_level,
        )
        min_distance = compute_min_peak_distance(
            signal_length,
            ratio=experiment.min_distance_ratio,
            absolute_min=experiment.min_distance_absolute,
        )
        for base_method in experiment.methods:
            pipeline = PhaseStableWFS(
                build_transform(
                    _method_transform_config(base_method, level, experiment)
                ),
                selection_config,
            )
            for record in group:
                features = _load_features(record)
                native = pipeline.run(
                    record.relevance_scores,
                    experiment.frame_budget,
                    min_distance,
                    features=features,
                )
                funnel = peak_filter_funnel(
                    native.transform.saliency,
                    height_factor=pipeline.config.height_factor,
                    prominence_factor=pipeline.config.prominence_factor,
                    min_distance=min_distance,
                    timestamps_sec=record.timestamps_sec,
                    wavelet=experiment.wavelet,
                    level=level,
                )
                if funnel["final_peak_indices"] != native.peaks.astype(int).tolist():
                    raise RuntimeError(
                        "peak funnel does not reproduce the native detector"
                    )
                candidate_mask = np.zeros(signal_length, dtype=bool)
                candidate_mask[np.asarray(funnel["local_peak_indices"], dtype=int)] = (
                    True
                )

                policy_traces: list[tuple[str, Any, dict[str, Any]]] = [
                    (
                        "adaptive",
                        native,
                        {
                            "type": "native_adaptive",
                            "candidate_set": "strict_local_maxima",
                            "height_factor": pipeline.config.height_factor,
                            "prominence_factor": pipeline.config.prominence_factor,
                        },
                    )
                ]
                native_count = int(native.peaks.size)
                topc_indices = (
                    np.asarray([], dtype=int)
                    if native_count == 0
                    else select_top_nms_indices(
                        native.transform.saliency,
                        count=native_count,
                        min_distance=min_distance,
                        valid_mask=candidate_mask,
                    )
                )
                topc = pipeline.run(
                    record.relevance_scores,
                    experiment.frame_budget,
                    min_distance,
                    features=features,
                    boundary_indices=np.sort(topc_indices),
                )
                policy_traces.append(
                    (
                        "topc",
                        topc,
                        {
                            "type": "count_preserving_top_c",
                            "count": native_count,
                            "count_source": "same_arm_native_adaptive",
                            "candidate_set": "strict_local_maxima",
                            "ranking_signal": "abs_coarse_detail",
                        },
                    )
                )

                frontier = select_nested_nms_frontier(
                    native.transform.saliency,
                    max_count=counts[-1],
                    min_distance=min_distance,
                )
                if frontier.size != counts[-1]:
                    raise RuntimeError(
                        "nested boundary frontier does not match max requested B"
                    )
                for boundary_count in counts:
                    policy_id = f"nested_b{boundary_count:02d}"
                    indices = np.sort(frontier[:boundary_count])
                    nested = pipeline.run(
                        record.relevance_scores,
                        experiment.frame_budget,
                        min_distance,
                        features=features,
                        boundary_indices=indices,
                    )
                    if nested.peaks.size != boundary_count:
                        raise RuntimeError(
                            f"nested policy B={boundary_count} did not select exact B"
                        )
                    policy_traces.append(
                        (
                            policy_id,
                            nested,
                            {
                                "type": "max_b_anchored_nested_top_b",
                                "count": boundary_count,
                                "max_frontier_count": counts[-1],
                                "candidate_set": "all_valid_interior_samples",
                                "ranking_signal": "abs_coarse_detail",
                            },
                        )
                    )

                for policy_id, trace, boundary_policy in policy_traces:
                    method = f"{base_method}_{policy_id}"
                    _, row = save_trace_npz(output_root, record, method, trace)
                    row.update(
                        {
                            "base_method": base_method,
                            "policy_id": policy_id,
                            "boundary_policy": {
                                **boundary_policy,
                                "min_peak_distance": min_distance,
                            },
                            "level": level,
                            "min_peak_distance": min_distance,
                        }
                    )
                    trace_rows.append(row)
                    policy_boundary_count = int(trace.peaks.size)
                    exposure_sec = float(funnel["exposure_sec"])
                    candidate_set = str(boundary_policy["candidate_set"])
                    local_max_count = int(funnel["local_max_count"])
                    interior_count = (
                        sum(
                            int(funnel["wavelet_support_margin_samples"])
                            <= int(index)
                            < signal_length
                            - int(funnel["wavelet_support_margin_samples"])
                            for index in trace.peaks
                        )
                        if funnel["interior_defined"]
                        else None
                    )
                    peak_rows.append(
                        {
                            "dataset": record.dataset,
                            "video_id": record.video_id,
                            "question_id": record.question_id,
                            "origin_id": record.origin_id,
                            "base_method": base_method,
                            "policy_id": policy_id,
                            **funnel,
                            "native_adaptive_boundary_count": int(
                                funnel["boundary_count"]
                            ),
                            "native_adaptive_peak_indices": funnel[
                                "final_peak_indices"
                            ],
                            "boundary_count": policy_boundary_count,
                            "boundaries_per_minute": float(
                                policy_boundary_count * 60.0 / exposure_sec
                            ),
                            "boundary_local_retention": (
                                float(policy_boundary_count / local_max_count)
                                if local_max_count
                                and candidate_set == "strict_local_maxima"
                                else None
                            ),
                            "interior_boundary_count": interior_count,
                            "edge_boundary_fraction": (
                                float(
                                    (policy_boundary_count - interior_count)
                                    / policy_boundary_count
                                )
                                if interior_count is not None
                                and policy_boundary_count > 0
                                else None
                            ),
                            "candidate_set": candidate_set,
                            "policy_boundary_indices": trace.peaks.astype(int).tolist(),
                            **_allocation_diagnostics(
                                trace,
                                experiment.frame_budget,
                            ),
                        }
                    )
                    fidelity = qvhighlights_selection_fidelity(
                        record,
                        trace.selected_indices,
                    )
                    if fidelity is not None:
                        fidelity_rows.append(
                            {
                                "dataset": record.dataset,
                                "video_id": record.video_id,
                                "question_id": record.question_id,
                                "origin_id": record.origin_id,
                                "method": method,
                                "base_method": base_method,
                                "policy_id": policy_id,
                                "boundary_count": int(trace.peaks.size),
                                **_allocation_diagnostics(
                                    trace,
                                    experiment.frame_budget,
                                ),
                                **fidelity,
                            }
                        )

    metric_rows = compute_trace_metrics(trace_rows, config=experiment)
    write_jsonl(output_root / "traces.jsonl", trace_rows)
    write_jsonl(output_root / "item_metrics.jsonl", metric_rows)
    write_jsonl(output_root / "peak_rows.jsonl", peak_rows)
    write_jsonl(output_root / "fidelity_rows.jsonl", fidelity_rows)
    _write_csv(output_root / "item_metrics.csv", metric_rows)
    _write_csv(output_root / "peak_rows.csv", peak_rows)
    _write_csv(output_root / "fidelity_rows.csv", fidelity_rows)
    metric_summary = {
        "aggregate": aggregate_item_metrics(metric_rows),
        "method_contrasts": {},
    }
    policy_ids = sorted({str(row["policy_id"]) for row in trace_rows})
    for policy_id in policy_ids:
        baseline_method = f"dwt_{policy_id}"
        treatment_method = f"swt_{policy_id}"
        method_names = {str(row["method"]) for row in metric_rows}
        if not {baseline_method, treatment_method}.issubset(method_names):
            continue
        metric_summary["method_contrasts"][policy_id] = paired_metric_bootstrap(
            metric_rows,
            baseline_method=baseline_method,
            treatment_method=treatment_method,
            metric_names=POLICY_STABILITY_METRICS,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
    method_policy = {
        str(row["method"]): (str(row["base_method"]), str(row["policy_id"]))
        for row in trace_rows
    }
    stability_policy_rows = [
        {
            **row,
            "base_method": method_policy[str(row["method"])][0],
            "policy_id": method_policy[str(row["method"])][1],
        }
        for row in metric_rows
    ]
    metric_summary["policy_factorial"] = summarize_policy_rows(
        stability_policy_rows,
        POLICY_STABILITY_METRICS,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return {
        "trace_rows": trace_rows,
        "item_metric_rows": metric_rows,
        "peak_rows": peak_rows,
        "fidelity_rows": fidelity_rows,
        "stability_summary": metric_summary,
        "peak_summary": summarize_policy_rows(
            peak_rows,
            PEAK_METRICS,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        ),
        "fidelity_summary": summarize_policy_rows(
            fidelity_rows,
            FIDELITY_METRICS,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        ),
    }


__all__ = [
    "FIDELITY_METRICS",
    "PEAK_METRICS",
    "POLICY_STABILITY_METRICS",
    "peak_filter_funnel",
    "qvhighlights_selection_fidelity",
    "run_policy_separation_experiment",
    "summarize_policy_rows",
]
