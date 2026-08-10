"""Uniform and Top-K context baselines for downstream origin experiments."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .artifacts import OriginSignalRecord, write_jsonl
from .metrics import selected_timestamp_metrics


BASELINE_METHODS = ("uniform", "topk")


def uniform_indices(length: int, budget: int) -> list[int]:
    if length <= 0 or budget <= 0:
        raise ValueError("length and budget must be positive")
    if length < budget:
        raise ValueError(f"candidate count {length} is smaller than frame budget {budget}")
    # Match the official WFS-SB uniform fallback's endpoint-inclusive policy.
    indices = np.linspace(0, length - 1, budget, dtype=int)
    return indices.tolist()


def topk_indices(scores: Sequence[float], budget: int) -> list[int]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional array")
    if budget <= 0 or values.size < budget:
        raise ValueError("budget must be positive and no larger than score length")
    # lexsort uses candidate time as a deterministic earlier-frame tie-break.
    ranked = np.lexsort((np.arange(values.size), -values))[:budget]
    return sorted(int(index) for index in ranked)


def _baseline_trace_row(
    record: OriginSignalRecord,
    method: str,
    selected_indices: Sequence[int],
) -> Dict[str, Any]:
    timestamps = np.asarray(record.timestamps_sec, dtype=float)
    actual_pts = np.asarray(record.actual_pts_sec, dtype=float)
    source_indices = np.asarray(record.source_frame_indices, dtype=int)
    selected = np.asarray(selected_indices, dtype=int)
    return {
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
        "peaks": [],
        "peaks_sec": [],
        "segments": [[0, len(record.relevance_scores)]],
        "valid_segments": [[0, len(record.relevance_scores)]],
        "importance_scores": [],
        "valid_importance_scores": [],
        "allocation": {},
        "selected_indices": selected.tolist(),
        "selected_timestamps_sec": timestamps[selected].tolist(),
        "selected_actual_pts_sec": actual_pts[selected].tolist(),
        "selected_source_frame_indices": source_indices[selected].tolist(),
        "selected_pixel_hashes": (
            []
            if not record.pixel_hashes
            else np.asarray(record.pixel_hashes, dtype=str)[selected].tolist()
        ),
        "used_fallback": False,
        "transform": None,
    }


def run_selection_baselines(
    records: Sequence[OriginSignalRecord],
    output_dir: str | Path,
    *,
    frame_budget: int = 16,
    methods: Sequence[str] = BASELINE_METHODS,
    selected_tolerance_sec: float = 1.0,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Write baseline traces and origin-pair selection stability metrics."""

    selected_methods = tuple(dict.fromkeys(str(method) for method in methods))
    if not selected_methods or any(method not in BASELINE_METHODS for method in selected_methods):
        raise ValueError(f"methods must be selected from {BASELINE_METHODS}")
    rows: list[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, OriginSignalRecord):
            raise TypeError("records must contain OriginSignalRecord objects")
        for method in selected_methods:
            if method == "uniform":
                selected = uniform_indices(len(record.relevance_scores), frame_budget)
            else:
                selected = topk_indices(record.relevance_scores, frame_budget)
            rows.append(_baseline_trace_row(record, method, selected))

    metrics = compute_baseline_selection_metrics(
        rows,
        tolerance_sec=selected_tolerance_sec,
    )
    root = Path(output_dir)
    write_jsonl(root / "baseline_traces.jsonl", rows)
    write_jsonl(root / "baseline_item_metrics.jsonl", metrics)
    return rows, metrics


def compute_baseline_selection_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_sec: float = 1.0,
) -> list[Dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["video_id"]),
            str(row["question_id"]),
            str(row["method"]),
        )
        groups[key].append(row)

    results: list[Dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: int(row["origin_id"]))
        if len(group) < 2:
            raise ValueError(f"baseline group {key} must have at least two origins")
        origin_ids = [int(row["origin_id"]) for row in group]
        if len(origin_ids) != len(set(origin_ids)):
            raise ValueError(f"duplicate origin in baseline group {key}")
        f1_values: list[float] = []
        distance_values: list[float] = []
        embedding_values: list[float] = []
        matrices: list[Optional[np.ndarray]] = []
        for row in group:
            path = row.get("visual_features_path")
            matrices.append(
                None
                if path is None
                else np.asarray(np.load(str(path), allow_pickle=False), dtype=float)
            )
        for first, second in combinations(range(len(group)), 2):
            kwargs: Dict[str, Any] = {}
            if matrices[first] is not None and matrices[second] is not None:
                kwargs = {
                    "first_embeddings": matrices[first][group[first]["selected_indices"]],
                    "second_embeddings": matrices[second][group[second]["selected_indices"]],
                }
            selected = selected_timestamp_metrics(
                group[first]["selected_actual_pts_sec"],
                group[second]["selected_actual_pts_sec"],
                tolerance=tolerance_sec,
                **kwargs,
            )
            f1_values.append(selected["f1"])
            if selected["optimal_mean_distance"] is not None:
                distance_values.append(selected["optimal_mean_distance"])
            if selected.get("matched_embedding_cosine") is not None:
                embedding_values.append(selected["matched_embedding_cosine"])
        results.append(
            {
                "dataset": key[0],
                "video_id": key[1],
                "question_id": key[2],
                "method": key[3],
                "num_origins": len(group),
                "selected_f1_mean": float(np.mean(f1_values)),
                "selected_f1_worst": float(np.min(f1_values)),
                "selected_distance_mean": (
                    None if not distance_values else float(np.mean(distance_values))
                ),
                "selected_embedding_cosine_mean": (
                    None if not embedding_values else float(np.mean(embedding_values))
                ),
            }
        )
    return results


__all__ = [
    "BASELINE_METHODS",
    "compute_baseline_selection_metrics",
    "run_selection_baselines",
    "topk_indices",
    "uniform_indices",
]
