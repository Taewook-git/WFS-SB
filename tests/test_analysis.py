from pathlib import Path

import numpy as np

from phase_stable.analysis import (
    ExperimentConfig,
    align_time_series,
    compute_matched_cardinality_metrics,
    controlled_shift_metrics,
    evaluate_prediction_rows,
    run_real_origin_experiment,
)
from phase_stable.artifacts import OriginSignalRecord


def _records() -> list[OriginSignalRecord]:
    records = []
    for origin_id, origin_sec in enumerate((0.05, 0.25, 0.45)):
        timestamps = origin_sec + np.arange(128, dtype=float)
        scores = 0.5 + 0.3 * np.sin(timestamps / 8.0)
        scores[timestamps > 50] += 0.1
        records.append(
            OriginSignalRecord(
                dataset="demo",
                video_id="video-1",
                question_id="question-1",
                origin_id=origin_id,
                origin_sec=origin_sec,
                timestamps_sec=tuple(timestamps),
                actual_pts_sec=tuple(timestamps),
                source_frame_indices=tuple(range(128)),
                relevance_scores=tuple(scores),
            )
        )
    return records


def test_align_time_series_uses_common_absolute_grid() -> None:
    times = [np.asarray(record.timestamps_sec) for record in _records()]
    arrays = [np.asarray(record.relevance_scores) for record in _records()]
    grid, aligned = align_time_series(times, arrays)
    assert aligned.shape == (3, grid.size)
    assert grid[0] >= max(time[0] for time in times)
    assert grid[-1] <= min(time[-1] for time in times)


def test_real_origin_experiment_writes_traces_and_metrics(tmp_path: Path) -> None:
    config = ExperimentConfig(
        methods=("dwt", "swt"),
        level=3,
        frame_budget=8,
        min_distance_absolute=3,
    )
    traces, metrics = run_real_origin_experiment(_records(), tmp_path, config=config)
    assert len(traces) == 6
    assert len(metrics) == 2
    assert (tmp_path / "traces.jsonl").is_file()
    assert (tmp_path / "item_metrics.jsonl").is_file()
    assert {row["method"] for row in metrics} == {"dwt", "swt"}
    matched = compute_matched_cardinality_metrics(traces, 2)
    assert len(matched) == 2
    assert all(row["matched_boundary_count"] == 2 for row in matched)


def test_controlled_shift_reports_both_methods() -> None:
    signal = np.zeros(256)
    signal[60:130] = 1.0
    result = controlled_shift_metrics(signal, level=4)
    assert set(result) == {"dwt", "swt"}
    assert result["swt"]["representation_consistency_mean"] > 0.99


def test_prediction_metrics_and_cluster_bootstrap() -> None:
    rows = []
    for method in ("dwt", "swt"):
        for video_index in range(3):
            for origin_id in range(3):
                gold = "A"
                if method == "swt":
                    prediction = "A"
                else:
                    prediction = "B" if origin_id == 2 and video_index == 0 else "A"
                rows.append(
                    {
                        "dataset": "demo",
                        "video_id": f"v{video_index}",
                        "question_id": f"q{video_index}",
                        "origin_id": origin_id,
                        "method": method,
                        "prediction": prediction,
                        "gold": gold,
                    }
                )
    result = evaluate_prediction_rows(rows, n_bootstrap=100, seed=7)
    assert result["methods"]["swt"]["robust_accuracy"] == 1.0
    assert result["comparison"]["estimate"][1] > 0


def test_constant_signal_is_reported_as_degenerate_without_crashing(tmp_path: Path) -> None:
    records = []
    for origin_id, origin_sec in enumerate((0.1, 0.3)):
        timestamps = origin_sec + np.arange(64, dtype=float)
        records.append(
            OriginSignalRecord(
                dataset="demo",
                video_id="constant-video",
                question_id="constant-question",
                origin_id=origin_id,
                origin_sec=origin_sec,
                timestamps_sec=tuple(timestamps),
                actual_pts_sec=tuple(timestamps),
                source_frame_indices=tuple(range(64)),
                relevance_scores=tuple(np.full(64, 0.5)),
            )
        )
    _, metrics = run_real_origin_experiment(
        records,
        tmp_path,
        config=ExperimentConfig(methods=("dwt", "swt"), level=2, frame_budget=8),
    )
    assert len(metrics) == 2
    assert all(row["saliency_zero_origin_rate"] >= 0 for row in metrics)
    assert all("saliency_temporal_variance_mean" in row for row in metrics)
