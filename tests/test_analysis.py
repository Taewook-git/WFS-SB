from pathlib import Path

import numpy as np
import pytest

import phase_stable.analysis as analysis_module
from phase_stable.analysis import (
    ExperimentConfig,
    align_time_series,
    compute_matched_cardinality_metrics,
    controlled_shift_metrics,
    evaluate_prediction_interaction_rows,
    evaluate_prediction_rows,
    run_matched_selection_experiment,
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


def _interaction_prediction_rows() -> tuple[list[dict], list[dict]]:
    items = (("v0", "q0"), ("v0", "q1"), ("v1", "q0"), ("v1", "q1"))
    adaptive_values = {
        "dwt": (("A", "A"), ("A", "A"), ("A", "B"), ("B", "B")),
        "swt": (("A", "A"), ("A", "A"), ("A", "B"), ("B", "B")),
    }
    matched_values = {
        "dwt_matched": adaptive_values["dwt"],
        "swt_matched": (
            ("A", "A"),
            ("A", "A"),
            ("A", "A"),
            ("B", "B"),
        ),
    }

    def rows_for(values: dict[str, tuple[tuple[str, str], ...]]) -> list[dict]:
        rows = []
        for method, predictions in values.items():
            for (video_id, question_id), item_predictions in zip(items, predictions):
                for origin_id, prediction in enumerate(item_predictions):
                    rows.append(
                        {
                            "dataset": "demo",
                            "video_id": video_id,
                            "question_id": question_id,
                            "origin_id": origin_id,
                            "method": method,
                            "prediction": prediction,
                            "gold": "A",
                        }
                    )
        return rows

    return rows_for(adaptive_values), rows_for(matched_values)


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


def test_prediction_interaction_uses_one_joint_video_bootstrap(monkeypatch) -> None:
    adaptive, matched = _interaction_prediction_rows()
    original_bootstrap = analysis_module.video_cluster_paired_bootstrap
    calls = 0

    def counted_bootstrap(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_bootstrap(*args, **kwargs)

    monkeypatch.setattr(
        analysis_module, "video_cluster_paired_bootstrap", counted_bootstrap
    )
    result = evaluate_prediction_interaction_rows(
        adaptive, matched, n_bootstrap=100, seed=11
    )

    assert calls == 1
    assert result["num_paired_items"] == 4
    assert result["origin_ids"] == [0, 1]
    assert result["cluster_unit"] == "dataset/video_id"
    expected = np.array([0.125, 0.25, -0.25, 0.25])
    np.testing.assert_allclose(result["adaptive"]["comparison"]["estimate"], 0.0)
    np.testing.assert_allclose(result["matched"]["comparison"]["estimate"], expected)
    np.testing.assert_allclose(result["interaction"]["estimate"], expected)
    assert result["interaction"]["effect_order"] == [
        "interaction_delta_mean_accuracy",
        "interaction_delta_robust_accuracy",
        "interaction_delta_pairwise_answer_disagreement",
        "interaction_delta_worst_origin_accuracy",
    ]
    diagnostics = result["interaction"]["leave_one_video_out"]
    assert [(row["dataset"], row["video_id"]) for row in diagnostics["rows"]] == [
        ("demo", "v0"),
        ("demo", "v1"),
    ]
    np.testing.assert_allclose(diagnostics["min"], [0.0, 0.0, -0.5, 0.0])
    np.testing.assert_allclose(diagnostics["max"], [0.25, 0.5, 0.0, 0.5])


def test_prediction_interaction_is_row_order_invariant() -> None:
    adaptive, matched = _interaction_prediction_rows()
    first = evaluate_prediction_interaction_rows(
        adaptive, matched, n_bootstrap=100, seed=17
    )
    second = evaluate_prediction_interaction_rows(
        list(reversed(adaptive)), list(reversed(matched)), n_bootstrap=100, seed=17
    )
    for section in ("adaptive", "matched"):
        for key in ("estimate", "ci_low", "ci_high"):
            np.testing.assert_allclose(
                first[section]["comparison"][key],
                second[section]["comparison"][key],
            )
    for key in ("estimate", "ci_low", "ci_high"):
        np.testing.assert_allclose(
            first["interaction"][key], second["interaction"][key]
        )
    first_loo = first["interaction"]["leave_one_video_out"]["rows"]
    second_loo = second["interaction"]["leave_one_video_out"]["rows"]
    assert [(row["dataset"], row["video_id"]) for row in first_loo] == [
        (row["dataset"], row["video_id"]) for row in second_loo
    ]
    for first_row, second_row in zip(first_loo, second_loo):
        np.testing.assert_allclose(first_row["estimate"], second_row["estimate"])


def test_prediction_interaction_rejects_any_four_arm_misalignment() -> None:
    adaptive, matched = _interaction_prediction_rows()

    item_mismatch = [dict(row) for row in matched]
    for row in item_mismatch:
        if row["video_id"] == "v1":
            row["video_id"] = "other"
    with pytest.raises(ValueError, match="prediction items do not align"):
        evaluate_prediction_interaction_rows(
            adaptive, item_mismatch, n_bootstrap=10
        )

    origin_mismatch = [dict(row) for row in matched]
    for row in origin_mismatch:
        if row["origin_id"] == 1:
            row["origin_id"] = 2
    with pytest.raises(ValueError, match="origin grids do not align"):
        evaluate_prediction_interaction_rows(
            adaptive, origin_mismatch, n_bootstrap=10
        )

    gold_mismatch = [dict(row) for row in matched]
    for row in gold_mismatch:
        if row["video_id"] == "v0" and row["question_id"] == "q0":
            row["gold"] = "B"
    with pytest.raises(ValueError, match="gold answers do not align"):
        evaluate_prediction_interaction_rows(
            adaptive, gold_mismatch, n_bootstrap=10
        )

    missing_arm = [row for row in matched if row["method"] != "swt_matched"]
    with pytest.raises(ValueError, match="matched predictions require methods"):
        evaluate_prediction_interaction_rows(adaptive, missing_arm, n_bootstrap=10)


def test_prediction_interaction_preserves_mixed_scalar_labels() -> None:
    adaptive, matched = _interaction_prediction_rows()
    for row in (*adaptive, *matched):
        row["prediction"] = 1 if row["prediction"] == "A" else 2
        row["gold"] = "1"

    result = evaluate_prediction_interaction_rows(
        adaptive, matched, n_bootstrap=20, seed=3
    )
    # Numeric predictions must remain distinct from string gold labels inside
    # the bootstrap bundles instead of being coerced to strings by NumPy.
    np.testing.assert_allclose(
        result["interaction"]["estimate"], [0.0, 0.0, -0.25, 0.0]
    )
    for regime in ("adaptive", "matched"):
        for metrics in result[regime]["methods"].values():
            assert metrics["mean_accuracy"] == 0.0
    np.testing.assert_allclose(result["adaptive"]["comparison"]["estimate"], 0.0)
    np.testing.assert_allclose(
        result["matched"]["comparison"]["estimate"], [0.0, 0.0, -0.25, 0.0]
    )

    invalid_origin = [dict(row) for row in adaptive]
    invalid_origin[0]["origin_id"] = 0.5
    with pytest.raises(ValueError, match="origin_id must be a non-negative integer"):
        evaluate_prediction_interaction_rows(
            invalid_origin, matched, n_bootstrap=10
        )


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


def test_matched_selection_writes_distinct_exact_count_traces(tmp_path: Path) -> None:
    config = ExperimentConfig(
        methods=("dwt", "swt"),
        level=3,
        frame_budget=8,
        min_distance_absolute=3,
    )
    traces, metrics = run_matched_selection_experiment(
        _records(), tmp_path, 4, config=config
    )
    assert len(traces) == 6
    assert len(metrics) == 2
    assert {row["method"] for row in traces} == {"dwt_matched", "swt_matched"}
    assert {row["base_method"] for row in traces} == {"dwt", "swt"}
    assert all(len(row["peaks"]) == 4 for row in traces)
    assert all(len(row["segments"]) == 5 for row in traces)
    assert all(row["matched_boundary_count"] == 4 for row in traces)
    assert all(row["boundary_policy"]["count_source"] == "fixed" for row in traces)
    assert all(len(row["selected_indices"]) == 8 for row in traces)


def test_matched_selection_accepts_only_calibrated_video_count_mapping(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        methods=("dwt", "swt"), level=3, frame_budget=8, min_distance_absolute=3
    )
    traces, _ = run_matched_selection_experiment(
        _records(), tmp_path / "mapped", {"video-1": 3}, config=config
    )
    assert all(len(row["peaks"]) == 3 for row in traces)
    assert all(
        row["boundary_policy"]["count_source"] == "counts_json" for row in traces
    )
    with pytest.raises(ValueError, match="no calibrated boundary count"):
        run_matched_selection_experiment(
            _records(), tmp_path / "missing", {"another-video": 3}, config=config
        )
