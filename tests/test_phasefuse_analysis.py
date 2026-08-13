from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from phase_stable.artifacts import OriginSignalRecord
from phase_stable.phasefuse_analysis import (
    categorize_prediction_stability,
    evaluate_phasefuse_analysis,
    evaluate_prediction_failure_calibration,
    evaluate_prediction_stability_rows,
    phase_uncertainty_failure_calibration,
    selected_set_consistency,
)


def _trace(
    method: str,
    video_id: str,
    origin_id: int,
    selected: Sequence[int],
    *,
    question_id: str = "q1",
    candidates: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, Any]:
    candidate_ids = list(candidates)
    candidate_times = [float(value) for value in range(len(candidate_ids))]
    selected_indices = list(selected)
    return {
        "dataset": "qvhighlights",
        "video_id": video_id,
        "question_id": question_id,
        "origin_id": origin_id,
        "method": method,
        "timestamps_sec": candidate_times,
        "actual_pts_sec": candidate_times,
        "source_frame_indices": candidate_ids,
        "selected_indices": selected_indices,
        "selected_actual_pts_sec": [
            candidate_times[index] for index in selected_indices
        ],
        "selected_source_frame_indices": [
            candidate_ids[index] for index in selected_indices
        ],
    }


def _qv_record(
    video_id: str, origin_id: int, *, question_id: str = "q1"
) -> OriginSignalRecord:
    times = (0.0, 1.0, 2.0, 3.0)
    return OriginSignalRecord(
        dataset="qvhighlights",
        video_id=video_id,
        question_id=question_id,
        origin_id=origin_id,
        origin_sec=float(origin_id) / 10.0,
        timestamps_sec=times,
        actual_pts_sec=times,
        source_frame_indices=(0, 1, 2, 3),
        relevance_scores=(0.9, 0.8, 0.2, 0.1),
        metadata={
            "query_metadata": {
                "relevant_windows_sec": [[0.0, 2.0]],
                "relevant_clip_ids": [0],
                "saliency_votes": [[4, 4, 4]],
            }
        },
    )


def test_phase_uncertainty_calibration_has_exact_oracle_and_reversed_values() -> None:
    oracle = phase_uncertainty_failure_calibration([0.0, 1.0], [0.0, 1.0])
    reversed_ranking = phase_uncertainty_failure_calibration([1.0, 0.0], [0.0, 1.0])

    assert oracle["aurc"] == pytest.approx(0.25)
    assert oracle["oracle_aurc"] == pytest.approx(0.25)
    assert oracle["excess_aurc"] == pytest.approx(0.0)
    assert oracle["brier_score"] == pytest.approx(0.0)
    assert reversed_ranking["aurc"] == pytest.approx(0.75)
    assert reversed_ranking["excess_aurc"] == pytest.approx(0.5)
    assert reversed_ranking["brier_score"] == pytest.approx(1.0)


def test_calibration_ties_are_row_order_invariant() -> None:
    first = phase_uncertainty_failure_calibration([0.2, 0.2, 0.8], [0.0, 1.0, 1.0])
    second = phase_uncertainty_failure_calibration([0.8, 0.2, 0.2], [1.0, 1.0, 0.0])

    assert first["aurc"] == pytest.approx(second["aurc"])
    assert first["risk"] == pytest.approx(second["risk"])


def test_selected_set_consistency_uses_canonical_source_frame_sets() -> None:
    rows = [
        _trace("main", "v1", 0, (0, 1)),
        _trace("main", "v1", 1, (1, 2)),
    ]

    result = selected_set_consistency(rows, method="main")

    assert len(result) == 1
    assert result[0]["selected_set_consistency"] == pytest.approx(1.0 / 3.0)
    assert result[0]["selected_set_overlap_mean"] == pytest.approx(0.5)
    assert result[0]["selected_all_origin_intersection_fraction"] == pytest.approx(0.5)
    assert result[0]["outer_origin_selected_set_uncertainty"] == pytest.approx(
        2.0 / 3.0
    )
    assert result[0]["selected_timestamp_f1_at_0p25s_mean"] == pytest.approx(0.5)
    assert result[0]["selected_timestamp_f1_at_0p5s_mean"] == pytest.approx(0.5)
    # At one-second tolerance, both selected timestamps can be matched.
    assert result[0]["selected_timestamp_f1_at_1s_mean"] == pytest.approx(1.0)
    assert result[0]["selected_timestamp_f1_mean"] == pytest.approx(1.0)
    assert result[0]["outer_origin_selector_uncertainty"] == pytest.approx(0.5)


def test_primary_selector_stability_uses_actual_pts_not_exact_frame_ids() -> None:
    first = _trace("main", "v1", 0, (0, 2))
    second = _trace("main", "v1", 1, (1, 3))
    second["actual_pts_sec"] = [0.0, 0.4, 2.0, 2.4]
    second["selected_actual_pts_sec"] = [0.4, 2.4]

    [result] = selected_set_consistency([first, second], method="main")

    assert result["selected_set_consistency"] == pytest.approx(0.0)
    assert result["selected_timestamp_f1_at_0p25s_mean"] == pytest.approx(0.0)
    assert result["selected_timestamp_f1_at_0p5s_mean"] == pytest.approx(1.0)
    assert result["selected_timestamp_f1_at_1s_mean"] == pytest.approx(1.0)
    assert result["outer_origin_selector_uncertainty"] == pytest.approx(0.0)
    assert result["outer_origin_selected_set_uncertainty"] == pytest.approx(1.0)


def test_evaluate_phasefuse_is_compute_matched_and_bootstraps_video_clusters() -> None:
    rows: list[dict[str, Any]] = []
    records: list[OriginSignalRecord] = []
    for video_id in ("v1", "v2"):
        for origin_id in (0, 1):
            rows.append(_trace("baseline", video_id, origin_id, (0, 1)))
            rows.append(_trace("treatment", video_id, origin_id, (2, 3)))
            records.append(_qv_record(video_id, origin_id))

    result = evaluate_phasefuse_analysis(
        rows,
        baseline_method="baseline",
        treatment_method="treatment",
        qv_records=records,
        n_bootstrap=40,
        seed=7,
    )

    assert result["num_paired_items"] == 2
    assert result["origin_ids"] == [0, 1]
    assert result["frame_budget"] == 2
    assert result["comparison"]["n_clusters"] == 2
    effect_index = {
        name: index for index, name in enumerate(result["comparison"]["effect_order"])
    }
    estimate = result["comparison"]["estimate"]
    assert estimate[effect_index["delta_selected_relevant_fraction"]] == pytest.approx(
        -1.0
    )
    assert estimate[effect_index["delta_qv_evidence_loss"]] == pytest.approx(1.0)
    assert estimate[
        effect_index["delta_outer_origin_selected_set_failure_brier"]
    ] == pytest.approx(1.0)
    assert result["methods"]["baseline"][
        "outer_origin_selected_set_failure_calibration"
    ]["brier_score"] == pytest.approx(0.0)
    assert result["methods"]["treatment"][
        "outer_origin_selected_set_failure_calibration"
    ]["brier_score"] == pytest.approx(1.0)
    assert result["inner_phase_uncertainty"]["methods"]["treatment"]["status"] == (
        "unavailable"
    )
    assert result["schema_version"] == 1
    assert result["selector_stability"]["primary"]["reference_metric"] == (
        "selected_timestamp_f1_at_0p5s_mean"
    )
    assert result["selector_stability"]["secondary"]["reference_metric"] == (
        "selected_set_consistency"
    )
    assert result["selector_stability"]["legacy_aliases"] == {
        "selected_timestamp_f1_mean": "selected_timestamp_f1_at_1s_mean",
        "selected_timestamp_f1_worst": "selected_timestamp_f1_at_1s_worst",
    }
    baseline_metrics = result["methods"]["baseline"]["metrics"]
    # Schema-v1 fields remain available alongside the new primary PTS metrics.
    assert {
        "selected_set_consistency",
        "selected_timestamp_f1_mean",
        "selected_timestamp_f1_worst",
        "outer_origin_selected_set_uncertainty",
    } <= set(baseline_metrics)
    assert "delta_selected_timestamp_f1_mean" in result["comparison"]["effect_order"]
    assert (
        "delta_outer_origin_selected_set_uncertainty"
        in result["comparison"]["effect_order"]
    )
    # Public output must be strict-JSON serializable, including bootstrap arrays.
    json.dumps(result, allow_nan=False)


def test_evaluate_phasefuse_is_row_order_invariant() -> None:
    rows = []
    for video_id in ("v1", "v2"):
        for origin_id in (0, 1):
            rows.extend(
                (
                    _trace("baseline", video_id, origin_id, (0, 1)),
                    _trace("treatment", video_id, origin_id, (0, 2)),
                )
            )
    first = evaluate_phasefuse_analysis(
        rows,
        baseline_method="baseline",
        treatment_method="treatment",
        n_bootstrap=30,
        seed=11,
    )
    second = evaluate_phasefuse_analysis(
        list(reversed(rows)),
        baseline_method="baseline",
        treatment_method="treatment",
        n_bootstrap=30,
        seed=11,
    )

    assert first["comparison"]["estimate"] == pytest.approx(
        second["comparison"]["estimate"]
    )
    assert first["comparison"]["ci_low"] == pytest.approx(
        second["comparison"]["ci_low"]
    )
    assert first["comparison"]["ci_high"] == pytest.approx(
        second["comparison"]["ci_high"]
    )


def test_inner_phase_uncertainty_is_reported_in_a_distinct_calibration_block() -> None:
    rows: list[dict[str, Any]] = []
    records: list[OriginSignalRecord] = []
    for video_id in ("v1", "v2"):
        for origin_id in (0, 1):
            baseline = _trace("baseline", video_id, origin_id, (0, 1))
            treatment = _trace("treatment", video_id, origin_id, (2, 3))
            baseline.update(
                {
                    "selected_phase_uncertainty_mean": 0.0,
                    "selected_phase_uncertainty_max": 0.0,
                    "phase_uncertainty_mean": 0.0,
                }
            )
            treatment.update(
                {
                    "selected_phase_uncertainty_mean": 1.0,
                    "selected_phase_uncertainty_max": 1.2,
                    "phase_uncertainty_mean": 0.8,
                }
            )
            rows.extend((baseline, treatment))
            records.append(_qv_record(video_id, origin_id))

    result = evaluate_phasefuse_analysis(
        rows,
        baseline_method="baseline",
        treatment_method="treatment",
        qv_records=records,
        n_bootstrap=20,
        seed=3,
    )

    inner = result["inner_phase_uncertainty"]
    assert "outer-origin selected-set uncertainty" in inner["definition"]
    assert inner["methods"]["baseline"]["raw_metrics"][
        "selected_phase_uncertainty_mean"
    ] == pytest.approx(0.0)
    assert inner["methods"]["treatment"]["raw_metrics"][
        "selected_phase_uncertainty_mean"
    ] == pytest.approx(1.0)
    assert (
        inner["methods"]["treatment"]["failure_calibration"]["probability_mapping"]
        == "u / (1 + u), fixed and unfitted"
    )
    assert inner["comparison"]["status"] == "available"
    effect_index = {
        name: index for index, name in enumerate(inner["comparison"]["effect_order"])
    }
    assert inner["comparison"]["estimate"][
        effect_index["delta_raw_selected_phase_uncertainty_mean"]
    ] == pytest.approx(1.0)
    assert inner["comparison"]["estimate"][
        effect_index["delta_inner_phase_failure_brier"]
    ] == pytest.approx(0.25)
    assert "phase_uncertainty_definition" not in result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("baseline_method", "baseline_metadata"),
    (
        ("dense_swt", {}),
        (
            "baseline",
            {
                "selector_kind": "direct_dense_single_stream",
                "phasefuse": {
                    "num_phases": 1,
                    "config": {"num_phases": 1},
                },
            },
        ),
    ),
)
def test_single_phase_inner_uncertainty_is_unavailable_not_numeric_zero(
    baseline_method: str, baseline_metadata: dict[str, Any]
) -> None:
    rows: list[dict[str, Any]] = []
    for origin_id in (0, 1):
        baseline = _trace(baseline_method, "v1", origin_id, (0, 1))
        baseline.update(
            {
                "selected_phase_uncertainty_mean": 0.0,
                "selected_phase_uncertainty_max": 0.0,
                "phase_uncertainty_mean": 0.0,
                "method_metadata": baseline_metadata,
            }
        )
        treatment = _trace("treatment", "v1", origin_id, (0, 2))
        treatment.update(
            {
                "selected_phase_uncertainty_mean": 0.2,
                "selected_phase_uncertainty_max": 0.3,
                "phase_uncertainty_mean": 0.1,
                "method_metadata": {
                    "selector_kind": "multiphase_fusion",
                    "phasefuse": {
                        "num_phases": 4,
                        "config": {"num_phases": 4},
                    },
                },
            }
        )
        rows.extend((baseline, treatment))

    result = evaluate_phasefuse_analysis(
        rows,
        baseline_method=baseline_method,
        treatment_method="treatment",
        n_bootstrap=5,
    )

    baseline_block = result["inner_phase_uncertainty"]["methods"][baseline_method]
    assert baseline_block["status"] == "unavailable"
    assert (
        "single" in baseline_block["reason"]
        or "num_phases<2" in baseline_block["reason"]
    )
    assert "raw_metrics" not in baseline_block
    assert result["inner_phase_uncertainty"]["methods"]["treatment"]["status"] == (
        "available"
    )
    assert result["inner_phase_uncertainty"]["comparison"]["status"] == "unavailable"


def test_inner_phase_optional_fields_reject_partial_artifacts() -> None:
    rows = [
        _trace(method, "v1", origin, (0, 1))
        for method in ("baseline", "treatment")
        for origin in (0, 1)
    ]
    treatment_rows = [row for row in rows if row["method"] == "treatment"]
    treatment_rows[0]["selected_phase_uncertainty_mean"] = 0.2

    with pytest.raises(ValueError, match="partially missing optional field"):
        evaluate_phasefuse_analysis(
            rows,
            baseline_method="baseline",
            treatment_method="treatment",
            n_bootstrap=5,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_origin", "rectangular origin grid"),
        ("candidate_mismatch", "candidate source frames do not align"),
        ("actual_pts_mismatch", "candidate actual_pts_sec do not align"),
        ("budget_mismatch", "frame budgets are not compute-matched"),
        ("duplicate_selection", "must be unique"),
    ),
)
def test_evaluate_phasefuse_rejects_nonmatched_or_invalid_traces(
    mutation: str, message: str
) -> None:
    rows = [
        _trace(method, video, origin, (0, 1))
        for method in ("baseline", "treatment")
        for video in ("v1", "v2")
        for origin in (0, 1)
    ]
    if mutation == "missing_origin":
        rows = [
            row
            for row in rows
            if not (
                row["method"] == "treatment"
                and row["video_id"] == "v2"
                and row["origin_id"] == 1
            )
        ]
    elif mutation == "candidate_mismatch":
        target = next(row for row in rows if row["method"] == "treatment")
        target["source_frame_indices"] = [0, 1, 2, 99]
    elif mutation == "actual_pts_mismatch":
        target = next(row for row in rows if row["method"] == "treatment")
        target["actual_pts_sec"] = [0.0, 1.0, 2.0, 3.1]
    elif mutation == "budget_mismatch":
        for row in rows:
            if row["method"] == "treatment":
                row["selected_indices"] = [0]
                row["selected_actual_pts_sec"] = [0.0]
                row["selected_source_frame_indices"] = [0]
    else:
        target = rows[0]
        target["selected_indices"] = [0, 0]
        target["selected_actual_pts_sec"] = [0.0, 0.0]
        target["selected_source_frame_indices"] = [0, 0]

    with pytest.raises(ValueError, match=message):
        evaluate_phasefuse_analysis(
            rows,
            baseline_method="baseline",
            treatment_method="treatment",
            n_bootstrap=5,
        )


def _prediction(
    video_id: str, origin_id: int, prediction: str, *, method: str = "main"
) -> dict[str, Any]:
    return {
        "dataset": "videomme",
        "video_id": video_id,
        "question_id": f"q-{video_id}",
        "origin_id": origin_id,
        "method": method,
        "prediction": prediction,
        "gold": "A",
    }


def test_prediction_categories_are_mutually_exhaustive() -> None:
    rows = [
        _prediction("correct", 0, "A"),
        _prediction("correct", 1, "A"),
        _prediction("wrong", 0, "B"),
        _prediction("wrong", 1, "B"),
        _prediction("mixed", 0, "A"),
        _prediction("mixed", 1, "B"),
    ]

    result = categorize_prediction_stability(rows)

    assert result["methods"]["main"]["counts"] == {
        "stable_correct": 1,
        "stable_wrong": 1,
        "mixed": 1,
    }
    metrics = result["methods"]["main"]["mllm_stability_metrics"]
    assert metrics["mean_accuracy"] == pytest.approx(0.5)
    assert metrics["robust_accuracy"] == pytest.approx(1 / 3)
    assert metrics["pairwise_answer_disagreement"] == pytest.approx(1 / 3)
    by_video = {row["video_id"]: row for row in result["item_rows"]}
    assert by_video["correct"]["failure_rate"] == pytest.approx(0.0)
    assert by_video["wrong"]["failure_rate"] == pytest.approx(1.0)
    assert by_video["mixed"]["failure_rate"] == pytest.approx(0.5)
    json.dumps(result, allow_nan=False)


def test_prediction_category_contrasts_use_paired_video_cluster_bootstrap() -> None:
    rows = [
        _prediction("v1", 0, "B", method="baseline"),
        _prediction("v1", 1, "B", method="baseline"),
        _prediction("v2", 0, "A", method="baseline"),
        _prediction("v2", 1, "B", method="baseline"),
        _prediction("v1", 0, "A", method="treatment"),
        _prediction("v1", 1, "A", method="treatment"),
        _prediction("v2", 0, "A", method="treatment"),
        _prediction("v2", 1, "A", method="treatment"),
    ]

    result = evaluate_prediction_stability_rows(
        rows,
        baseline_method="baseline",
        treatment_method="treatment",
        n_bootstrap=30,
        seed=5,
    )

    assert result["comparison"]["n_clusters"] == 2
    assert result["comparison"]["effect_order"] == [
        "delta_stable_correct_fraction",
        "delta_stable_wrong_fraction",
        "delta_mixed_fraction",
        "delta_mean_accuracy",
        "delta_pairwise_answer_disagreement",
        "delta_failure_rate",
    ]
    assert result["comparison"]["estimate"] == pytest.approx(
        [1.0, -0.5, -0.5, 0.75, -0.5, -0.75]
    )
    json.dumps(result, allow_nan=False)


def test_prediction_failure_calibration_strictly_joins_trace_items() -> None:
    predictions = [
        _prediction("correct", 0, "A"),
        _prediction("correct", 1, "A"),
        _prediction("wrong", 0, "B"),
        _prediction("wrong", 1, "B"),
    ]
    traces = [
        _trace("main", "correct", 0, (0, 1), question_id="q-correct"),
        _trace("main", "correct", 1, (0, 1), question_id="q-correct"),
        _trace("main", "wrong", 0, (0, 1), question_id="q-wrong"),
        _trace("main", "wrong", 1, (2, 3), question_id="q-wrong"),
    ]
    for row in traces:
        row["dataset"] = "videomme"

    result = evaluate_prediction_failure_calibration(
        traces, predictions, methods=("main",)
    )

    calibration = result["outer_origin_selected_set_failure_calibration"]["main"]
    assert calibration["aurc"] == pytest.approx(0.25)
    assert calibration["brier_score"] == pytest.approx(0.0)
    assert calibration["failure_definition"] == (
        "fraction of origins answered incorrectly"
    )


def test_qv_alignment_rejects_silent_partial_intersection() -> None:
    rows = [
        _trace(method, "v1", origin, (0, 1))
        for method in ("baseline", "treatment")
        for origin in (0, 1)
    ]
    records = [_qv_record("v1", 0)]

    with pytest.raises(ValueError, match="do not align exactly"):
        evaluate_phasefuse_analysis(
            rows,
            baseline_method="baseline",
            treatment_method="treatment",
            qv_records=records,
            n_bootstrap=5,
        )
