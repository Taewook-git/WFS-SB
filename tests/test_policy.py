from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import find_peaks

import phase_stable.policy as policy_module
from phase_stable.analysis import ExperimentConfig
from phase_stable.artifacts import OriginSignalRecord
from phase_stable.metrics import video_cluster_paired_bootstrap
from phase_stable.pipeline import (
    select_nested_nms_frontier,
    select_top_nms_indices,
)
from phase_stable.policy import (
    peak_filter_funnel,
    qvhighlights_selection_fidelity,
    run_policy_separation_experiment,
    summarize_policy_rows,
)


def test_peak_filter_funnel_matches_stage_oracle() -> None:
    saliency = np.asarray([0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0])
    result = peak_filter_funnel(
        saliency,
        height_factor=0.5,
        prominence_factor=0.05,
        min_distance=3,
        timestamps_sec=np.arange(saliency.size, dtype=float),
        wavelet="haar",
        level=1,
    )

    assert result["local_peak_indices"] == [1, 3, 5, 7]
    assert result["final_peak_indices"] == [3, 7]
    assert result["local_max_count"] == 4
    assert result["height_pass_count"] == 3
    assert result["height_distance_pass_count"] == 2
    assert result["height_prominence_pass_count"] == 3
    assert result["boundary_count"] == 2
    assert result["height_threshold"] == pytest.approx(1.8354669339114056)
    assert result["prominence_threshold"] == pytest.approx(0.2)
    assert result["sample_step_sec"] == pytest.approx(1.0)
    assert result["exposure_sec"] == pytest.approx(9.0)
    assert result["local_max_per_minute"] == pytest.approx(80.0 / 3.0)
    assert result["boundaries_per_minute"] == pytest.approx(40.0 / 3.0)
    assert result["height_local_retention"] == pytest.approx(3.0 / 4.0)
    assert result["distance_height_retention"] == pytest.approx(2.0 / 3.0)
    assert result["prominence_distance_retention"] == pytest.approx(1.0)
    assert result["boundary_local_retention"] == pytest.approx(0.5)


def test_peak_filter_funnel_reports_constant_signal_without_fake_peaks() -> None:
    result = peak_filter_funnel(
        np.zeros(9, dtype=float),
        height_factor=0.5,
        prominence_factor=0.05,
        min_distance=3,
        timestamps_sec=np.arange(9, dtype=float) / 2.0,
        wavelet="haar",
        level=1,
    )

    assert result["local_peak_indices"] == []
    assert result["final_peak_indices"] == []
    assert result["local_max_count"] == 0
    assert result["height_pass_count"] == 0
    assert result["height_distance_pass_count"] == 0
    assert result["height_prominence_pass_count"] == 0
    assert result["boundary_count"] == 0
    assert result["local_max_per_minute"] == 0.0
    assert result["boundaries_per_minute"] == 0.0
    # Undefined ratios stay missing instead of manufacturing evidence of
    # perfect or zero retention from a zero denominator.
    assert result["boundary_local_retention"] is None
    assert result["height_local_retention"] is None
    assert result["distance_height_retention"] is None
    assert result["prominence_distance_retention"] is None
    assert result["peak_mean"] is None
    assert result["mean_final_prominence"] is None
    assert result["median_final_width_samples"] is None
    assert result["median_final_spacing_samples"] is None


def test_top_c_is_exact_and_uses_only_strict_local_maxima() -> None:
    saliency = np.asarray([0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0])
    local_peaks = find_peaks(saliency)[0]
    candidate_mask = np.zeros(saliency.size, dtype=bool)
    candidate_mask[local_peaks] = True

    selected = select_top_nms_indices(
        saliency,
        count=3,
        min_distance=2,
        valid_mask=candidate_mask,
    )

    np.testing.assert_array_equal(local_peaks, [1, 3, 5])
    np.testing.assert_array_equal(selected, [1, 3, 5])
    assert selected.size == 3
    assert set(selected).issubset(set(local_peaks))
    assert np.all(np.diff(selected) >= 2)

    # Without the local-maximum mask the exact solver can use a flank to make
    # an otherwise infeasible quota.  The policy must never silently change
    # candidate universes in this case.
    counterexample = np.asarray([0.0, 4.0, 10.0, 1.0, 4.0, 0.0])
    unrestricted = select_top_nms_indices(counterexample, count=2, min_distance=3)
    np.testing.assert_array_equal(unrestricted, [1, 4])
    counterexample_mask = np.zeros(counterexample.size, dtype=bool)
    counterexample_mask[find_peaks(counterexample)[0]] = True
    with pytest.raises(ValueError, match="cannot select 2 boundaries"):
        select_top_nms_indices(
            counterexample,
            count=2,
            min_distance=3,
            valid_mask=counterexample_mask,
        )


def test_nested_frontier_prefixes_are_not_independent_top_b_optima() -> None:
    values = np.asarray([0.0, 4.0, 10.0, 1.0, 4.0, 0.0])

    independent_b1 = select_top_nms_indices(values, count=1, min_distance=3)
    independent_b2 = select_top_nms_indices(values, count=2, min_distance=3)
    frontier = select_nested_nms_frontier(values, max_count=2, min_distance=3)

    np.testing.assert_array_equal(independent_b1, [2])
    np.testing.assert_array_equal(independent_b2, [1, 4])
    np.testing.assert_array_equal(frontier, [1, 4])
    np.testing.assert_array_equal(frontier[:1], [1])
    assert set(frontier[:1]) < set(frontier[:2])
    assert not np.array_equal(frontier[:1], independent_b1)
    assert np.all(np.diff(np.sort(frontier[:2])) >= 3)


def _qv_record() -> OriginSignalRecord:
    timestamps = tuple(float(value) for value in range(8))
    return OriginSignalRecord(
        dataset="qvhighlights",
        video_id="video-1",
        question_id="query-1",
        origin_id=0,
        origin_sec=0.0,
        timestamps_sec=timestamps,
        actual_pts_sec=timestamps,
        source_frame_indices=tuple(range(8)),
        relevance_scores=tuple(np.linspace(0.0, 1.0, 8)),
        metadata={
            "query_metadata": {
                "relevant_windows_sec": [[2.0, 4.0], [6.0, 8.0]],
                "relevant_clip_ids": [1, 3],
                "saliency_votes": [[4, 4, 4], [2, 2, 2]],
            }
        },
    )


def test_qvhighlights_selection_fidelity_uses_half_open_windows_and_clip_centers() -> (
    None
):
    result = qvhighlights_selection_fidelity(_qv_record(), [2, 4, 6])

    assert result is not None
    assert result == pytest.approx(
        {
            "selected_count": 3,
            "selected_relevant_count": 2,
            "selected_relevant_fraction": 2.0 / 3.0,
            "relevant_window_count": 2,
            "relevant_window_recall": 1.0,
            "relevant_clip_count": 2,
            "relevant_clip_recall": 1.0,
            "mean_selected_saliency_vote": 2.0,
            "mean_gt_clip_nearest_selected_sec": 1.0,
        }
    )


def _summary_rows() -> list[dict[str, object]]:
    item_effects = (("v1", "q1", 1.0), ("v1", "q2", 1.0), ("v2", "q1", 3.0))
    rows: list[dict[str, object]] = []
    for method in ("dwt", "swt"):
        for video_id, question_id, treatment_value in item_effects:
            rows.append(
                {
                    "dataset": "demo",
                    "video_id": video_id,
                    "question_id": question_id,
                    "origin_id": 0,
                    "base_method": method,
                    "policy_id": "adaptive",
                    "score": 0.0 if method == "dwt" else treatment_value,
                }
            )
    return rows


def test_policy_summary_bootstraps_whole_video_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bootstrap_with_samples(
        cluster_ids: Sequence[object],
        baseline: Sequence[float],
        treatment: Sequence[float],
        **kwargs: object,
    ) -> dict[str, object]:
        return video_cluster_paired_bootstrap(
            cluster_ids,
            baseline,
            treatment,
            statistic=kwargs["statistic"],  # type: ignore[arg-type]
            n_bootstrap=int(kwargs["n_bootstrap"]),
            confidence=float(kwargs["confidence"]),
            seed=int(kwargs["seed"]),
            return_samples=True,
        )

    monkeypatch.setattr(
        policy_module,
        "video_cluster_paired_bootstrap",
        bootstrap_with_samples,
    )
    summary = summarize_policy_rows(
        _summary_rows(),
        ("score",),
        n_bootstrap=200,
        seed=123,
    )

    comparison = summary["method_contrasts"]["adaptive"]
    assert comparison["effect_order"] == ["score"]
    assert comparison["estimate"] == pytest.approx([5.0 / 3.0])
    assert comparison["n_clusters"] == 2
    samples = np.asarray(comparison["samples"], dtype=float).reshape(-1)
    assert set(np.round(samples, 12)).issubset({1.0, round(5.0 / 3.0, 12), 3.0})


def test_policy_summary_jointly_bootstraps_method_policy_interaction() -> None:
    rows = []
    for video_id in ("v1", "v2"):
        for method, policy, value in (
            ("dwt", "adaptive", 0.0),
            ("swt", "adaptive", 1.0),
            ("dwt", "topc", 0.0),
            ("swt", "topc", 3.0),
        ):
            rows.append(
                {
                    "dataset": "demo",
                    "video_id": video_id,
                    "question_id": "q1",
                    "origin_id": 0,
                    "base_method": method,
                    "policy_id": policy,
                    "score": value,
                }
            )

    summary = summarize_policy_rows(rows, ("score",), n_bootstrap=20, seed=4)
    interaction = summary["method_policy_interactions"]["topc-minus-adaptive"]

    assert interaction["effect_order"] == ["score"]
    assert interaction["estimate"] == pytest.approx([2.0])
    assert interaction["ci_low"] == pytest.approx([2.0])
    assert interaction["ci_high"] == pytest.approx([2.0])
    assert interaction["n_clusters"] == 2


def test_policy_experiment_preserves_topc_count_and_nested_prefixes(
    tmp_path: Path,
) -> None:
    length = 32
    records = []
    for origin_id, offset in enumerate((0.1, 0.6)):
        timestamps = offset + np.arange(length, dtype=float)
        scores = 0.5 + 0.3 * np.sin(np.arange(length) * 0.73 + offset)
        scores += 0.15 * np.sin(np.arange(length) * 1.91 - offset)
        feature_path = tmp_path / f"features-{origin_id}.npy"
        np.save(
            feature_path,
            np.stack((scores, np.square(scores), np.cos(scores)), axis=1),
        )
        records.append(
            OriginSignalRecord(
                dataset="qvhighlights",
                video_id="video-1",
                question_id="query-1",
                origin_id=origin_id,
                origin_sec=offset,
                timestamps_sec=tuple(timestamps),
                actual_pts_sec=tuple(timestamps),
                source_frame_indices=tuple(range(length)),
                relevance_scores=tuple(scores),
                visual_features_path=str(feature_path),
                metadata={
                    "query_metadata": {
                        "relevant_windows_sec": [[2.0, 6.0]],
                        "relevant_clip_ids": [1, 2],
                        "saliency_votes": [[4, 4, 3], [3, 2, 2]],
                    }
                },
            )
        )

    result = run_policy_separation_experiment(
        records,
        tmp_path / "output",
        b_values=(1, 2),
        config=ExperimentConfig(
            wavelet="haar",
            level=2,
            frame_budget=4,
            min_distance_ratio=0.0,
            min_distance_absolute=2,
        ),
        n_bootstrap=10,
        seed=7,
    )

    assert len(result["trace_rows"]) == 16
    assert len(result["peak_rows"]) == 16
    assert len(result["fidelity_rows"]) == 16
    by_arm = {
        (row["origin_id"], row["base_method"], row["policy_id"]): row
        for row in result["trace_rows"]
    }
    for origin_id in (0, 1):
        for method in ("dwt", "swt"):
            adaptive = by_arm[(origin_id, method, "adaptive")]
            topc = by_arm[(origin_id, method, "topc")]
            nested_1 = by_arm[(origin_id, method, "nested_b01")]
            nested_2 = by_arm[(origin_id, method, "nested_b02")]
            assert len(topc["peaks"]) == len(adaptive["peaks"])
            assert topc["boundary_policy"]["candidate_set"] == "strict_local_maxima"
            assert len(nested_1["peaks"]) == 1
            assert len(nested_2["peaks"]) == 2
            assert set(nested_1["peaks"]) < set(nested_2["peaks"])
