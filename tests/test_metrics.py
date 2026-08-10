import numpy as np
import pytest

from phase_stable.metrics import (
    adjusted_rand_index,
    boundaries_to_segment_labels,
    boundary_count_cv,
    js_divergence,
    make_time_grid,
    match_timestamps,
    mllm_stability_metrics,
    normalized_l1_distance,
    pairwise_cosine_consistency,
    pairwise_normalized_l1,
    scale_energy_drift,
    scale_energy_proportions,
    segmentation_consistency,
    selected_timestamp_metrics,
    tolerant_boundary_metrics,
    variation_of_information,
    video_cluster_paired_bootstrap,
)


def test_all_pair_cosine_mean_worst_and_validation():
    values = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = pairwise_cosine_consistency(values)
    assert result == pytest.approx({"mean": 1.0 / 3.0, "worst": 0.0, "n_pairs": 3})

    with pytest.raises(ValueError, match="zero-vector"):
        pairwise_cosine_consistency([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="NaN"):
        pairwise_cosine_consistency([[1.0, 0.0], [np.nan, 1.0]])
    with pytest.raises(ValueError, match="at least two"):
        pairwise_cosine_consistency([[1.0, 2.0]])


def test_normalized_l1_and_pairwise_summary():
    assert normalized_l1_distance([0, 0], [0, 0]) == 0.0
    assert normalized_l1_distance([1, 0], [0, 1]) == 1.0
    assert normalized_l1_distance([1, 1], [2, 0]) == pytest.approx(0.5)
    summary = pairwise_normalized_l1([[1, 0], [0, 1], [1, 0]])
    assert summary == pytest.approx({"mean": 2 / 3, "worst": 1.0, "n_pairs": 3})
    with pytest.raises(ValueError, match="same shape"):
        normalized_l1_distance([1], [1, 2])


def test_js_divergence_known_cases_and_invalid_mass():
    assert js_divergence([1, 0], [1, 0]) == pytest.approx(0.0)
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-negative"):
        js_divergence([1, -1], [0, 1])
    with pytest.raises(ValueError, match="positive total mass"):
        js_divergence([0, 0], [1, 0])


def test_scale_energy_proportion_and_drift():
    representation = np.array([[1.0, 1.0], [2.0, 0.0]])
    proportions = scale_energy_proportions(representation)
    assert proportions == pytest.approx([1 / 3, 2 / 3])

    same = scale_energy_drift(representation, representation)
    assert same == pytest.approx({"l1": 0.0, "js_divergence": 0.0, "js_distance": 0.0})
    orthogonal = scale_energy_drift([1, 0], [0, 1], inputs_are_proportions=True)
    assert orthogonal == pytest.approx(
        {"l1": 2.0, "js_divergence": 1.0, "js_distance": 1.0}
    )
    with pytest.raises(ValueError, match="all-zero"):
        scale_energy_proportions(np.zeros((2, 4)))


def test_timestamp_matching_maximizes_cardinality_before_distance():
    # Nearest-first can consume 1.0 -> 1.1 and leave 2.0 unmatched.  The
    # maximum-cardinality solution is 0.0 -> 1.1 and 1.0 -> 2.0.
    result = match_timestamps([0.0, 1.0], [1.1, 2.0], tolerance=1.1)
    assert result["count"] == 2
    assert result["first_indices"].tolist() == [0, 1]
    assert result["second_indices"].tolist() == [0, 1]
    assert result["distances"] == pytest.approx([1.1, 1.0])


def test_tolerant_boundary_metrics_and_empty_convention():
    result = tolerant_boundary_metrics(
        [1.0, 2.0, 4.0], [1.1, 3.9], tolerance=0.2
    )
    assert result["matched"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(0.8)
    assert result["mean_distance"] == pytest.approx(0.1)

    both_empty = tolerant_boundary_metrics([], [], tolerance=1.0)
    assert (both_empty["precision"], both_empty["recall"], both_empty["f1"]) == (
        1.0,
        1.0,
        1.0,
    )
    one_empty = tolerant_boundary_metrics([], [1.0], tolerance=1.0)
    assert (one_empty["precision"], one_empty["recall"], one_empty["f1"]) == (
        0.0,
        0.0,
        0.0,
    )
    assert one_empty["mean_distance"] is None
    with pytest.raises(ValueError, match="sorted"):
        tolerant_boundary_metrics([2, 1], [1], tolerance=1)


def test_boundary_count_cv_handles_stable_zero_but_rejects_bad_inputs():
    assert boundary_count_cv([2, 2, 2]) == 0.0
    assert boundary_count_cv([0, 0, 0]) == 0.0
    assert boundary_count_cv([1, 3]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="non-negative"):
        boundary_count_cv([1, -1])
    with pytest.raises(ValueError, match="empty"):
        boundary_count_cv([])


def test_segment_grid_labels_ari_and_vi_without_sklearn():
    grid = make_time_grid(0.0, 4.0, 1.0)
    labels = boundaries_to_segment_labels([2.0], grid)
    assert grid.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert labels.tolist() == [0, 0, 1, 1, 1]
    relabeled = np.array([7, 7, 9, 9, 9])
    assert adjusted_rand_index(labels, relabeled) == pytest.approx(1.0)
    assert variation_of_information(labels, relabeled) == pytest.approx(0.0)

    single_cluster = np.zeros(4, dtype=int)
    two_clusters = np.array([0, 0, 1, 1])
    assert adjusted_rand_index(single_cluster, two_clusters) == pytest.approx(0.0)
    assert variation_of_information(single_cluster, two_clusters) == pytest.approx(1.0)
    consistency = segmentation_consistency([2.0], [2.0], grid)
    assert consistency == pytest.approx({"ari": 1.0, "vi": 0.0})

    with pytest.raises(ValueError, match="grid must not be empty"):
        boundaries_to_segment_labels([], [])
    with pytest.raises(ValueError, match="within"):
        boundaries_to_segment_labels([5], grid)
    with pytest.raises(ValueError, match="NaN"):
        adjusted_rand_index([0, np.nan], [0, 1])


def test_make_time_grid_appends_noninteger_endpoint():
    assert make_time_grid(0, 1, 0.4).tolist() == pytest.approx([0, 0.4, 0.8, 1.0])
    assert make_time_grid(0, 1, 0.5, include_endpoint=False).tolist() == [0.0, 0.5]
    with pytest.raises(ValueError, match="positive"):
        make_time_grid(0, 1, 0)


def test_selected_timestamp_metrics_with_optimal_embedding_match():
    result = selected_timestamp_metrics(
        [0.0, 3.0],
        [0.2, 2.8],
        tolerance=0.5,
        first_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        second_embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    assert result["f1"] == 1.0
    assert result["optimal_mean_distance"] == pytest.approx(0.2)
    assert result["matched_embedding_cosine"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="supplied together"):
        selected_timestamp_metrics([0], [0], first_embeddings=[[1, 0]])


def test_mllm_stability_metrics_exact_values_and_constant_answers():
    predictions = np.array(
        [
            ["A", "A", "A"],
            ["B", "C", "B"],
            ["D", "D", "C"],
        ]
    )
    result = mllm_stability_metrics(predictions, ["A", "B", "D"])
    assert result["mean_accuracy"] == pytest.approx(7 / 9)
    assert result["robust_accuracy"] == pytest.approx(1 / 3)
    assert result["worst_origin_accuracy"] == pytest.approx(2 / 3)
    assert result["accuracy_std"] == pytest.approx(np.std([1, 2 / 3, 2 / 3]))
    assert result["answer_agreement"] == pytest.approx(5 / 9)
    assert result["pairwise_answer_disagreement"] == pytest.approx(4 / 9)

    constant = mllm_stability_metrics([["A", "A"], ["A", "A"]], ["A", "B"])
    assert constant["answer_agreement"] == 1.0
    assert constant["pairwise_answer_disagreement"] == 0.0
    assert constant["mean_accuracy"] == 0.5
    with pytest.raises(ValueError, match="missing"):
        mllm_stability_metrics([["A", None]], ["A"])


def test_video_cluster_paired_bootstrap_resamples_whole_video_clusters():
    # Cluster v1 has two questions and v2 one. Treatment-baseline effects are
    # [1, 1] for v1 and [3] for v2; cluster resamples can only produce estimates
    # 1, 5/3, or 3, never a row-level mixture such as 7/3.
    result = video_cluster_paired_bootstrap(
        ["v1", "v1", "v2"],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 3.0],
        n_bootstrap=200,
        seed=123,
        return_samples=True,
    )
    assert result["estimate"] == pytest.approx(5 / 3)
    assert result["n_clusters"] == 2
    assert set(np.round(result["samples"], 12)).issubset({1.0, round(5 / 3, 12), 3.0})


def test_video_cluster_bootstrap_supports_generic_vector_statistic():
    def effects(baseline, treatment):
        difference = treatment - baseline
        return np.array([np.mean(difference), np.max(difference)])

    result = video_cluster_paired_bootstrap(
        ["v1", "v2"],
        [0.0, 2.0],
        [1.0, 4.0],
        statistic=effects,
        n_bootstrap=20,
        seed=1,
    )
    assert result["estimate"] == pytest.approx([1.5, 2.0])
    assert np.asarray(result["ci_low"]).shape == (2,)
    with pytest.raises(ValueError, match="NaN"):
        video_cluster_paired_bootstrap(
            ["v1", "v2"], [0.0, np.nan], [1.0, 2.0], n_bootstrap=2
        )
