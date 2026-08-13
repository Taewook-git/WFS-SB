from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from phase_stable.phasefuse import (
    PhaseFuse,
    PhaseFuseConfig,
    align_phase_values,
    allocate_exact_budget,
    interleaved_phase_ids,
    robust_phase_consensus,
    run_phasefuse,
    select_mmr_indices,
    select_weighted_interval_indices,
)
from phase_stable.transforms import SWTTransform, TransformConfig


class IdentityTransform:
    def transform(self, signal):
        values = np.asarray(signal, dtype=float)
        return SimpleNamespace(method="identity", saliency=np.abs(values))


class BadTransform:
    def __init__(self, saliency):
        self.saliency = saliency

    def transform(self, signal):
        return SimpleNamespace(method="bad", saliency=self.saliency)


def _run_flat(
    *,
    num_samples: int = 20,
    config: PhaseFuseConfig | None = None,
    **kwargs,
):
    timestamps = np.arange(num_samples, dtype=float)
    relevance = np.ones(num_samples, dtype=float)
    features = np.column_stack((timestamps, np.ones(num_samples)))
    return run_phasefuse(
        timestamps,
        relevance,
        features,
        config=config or PhaseFuseConfig(num_phases=4, frame_budget=8),
        transform=IdentityTransform(),
        **kwargs,
    )


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="num_phases"):
        PhaseFuseConfig(num_phases=0)
    with pytest.raises(ValueError, match="mmr_lambda"):
        PhaseFuseConfig(mmr_lambda=1.1)
    with pytest.raises(ValueError, match="allocation_temperature"):
        PhaseFuseConfig(allocation_temperature=0.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        PhaseFuseConfig(frame_budget=3, min_frames_per_segment=4)
    with pytest.raises(ValueError, match="uniform_reserve"):
        PhaseFuseConfig(frame_budget=3, uniform_reserve=4)
    with pytest.raises(ValueError, match="selection_strategy"):
        PhaseFuseConfig(selection_strategy="boundaries")
    with pytest.raises(ValueError, match="component_scaling"):
        PhaseFuseConfig(component_scaling="minmax")
    with pytest.raises(ValueError, match="min_selection_distance_sec"):
        PhaseFuseConfig(min_selection_distance_sec=-0.1)


def test_interleaved_phase_ids_handles_remainder():
    np.testing.assert_array_equal(
        interleaved_phase_ids(7, 3),
        np.asarray([0, 1, 2, 0, 1, 2, 0]),
    )


def test_absolute_time_alignment_does_not_extrapolate():
    aligned, support = align_phase_values(
        [0.0, 1.0, 2.0, 4.0, 7.0],
        [[0, 2, 4], [1, 3]],
        [[0.0, 2.0, 7.0], [10.0, 40.0]],
    )
    assert np.isnan(aligned[1, 0])
    assert np.isnan(aligned[1, 4])
    assert not support[1, 0]
    assert not support[1, 4]
    assert aligned[0, 3] == pytest.approx(4.0)
    assert aligned[1, 2] == pytest.approx(20.0)


def test_robust_consensus_limits_one_phase_outlier():
    values = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [100.0, 200.0, 300.0],
        ]
    )
    consensus, uncertainty = robust_phase_consensus(values)
    np.testing.assert_allclose(consensus, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(uncertainty, 0.0)


def test_weighted_interval_dp_beats_greedy_and_honors_ties():
    timestamps = np.arange(5, dtype=float)
    np.testing.assert_array_equal(
        select_weighted_interval_indices(
            timestamps,
            [0.0, 6.0, 10.0, 6.0, 0.0],
            2,
            0.0,
            min_index_distance=2,
        ),
        [1, 3],
    )
    # Equal total weight prefers fewer boundaries.
    np.testing.assert_array_equal(
        select_weighted_interval_indices(
            timestamps,
            [0.0, 1.0, 2.0, 1.0, 0.0],
            2,
            0.0,
            min_index_distance=2,
        ),
        [2],
    )
    # Equal singleton scores prefer the earlier dense index.
    np.testing.assert_array_equal(
        select_weighted_interval_indices(timestamps, [0.0, 1.0, 1.0, 0.0, 0.0], 1, 0.0),
        [1],
    )


def test_weighted_interval_does_not_mutate_valid_mask():
    mask = np.ones(5, dtype=bool)
    select_weighted_interval_indices(
        np.arange(5, dtype=float), np.ones(5), 1, 0.0, valid_mask=mask
    )
    assert np.all(mask)


def test_weighted_interval_jointly_optimizes_segment_capacity():
    selectable = np.zeros(12, dtype=bool)
    selectable[[0, 1, 4, 5, 8, 9]] = True
    weights = np.zeros(12)
    weights[[4, 5, 8]] = [98.0, 100.0, 99.0]
    selected = select_weighted_interval_indices(
        np.arange(12, dtype=float),
        weights,
        2,
        0.0,
        min_index_distance=2,
        segment_capacity_mask=selectable,
        min_segment_capacity=2,
    )
    np.testing.assert_array_equal(selected, [4, 8])


def test_allocate_exact_budget_is_capacity_safe_and_deterministic():
    allocation = allocate_exact_budget([2, 10, 3], [0.0, 2.0, 1.0], 10)
    assert allocation.tolist() == [1, 6, 3]
    assert int(np.sum(allocation)) == 10
    assert np.all(allocation <= np.asarray([2, 10, 3]))
    np.testing.assert_array_equal(allocate_exact_budget([3, 3], [0.0, 0.0], 3), [2, 1])
    np.testing.assert_array_equal(
        allocate_exact_budget([10, 10, 10], np.log([0.34, 0.33, 0.33]), 8),
        [3, 3, 2],
    )


@pytest.mark.parametrize(
    ("capacities", "budget"),
    [([1, 1], 1), ([1, 1], 3), ([0, 3], 2)],
)
def test_allocate_exact_budget_rejects_infeasible_requests(capacities, budget):
    with pytest.raises(ValueError):
        allocate_exact_budget(capacities, np.zeros(len(capacities)), budget)


def test_mmr_is_exact_deterministic_and_handles_zero_vectors():
    candidates = np.arange(6)
    relevance = np.ones(6)
    features = np.zeros((6, 3))
    timestamps = np.arange(6, dtype=float)
    first = select_mmr_indices(
        candidates,
        relevance,
        3,
        features=features,
        timestamps_sec=timestamps,
    )
    second = select_mmr_indices(
        candidates,
        relevance,
        3,
        features=features,
        timestamps_sec=timestamps,
    )
    np.testing.assert_array_equal(first, second)
    assert first.size == 3
    assert np.unique(first).size == 3
    assert np.all(np.diff(first) > 0)


def test_end_to_end_returns_exact_budget_and_strict_json_trace():
    trace = _run_flat()
    assert trace.selected_indices.size == 8
    np.testing.assert_array_equal(trace.selected_dense_indices, trace.selected_indices)
    assert np.unique(trace.selected_indices).size == 8
    assert trace.boundary_indices.size == 0
    assert trace.used_fallback
    assert trace.fallback_reason == "no_positive_boundary_evidence"
    assert int(np.sum(trace.allocation)) == 8
    json.dumps(trace.to_dict(include_arrays=True), allow_nan=False)


def test_invalid_target_edges_are_excluded_from_selection_domain():
    n = 24
    timestamps = np.arange(n, dtype=float)
    valid = np.zeros(n, dtype=bool)
    valid[4:20] = True
    baseline = np.linspace(0.0, 1.0, n)
    trace = run_phasefuse(
        timestamps,
        baseline,
        None,
        config=PhaseFuseConfig(num_phases=2, frame_budget=6),
        transform=IdentityTransform(),
        valid_mask=valid,
    )
    assert np.all(valid[trace.selected_indices])
    assert not np.any(trace.selectable_mask[~valid])
    np.testing.assert_array_equal(trace.fusion_domain_mask, valid)
    assert np.all(np.isnan(trace.normalized_aligned_phase_saliency[:, :4]))
    assert np.all(np.isnan(trace.normalized_aligned_phase_saliency[:, 20:]))


def test_valid_mask_is_target_domain_not_phase_source_crop():
    timestamps = np.arange(32, dtype=float) / 4.0
    valid = (timestamps >= 0.75) & (timestamps <= 7.0)
    trace = run_phasefuse(
        timestamps,
        np.sin(timestamps) ** 2,
        None,
        config=PhaseFuseConfig(num_phases=4, frame_budget=8),
        transform=IdentityTransform(),
        valid_mask=valid,
    )
    np.testing.assert_array_equal(trace.full_support_mask, valid)
    assert trace.phase_indices[0].tolist() == list(range(0, 32, 4))
    assert trace.phase_indices[3].tolist() == list(range(3, 32, 4))
    assert np.all(valid[trace.selected_indices])


def test_trace_json_accepts_numpy_scalar_config_values():
    trace = _run_flat(
        config=PhaseFuseConfig(
            num_phases=np.int64(2),
            frame_budget=np.int64(4),
        )
    )
    json.dumps(trace.to_dict(), allow_nan=False)


def test_transform_factory_is_called_once_per_phase():
    phases = []

    def factory(phase):
        phases.append(phase)
        return IdentityTransform()

    n = 16
    trace = run_phasefuse(
        np.arange(n, dtype=float),
        np.linspace(0.0, 1.0, n),
        None,
        config=PhaseFuseConfig(num_phases=4, frame_budget=4),
        transform_factory=factory,
    )
    assert phases == [0, 1, 2, 3]
    assert trace.transform_methods == ("identity",) * 4


def test_existing_ti_dwt_transform_is_pluggable():
    n = 32
    timestamps = np.arange(n, dtype=float) / 4.0
    relevance = np.sin(timestamps) ** 2
    trace = run_phasefuse(
        timestamps,
        relevance,
        None,
        config=PhaseFuseConfig(num_phases=4, frame_budget=8),
        transform=SWTTransform(TransformConfig(method="swt", level=1)),
    )
    assert trace.selected_indices.size == 8
    assert trace.transform_methods == ("swt",) * 4


def test_short_phase_falls_back_to_single_stream():
    trace = _run_flat(
        num_samples=4,
        config=PhaseFuseConfig(num_phases=4, frame_budget=2),
    )
    assert trace.used_fallback
    assert trace.fallback_reason == "insufficient_phase_samples:single_stream"
    assert len(trace.phase_indices) == 1
    np.testing.assert_array_equal(trace.phase_ids, 0)
    np.testing.assert_array_equal(trace.input_phase_ids, [0, 1, 2, 3])
    assert trace.selected_indices.size == 2


def test_two_sample_single_token_case_is_budget_safe():
    trace = _run_flat(
        num_samples=2,
        config=PhaseFuseConfig(num_phases=1, frame_budget=1),
    )
    assert trace.boundary_indices.size == 0
    assert trace.selected_indices.size == 1
    assert trace.fallback_reason == "boundary_budget_zero"


def test_source_frame_duplicates_use_earliest_valid_dense_representative():
    n = 12
    sources = np.asarray([0, 0, *range(1, 11)])
    valid = np.ones(n, dtype=bool)
    valid[0] = False
    trace = _run_flat(
        num_samples=n,
        config=PhaseFuseConfig(num_phases=1, frame_budget=11),
        source_frame_indices=sources,
        valid_mask=valid,
    )
    assert 0 not in trace.selected_indices
    assert 1 in trace.selected_indices
    assert trace.selected_source_frame_indices is not None
    assert np.unique(trace.selected_source_frame_indices).size == 11
    assert trace.to_dict()["selected_source_frame_indices"] == list(range(11))
    assert trace.to_dict()["num_input_valid_candidates"] == 11
    assert trace.to_dict()["num_selectable_candidates"] == 11


def test_duplicate_source_deduplication_does_not_suppress_boundary_evidence():
    relevance = np.zeros(10)
    relevance[1] = 10.0
    trace = run_phasefuse(
        np.arange(10, dtype=float),
        relevance,
        None,
        config=PhaseFuseConfig(
            num_phases=1,
            frame_budget=4,
            min_boundary_distance_sec=0.0,
            boundary_threshold_mad=0.0,
            uncertainty_penalty=0.0,
            relevance_weight=0.0,
            phase_vote_weight=0.0,
        ),
        transform=IdentityTransform(),
        source_frame_indices=[0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    assert trace.boundary_indices.tolist() == [1]
    assert 1 not in trace.selected_indices


def test_source_frame_deduplication_must_leave_enough_candidates():
    with pytest.raises(ValueError, match="deduplicated"):
        _run_flat(
            num_samples=8,
            config=PhaseFuseConfig(num_phases=1, frame_budget=5),
            source_frame_indices=[0, 0, 1, 1, 2, 2, 3, 3],
        )


@pytest.mark.parametrize(
    "bad_features",
    [np.zeros(10), np.zeros((9, 2)), np.full((10, 2), np.nan)],
)
def test_end_to_end_rejects_bad_features(bad_features):
    with pytest.raises(ValueError, match="features"):
        run_phasefuse(
            np.arange(10, dtype=float),
            np.ones(10),
            bad_features,
            config=PhaseFuseConfig(num_phases=2, frame_budget=4),
            transform=IdentityTransform(),
        )


@pytest.mark.parametrize(
    "bad_saliency",
    [np.asarray([1.0]), np.asarray([1.0, np.nan, 1.0, 1.0])],
)
def test_end_to_end_rejects_invalid_transform_output(bad_saliency):
    with pytest.raises(ValueError, match="transform saliency"):
        run_phasefuse(
            np.arange(8, dtype=float),
            np.ones(8),
            None,
            config=PhaseFuseConfig(num_phases=2, frame_budget=4),
            transform=BadTransform(bad_saliency),
        )


def test_phasefuse_wrapper_matches_function_and_repeats_deterministically():
    selector = PhaseFuse(
        IdentityTransform(), PhaseFuseConfig(num_phases=2, frame_budget=4)
    )
    timestamps = np.arange(12, dtype=float)
    relevance = np.sin(timestamps) ** 2
    first = selector.run(timestamps, relevance)
    second = selector.run(timestamps, relevance)
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    np.testing.assert_array_equal(first.boundary_indices, second.boundary_indices)


def test_default_segmented_selection_preserves_v1_fingerprint():
    timestamps = np.arange(24, dtype=float) * 0.5
    relevance = np.asarray(
        [
            0.0,
            0.2,
            0.1,
            0.4,
            0.3,
            0.8,
            0.2,
            0.1,
            0.9,
            0.4,
            0.2,
            0.7,
            0.1,
            0.3,
            0.6,
            0.2,
            0.8,
            0.1,
            0.5,
            0.2,
            0.9,
            0.1,
            0.4,
            0.2,
        ]
    )
    features = np.column_stack((np.sin(timestamps), np.cos(timestamps)))
    trace = run_phasefuse(
        timestamps,
        relevance,
        features,
        config=PhaseFuseConfig(
            num_phases=2,
            frame_budget=6,
            min_boundary_distance_sec=1.0,
        ),
        transform=IdentityTransform(),
    )
    assert trace.config.selection_strategy == "segmented"
    assert trace.boundary_indices.tolist() == [5, 8, 11, 16, 20]
    assert trace.selected_indices.tolist() == [3, 5, 8, 11, 16, 20]
    assert trace.anchor_indices.size == 0


def test_percentile_component_scaling_is_bounded_and_outlier_magnitude_invariant():
    timestamps = np.arange(20, dtype=float)
    base = np.linspace(0.0, 1.0, 20)
    extreme = base.copy()
    extreme[-1] = 1_000_000.0
    config = PhaseFuseConfig(
        num_phases=1,
        frame_budget=6,
        selection_strategy="global_coverage",
        component_scaling="percentile",
    )
    first = run_phasefuse(
        timestamps, base, None, config=config, transform=IdentityTransform()
    )
    second = run_phasefuse(
        timestamps, extreme, None, config=config, transform=IdentityTransform()
    )
    assert np.all((first.normalized_relevance >= 0) & (first.normalized_relevance <= 1))
    assert np.all(
        (second.normalized_relevance >= 0) & (second.normalized_relevance <= 1)
    )
    np.testing.assert_allclose(first.normalized_relevance, second.normalized_relevance)
    np.testing.assert_allclose(
        first.normalized_aligned_phase_saliency,
        second.normalized_aligned_phase_saliency,
    )
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)


def test_global_coverage_reserves_exact_uniform_anchors_without_boundaries():
    trace = _run_flat(
        num_samples=32,
        config=PhaseFuseConfig(
            num_phases=1,
            frame_budget=8,
            selection_strategy="global_coverage",
            uniform_reserve=4,
            component_scaling="percentile",
        ),
    )
    assert trace.anchor_indices.tolist() == [4, 12, 19, 27]
    assert set(trace.anchor_indices).issubset(set(trace.selected_indices))
    assert trace.anchor_indices.size == 4
    assert trace.boundary_indices.size == 0
    assert not np.any(trace.boundary_candidate_mask)
    assert trace.segments == ((0, 32),)
    assert trace.allocation.tolist() == [8]
    assert trace.to_dict()["anchor_indices"] == [4, 12, 19, 27]


def test_global_mmr_is_seeded_by_anchors_and_avoids_near_cosine_duplicate():
    timestamps = np.arange(12, dtype=float)
    relevance = np.zeros(12)
    relevance[[2, 3, 8]] = [1.0, 1.0, 0.9]
    features = np.zeros((12, 3))
    features[:, 2] = 1.0
    features[2] = [1.0, 0.0, 0.0]
    features[3] = [0.999, 0.001, 0.0]
    features[8] = [0.0, 1.0, 0.0]
    trace = run_phasefuse(
        timestamps,
        relevance,
        features,
        config=PhaseFuseConfig(
            num_phases=1,
            frame_budget=4,
            selection_strategy="global_coverage",
            uniform_reserve=2,
            component_scaling="percentile",
            selection_event_weight=0.0,
            mmr_lambda=0.25,
            mmr_visual_weight=1.0,
            min_selection_distance_sec=1.5,
        ),
        transform=IdentityTransform(),
    )
    assert set(trace.anchor_indices).issubset(trace.selected_indices)
    assert not ({2, 3} <= set(trace.selected_indices))
    assert np.min(np.diff(trace.selected_timestamps_sec)) >= 1.5


def test_global_minimum_distance_has_deterministic_relaxation_fallback():
    config = PhaseFuseConfig(
        num_phases=1,
        frame_budget=5,
        selection_strategy="global_coverage",
        uniform_reserve=2,
        min_selection_distance_sec=100.0,
    )
    first = _run_flat(num_samples=10, config=config)
    second = _run_flat(num_samples=10, config=config)
    assert first.selected_indices.size == 5
    assert first.fallback_reason == "min_selection_distance_relaxed"
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)


def test_global_coverage_is_tolerant_to_subframe_outer_origin_shift():
    config = PhaseFuseConfig(
        num_phases=4,
        frame_budget=10,
        selection_strategy="global_coverage",
        uniform_reserve=5,
        component_scaling="percentile",
        selection_event_weight=0.25,
        uncertainty_penalty=0.0,
        phase_vote_weight=0.0,
        min_selection_distance_sec=0.5,
    )

    def select(offset):
        timestamps = offset + np.arange(80, dtype=float) * 0.25
        relevance = (
            0.2
            + 0.8 * np.exp(-np.square((timestamps - 5.0) / 0.8))
            + 0.6 * np.exp(-np.square((timestamps - 13.0) / 1.2))
            + 0.1 * np.square(np.sin(timestamps * 0.7))
        )
        features = np.column_stack(
            (np.sin(timestamps * 0.4), np.cos(timestamps * 0.4), relevance)
        )
        return run_phasefuse(
            timestamps,
            relevance,
            features,
            config=config,
            transform=IdentityTransform(),
        )

    baseline = select(0.0)
    shifted = select(0.125)
    baseline_to_shifted = [
        np.min(np.abs(value - shifted.selected_timestamps_sec))
        for value in baseline.selected_timestamps_sec
    ]
    shifted_to_baseline = [
        np.min(np.abs(value - baseline.selected_timestamps_sec))
        for value in shifted.selected_timestamps_sec
    ]
    assert max((*baseline_to_shifted, *shifted_to_baseline)) <= 0.25


def test_global_coverage_is_invariant_to_phase_label_permutation():
    timestamps = np.arange(40, dtype=float) * 0.25
    relevance = 0.3 + np.square(np.sin(timestamps))
    phase_ids = interleaved_phase_ids(timestamps.size, 4)
    permutation = np.asarray([2, 0, 3, 1])
    config = PhaseFuseConfig(
        num_phases=4,
        frame_budget=8,
        selection_strategy="global_coverage",
        uniform_reserve=4,
        component_scaling="percentile",
        uncertainty_penalty=0.0,
    )
    original = run_phasefuse(
        timestamps,
        relevance,
        None,
        config=config,
        transform=IdentityTransform(),
        phase_ids=phase_ids,
    )
    relabeled = run_phasefuse(
        timestamps,
        relevance,
        None,
        config=config,
        transform=IdentityTransform(),
        phase_ids=permutation[phase_ids],
    )
    np.testing.assert_allclose(original.consensus, relabeled.consensus)
    np.testing.assert_allclose(original.uncertainty, relabeled.uncertainty)
    np.testing.assert_array_equal(original.anchor_indices, relabeled.anchor_indices)
    np.testing.assert_array_equal(original.selected_indices, relabeled.selected_indices)


def test_argument_validation_for_phase_and_source_maps():
    common = {
        "timestamps_sec": np.arange(8, dtype=float),
        "relevance_scores": np.ones(8),
        "features": None,
        "config": PhaseFuseConfig(num_phases=2, frame_budget=3),
        "transform": IdentityTransform(),
    }
    with pytest.raises(ValueError, match="every configured phase"):
        run_phasefuse(**common, phase_ids=np.zeros(8, dtype=int))
    with pytest.raises(ValueError, match="integer vector"):
        run_phasefuse(**common, source_frame_indices=np.arange(8, dtype=float))
    with pytest.raises(ValueError, match="non-negative"):
        run_phasefuse(**common, source_frame_indices=[-1, *range(7)])
    with pytest.raises(ValueError, match="non-decreasing"):
        run_phasefuse(**common, source_frame_indices=[0, 1, 2, 4, 3, 5, 6, 7])
    with pytest.raises(ValueError, match="exactly one"):
        run_phasefuse(**common, transform_factory=lambda _: IdentityTransform())
