import numpy as np
import pytest

from phase_stable.canonical_residual import (
    RC12Config,
    canonical_lattice,
    percentile_rank,
    select_canonical_uniform_targets,
    select_rc12_targets,
)


def test_percentile_rank_is_tie_aware_and_constant_safe():
    assert percentile_rank([2.0, 1.0, 1.0, 4.0]).tolist() == pytest.approx(
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0]
    )
    assert percentile_rank([5.0, 5.0]).tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("scout_step", "expected_decision_step"),
    [(0.25, 0.5), (1.0, 1.0)],
)
def test_lattice_never_upsamples_scout_and_is_capped_at_two_hz(
    scout_step, expected_decision_step
):
    scout = np.arange(0.0, 30.0 + scout_step / 2.0, scout_step)
    lattice, measured_scout_step, decision_step = canonical_lattice(
        scout,
        support_start_sec=1.1,
        support_stop_sec=28.9,
        config=RC12Config(),
    )
    assert measured_scout_step == pytest.approx(scout_step)
    assert decision_step == pytest.approx(expected_decision_step)
    assert lattice[0] % decision_step == pytest.approx(0.0)
    assert np.allclose(np.diff(lattice), decision_step)


def test_rc12_returns_twelve_anchors_four_residuals_and_exact_budget():
    scout = np.arange(0.0, 40.0, 0.25)
    relevance = np.exp(-0.5 * np.square((scout - 10.0) / 2.0))
    decision = select_rc12_targets(
        scout,
        relevance,
        support_start_sec=1.0,
        support_stop_sec=38.0,
    )
    assert decision.anchor_indices.size == 12
    assert decision.residual_indices.size == 4
    assert decision.selected_indices.size == 16
    assert set(decision.anchor_indices) <= set(decision.selected_indices)
    assert np.all(np.diff(decision.target_timestamps_sec) > 0.0)


def test_residual_ties_prefer_coverage_then_earlier_time():
    config = RC12Config(
        frame_budget=4,
        anchor_count=2,
        max_lattice_hz=1.0,
        smoothing_sigma_sec=1.0,
        score_quantum=1.0,
        min_residual_distance_sec=1.0,
        coverage_tiebreak_cap_sec=100.0,
    )
    scout = np.arange(0.0, 10.0, 1.0)
    decision = select_rc12_targets(
        scout,
        np.ones(scout.size),
        support_start_sec=0.0,
        support_stop_sec=9.0,
        config=config,
    )
    assert decision.anchor_indices.tolist() == [2, 7]
    assert decision.residual_indices.tolist() == [0, 4]
    assert decision.selected_indices.tolist() == [0, 2, 4, 7]


def test_minimum_distance_relaxes_only_when_exact_budget_is_infeasible():
    config = RC12Config(
        frame_budget=5,
        anchor_count=4,
        max_lattice_hz=1.0,
        smoothing_sigma_sec=1.0,
        score_quantum=0.1,
        min_residual_distance_sec=10.0,
    )
    scout = np.arange(0.0, 8.0, 1.0)
    decision = select_rc12_targets(
        scout,
        np.linspace(0.0, 1.0, scout.size),
        support_start_sec=0.0,
        support_stop_sec=7.0,
        config=config,
    )
    assert decision.selected_indices.size == 5
    assert decision.distance_relaxation_used


def test_decisions_are_canonical_targets_not_scout_frame_indices():
    base_times = np.arange(0.0, 30.0, 0.25)
    shifted_times = base_times + 0.1
    relevance = np.sin(base_times / 5.0) + 1.0
    first = select_rc12_targets(
        base_times,
        relevance,
        support_start_sec=1.0,
        support_stop_sec=28.0,
    )
    second = select_rc12_targets(
        shifted_times,
        relevance,
        support_start_sec=1.0,
        support_stop_sec=28.0,
    )
    assert np.all(first.target_timestamps_sec % 0.5 == 0.0)
    assert np.all(second.target_timestamps_sec % 0.5 == 0.0)
    assert not np.shares_memory(first.target_timestamps_sec, base_times)
    assert not np.shares_memory(second.target_timestamps_sec, shifted_times)


def test_canonical_uniform_returns_full_lattice_and_exact_target_indices():
    scout = np.arange(0.0, 40.0, 0.25)
    lattice, selected = select_canonical_uniform_targets(
        scout,
        support_start_sec=1.0,
        support_stop_sec=38.0,
    )
    assert selected.shape == (16,)
    assert np.all(np.diff(selected) > 0)
    assert lattice.size > selected.size
    assert np.all(lattice % 0.5 == 0.0)
