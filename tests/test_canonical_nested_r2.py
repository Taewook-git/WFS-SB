from __future__ import annotations

import numpy as np
import pytest

from phase_stable.canonical_nested_r2 import (
    NestedR2Config,
    _Proposal,
    _choose_relocations,
    select_nested_r2_targets,
)
from phase_stable.canonical_residual import (
    RC12Config,
    select_canonical_uniform_targets,
    select_rc12_targets,
)
from scripts.decode_rc12_exact import _arm
from scripts.evaluate_canonical_gate import evaluate_gate


def _scout(stop: float = 320.0) -> np.ndarray:
    return np.arange(0.0, stop + 0.125, 0.25)


def test_constant_relevance_is_exact_canonical_uniform_noop():
    scout = _scout()
    lattice, uniform = select_canonical_uniform_targets(
        scout, support_start_sec=0.0, support_stop_sec=320.0
    )
    decision = select_nested_r2_targets(
        scout,
        np.ones_like(scout),
        support_start_sec=0.0,
        support_stop_sec=320.0,
    )

    assert np.array_equal(decision.lattice_timestamps_sec, lattice)
    assert np.array_equal(decision.selected_indices, uniform)
    assert decision.relocations == ()
    assert decision.preserved_base_indices.size == 16
    assert decision.residual_indices.size == 0


def test_two_safe_peaks_relocate_only_inside_nonadjacent_uniform_regions():
    scout = _scout()
    lattice, uniform = select_canonical_uniform_targets(
        scout, support_start_sec=0.0, support_stop_sec=320.0
    )
    peak_times = lattice[uniform[[3, 11]]] + 4.0
    relevance = sum(
        np.exp(-0.5 * np.square((scout - peak) / 1.0)) for peak in peak_times
    )
    decision = select_nested_r2_targets(
        scout,
        relevance,
        support_start_sec=0.0,
        support_stop_sec=320.0,
    )

    assert len(decision.relocations) == 2
    assert decision.preserved_base_indices.size == 14
    assert len(set(uniform) & set(decision.selected_indices)) == 14
    donor_slots = [item.donor_slot for item in decision.relocations]
    assert all(slot not in {0, 15} for slot in donor_slots)
    assert abs(donor_slots[0] - donor_slots[1]) > 1
    for item in decision.relocations:
        assert decision.region_owner_slots[item.residual_index] == item.donor_slot
        assert item.quantized_gain > 0.0
        assert (
            abs(
                decision.lattice_timestamps_sec[item.residual_index]
                - decision.lattice_timestamps_sec[item.donor_index]
            )
            >= 2.0
        )


def test_nested_r2_reuses_rc12_lattice_and_score_arrays_exactly():
    scout = _scout(80.0)
    relevance = np.sin(scout / 5.0) + np.cos(scout / 11.0)
    nested = select_nested_r2_targets(
        scout,
        relevance,
        support_start_sec=1.0,
        support_stop_sec=79.0,
    )
    rc12 = select_rc12_targets(
        scout,
        relevance,
        support_start_sec=1.0,
        support_stop_sec=79.0,
        config=RC12Config(),
    )

    for name in (
        "lattice_timestamps_sec",
        "interpolated_relevance",
        "percentile_relevance",
        "smoothed_relevance",
        "quantized_scores",
    ):
        assert np.array_equal(getattr(nested, name), getattr(rc12, name))


def test_random_signals_preserve_budget_regions_edges_and_fourteen_slot_floor():
    scout = _scout(160.0)
    for seed in range(40):
        relevance = np.random.default_rng(seed).normal(size=scout.size)
        decision = select_nested_r2_targets(
            scout,
            relevance,
            support_start_sec=0.0,
            support_stop_sec=160.0,
        )
        base = decision.base_uniform_indices
        selected = decision.selected_indices
        assert selected.size == np.unique(selected).size == 16
        assert np.all(np.diff(selected) > 0)
        assert len(set(base) & set(selected)) >= 14
        assert base[0] in selected and base[-1] in selected
        owner_counts = np.bincount(decision.region_owner_slots[selected], minlength=16)
        assert np.array_equal(owner_counts, np.ones(16, dtype=int))
        donor_slots = [item.donor_slot for item in decision.relocations]
        assert all(
            abs(left - right) > 1
            for index, left in enumerate(donor_slots)
            for right in donor_slots[index + 1 :]
        )


def test_joint_rule_prefers_fewer_changes_on_exact_total_gain_tie():
    lattice = np.arange(0.0, 100.0, 1.0)
    proposals = (
        _Proposal(2, 20, 22, 0.0, 0.2, 0.2, 5.0),
        _Proposal(1, 10, 12, 0.0, 0.1, 0.1, 5.0),
        _Proposal(3, 30, 32, 0.0, 0.1, 0.1, 5.0),
    )
    chosen = _choose_relocations(proposals, lattice=lattice, config=NestedR2Config())
    assert [item.donor_slot for item in chosen] == [2]


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"max_relocations": 1}, "max_relocations"),
        ({"min_relocation_distance_sec": 1.0}, "min_relocation_distance_sec"),
        ({"scoring_config": RC12Config(smoothing_sigma_sec=2.0)}, "scoring_config"),
    ],
)
def test_configuration_cannot_be_tuned_in_place(kwargs, name):
    with pytest.raises(ValueError, match=name):
        NestedR2Config(**kwargs)


def test_provenance_states_nested_control_and_no_ti_dwt():
    scout = _scout(80.0)
    decision = select_nested_r2_targets(
        scout,
        np.sin(scout / 3.0),
        support_start_sec=0.0,
        support_stop_sec=80.0,
    )
    payload = decision.to_dict(include_arrays=True)

    assert payload["method"] == "nested_r2"
    assert payload["ti_dwt_role"] == "not_used"
    assert payload["phase_effect_claim"] is False
    assert payload["coverage_contract"]["minimum_unchanged_control_slots"] == 14
    assert payload["anchor_indices"] == payload["preserved_base_indices"]
    assert set(payload["anchor_indices"]).isdisjoint(payload["residual_indices"])
    assert (
        sorted(payload["anchor_indices"] + payload["residual_indices"])
        == payload["selected_indices"]
    )


def test_existing_downstream_safety_gate_accepts_nested_r2_without_rule_changes():
    summary = {
        "stability": {
            "baseline_method": "canonical_uniform",
            "treatment_method": "phasefuse_nested_r2",
            "comparison": {
                "effect_definition": "treatment - baseline",
                "effect_order": [
                    "delta_mean_accuracy",
                    "delta_pairwise_answer_disagreement",
                ],
                "estimate": [0.0, 0.0],
                "ci_low": [-0.029, -0.01],
                "ci_high": [0.01, 0.029],
            },
        }
    }
    result = evaluate_gate(summary, "phasefuse_nested_r2")
    assert result["status"] == "pass"
    assert result["accuracy_noninferiority"]["strict_threshold"] == -0.03
    assert result["pad_safety"]["strict_threshold"] == 0.03


def test_exact_decode_adapter_rejects_nested_provenance_drift_and_labels_donors():
    lattice = list(map(float, range(40)))
    base = list(range(1, 32, 2))
    donors = [base[2], base[10]]
    anchors = [index for index in base if index not in donors]
    residuals = [6, 22]
    selected = sorted(anchors + residuals)
    row = {
        "method": "phasefuse_nested_r2",
        "lattice_timestamps_sec": lattice,
        "target_indices": selected,
        "target_timestamps_sec": [lattice[index] for index in selected],
        "quantized_scores": [0.0] * len(lattice),
        "decision_metadata": {
            "anchor_indices": anchors,
            "residual_indices": residuals,
            "base_uniform_indices": base,
            "donor_indices": donors,
        },
    }

    request = _arm(row)
    assert request.role == "nested_r2"
    assert all(
        request.candidate_sources[index] == "nested_r2_uniform_donor_candidate"
        for index in donors
    )
    corrupted = {
        **row,
        "decision_metadata": {
            **row["decision_metadata"],
            "donor_indices": donors + [base[3]],
        },
    }
    with pytest.raises(ValueError, match="base/donor provenance"):
        _arm(corrupted)
