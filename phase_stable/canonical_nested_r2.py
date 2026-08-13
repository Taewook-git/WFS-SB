"""Frozen nested-R2 canonical selector.

Nested-R2 starts from the exact K=16 canonical-uniform decision.  It may
replace at most two *interior* uniform slots with query-relevance targets, but
each replacement must stay inside the removed slot's canonical Voronoi region.
The edge slots are immutable and two replaced slots may not be adjacent.  The
policy therefore retains at least fourteen exact control timestamps and keeps
one final target in every original uniform region.

Relevance preprocessing is not reimplemented here.  It is obtained from the
frozen RC12 selector so interpolation, tie-aware percentile ranking, Gaussian
smoothing, quantisation, the origin-zero lattice, and their numerical details
remain byte-for-byte shared with RC12.  TI-DWT, visual features, MMR, phase
consensus, and downstream outcomes never enter this decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .canonical_residual import (
    RC12Config,
    select_canonical_uniform_targets,
    select_rc12_targets,
)


@dataclass(frozen=True)
class NestedR2Config:
    """One-shot, non-tunable nested-R2 contract.

    The validation deliberately rejects every value other than the frozen
    defaults.  A future policy change therefore requires a new method name and
    cannot silently tune nested-R2 after seeing confirmation outcomes.
    """

    scoring_config: RC12Config = field(default_factory=RC12Config)
    base_anchor_count: int = 16
    max_relocations: int = 2
    protect_edge_anchors: bool = True
    forbid_adjacent_relocations: bool = True
    require_strict_quantized_gain: bool = True
    min_relocation_distance_sec: float = 2.0
    coverage_tiebreak_cap_sec: float = 8.0

    def __post_init__(self) -> None:
        expected = NestedR2Config.__dataclass_fields__
        frozen_values = {
            "scoring_config": RC12Config(),
            "base_anchor_count": 16,
            "max_relocations": 2,
            "protect_edge_anchors": True,
            "forbid_adjacent_relocations": True,
            "require_strict_quantized_gain": True,
            "min_relocation_distance_sec": 2.0,
            "coverage_tiebreak_cap_sec": 8.0,
        }
        # Referencing the dataclass fields makes accidental schema drift visible
        # to coverage while the exact-value comparison freezes the policy.
        if set(expected) != set(frozen_values):
            raise RuntimeError("nested-R2 configuration schema drift")
        for name, value in frozen_values.items():
            if getattr(self, name) != value:
                raise ValueError(f"nested-R2 freezes {name}={value!r}")


@dataclass(frozen=True)
class NestedR2Relocation:
    donor_slot: int
    donor_index: int
    residual_index: int
    donor_score: float
    residual_score: float
    quantized_gain: float
    nearest_retained_anchor_sec: float

    def to_dict(self, lattice: np.ndarray) -> dict[str, Any]:
        return {
            "donor_slot": int(self.donor_slot),
            "donor_index": int(self.donor_index),
            "donor_timestamp_sec": float(lattice[self.donor_index]),
            "residual_index": int(self.residual_index),
            "residual_timestamp_sec": float(lattice[self.residual_index]),
            "donor_score": float(self.donor_score),
            "residual_score": float(self.residual_score),
            "quantized_gain": float(self.quantized_gain),
            "nearest_retained_anchor_sec": float(self.nearest_retained_anchor_sec),
        }


@dataclass(frozen=True)
class NestedR2Decision:
    config: NestedR2Config
    support_start_sec: float
    support_stop_sec: float
    scout_step_sec: float
    decision_step_sec: float
    lattice_timestamps_sec: np.ndarray
    interpolated_relevance: np.ndarray
    percentile_relevance: np.ndarray
    smoothed_relevance: np.ndarray
    quantized_scores: np.ndarray
    base_uniform_indices: np.ndarray
    region_owner_slots: np.ndarray
    preserved_base_indices: np.ndarray
    donor_indices: np.ndarray
    residual_indices: np.ndarray
    selected_indices: np.ndarray
    relocations: tuple[NestedR2Relocation, ...]

    @property
    def target_timestamps_sec(self) -> np.ndarray:
        return self.lattice_timestamps_sec[self.selected_indices]

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "method": "nested_r2",
            "config": asdict(self.config),
            "support_start_sec": float(self.support_start_sec),
            "support_stop_sec": float(self.support_stop_sec),
            "scout_step_sec": float(self.scout_step_sec),
            "decision_step_sec": float(self.decision_step_sec),
            "base_uniform_indices": self.base_uniform_indices.astype(int).tolist(),
            "base_uniform_timestamps_sec": self.lattice_timestamps_sec[
                self.base_uniform_indices
            ]
            .astype(float)
            .tolist(),
            # Alias retained for the existing exact-decode provenance adapter.
            "anchor_indices": self.preserved_base_indices.astype(int).tolist(),
            "preserved_base_indices": self.preserved_base_indices.astype(int).tolist(),
            "preserved_base_timestamps_sec": self.lattice_timestamps_sec[
                self.preserved_base_indices
            ]
            .astype(float)
            .tolist(),
            "donor_indices": self.donor_indices.astype(int).tolist(),
            "residual_indices": self.residual_indices.astype(int).tolist(),
            "residual_timestamps_sec": self.lattice_timestamps_sec[
                self.residual_indices
            ]
            .astype(float)
            .tolist(),
            "selected_indices": self.selected_indices.astype(int).tolist(),
            "target_timestamps_sec": self.target_timestamps_sec.astype(float).tolist(),
            "relocation_count": len(self.relocations),
            "unchanged_base_slot_count": int(self.preserved_base_indices.size),
            "relocations": [
                relocation.to_dict(self.lattice_timestamps_sec)
                for relocation in self.relocations
            ],
            "coverage_contract": {
                "same_uniform_voronoi_region": True,
                "edge_anchors_immutable": True,
                "adjacent_donor_slots_forbidden": True,
                "minimum_unchanged_control_slots": 14,
                "distance_relaxation_allowed": False,
            },
            "decision_signal": "frozen_rc12_smoothed_query_relevance_only",
            "ti_dwt_role": "not_used",
            "phase_effect_claim": False,
        }
        if include_arrays:
            payload["lattice_timestamps_sec"] = self.lattice_timestamps_sec.astype(
                float
            ).tolist()
            payload["region_owner_slots"] = self.region_owner_slots.astype(int).tolist()
            payload["interpolated_relevance"] = self.interpolated_relevance.astype(
                float
            ).tolist()
            payload["percentile_relevance"] = self.percentile_relevance.astype(
                float
            ).tolist()
            payload["smoothed_relevance"] = self.smoothed_relevance.astype(
                float
            ).tolist()
            payload["quantized_scores"] = self.quantized_scores.astype(float).tolist()
        return payload


@dataclass(frozen=True)
class _Proposal:
    donor_slot: int
    donor_index: int
    residual_index: int
    donor_score: float
    residual_score: float
    gain: float
    nearest_retained_anchor_sec: float


def _uniform_region_owners(
    lattice: np.ndarray, base_uniform_indices: np.ndarray
) -> np.ndarray:
    """Assign every lattice tick to its nearest A16 slot; midpoint ties go early."""

    distances = np.abs(
        lattice[:, np.newaxis] - lattice[base_uniform_indices][np.newaxis, :]
    )
    owners = np.argmin(distances, axis=1).astype(int)
    if not np.array_equal(owners[base_uniform_indices], np.arange(16)):
        raise RuntimeError("canonical-uniform anchors do not own distinct regions")
    return owners


def _best_proposal_for_slot(
    *,
    donor_slot: int,
    lattice: np.ndarray,
    base: np.ndarray,
    owners: np.ndarray,
    scores: np.ndarray,
    config: NestedR2Config,
) -> _Proposal | None:
    donor_index = int(base[donor_slot])
    donor_score = float(scores[donor_index])
    retained = np.delete(base, donor_slot)
    base_set = set(int(index) for index in base)
    candidates: list[_Proposal] = []
    for candidate_index in np.flatnonzero(owners == donor_slot):
        candidate_index = int(candidate_index)
        if candidate_index in base_set:
            continue
        donor_distance = abs(float(lattice[candidate_index] - lattice[donor_index]))
        retained_distances = np.abs(lattice[retained] - lattice[candidate_index])
        if donor_distance < config.min_relocation_distance_sec:
            continue
        if np.any(retained_distances < config.min_relocation_distance_sec):
            continue
        residual_score = float(scores[candidate_index])
        gain = residual_score - donor_score
        if gain <= 1e-12:
            continue
        nearest = float(np.min(retained_distances))
        candidates.append(
            _Proposal(
                donor_slot=donor_slot,
                donor_index=donor_index,
                residual_index=candidate_index,
                donor_score=donor_score,
                residual_score=residual_score,
                gain=gain,
                nearest_retained_anchor_sec=nearest,
            )
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda proposal: (
            proposal.residual_score,
            min(
                proposal.nearest_retained_anchor_sec,
                config.coverage_tiebreak_cap_sec,
            ),
            -proposal.residual_index,
        ),
    )


def _choose_relocations(
    proposals: Sequence[_Proposal],
    *,
    lattice: np.ndarray,
    config: NestedR2Config,
) -> tuple[_Proposal, ...]:
    """Choose the positive-gain size-0/1/2 subset with a deterministic optimum."""

    valid: list[tuple[_Proposal, ...]] = [()]
    valid.extend((proposal,) for proposal in proposals)
    for left, right in combinations(proposals, 2):
        if abs(left.donor_slot - right.donor_slot) <= 1:
            continue
        if (
            abs(float(lattice[left.residual_index] - lattice[right.residual_index]))
            < config.min_relocation_distance_sec
        ):
            continue
        valid.append(tuple(sorted((left, right), key=lambda item: item.donor_slot)))

    def priority(choice: tuple[_Proposal, ...]) -> tuple[Any, ...]:
        # Fewer perturbations win an exact total-gain tie.  Remaining ties use
        # the inherited coverage cap and then earlier physical time.
        ordered = tuple(sorted(choice, key=lambda item: item.donor_slot))
        return (
            round(sum(item.gain for item in ordered), 12),
            -len(ordered),
            round(sum(item.residual_score for item in ordered), 12),
            round(
                sum(
                    min(
                        item.nearest_retained_anchor_sec,
                        config.coverage_tiebreak_cap_sec,
                    )
                    for item in ordered
                ),
                12,
            ),
            tuple(-item.donor_slot for item in ordered),
            tuple(-item.residual_index for item in ordered),
        )

    return max(valid, key=priority)


def select_nested_r2_targets(
    scout_timestamps_sec: Sequence[float],
    relevance_scores: Sequence[float],
    *,
    support_start_sec: float,
    support_stop_sec: float,
    config: NestedR2Config | None = None,
) -> NestedR2Decision:
    """Return a coverage-safe perturbation nested inside canonical A16."""

    resolved = config or NestedR2Config()
    # This call is the single source of truth for the frozen RC12 score arrays.
    scoring = select_rc12_targets(
        scout_timestamps_sec,
        relevance_scores,
        support_start_sec=support_start_sec,
        support_stop_sec=support_stop_sec,
        config=resolved.scoring_config,
    )
    lattice, base = select_canonical_uniform_targets(
        scout_timestamps_sec,
        support_start_sec=support_start_sec,
        support_stop_sec=support_stop_sec,
        config=resolved.scoring_config,
    )
    if not np.array_equal(lattice, scoring.lattice_timestamps_sec):
        raise RuntimeError("nested-R2 scoring and canonical-uniform lattices drifted")
    if base.size != resolved.base_anchor_count:
        raise RuntimeError("nested-R2 requires exact canonical-uniform A16")

    owners = _uniform_region_owners(lattice, base)
    proposals = [
        proposal
        for donor_slot in range(1, base.size - 1)
        if (
            proposal := _best_proposal_for_slot(
                donor_slot=donor_slot,
                lattice=lattice,
                base=base,
                owners=owners,
                scores=scoring.quantized_scores,
                config=resolved,
            )
        )
        is not None
    ]
    chosen = _choose_relocations(proposals, lattice=lattice, config=resolved)
    if len(chosen) > resolved.max_relocations:
        raise RuntimeError("nested-R2 relocation budget exceeded")

    donor_indices = np.asarray(
        sorted(proposal.donor_index for proposal in chosen), dtype=int
    )
    residual_indices = np.asarray(
        sorted(proposal.residual_index for proposal in chosen), dtype=int
    )
    donor_set = set(int(index) for index in donor_indices)
    preserved = np.asarray(
        [int(index) for index in base if int(index) not in donor_set], dtype=int
    )
    selected = np.asarray(
        sorted([*preserved.tolist(), *residual_indices.tolist()]), dtype=int
    )
    relocations = tuple(
        NestedR2Relocation(
            donor_slot=proposal.donor_slot,
            donor_index=proposal.donor_index,
            residual_index=proposal.residual_index,
            donor_score=proposal.donor_score,
            residual_score=proposal.residual_score,
            quantized_gain=proposal.gain,
            nearest_retained_anchor_sec=proposal.nearest_retained_anchor_sec,
        )
        for proposal in sorted(chosen, key=lambda item: item.donor_slot)
    )

    if selected.size != 16 or np.unique(selected).size != 16:
        raise RuntimeError("nested-R2 must return sixteen distinct lattice targets")
    if preserved.size < 14 or not set(preserved) <= set(base):
        raise RuntimeError("nested-R2 must retain at least fourteen exact A16 slots")
    if set(preserved) & set(residual_indices):
        raise RuntimeError("nested-R2 anchor/residual provenance overlaps")
    for relocation in relocations:
        if owners[relocation.residual_index] != relocation.donor_slot:
            raise RuntimeError("nested-R2 residual escaped its donor region")

    return NestedR2Decision(
        config=resolved,
        support_start_sec=float(support_start_sec),
        support_stop_sec=float(support_stop_sec),
        scout_step_sec=scoring.scout_step_sec,
        decision_step_sec=scoring.decision_step_sec,
        lattice_timestamps_sec=lattice,
        interpolated_relevance=scoring.interpolated_relevance,
        percentile_relevance=scoring.percentile_relevance,
        smoothed_relevance=scoring.smoothed_relevance,
        quantized_scores=scoring.quantized_scores,
        base_uniform_indices=base,
        region_owner_slots=owners,
        preserved_base_indices=preserved,
        donor_indices=donor_indices,
        residual_indices=residual_indices,
        selected_indices=selected,
        relocations=relocations,
    )


__all__ = [
    "NestedR2Config",
    "NestedR2Decision",
    "NestedR2Relocation",
    "select_nested_r2_targets",
]
